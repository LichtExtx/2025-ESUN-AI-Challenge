# 資料預處理執行腳本 (`preprocessing.py`)

本腳本為 **2025 E-SUN AI 挑戰賽** 的第一階段 (Stage 1) 預處理程式。

它的主要功能為：

* 載入原始 `.csv` 資料
* 執行 **完整特徵工程**（包含外幣特徵與圖特徵）
* 處理 **Plan B** 標籤隔離機制
* 將所有訓練所需檔案（特徵矩陣、圖結構、標籤、遮罩、標準化器等）輸出至指定目錄
* 第二階段 (`Model/train.py`) 將直接載入本腳本產生的所有結果

---

## 關鍵依賴 (Key Dependencies)

### Python 套件
* pandas  
* numpy  
* scikit-learn（joblib, RobustScaler）  
* torch（僅用於 `seed_everything`，主要計算依然是 pandas/numpy）  

### 原始資料檔案
本腳本需要以下原始輸入：
* `acct_transaction.csv`
* `acct_alert.csv`
* （Plan B 模式）`acct_predict.csv`

---

## 預處理工作流程 (Preprocessing Workflow)

執行本腳本後，將依序進行以下步驟：

1. **載入資料 (Load Data)**  
   讀取交易資料、警示帳戶標籤，以及（Plan B）預測帳戶清單。

2. **建立索引 (Build Index)**  
   建立 `id2idx`，將帳戶 ID 映射為節點索引。

3. **特徵工程 (Feature Engineering)**  
   呼叫 `agg_account_features`，計算完整特徵（含外幣特徵）。

4. **標籤處理 (Label Assignment — Plan B)**  
   * 若啟用 `--use_planb`：  
     → `predicts` 清單內帳戶之標籤會被設為 **NaN**，不會分配到 train/val。

5. **建立圖結構 (Build Graph)**  
   * 依照所有交易建立 `edge_index`
   * 計算圖特徵（例如 `in_degree`、`pagerank_approx`）

6. **特徵處理 (Feature Processing)**  
   * 移除冗餘欄位 `FEATURES_TO_REMOVE`
   * 使用 `RobustScaler` 進行特徵標準化（fit + transform）

7. **建立 Train / Val 遮罩 (Create Masks)**  
   * 基於 `trainable_mask` 進行分層抽樣  
   * 產生 `train_mask.npy`、`val_mask.npy`

8. **儲存所有檔案 (Save Artifacts)**  
   * 特徵矩陣、標籤、邊、遮罩、scaler、特徵名稱等  
   * 全部儲存至 `--out_dir`

---

## 檔案說明 (File Definitions)

### 1. 必要輸入 (Required Inputs)

| 參數 | 說明 | 範例 |
|------|------|------|
| `--transactions` | (必要) 完整交易資料 CSV | `Data/acct_transaction.csv` |
| `--alerts` | (必要) 警示帳戶資料 CSV | `Data/acct_alert.csv` |
| `--out_dir` | (必要) 預處理輸出資料夾 | `Preprocessing/processed_data` |
| `--predicts` | (Plan B) 預測清單 CSV | `Data/acct_predict.csv` |
| `--val_ratio` | 驗證集比例 (預設 0.2) | `0.2` |
| `--use_planb` | 啟用 Plan B 標籤隔離模式 | 無值 |

---

### 2. 輸出檔案 (Output Artifacts)

以下檔案將儲存在 `--out_dir`：

* **X_scaled.npy**  
  最終 scaled 特徵矩陣 `(num_nodes, num_features)`

* **y_labels.npy**  
  全部帳戶標籤

* **edge_index.npy**  
  圖的邊 `(2, num_edges)`

* **train_mask.npy**  
  訓練遮罩

* **val_mask.npy**  
  驗證遮罩

* **scaler.joblib**  
  已 fit 過之 RobustScaler

* **feature_columns.json**  
  最終特徵欄位名稱清單

* **id2idx.json**  
  帳戶 ID → 節點索引映射

* **feature_std.csv / feature_comparison.csv**  
  特徵分析報告（非模型必須，但方便人工調查）

---

## 使用範例 (Usage Example)

```bash
# 執行第一階段：預處理
# (此步驟會讀取 CSV 並產生所有訓練所需的 .npy / .json / .joblib)

python Preprocessing/preprocessing.py \
    --transactions Data/acct_transaction.csv \
    --alerts Data/acct_alert.csv \
    --predicts Data/acct_predict.csv \
    --out_dir Preprocessing/processed_data \
    --use_planb
