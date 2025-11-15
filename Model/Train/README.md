# 模型訓練腳本 (`trainv4_SAGE.py`)

本腳本為 **2025 E-SUN AI 挑戰賽** 的核心模型訓練程式，負責使用 **GraphSAGE（v4 版）** 對帳戶交易圖進行訓練，並輸出可供 `predictv4_SAGE.py` 使用的完整模型資料夾（Model Artifacts Directory）。

---

## 關鍵依賴 (Key Dependencies)

### 1. Python 套件
* pandas, numpy  
* scikit-learn（joblib, RobustScaler）  
* torch  
* torch_geometric（圖神經網路核心函式庫）

### 2. 原始資料檔案
模型訓練腳本需使用：
* `acct_transaction.csv`
* `acct_alert.csv`
* （Plan B）`acct_predict.csv`

---

## 訓練工作流程 (Training Workflow)

執行本腳本後將依序完成：

1. **載入資料 (Load Data)**  
   讀取 `transactions`、`alerts`，以及（Plan B）`predicts`。

2. **建立索引 (Build Index)**  
   建立 `id2idx` 將帳戶 ID 映射至節點索引。

3. **特徵工程 (Feature Engineering)**  
   呼叫 `agg_account_features` 計算帳戶特徵（含外幣特徵）。

4. **標籤處理 (Label Assignment - Plan B)**  
   * 若啟用 `--use_planb`：  
     → `predicts` 清單內所有帳戶的標籤被設為 **NaN**（訓練及驗證均不使用）。

5. **建立圖結構 (Build Graph)**  
   * 使用交易紀錄建立 `edge_index`  
   * 計算 `pagerank_approx`, `in_degree` 等圖特徵並加入帳戶特徵

6. **特徵處理 (Feature Processing)**  
   * 移除 `FEATURES_TO_REMOVE`  
   * 使用 `RobustScaler` 進行 `fit` + `transform`

7. **儲存前置模型資料 (Save Artifacts)**  
   * 儲存 `scaler.joblib`, `feature_columns.json`, `id2idx.json`

8. **劃分資料集 (Split Data)**  
   * 使用 `trainable_mask`  
   * 進行分層抽樣產生 `train_mask.npy`, `val_mask.npy`

9. **模型訓練 (Model Training)**  
   * 初始化 `EnhancedGraphSAGEModel`  
   * AdamW 優化器  
   * FocalLoss  
   * 使用 `NeighborLoader` 進行 mini-batch 訓練  
   * 驗證使用 `evaluate_with_auprc`  
   * 根據最高 F1-Score 儲存最佳模型  
   * 內含 early stopping（`patience`）與 ReduceLROnPlateau

10. **儲存訓練摘要 (Save Summary)**  
    儲存 `training_summary.json`：  
    * F1, AUPRC  
    * best_threshold  
    * 模型超參數  
    * FocalLoss 參數

---

## Plan B 模式 (`--use_planb`)

此模式是比賽核心策略，確保預測帳戶不會洩漏標籤資訊。

### 1. 特徵一致性 (Feature Consistency)
* 使用 **完整交易資料**（不論帳戶是否為預測對象）計算特徵  
* 確保所有帳戶的特徵來源一致

### 2. 標籤隔離 (Label Isolation)
* `acct_predict.csv` 內所有帳戶 → 標籤設為 NaN  
* 這些帳戶不會被分到 train/val  
* 模型訓練 **永不會看到這些帳戶的真實標籤**

### 3. 圖結構完整性 (Graph Integrity)
* 預測帳戶依然會被保留於圖中  
* GNN 可透過鄰居採樣取得其周邊結構  
* 但預測帳戶本身 **不參與 loss**

---

## 結論

**Plan B 模式會讓模型在「不知道預測帳戶標籤」的前提下，仍能利用完整的圖訊息與特徵分布進行學習。**

---

## 檔案說明 (File Definitions)

### 1. 命令列參數 (Arguments)

| 參數 | 說明 | 範例 |
|------|------|------|
| `--transactions` | (必要) 交易資料 CSV | Data/acct_transaction.csv |
| `--alerts` | (必要) 標籤資料 CSV | Data/acct_alert.csv |
| `--out_dir` | (必要) 模型輸出目錄 | outputv4_SAGE/ |
| `--predicts` | (Plan B) 預測清單 | Data/acct_predict.csv |
| `--epochs` | 訓練輪數（預設 200） | 200 |
| `--lr` | 學習率（預設 0.0005） | 0.0005 |
| `--hidden` | 隱藏層維度（預設 1024） | 1024 |
| `--num_layers` | GraphSAGE 層數（預設 3） | 3 |
| `--dropout` | Dropout 比例（預設 0.5） | 0.5 |
| `--batch_size` | 訓練批次大小（預設 512） | 512 |
| `--val_ratio` | 驗證集比例（預設 0.2） | 0.2 |
| `--patience` | 早停耐心值（預設 20） | 20 |
| `--use_planb` | 啟用 Plan B 模式 | 無值 |

---

## 2. 輸出檔案 (Output Artifacts)

`--out_dir` 內將生成：

* **best_model.pth**  
  * 儲存最佳模型權重（`state_dict`）  
  * best_threshold  
  * 當次最佳 F1 / AUPRC

* **scaler.joblib**  
  已 fit 完成的 `RobustScaler`，供預測端使用。

* **feature_columns.json**  
  最終訓練使用的特徵欄位順序（預測時對齊必須一致）。

* **id2idx.json**  
  帳戶 ID → 節點索引映射表。

* **training_summary.json**  
  內容包含：
  * 訓練模式  
  * F1、AUPRC  
  * best_threshold  
  * 訓練參數  
  * FocalLoss 設定

* **feature_std.csv / feature_comparison.csv**  
  （分析用）特徵標準差與警示/正常帳戶分布比較。

---

## 使用範例 (Usage Example)

```bash
# 執行 Plan B 模式訓練
# (需提供 predicts 參數)

python trainv4_SAGE.py \
    --transactions Data/acct_transaction.csv \
    --alerts Data/acct_alert.csv \
    --predicts Data/acct_predict.csv \
    --out_dir outputv4_SAGE \
    --lr 0.0005 \
    --hidden 1024 \
    --dropout 0.5 \
    --batch_size 512 \
    --epochs 180 \
    --use_planb
