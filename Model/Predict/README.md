# 模型預測腳本 (predictv4_SAGE.py)

本腳本為 2025 E-SUN AI 挑戰賽的主要預測程式。使用特徵工程（包含外幣特徵）和 GraphSAGE 圖神經網路模型來產生警示帳戶的預測。

此腳本的設計完全依賴 `trainv4_SAGE.py` 訓練時產生的模型檔案目錄 (Model Artifacts Directory)，以確保特徵工程、模型架構和標準化過程在訓練與預測時完全一致。

## 關鍵依賴 (Key Dependencies)

1.  Python 套件:
    * `pandas`, `numpy`, `scikit-learn` (for `joblib`)
    * `torch`
    * `torch_geometric`: 核心的圖神經網路函式庫。
2.  本地腳本 (Local Script):
    * `trainv4_SAGE.py`: 本腳本會導入 (import) `trainv4_SAGE.py` 中的特徵工程函數 (如 `agg_account_features`)。因此，這兩個檔案必須放在同一目錄下。
3.  模型檔案目錄 (Model Artifacts):
    * 您必須指定一個**已訓練完成**的模型目錄 (透過 `--model_dir` 參數)。

---

## 預測工作流程 (Prediction Workflow)

本腳本執行時，會依序完成以下步驟：

1.  載入模型檔案 (Load Artifacts):
    * 腳本會讀取 `--model_dir` 目錄下的所有必要檔案。
    * 自動載入超參數: 從 `training_summary.json` 讀取訓練時的超參數 (如 `hidden`, `dropout`, `num_layers`)，並覆蓋命令列的預設值，以確保模型架構一致。
    * 載入特徵對齊資訊: 從 `feature_columns.json` 讀取訓練時使用的「最終特徵欄位列表」，這是特徵對齊的關鍵。
2.  載入資料 (Load Data):
    * 讀取完整的交易資料 (`--transactions`)。
    * 讀取需要預測的帳戶清單 (`--acct_predict`)。
3.  重現特徵工程 (Reproduce Feature Engineering):
    * 呼叫 `trainv4_SAGE.py` 中的 `agg_account_features` 等函數，**從零開始**在完整的交易資料上建立特徵（包含外幣特徵）。
    * Plan B 模式: 如果指定了 `--use_planb`，會完全複製 Plan B 的特徵工程邏輯（使用全部交易計算特徵）。
    * 建立圖結構: 根據完整交易資料建立 `edge_index`（雙向邊）。
    * 計算圖特徵: 計算 `pagerank_approx`, `in_degree` 等圖特徵。
4.  特徵對齊與標準化 (Feature Alignment & Scaling):
    * 驗證對齊: 腳本會**嚴格比對**當前產生的特徵欄位與 `feature_columns.json` 是否一致。
    * 修正: 自動補齊缺失的特徵（設為 0）並移除多餘的特徵。
    * 排序: 確保特徵順序與訓練時完全相同。
    * 標準化: 載入 `scaler.joblib`，並使用**訓練時的平均值與標準差**來 `transform` 預測資料。
5.  模型預測 (Inference):
    * 初始化 `EnhancedGraphSAGEModel` (使用載入的超參數)。
    * 載入 `best_model.pth` 的模型權重。
    * 使用 `torch_geometric.loader.NeighborLoader` 進行小批次 (Mini-Batch) 推論，以處理大型圖資料，避免記憶體溢出。
6.  輸出結果 (Output Generation):
    * 將模型的 `sigmoid` 輸出（機率）與閾值 (`--threshold` 或 `best_threshold`) 比較，產生 `0/1` 標籤。
    * 儲存兩個檔案：一個只含 `acct` 和 `label` (符合繳交格式)，另一個包含 `prob` (供分析使用)。

---

## 檔案說明 (File Definitions)

### 1. 必要輸入 (Required Inputs)

本腳本需要三個命令列參數作為輸入：

| 參數 | 說明 | 範例 |
| `--transactions` | (必要) 完整的交易資料 `.csv` 檔案。 | `Data/acct_transaction.csv` |
| `--acct_predict` | (必要) 需預測的帳戶清單 `.csv` 檔案。 | `Data/acct_predict.csv` |
| `--model_dir` | (必要) 包含所有模型檔案的目錄。 | `outputv4_SAGE/` |

### 2. 模型目錄結構 (`--model_dir`)

`--model_dir` 參數指定的目錄**必須**包含以下由 `trainv4_SAGE.py` 產生的檔案：

* `best_model.pth`: 訓練好的 PyTorch 模型權重。
* `scaler.joblib`: 訓練好的 `StandardScaler` (或 `MinMaxScaler`)。
* `id2idx.json`: 帳戶 ID 到圖節點索引 (Node Index) 的映射。
* `feature_columns.json`: 訓練時最終使用的特徵欄位列表（已包含圖特徵），用於預測時對齊。
* `training_summary.json`: 訓練摘要，包含模型超參數 (Hyperparameters) 和最佳閾值 (best_threshold)。

### 3. 輸出檔案 (Outputs)

本腳本會產生兩個 `.csv` 檔案：

1.  主要輸出檔 (Main Output):
    * 路徑: 由 `--output` 參數指定 (例如: `outputv4_SAGE/output.csv`)。
    * 欄位: `acct`, `label`
    * 用途: 符合競賽繳交格式的最終結果。
2.  機率輸出檔 (Probability Output):
    * 路徑: 自動在 `--output` 路徑後加上 `_probs` (例如: `outputv4_SAGE/output_probs.csv`)。
    * 欄位: `acct`, `label`, `prob`
    * 用途: 包含原始預測機率，用於後續分析與模型除錯。

---

## 使用範例 (Usage Example)

```bash
# 確保 trainv4_SAGE.py 與 predictv4_SAGE.py 在同一目錄
#
# --transactions: 完整的交易資料
# --acct_predict: 要預測的帳戶清單
# --model_dir:    訓練好的 v4 SAGE 模型目錄
# --output:       最終繳交的檔案路徑
# --use_planb:    (選用) 如果 v4 模型是 Plan B 訓練的，請加上此旗標
# --threshold:    (選用) 指定分類閾值，若不指定，則自動使用模型內的 best_threshold

python predictv4_SAGE.py \
    --transactions Data/acct_transaction.csv \
    --acct_predict Data/acct_predict.csv \
    --model_dir outputv4_SAGE \
    --output outputv4_SAGE/output.csv \
    --use_planb \
    --threshold 0.25
