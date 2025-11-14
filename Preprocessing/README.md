資料預處理執行腳本 (preprocessing.py)
本腳本為 2025 E-SUN AI 挑戰賽的第一階段 (Stage 1) 預處理程式。

此腳本的唯一職責是載入原始 .csv 資料，執行 'v4' 版的完整特徵工程（包含外幣特徵、圖特徵），處理 "Plan B" 標籤隔離，並將所有用於訓練的檔案（特徵矩陣、圖結構、標籤、遮罩、標準化器等）儲存到指定的輸出目錄。

Model/train.py（第二階段）將會直接載入此腳本的輸出檔案來進行模型訓練。

關鍵依賴 (Key Dependencies)
Python 套件:

pandas, numpy, scikit-learn (for joblib, RobustScaler)

torch (僅用於 seed_everything，主要計算仍為 numpy/pandas)

原始資料檔案 (Raw Data Files):

本腳本需要 acct_transaction.csv, acct_alert.csv，以及在 Plan B 模式下的 acct_predict.csv。

預處理工作流程 (Preprocessing Workflow)
本腳本執行時，會依序完成以下步驟：

載入資料 (Load Data): 讀取 transactions (交易資料)、alerts (標籤)、以及 predicts (預測清單，Plan B 模式使用)。

建立索引 (Build Index): 建立 id2idx，將所有帳戶 ID 映射到唯一的節點索引。

特徵工程 (Feature Engineering): 呼叫 agg_account_features，在完整的交易資料上計算所有帳戶的特徵（包含外幣特徵）。

標籤分配 (Label Assignment - Plan B):

關鍵步驟: 如果啟用了 --use_planb，在 predicts 清單中的帳戶，其標籤會被設為 NaN (缺失值)，因此它們不會被劃分到訓練集或驗證集中。

建立圖結構 (Build Graph):

根據完整的交易資料建立圖的邊 (edge_index)。

計算圖特徵 (如 pagerank_approx, in_degree) 並與帳戶特徵合併。

特徵處理 (Feature Processing):

移除 FEATURES_TO_REMOVE 列表中的冗餘特徵。

使用 RobustScaler 對最終的特徵集進行擬合 (fit) 與轉換 (transform)。

建立訓練/驗證遮罩 (Create Masks):

根據 trainable_mask (在 Plan B 模式下會排除預測帳戶) 進行分層抽樣，產生 train_mask.npy 和 val_mask.npy。

儲存所有檔案 (Save Artifacts):

將所有產出 (特徵矩陣、標籤、圖、遮罩、標準化器、特徵列表) 儲存至 --out_dir。

檔案說明 (File Definitions)
1. 必要輸入 (Required Inputs)
本腳本需要以下命令列參數作為輸入：
參數,說明,範例
--transactions,(必要) 完整的交易資料 .csv 檔案。,Data/acct_transaction.csv
--alerts,(必要) 警示帳戶標籤 .csv 檔案。,Data/acct_alert.csv
--out_dir,(必要) 儲存所有預處理檔案的目錄。,Preprocessing/processed_data
--predicts,(Plan B 選用) 預測帳戶清單 .csv。,Data/acct_predict.csv
--val_ratio,(選用) 驗證集比例 (預設: 0.2)。,0.2
--use_planb,"(選用) 啟用 ""Plan B"" 標籤隔離模式。",(無值)

2. 輸出檔案 (Output Artifacts)
--out_dir 參數指定的目錄將會產生以下檔案 (供 Model/train.py 讀取)：

X_scaled.npy: (Numpy 陣列) 最終的特徵矩陣，形狀為 (num_nodes, num_features)。

y_labels.npy: (Numpy 陣列) 全部的標籤，形狀為 (num_nodes,)。

edge_index.npy: (Numpy 陣列) 完整的圖邊索引，形狀為 (2, num_edges)。

train_mask.npy: (Numpy 陣列) 訓練遮罩 (布林值)，形狀為 (num_nodes,)。

val_mask.npy: (Numpy 陣列) 驗證遮罩 (布林值)，形狀為 (num_nodes,)。

scaler.joblib: 已使用最終特徵集擬合 (fit) 過的 RobustScaler 物件。

feature_columns.json: 一個 list，包含最終特徵欄位名稱及其順序。

id2idx.json: 帳戶 ID (str) 到節點索引 (int) 的映射。

feature_std.csv / feature_comparison.csv: (分析用) 特徵分析報告。

使用範例 (Usage Example)
# 執行第一階段：預處理
# (此步驟會讀取 CSV 並產生 .npy 檔案)

python Preprocessing/preprocessing.py \
    --transactions Data/acct_transaction.csv \
    --alerts Data/acct_alert.csv \
    --predicts Data/acct_predict.csv \
    --out_dir Preprocessing/processed_data \
    --use_planb