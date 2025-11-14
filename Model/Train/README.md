# 模型訓練腳本 (trainv4_SAGE.py)

本腳本為 2025 E-SUN AI 挑戰賽的核心模型訓練程式。它負責從原始交易資料中學習並建立一個 GraphSAGE（v4 特徵版）模型。

此腳本的最終產出是一個包含所有必要檔案的模型目錄 (Model Artifacts Directory)，該目錄將被 predictv4_SAGE.py 腳本用於執行預測。

## 關鍵依賴 (Key Dependencies)
1. Python 套件:
    * pandas, numpy, scikit-learn (for joblib, RobustScaler)
    * torch
    * torch_geometric: 核心的圖神經網路函式庫。

2. 原始資料檔案 (Raw Data Files):
    * 訓練腳本需要 acct_transaction.csv, acct_alert.csv，以及在 Plan B 模式下的 acct_predict.csv。

## 訓練工作流程 (Training Workflow)
本腳本執行時，會依序完成以下步驟：

1. 載入資料 (Load Data): 讀取 transactions (交易資料)、alerts (標籤)、以及 predicts (預測清單，Plan B 模式使用)。

2. 建立索引 (Build Index): 建立 id2idx，將所有帳戶 ID 映射到唯一的節點索引。

3. 特徵工程 (Feature Engineering): 呼叫 agg_account_features，在完整的交易資料上計算所有帳戶的特徵（包含外幣特徵）。

4. 標籤分配 (Label Assignment - Plan B):

*   關鍵步驟: 如果啟用了 --use_planb，在 predicts 清單中的帳戶，其標籤會被設為 NaN (缺失值)，因此它們不會被劃分到訓練集或驗證集中。

5. 建立圖結構 (Build Graph):

*   根據完整的交易資料建立圖的邊 (edge_index)。

*   計算圖特徵 (如 pagerank_approx, in_degree) 並與帳戶特徵合併。

6. 特徵處理 (Feature Processing):

*   移除 FEATURES_TO_REMOVE 列表中的冗餘特徵。

*   使用 RobustScaler 對最終的特徵集進行擬合 (fit) 與轉換 (transform)。

7. 儲存模型檔案 (Save Artifacts):

*   立即儲存 scaler.joblib (已擬合), feature_columns.json (最終特徵列表), 和 id2idx.json。

8. 劃分資料集 (Split Data): 根據 trainable_mask (在 Plan B 模式下會排除預測帳戶) 進行分層抽樣，劃分 train_mask 和 val_mask。

9. 模型訓練 (Model Training):

*   初始化 EnhancedGraphSAGEModel、AdamW 優化器和 FocalLoss。

*   使用 NeighborLoader 進行小批次 (mini-batch) 訓練。

*   使用 evaluate_with_auprc 評估驗證集，並根據 F1-Score 儲存最佳模型。

*   包含早停 (patience) 和學習率調度 (ReduceLROnPlateau)。

10. 儲存摘要 (Save Summary): 將最佳指標 (F1, AUPRC, best_threshold) 和模型超參數存入 training_summary.json。

## "Plan B" 訓練模式 (--use_planb)
這是此訓練腳本的關鍵模式，專為確保預測的公平性與一致性而設計。

當啟用 --use_planb 時：

1. 特徵一致性 (Feature Consistency):

*   會使用全部的交易資料 (acct_transaction.csv) 來計算所有帳戶的特徵。

*   這確保了「待預測帳戶」的特徵與「訓練帳戶」的特徵是在相同的資料基礎上計算的。

2. 標籤隔離 (Label Isolation):

*   acct_predict.csv 清單中的所有帳戶，其標籤會被強制設為 NaN。

*   這導致它們在劃分訓練/驗證集時被排除在外 (trainable_mask = False)。

*   結果: 模型在訓練過程中絕對不會看到預測帳戶的真實標籤（無論它們是否為警示帳戶）。

3. 圖結構完整性 (Graph Integrity):

*   在建立圖時，會保留預測帳戶與其他帳戶之間的所有邊。

*   這允許 GNN 模型在訓練時，仍然可以透過鄰居採樣 (Neighbor Sampling) 來學習預測帳戶周圍的圖結構資訊，即使預測帳戶本身不參與 Loss 計算。

## 結論： Plan B 模式訓練出的模型，是在不知道預測帳戶標籤的情況下，學習如何利用完整的圖結構和特徵分布。

## 檔案說明 (File Definitions)

1. 命令列參數 (Arguments)
本腳本需要以下命令列參數作為輸入：
參數,說明,範例
--transactions,(必要) 完整的交易資料 .csv 檔案。,Data/acct_transaction.csv
--alerts,(必要) 警示帳戶標籤 .csv 檔案。,Data/acct_alert.csv
--out_dir,(必要) 輸出模型檔案的目錄。,outputv4_SAGE/
--predicts,(Plan B 選用) 預測帳戶清單 .csv。,Data/acct_predict.csv
--epochs,(選用) 訓練輪數 (預設: 200),200
--lr,(選用) 學習率 (預設: 0.0005),0.0005
--hidden,(選用) GNN 隱藏層維度 (預設: 1024),1024
--num_layers,(選用) GNN 層數 (預設: 3),3
--dropout,(選用) Dropout 率 (預設: 0.5),0.5
--batch_size,(選用) 訓練批次大小 (預設: 512),512
--val_ratio,(選用) 驗證集比例 (預設: 0.2),0.2
--patience,(選用) 早停耐心值 (預設: 20),20
--use_planb,"(選用) 啟用 ""Plan B"" 標籤隔離模式",(無值)

2. 輸出檔案 (Output Artifacts)
--out_dir 參數指定的目錄將會產生以下檔案 (供 predictv4_SAGE.py 使用)：

best_model.pth: 儲存驗證集 F1-Score 最高的模型權重 (state_dict)、最佳閾值 (best_threshold) 和當時的指標。

scaler.joblib: 已使用最終特徵集 (X_final) 擬合 (fit) 過的 RobustScaler 物件。

feature_columns.json: 一個 list，包含最終用於訓練的特徵欄位名稱及其順序。這是確保預測時特徵對齊的關鍵檔案。

id2idx.json: 帳戶 ID (str) 到圖節點索引 (Node Index) 的映射。

training_summary.json: 包含訓練模式、最佳指標 (F1, AUPRC)、模型超參數和 Focal Loss 參數的 JSON 檔案。

feature_std.csv / feature_comparison.csv: (分析用) 特徵的標準差與警示/正常帳戶的均值對比。

使用範例 (Usage Example)
# 執行 Plan B 模式訓練
# (確保 `predicts` 參數已指定)

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