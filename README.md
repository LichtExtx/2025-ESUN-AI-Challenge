2025-ESUN-AI-Challenge
TEAM_8302 | 2025 E.SUN AI Challenge

An AI program for detecting suspicious (alert) bank accounts.

1. 建模概述 (Modeling Overview)
本專案旨在透過帳戶的交易行為來偵測潛在的警示帳戶。我們採用了一個兩階段式 (Two-Stage) 的圖神經網路 (Graph Neural Network, GNN) 流程。

核心模型為 GraphSAGE (v4 版特徵工程)，此版本特別強化了對外幣交易、資金匯集、交易時間熵及對手方集中度 (Gini) 的特徵提取。

為了確保模型的可重現性並模擬真實世界的預測情境，我們採用了 "Plan B" 訓練策略：

特徵一致性： 使用全部交易資料（包含訓練與預測帳戶）來建立特徵。

標籤隔離： 在訓練過程中，完全不使用「待預測帳戶」的標籤，確保模型不會洩漏 (Leak) 預測集的答案。

結構完整性： 保留完整的圖結構，使 GNN 能夠學習到「待預測帳戶」周圍的鄰居資訊。

2. 環境需求 (Environment Requirements)
Python 版本
Python 3.10.18

必要套件 (Requirements)
所有必要的 Python 套件及其相容版本皆已列在 requirements.txt 中。請使用以下指令安裝：

pip install -r requirements.txt

主要依賴套件包含：

torch

torch_geometric

pandas

numpy

scikit-learn (joblib)

3. 專案結構 (Project Structure)
本專案的程式碼依照競賽要求，明確劃分為「預處理」和「模型」兩大階段。
.
├── Data/
│   ├── acct_transaction.csv    # (原始資料) 完整的交易紀錄
│   ├── acct_alert.csv          # (原始資料) 真實警示帳戶標籤
│   └── acct_predict.csv        # (原始資料) 需預測的帳戶清單
│
├── Preprocessing/
│   ├── preprocessing.py        # 【階段一】可執行的預處理腳本 (執行特徵工程)
│   ├── README.md               # (說明) 預處理腳本的詳細說明
│   └── processed_data/         # (範例輸出目錄) 存放階段一產出的檔案
│       ├── X_scaled.npy
│       ├── y_labels.npy
│       ├── edge_index.npy
│       ├── train_mask.npy
│       ├── val_mask.npy
│       ├── scaler.joblib
│       ├── feature_columns.json
│       └── id2idx.json
│
├── Model/
│   ├── train.py                # 【階段二】可執行的模型訓練腳本
│   ├── predictv4_SAGE.py       # 【預測】可執行的最終預測腳本
│   ├── README.md               # (說明) 訓練與預測腳本的詳細說明
│   └── outputv4_SAGE/          # (範例輸出目錄) 存放階段二產出的模型
│       ├── best_model.pth
│       ├── training_summary.json
│       ├── scaler.joblib         (從 processed_data/ 複製而來)
│       ├── feature_columns.json  (從 processed_data/ 複製而來)
│       └── id2idx.json           (從 processed_data/ 複製而來)
│
├── tools/
│   ├── analyze_alert_timing.py # (輔助) 分析真實警示帳戶的時間特性
│   ├── analyze_compare.py      # (輔助) 比較兩個預測檔的差異
│   ├── tool_split_alert_total.py # (輔助) 產生人工查核用的 .txt 報告
│   ├── analyze_account.py      # (輔助) 檢查帳戶重疊率
│   └── README.md               # (說明) 輔助工具的詳細說明
│
├── requirements.txt            # (環境) Python 套件需求
└── README.md                   # (本檔案) 專案總說明

好的，這是一份幫您重新排版過的 README.md 內容。

我主要運用了 Headings (標題)、Code Blocks (程式碼區塊)、Blockquotes (引言) 和 Bulleted Lists (項目符號) 來增加易讀性，同時修正了目錄結構中幾個看起來像亂碼的錯字，並將步驟編號修正。

2025-ESUN-AI-Challenge
TEAM_8302 | 2025 E.SUN AI Challenge

An AI program for detecting suspicious (alert) bank accounts.

1. 建模概述 (Modeling Overview)
本專案旨在透過帳戶的交易行為來偵測潛在的警示帳戶。我們採用了一個兩階段式 (Two-Stage) 的圖神經網路 (Graph Neural Network, GNN) 流程。

核心模型為 GraphSAGE (v4 版特徵工程)，此版本特別強化了對外幣交易、資金匯集、交易時間熵及對手方集中度 (Gini) 的特徵提取。

為了確保模型的可重現性並模擬真實世界的預測情境，我們採用了 "Plan B" 訓練策略：

特徵一致性： 使用全部交易資料（包含訓練與預測帳戶）來建立特徵。

標籤隔離： 在訓練過程中，完全不使用「待預測帳戶」的標籤，確保模型不會洩漏 (Leak) 預測集的答案。

結構完整性： 保留完整的圖結構，使 GNN 能夠學習到「待預測帳戶」周圍的鄰居資訊。

2. 環境需求 (Environment Requirements)
Python 版本
Python 3.10.18

必要套件 (Requirements)
所有必要的 Python 套件及其相容版本皆已列在 requirements.txt 中。請使用以下指令安裝：

Bash

pip install -r requirements.txt
主要依賴套件包含：

torch

torch_geometric

pandas

numpy

scikit-learn (joblib)

3. 專案結構 (Project Structure)
本專案的程式碼依照競賽要求，明確劃分為「預處理」和「模型」兩大階段。

Plaintext

.
├── Data/
│   ├── acct_transaction.csv    # (原始資料) 完整的交易紀錄
│   ├── acct_alert.csv          # (原始資料) 真實警示帳戶標籤
│   └── acct_predict.csv        # (原始資料) 需預測的帳戶清單
│
├── Preprocessing/
│   ├── preprocessing.py        # 【階段一】可執行的預處理腳本 (執行特徵工程)
│   ├── README.md               # (說明) 預處理腳本的詳細說明
│   └── processed_data/         # (範例輸出目錄) 存放階段一產出的檔案
│       ├── X_scaled.npy
│       ├── y_labels.npy
│       ├── edge_index.npy
│       ├── train_mask.npy
│       ├── val_mask.npy
│       ├── scaler.joblib
│       ├── feature_columns.json
│       └── id2idx.json
│
├── Model/
│   ├── train.py                # 【階段二】可執行的模型訓練腳本
│   ├── predictv4_SAGE.py       # 【預測】可執行的最終預測腳本
│   ├── README.md               # (說明) 訓練與預測腳本的詳細說明
│   └── outputv4_SAGE/          # (範例輸出目錄) 存放階段二產出的模型
│       ├── best_model.pth
│       ├── training_summary.json
│       ├── scaler.joblib         (從 processed_data/ 複製而來)
│       ├── feature_columns.json  (從 processed_data/ 複製而來)
│       └── id2idx.json           (從 processed_data/ 複製而來)
│
├── tools/
│   ├── analyze_alert_timing.py # (輔助) 分析真實警示帳戶的時間特性
│   ├── analyze_compare.py      # (輔助) 比較兩個預測檔的差異
│   ├── tool_split_alert_total.py # (輔助) 產生人工查核用的 .txt 報告
│   ├── analyze_account.py      # (輔助) 檢查帳戶重疊率
│   └── README.md               # (說明) 輔助工具的詳細說明
│
├── requirements.txt            # (環境) Python 套件需求
└── README.md                   # (本檔案) 專案總說明
4. 如何重現 (How to Reproduce)
請依照以下三步驟執行，即可完整重現從資料處理到最終預測的完整流程。

注意： 執行前請確保 Data/ 目錄下已放置所有競賽提供的 .csv 檔案。

步驟一：(階段一) 執行資料預處理
此步驟會讀取 Data/ 中的原始 CSV 檔案，執行 v4 特徵工程，並將處理後的檔案（.npy, .joblib 等）儲存到 Preprocessing/processed_data/ 目錄中。
# 執行預處理 (使用 Plan B 模式)
python Preprocessing/preprocessing.py \
    --transactions Data/acct_transaction.csv \
    --alerts Data/acct_alert.csv \
    --predicts Data/acct_predict.csv \
    --out_dir Preprocessing/processed_data \
    --use_planb

步驟二：(階段二) 執行模型訓練
此步驟會讀取 Preprocessing/processed_data/ 中的檔案，進行模型訓練，並將最佳模型 (best_model.pth) 和訓練摘要 (training_summary.json) 儲存到 Model/outputv4_SAGE/ 目錄中。

# 執行模型訓練
python Model/train.py \
    --data_dir Preprocessing/processed_data \
    --out_dir Model/outputv4_SAGE \
    --lr 0.0005 \
    --hidden 1024 \
    --dropout 0.5 \
    --batch_size 512 \
    --epochs 180

(超參數可自行調整，此處為範例)

步驟三：(預測) 產生最終繳交檔案
此步驟會載入 Model/outputv4_SAGE/ 中訓練好的模型，對 Data/acct_predict.csv 清單中的帳戶進行預測，並產生最終的 output.csv 繳交檔案。

# 執行預測 (使用 Plan B 模式)
python Model/predictv4_SAGE.py \
    --transactions Data/acct_transaction.csv \
    --acct_predict Data/acct_predict.csv \
    --model_dir Model/outputv4_SAGE \
    --output output.csv \
    --use_planb \
    --threshold 0.25
(--threshold 參數為選填，若不指定，將自動使用 training_summary.json 中的 best_threshold)

5. 超參數設定及資源配置 (Hyperparameters & Resource Config)
超參數 (Hyperparameters)
主要的模型超參數可在 Model/train.py 腳本的命令列參數中設定，或在 Model/outputv4_SAGE/training_summary.json 中查看：

--lr (學習率): 0.0005

--hidden (隱藏層維度): 1024

--num_layers (GNN 層數): 3

--dropout (Dropout 率): 0.5

--weight_decay (權重衰減): 5e-6

--batch_size (批次大小): 512

--epochs (最大輪數): 180 (搭配 patience=20 的早停)

Loss Function: Focal Loss (alpha 自動計算, gamma=2.0)

資源配置 (Resource Configuration)
GPU: 本模型使用 torch_geometric，建議在配有 NVIDIA GPU (例如 V100, A100, or RTX 3080+) 的環境中執行訓練與預測。

CPU: 預處理 (preprocessing.py) 階段主要依賴 CPU 和 RAM。

RAM: 由於特徵工程和圖建立需要載入完整交易資料，建議配備至少 64GB 以上的記憶體。

6. 實驗結果 (Experiment Results)
本模型採用 "Plan B" 策略進行訓練。訓練過程中的驗證集 (Validation Set) 指標 (F1, AUPRC, Precision, Recall) 以及最終選用的 best_threshold (最佳閾值)，皆已詳細記錄於 Model/outputv4_SAGE/training_summary.json 檔案中。
