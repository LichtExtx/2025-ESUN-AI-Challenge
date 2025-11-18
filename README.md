An AI program for detecting suspicious (alert) bank accounts.

---

# 1. 建模概述 (Modeling Overview)

本專案旨在從帳戶的交易行為中偵測出潛在的警示帳戶。我們採用 **Graph Neural Network (GNN)**，並以 **GraphSAGE** 作為核心模型。

此版本特別強化以下特徵工程：

- 外幣交易行為  
- 資金匯集特徵  
- 交易時間熵  
- 對手方集中度 (Gini)

為了模擬真實預測場景並避免資料洩漏，我們採用了 **Plan B 訓練策略**：

### 特徵一致性  
使用全部交易資料（包含預測帳戶）建立特徵。

### 標籤隔離  
預測帳戶的標籤在訓練中被全部設為 NaN，避免洩漏。

### 結構完整性  
保留完整圖結構，使模型仍能利用其鄰居資訊。

---

# 2. 環境需求 (Environment Requirements)

### Python 版本
Python 3.10.18

### 安裝必要套件
pip install -r requirements.txt

### 主要依賴
- torch  
- torch_geometric  
- pandas  
- numpy  
- scikit-learn (joblib)

---

# 3. 專案結構 (Project Structure)
.
├── Data/
│ ├── acct_transaction.csv # 完整交易紀錄
│ ├── acct_alert.csv # 真實警示帳戶標籤
│ └── acct_predict.csv # 需預測的帳戶清單
│
├── Preprocessing/
│ ├── preprocessing.py # 階段一：特徵工程
│ ├── README.md
│ └── processed_data/
│ ├── X_scaled.npy
│ ├── y_labels.npy
│ ├── edge_index.npy
│ ├── train_mask.npy
│ ├── val_mask.npy
│ ├── scaler.joblib
│ ├── feature_columns.json
│ └── id2idx.json
│
├── Model/
│ ├── train.py # 階段二：模型訓練
│ ├── trainv4_SAGE.py # (整合版) 預處理 + 訓練
│ ├── predictv4_SAGE.py # 預測腳本
│ ├── README.md
│ └── outputv4_SAGE/
│ ├── best_model.pth
│ ├── training_summary.json
│ ├── scaler.joblib
│ ├── feature_columns.json
│ └── id2idx.json
│
├── tools/
│ ├── analyze_alert_timing.py
│ ├── analyze_compare.py
│ ├── tool_split_alert_total.py
│ ├── analyze_account.py
│ └── README.md
│
├── requirements.txt
└── README.md

---

# 4. 如何重現 (How to Reproduce)

你可選擇 **兩種完全等效的流程**：

---

# 方式 A：一鍵式執行（trainv4_SAGE.py 已含預處理）

### **Step 1 — 預處理 + 訓練（自動完成）**
python Model/trainv4_SAGE.py
--transactions Data/acct_transaction.csv
--alerts Data/acct_alert.csv
--predicts Data/acct_predict.csv
--out_dir Model/outputv4_SAGE
--use_planb

### **Step 2 — 預測**
python Model/predictv4_SAGE.py
--transactions Data/acct_transaction.csv
--acct_predict Data/acct_predict.csv
--model_dir Model/outputv4_SAGE
--output output.csv
--use_planb
---

# 方式 B：分階段執行

---

## **步驟一：預處理 (Stage 1)**

python Preprocessing/preprocessing.py
--transactions Data/acct_transaction.csv
--alerts Data/acct_alert.csv
--predicts Data/acct_predict.csv
--out_dir Preprocessing/processed_data
--use_planb
---

## **步驟二：模型訓練 (Stage 2)**

python Model/train.py
--data_dir Preprocessing/processed_data
--out_dir Model/outputv4_SAGE
--lr 0.0005
--hidden 1024
--dropout 0.5
--batch_size 512
--epochs 180
---

## **步驟三：預測 (Stage 3)**

python Model/predictv4_SAGE.py
--transactions Data/acct_transaction.csv
--acct_predict Data/acct_predict.csv
--model_dir Model/outputv4_SAGE
--output output.csv
--use_planb
--threshold 0.25
> 若未指定 `--threshold`，模型會自動使用 `training_summary.json` 的最佳閾值。

---

# 補充：兩種執行流程「效果完全相同」

無論你選擇：

### **A. 執行 trainv4_SAGE.py（整合版）**  
或  
### **B. 執行 preprocessing.py → train.py → predictv4_SAGE.py（分段版）**

最終得到的：

- `best_model.pth`
- `training_summary.json`
- 預測結果 `output.csv`

**完全一致。**

兩種只是執行方式不同，不影響模型表現。

---

# 5. 超參數設定與資源配置 (Hyperparameters & Resource Config)

### 主要超參數
- `lr`：0.0005  
- `hidden`：1024  
- `num_layers`：3  
- `dropout`：0.5  
- `weight_decay`：5e-6  
- `batch_size`：512  
- `epochs`：180（含 early stopping）  
- Loss：**Focal Loss (gamma=2)**  

### 資源需求
- **GPU**：建議 GeForce RTX 3060 系列 以上規格  
- **CPU**：預處理階段依賴 CPU  
- **RAM**：至少 4GB（需載入完整交易資料）

---

# 6. 實驗結果 (Experiment Results)

所有訓練過程（F1、AUPRC、Precision、Recall、best_threshold、超參數設定）  
均紀錄於：

Model/outputv4_SAGE/training_summary.json
