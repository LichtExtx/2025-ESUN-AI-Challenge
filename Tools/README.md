# 分析工具 (Analysis Tools)

此資料夾包含的 Python 腳本是 E-SUN AI 挑戰賽專案中使用的輔助工具。

它們主要用於：
* 資料探索 (EDA): 深入分析原始資料的特性。
* 模型比較 (Validation): 比較不同模型版本預測結果的差異。
* 人工查核 (Manual Review): 產生易於閱讀的報告以供人工審閱。

---

## 腳本說明

### 1. `analyze_alert_timing.py`

* 功能描述:
    用於深度分析「真實警示帳戶 (`acct_alert.csv`)」的行為與時間特性。
* 核心功能:
    1.  時間差分析: 計算每個警示帳戶的「最後交易日期」與「通報日期 (event_date)」之間的時間差。
    2.  延遲分組: 根據時間差將帳戶分為「同日通報」、「1-10天」、「11-30天」、「>60天」等群組。
    3.  特徵分析: 分析不同延遲組別的交易特徵（例如：平均金額、交易頻率、夜間交易比例、帳戶壽命等）。
    4.  異常偵測: 偵測例如「高交易密度」、「短壽命帳戶」或「無交易紀錄」等異常模式。
* 使用方式:
    ```bash
    python analyze_alert_timing.py --transactions /path/to/acct_transaction.csv --alerts /path/to/acct_alert.csv --output_dir alert_analysis_results
    ```
* 輸出:
    在指定的 `--output_dir` (例如 `alert_analysis_results/`) 目錄下生成多份 `.csv` 分析報告。

---

### 2. `analyze_compare.py`

* 功能描述:
    此工具用於比較「兩個不同版本」的模型預測輸出檔案 (`.csv`)。
* 核心功能:
    1.  讀取兩個指定的預測檔 (v1, v2)。
    2.  找出並分別儲存「僅 v1 預測為 1」、「僅 v2 預測為 1」以及「v1 和 v2 皆預測為 1」的帳戶清單。
    3.  這對於模型迭代時，分析新舊版本之間的差異非常有用。
* 使用方式:
    1.  手動修改此腳本內部的變數：
        * `v1_path`: 設定第一個預測檔的路徑。
        * `v2_path`: 設定第二個預測檔的路徑。
        * `output_prefix`: 設定輸出檔名的前綴 (例如 `compare_results`)。
    2.  執行腳本：
        ```bash
        python analyze_compare.py
        ```
* 輸出:
    在當前目錄（或 `output_prefix` 指定的路徑）下生成 3 個 `.csv` 檔案 (例如 `compare_results_only_v1.csv`...)。

---

### 3. `tool_split_alert_total.py`

* 功能描述:
    此工具用於產生一份「人工查核」用的總報告（`.txt` 檔）。
* 核心功能:
    1.  讀取一份「模型預測的警示帳戶清單」(例如 `acct_predict.csv`)。
    2.  讀取「完整的交易紀錄 (`acct_transaction.csv`)」。
    3.  針對清單上的**每一個**預測帳戶，找出其**所有**相關交易（轉入或轉出）。
    4.  將每個帳戶的交易按日期排序，並以分隔線區隔，全部寫入一個 `.txt` 檔案中，方便人工逐一審閱其完整的交易歷程。
* 使用方式:
    1.  手動修改此腳本內部的檔案路徑變數（`alerts_file`, `transactions_file`, `output_file`）。
    2.  執行腳本：
        ```bash
        python tool_split_alert_total.py
        ```
* 輸出:
    生成一個 `.txt` 檔案 (例如 `Data/all_predict_accounts.txt`)。

---

### 4. `analyze_account.py`

* 功能描述:
    這是一個用於快速資料勘查 (Sanity Check) 的腳本。
* 核心功能:
    1.  讀取「交易資料」和「真實警示帳戶資料」。
    2.  統計並印出：交易資料中的總帳戶數、警示帳戶的總數。
    3.  計算並印出：有多少比例的「警示帳戶」是存在於交易資料中的，又有多少是不存在的（即沒有任何交易紀錄）。
* 使用方式:
    1.  手動修改此腳本內部的檔案路徑。
    2.  執行腳本：
        ```bash
        python analyze_account.py
        ```
* 輸出:
    直接在終端機 (Console) 印出統計數字。
