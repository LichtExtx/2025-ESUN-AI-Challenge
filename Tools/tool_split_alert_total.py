import os
import pandas as pd

# === 檔案設定 ===
alerts_file = "Data/acct_predict.csv"            # 警示帳戶清單
transactions_file = "Data/acct_transaction.csv"  # 交易紀錄
output_file = "Data/all_predict_accounts.txt"      # 輸出總檔案

# === 讀取資料 ===
alerts_df = pd.read_csv(alerts_file)
trans_df = pd.read_csv(transactions_file)

# 確保欄位格式一致
alerts_df['acct'] = alerts_df['acct'].astype(str)
trans_df['from_acct'] = trans_df['from_acct'].astype(str)
trans_df['to_acct'] = trans_df['to_acct'].astype(str)

# === 開始輸出 ===
with open(output_file, "w", encoding="utf-8") as f_out:
    for i, acct in enumerate(alerts_df['acct'].unique(), start=1):
        acct_txn = trans_df[
            (trans_df['from_acct'] == acct) | (trans_df['to_acct'] == acct)
        ].copy()
        
        # 依交易日期排序
        if 'txn_date' in acct_txn.columns:
            acct_txn = acct_txn.sort_values(by='txn_date')
        
        # 寫入分隔區塊
        f_out.write("=" * 80 + "\n")
        f_out.write(f"預測帳戶 {i}/{len(alerts_df)}: {acct}\n")
        f_out.write("=" * 80 + "\n")
        
        # 若沒有交易紀錄
        if acct_txn.empty:
            f_out.write("(此帳戶無交易紀錄)\n\n")
            continue
        
        # 寫入欄位名稱 + 每筆交易（保持原始格式）
        f_out.write(",".join(acct_txn.columns) + "\n")
        for _, row in acct_txn.iterrows():
            row_str = ",".join([str(v) for v in row.values])
            f_out.write(row_str + "\n")
        
        f_out.write("\n\n")  # 區隔帳戶之間
    
print(f"完成！共輸出 {len(alerts_df['acct'].unique())} 個帳戶的交易紀錄。")
print(f"檔案位置：{os.path.abspath(output_file)}")