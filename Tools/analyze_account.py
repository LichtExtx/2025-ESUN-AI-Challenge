import pandas as pd

trans_df = pd.read_csv('Data/acct_transaction.csv')
alerts_df = pd.read_csv('Data/acct_alert.csv')

trans_accts = set(pd.unique(trans_df[['from_acct', 'to_acct']].values.ravel()))
trans_accts = set(str(a) for a in trans_accts)

alert_accts = set(alerts_df['acct'].astype(str).values)

print(f"交易中的帳戶數: {len(trans_accts)}")
print(f"警示帳戶總數: {len(alert_accts)}")
print(f"在交易中的警示帳戶: {len(alert_accts & trans_accts)}")
print(f"不在交易中的警示帳戶: {len(alert_accts - trans_accts)}")

# 採樣檢查
print("\n警示帳戶樣本:")
print(list(alert_accts)[:10])
print("\n交易帳戶樣本:")
print(list(trans_accts)[:10])