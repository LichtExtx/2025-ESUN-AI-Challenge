import pandas as pd

# === 輸入你要比較的兩個檔案 ===
v1_path = "outputfile/output_193_0.428.csv"#"outputv2_SAGE/output.csv"
v2_path = "outputfile/output_206_0.434.csv"#outputv2_SAGE/output0.3.csv"
output_prefix = "compare_results"  # 結果輸出檔名前綴

# === 讀取兩份預測結果 ===
v1 = pd.read_csv(v1_path)
v2 = pd.read_csv(v2_path)

# === 確保格式正確 ===
if not {'acct', 'label'}.issubset(v1.columns) or not {'acct', 'label'}.issubset(v2.columns):
    raise ValueError("CSV 檔案必須包含 'acct' 和 'label' 欄位")

# === 合併比較 ===
merged = v1.merge(v2, on='acct', suffixes=('_v1', '_v2'))

# === 統計差異 ===
only_v1 = merged[(merged['label_v1'] == 1) & (merged['label_v2'] == 0)]
only_v2 = merged[(merged['label_v1'] == 0) & (merged['label_v2'] == 1)]
both_v1_v2 = merged[(merged['label_v1'] == 1) & (merged['label_v2'] == 1)]

print(f"差異統計：")
print(f" 版本1獨有的警示帳戶數：{len(only_v1)}")
print(f" 版本2獨有的警示帳戶數：{len(only_v2)}")
print(f" 兩版本皆為警示帳戶數：{len(both_v1_v2)}")

# === 儲存結果方便人工比對 ===
only_v1.to_csv(f"{output_prefix}_only_v1.csv", index=False)
only_v2.to_csv(f"{output_prefix}_only_v2.csv", index=False)
both_v1_v2.to_csv(f"{output_prefix}_common_positive.csv", index=False)

print(f"\n已輸出：")
print(f" - {output_prefix}_only_v1.csv   （只在版本1為1）")
print(f" - {output_prefix}_only_v2.csv   （只在版本2為1）")
print(f" - {output_prefix}_common_positive.csv （兩者皆為1）")
