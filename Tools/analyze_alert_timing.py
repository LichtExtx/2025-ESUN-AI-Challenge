#!/usr/bin/env python3
"""
整合版警示帳戶時間與特徵分析

功能:
1. 分析警示帳戶通報時間差 (最後交易 vs 通報日期)
2. 分析不同延遲組別的交易特徵
3. 偵測異常模式
4. 生成完整統計報告

使用方式:
python analyze_alert_timing.py --transactions Data/acct_transaction.csv --alerts Data/acct_alert.csv --output_dir alert_analysis_results
"""

import argparse
import os
import pandas as pd
import numpy as np
from collections import defaultdict

# ============================================================================
# Part 1: 時間差分析 (原 analyze_alert_timing.py)
# ============================================================================

def analyze_alert_timing(alerts_df, trans_df):
    """分析警示帳戶的通報時間"""
    print("=" * 80)
    print("Step 1: 警示帳戶通報時間分析")
    print("=" * 80)
    
    results = []
    print(f"\n開始分析 {len(alerts_df):,} 個警示帳戶...")
    
    for idx, row in alerts_df.iterrows():
        acct = str(row['acct'])
        event_date = int(row['event_date'])
        
        # 找出該帳戶的所有交易
        acct_txns = trans_df[
            (trans_df['from_acct'] == acct) | (trans_df['to_acct'] == acct)
        ].copy()
        
        if len(acct_txns) == 0:
            results.append({
                'acct': acct,
                'event_date': event_date,
                'last_txn_date': None,
                'first_txn_date': None,
                'total_txns': 0,
                'days_since_last_txn': None,
                'time_gap_category': 'NO_TRANSACTION',
                'account_lifespan': 0,
            })
            continue
        
        # 找出第一筆和最後一筆交易
        first_txn_date = acct_txns['txn_date'].min()
        last_txn_date = acct_txns['txn_date'].max()
        total_txns = len(acct_txns)
        
        # 計算時間差
        days_since_last = event_date - last_txn_date
        account_lifespan = last_txn_date - first_txn_date + 1
        
        # 分類
        if days_since_last < 0:
            category = 'FUTURE_TXN'
        elif days_since_last == 0:
            category = 'SAME_DAY'
        elif days_since_last <= 10:
            category = '0-10_DAYS'
        elif days_since_last <= 30:
            category = '11-30_DAYS'
        elif days_since_last <= 60:
            category = '31-60_DAYS'
        else:
            category = '>60_DAYS'
        
        results.append({
            'acct': acct,
            'event_date': event_date,
            'first_txn_date': first_txn_date,
            'last_txn_date': last_txn_date,
            'total_txns': total_txns,
            'days_since_last_txn': days_since_last,
            'time_gap_category': category,
            'account_lifespan': account_lifespan,
        })
        
        if (idx + 1) % 100 == 0:
            print(f"  進度: {idx+1:,}/{len(alerts_df):,}")
    
    result_df = pd.DataFrame(results)
    return result_df


def print_timing_statistics(result_df):
    """列印時間差統計"""
    print("\n" + "=" * 80)
    print("時間差統計結果")
    print("=" * 80)
    
    # 時間差分布
    print("\n【時間差分布】(通報日期 - 最後交易日期):")
    category_counts = result_df['time_gap_category'].value_counts()
    
    category_order = [
        'FUTURE_TXN', 'SAME_DAY', '0-10_DAYS', '11-30_DAYS', 
        '31-60_DAYS', '>60_DAYS', 'NO_TRANSACTION'
    ]
    
    total = len(result_df)
    for cat in category_order:
        if cat in category_counts.index:
            count = category_counts[cat]
            pct = count / total * 100
            print(f"  {cat:20s}: {count:>6,} ({pct:>5.1f}%)")
    
    # 關鍵統計
    print("\n【關鍵發現】")
    
    # 1. 超過10天
    over_10 = result_df[result_df['days_since_last_txn'] > 10]
    print(f"\n 延遲 > 10 天:")
    print(f"   帳戶數: {len(over_10):,} / {total:,} ({len(over_10)/total*100:.1f}%)")
    if len(over_10) > 0:
        print(f"   平均延遲: {over_10['days_since_last_txn'].mean():.1f} 天")
        print(f"   最大延遲: {over_10['days_since_last_txn'].max():.0f} 天")
    
    # 2. 超過30天
    over_30 = result_df[result_df['days_since_last_txn'] > 30]
    print(f"\n 延遲 > 30 天:")
    print(f"   帳戶數: {len(over_30):,} / {total:,} ({len(over_30)/total*100:.1f}%)")
    if len(over_30) > 0:
        print(f"   平均延遲: {over_30['days_since_last_txn'].mean():.1f} 天")
    
    # 3. 超過60天
    over_60 = result_df[result_df['days_since_last_txn'] > 60]
    print(f"\n 延遲 > 60 天 (極端延遲):")
    print(f"   帳戶數: {len(over_60):,} / {total:,} ({len(over_60)/total*100:.1f}%)")
    if len(over_60) > 0:
        print(f"   平均延遲: {over_60['days_since_last_txn'].mean():.1f} 天")
        print(f"\n   前5個最長延遲:")
        top_delays = over_60.nlargest(5, 'days_since_last_txn')
        for _, row in top_delays.iterrows():
            print(f"     {row['acct'][:16]}... | 延遲 {row['days_since_last_txn']:.0f} 天")
    
    # 4. 帳戶活躍度
    active = result_df[result_df['total_txns'] > 0]
    if len(active) > 0:
        print(f"\n【帳戶活躍度統計】")
        print(f"   平均交易數: {active['total_txns'].mean():.1f} 筆")
        print(f"   平均壽命: {active['account_lifespan'].mean():.1f} 天")
        print(f"   中位數交易數: {active['total_txns'].median():.0f} 筆")
        print(f"   中位數壽命: {active['account_lifespan'].median():.0f} 天")
    
    # 5. 時間差數值分布
    valid_gaps = result_df[result_df['days_since_last_txn'].notna()]
    if len(valid_gaps) > 0:
        print(f"\n【時間差數值分布】")
        print(f"   平均: {valid_gaps['days_since_last_txn'].mean():.1f} 天")
        print(f"   中位數: {valid_gaps['days_since_last_txn'].median():.1f} 天")
        print(f"   25%: {valid_gaps['days_since_last_txn'].quantile(0.25):.0f} 天")
        print(f"   75%: {valid_gaps['days_since_last_txn'].quantile(0.75):.0f} 天")


# ============================================================================
# Part 2: 延遲組別特徵分析 (原 tool_feature_analysis.py)
# ============================================================================

def analyze_temporal_patterns(trans_df, timing_df):
    """分析不同延遲組別的交易特徵"""
    print("\n" + "=" * 80)
    print("Step 2: 延遲組別特徵分析")
    print("=" * 80)
    
    # 定義延遲組別
    timing_df['delay_group'] = pd.cut(
        timing_df['days_since_last_txn'],
        bins=[-1, 0, 10, 30, 60, 1000],
        labels=['同日通報', '1-10天', '11-30天', '31-60天', '>60天']
    )
    
    results = []
    
    for group_name, group_df in timing_df.groupby('delay_group'):
        print(f"\n分析 [{group_name}] 組別...")
        print(f"  帳戶數: {len(group_df)}")
        
        group_accounts = set(group_df['acct'].values)
        
        # 該組帳戶的所有交易
        group_txns = trans_df[
            (trans_df['from_acct'].isin(group_accounts)) |
            (trans_df['to_acct'].isin(group_accounts))
        ].copy()
        
        if len(group_txns) == 0:
            continue
        
        # 特徵 1: 活躍度
        avg_txn_per_acct = len(group_txns) / len(group_accounts)
        
        # 特徵 2: 金額分布
        avg_amount = group_txns['txn_amt'].mean()
        med_amount = group_txns['txn_amt'].median()
        large_txn_ratio = (group_txns['txn_amt'] >= 50000).sum() / len(group_txns)
        
        # 特徵 3: 外幣使用
        foreign_ratio = (group_txns['currency_type'] != 'TWD').sum() / len(group_txns)
        
        # 特徵 4: 重複金額
        amount_counts = group_txns['txn_amt'].round(2).value_counts()
        duplicate_amounts = (amount_counts > 1).sum()
        max_duplicate = amount_counts.max() if len(amount_counts) > 0 else 0
        
        # 特徵 5: 時間集中度
        night_ratio = 0
        if 'txn_time' in group_txns.columns:
            def parse_hour(x):
                try:
                    if pd.isna(x):
                        return 0
                    return int(str(x).split(':')[0])
                except:
                    return 0
            
            hours = group_txns['txn_time'].apply(parse_hour)
            night_ratio = ((hours >= 22) | (hours <= 6)).sum() / len(hours) if len(hours) > 0 else 0
        
        # 特徵 6: 帳戶壽命
        avg_lifespan = group_df['account_lifespan'].mean()
        
        # 特徵 7: 交易密度
        total_lifespan = group_df['account_lifespan'].sum()
        avg_density = group_df['total_txns'].sum() / total_lifespan if total_lifespan > 0 else 0
        
        results.append({
            'delay_group': group_name,
            'account_count': len(group_accounts),
            'avg_txn_per_acct': avg_txn_per_acct,
            'avg_amount': avg_amount,
            'median_amount': med_amount,
            'large_txn_ratio': large_txn_ratio,
            'foreign_ratio': foreign_ratio,
            'duplicate_amounts': duplicate_amounts,
            'max_duplicate': max_duplicate,
            'night_txn_ratio': night_ratio,
            'avg_lifespan': avg_lifespan,
            'avg_txn_density': avg_density,
        })
    
    result_df = pd.DataFrame(results)
    
    print("\n" + "=" * 80)
    print("延遲組別特徵摘要")
    print("=" * 80)
    print(result_df.to_string(index=False))
    
    return result_df, timing_df


# ============================================================================
# Part 3: 異常模式偵測
# ============================================================================

def detect_anomaly_patterns(trans_df, timing_df):
    """偵測延遲通報帳戶的異常模式"""
    print("\n" + "=" * 80)
    print("Step 3: 異常模式偵測")
    print("=" * 80)
    
    delayed = timing_df[timing_df['days_since_last_txn'] > 30].copy()
    print(f"\n延遲組 (>30天): {len(delayed)} 帳戶")
    
    patterns = []
    
    # 模式 1: 高交易密度
    if 'account_lifespan' in delayed.columns and 'total_txns' in delayed.columns:
        delayed['txn_density'] = delayed['total_txns'] / (delayed['account_lifespan'] + 1)
        high_density = delayed[delayed['txn_density'] > 1.0]
        
        if len(high_density) > 0:
            patterns.append({
                'pattern': '高交易密度',
                'description': '短時間大量交易(密度>1筆/天)',
                'count': len(high_density),
                'severity': 'HIGH',
                'examples': high_density['acct'].head(3).tolist()
            })
    
    # 模式 2: 短壽命帳戶
    if 'account_lifespan' in delayed.columns:
        short_life = delayed[delayed['account_lifespan'] < 10]
        if len(short_life) > 0:
            patterns.append({
                'pattern': '短壽命帳戶',
                'description': '壽命<10天',
                'count': len(short_life),
                'severity': 'MEDIUM',
                'examples': short_life['acct'].head(3).tolist()
            })
    
    # 模式 3: 極長延遲
    extreme_delay = delayed[delayed['days_since_last_txn'] > 60]
    if len(extreme_delay) > 0:
        patterns.append({
            'pattern': '極長延遲',
            'description': '延遲>60天(可能事後檢舉)',
            'count': len(extreme_delay),
            'severity': 'LOW',
            'examples': extreme_delay['acct'].head(3).tolist()
        })
    
    # 模式 4: 無交易記錄
    no_txn = timing_df[timing_df['time_gap_category'] == 'NO_TRANSACTION']
    if len(no_txn) > 0:
        patterns.append({
            'pattern': '無交易記錄',
            'description': '警示帳戶卻無任何交易',
            'count': len(no_txn),
            'severity': 'HIGH',
            'examples': no_txn['acct'].head(3).tolist()
        })
    
    # 列印發現
    if patterns:
        print("\n【偵測到的異常模式】")
        for p in patterns:
            print(f"\n{p['pattern']} [{p['severity']}]")
            print(f"  描述: {p['description']}")
            print(f"  數量: {p['count']} 帳戶")
            if p['examples']:
                print(f"  範例: {p['examples'][0][:16]}...")
    else:
        print("\n✓ 未發現明顯異常模式")
    
    return patterns


# ============================================================================
# Part 4: 特徵差異對比
# ============================================================================

def compare_immediate_vs_delayed(timing_df):
    """比較即時通報 vs 延遲通報的差異"""
    print("\n" + "=" * 80)
    print("Step 4: 即時 vs 延遲通報特徵對比")
    print("=" * 80)
    
    immediate = timing_df[timing_df['days_since_last_txn'] <= 10]
    delayed = timing_df[timing_df['days_since_last_txn'] > 10]
    
    if len(immediate) == 0 or len(delayed) == 0:
        print("\n無法進行對比 (其中一組無數據)")
        return None
    
    features = ['total_txns', 'account_lifespan']
    
    print(f"\n{'特徵':<20} | {'即時組均值':<15} | {'延遲組均值':<15} | {'差異倍數':<10}")
    print("-" * 70)
    
    comparison = []
    for feat in features:
        if feat in timing_df.columns:
            imm_mean = immediate[feat].mean()
            del_mean = delayed[feat].mean()
            ratio = del_mean / imm_mean if imm_mean > 0 else 0
            
            print(f"{feat:<20} | {imm_mean:<15.2f} | {del_mean:<15.2f} | {ratio:<10.2f}x")
            
            comparison.append({
                'feature': feat,
                'immediate_mean': imm_mean,
                'delayed_mean': del_mean,
                'ratio': ratio
            })
    
    return pd.DataFrame(comparison) if comparison else None


# ============================================================================
# Main Function
# ============================================================================

def main(args):
    # 建立輸出目錄
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 載入資料
    print("=" * 80)
    print("載入資料")
    print("=" * 80)
    
    print("\n載入警示帳戶...")
    alerts_df = pd.read_csv(args.alerts)
    alerts_df['acct'] = alerts_df['acct'].astype(str)
    print(f"✓ 警示帳戶: {len(alerts_df):,}")
    
    print("\n載入交易資料...")
    trans_df = pd.read_csv(args.transactions)
    trans_df['from_acct'] = trans_df['from_acct'].astype(str)
    trans_df['to_acct'] = trans_df['to_acct'].astype(str)
    print(f"✓ 交易記錄: {len(trans_df):,}")
    
    # ========== Step 1: 時間差分析 ==========
    timing_df = analyze_alert_timing(alerts_df, trans_df)
    print_timing_statistics(timing_df)
    
    # 儲存主要結果
    timing_file = os.path.join(args.output_dir, 'alert_timing_analysis.csv')
    timing_df.to_csv(timing_file, index=False)
    print(f"\n✓ 時間差分析: {timing_file}")
    
    # 儲存分類檔案
    for threshold, label in [(10, 'over10days'), (30, 'over30days'), (60, 'over60days')]:
        subset = timing_df[timing_df['days_since_last_txn'] > threshold]
        if len(subset) > 0:
            subset_file = os.path.join(args.output_dir, f'alert_timing_{label}.csv')
            subset.to_csv(subset_file, index=False)
            print(f"✓ 延遲>{threshold}天: {subset_file}")
    
    # ========== Step 2: 延遲組別特徵分析 ==========
    temporal_features, enriched_timing = analyze_temporal_patterns(trans_df, timing_df)
    
    temporal_file = os.path.join(args.output_dir, 'temporal_features_by_group.csv')
    temporal_features.to_csv(temporal_file, index=False)
    print(f"\n✓ 延遲組別特徵: {temporal_file}")
    
    # ========== Step 3: 異常模式偵測 ==========
    patterns = detect_anomaly_patterns(trans_df, enriched_timing)
    
    if patterns:
        patterns_df = pd.DataFrame(patterns)
        patterns_file = os.path.join(args.output_dir, 'anomaly_patterns.csv')
        patterns_df.to_csv(patterns_file, index=False)
        print(f"\n✓ 異常模式: {patterns_file}")
    
    # ========== Step 4: 特徵差異對比 ==========
    comparison = compare_immediate_vs_delayed(enriched_timing)
    
    if comparison is not None:
        comparison_file = os.path.join(args.output_dir, 'immediate_vs_delayed_comparison.csv')
        comparison.to_csv(comparison_file, index=False)
        print(f"\n✓ 特徵對比: {comparison_file}")
    
    # ========== 統計摘要 ==========
    category_stats = timing_df['time_gap_category'].value_counts().reset_index()
    category_stats.columns = ['category', 'count']
    category_stats['percentage'] = category_stats['count'] / len(timing_df) * 100
    
    stats_file = os.path.join(args.output_dir, 'category_statistics.csv')
    category_stats.to_csv(stats_file, index=False)
    print(f"✓ 分類統計: {stats_file}")
    
    # 完成
    print("\n" + "=" * 80)
    print("✓ 分析完成！所有結果已儲存至:")
    print(f"  {args.output_dir}/")
    print("=" * 80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='整合版警示帳戶時間與特徵分析',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--transactions', type=str, required=True,
                       help='交易資料路徑')
    parser.add_argument('--alerts', type=str, required=True,
                       help='警示帳戶資料路徑')
    parser.add_argument('--output_dir', type=str, default='alert_analysis_results',
                       help='輸出目錄')
    
    args = parser.parse_args()
    main(args)