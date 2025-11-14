#!/usr/bin/env python3
"""
資料預處理 (Preprocessing) 執行腳本 - v4 版

此腳本負責執行兩階段流程中的【第一階段：預處理】。

它會載入原始的 .csv 資料，執行 'v4' 版的完整特徵工程，
包含帳戶特徵聚合、圖特徵計算、特徵標準化，並處理 'Plan B' 模式的
標籤與遮罩 (mask) 劃分。

最終，它會將所有用於模型訓練的必要檔案 (特徵矩陣、標籤、圖、
遮罩、標準化器等) 儲存到指定的輸出目錄中。

使用範例:
python Preprocessing/preprocessing.py \
    --transactions Data/acct_transaction.csv \
    --alerts Data/acct_alert.csv \
    --predicts Data/acct_predict.csv \
    --out_dir Preprocessing/processed_data \
    --use_planb
"""

import os
import argparse
import time
import json
import joblib
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, precision_recall_curve, auc
import torch
import random
import warnings
import gc

warnings.filterwarnings('ignore')

# 冗餘特徵（完全移除）
FEATURES_TO_REMOVE = [
    # 原始版本（保留 log 版本）
    'out_sum', 'in_sum', 'net_flow', 'avg_staying_time', 'single_day_turnover',
    'in_sum_log', 'out_sum_log',
    # 原始熵（保留 norm 版本）
    'time_entropy', 'hour_entropy', 'dow_entropy',
    
    # 基本計數（圖特徵已包含）
    'out_count', 'in_count',
    
    # 類別相關（已關閉）
    'chan_01', 'chan_02', 'chan_03', 'chan_04', 'chan_05', 'chan_06', 'chan_99', 'chan_UNK',
    'cur_EUR', 'cur_TWD', 'cur_GBP', 'cur_JPY', 'cur_HKD', 'cur_USD', 'cur_AUD', 'cur_CNY',

    # v4 新增：這些特徵在加入圖特徵後，在 X_final 階段被移除，以保持特徵集乾淨
    'inbound_only_ratio', 'outbound_only_ratio', 'peak_hour_concentration', 'unique_from', 'unique_to',
    'graph_out_degree', 'graph_in_degree', 'graph_out_degree_norm', 'graph_in_degree_norm', 'graph_pagerank_approx', 
    'micro_out', 'micro_in', 'early_txn_count', 'recent_txn_count', 'account_lifespan', 
    'near_100k_out','near_100k_in','near_500k_out','near_500k_in', 'ultra_micro_500_out', 'ultra_micro_500_in',
    'avg_out_amount', 'avg_in_amount', 'amount_volatility', 'outbound_only_ratio', 'night_txn_ratio', 'extreme_outbound', 'foreign_amount_entropy', 'multi_currency_ops',
]

# ============ 輔助函數 (Helpers) ============

def seed_everything(seed=42):
    """
    設定隨機種子以確保可復現性。

    Args:
        seed (int, optional): 隨機種子。預設為 42。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def build_node_index(trans_df):
    """
    從交易資料中建立所有帳戶的索引 (id2idx 映射)。

    Args:
        trans_df (pd.DataFrame): 包含 'from_acct' 和 'to_acct' 欄位的交易 DataFrame。

    Returns:
        dict: 帳戶 ID (str) 到節點索引 (int) 的映射。
    """
    accts = pd.unique(trans_df[['from_acct', 'to_acct']].values.ravel())
    accts = [str(a) for a in accts if pd.notnull(a)]
    return {a: i for i, a in enumerate(sorted(accts))}

def parse_time_features(time_str):
    """
    將 'HH:MM' 格式的時間字串解析為 (小時, 分鐘)。

    Args:
        time_str (str): 時間字串。

    Returns:
        tuple (int, int): (小時, 分鐘)。如果解析失敗則返回 (0, 0)。
    """
    try:
        if pd.isna(time_str) or time_str == '':
            return 0, 0
        time_str = str(time_str).strip()
        if ':' in time_str:
            parts = time_str.split(':')
            hour = int(parts[0]) % 24
            minute = int(parts[1]) % 60 if len(parts) > 1 else 0
        else:
            hour, minute = 0, 0
        return hour, minute
    except:
        return 0, 0

def parse_date(date_str):
    """
    將日期字串 (或數字) 解析為整數日期。

    Args:
        date_str (str or float): 日期字串。

    Returns:
        int: 整數日期。如果解析失敗則返回 0。
    """
    try:
        if pd.isna(date_str) or date_str == '':
            return 0
        date_num = int(float(date_str))
        return date_num if date_num > 0 else 0
    except:
        return 0

def compute_gini(values):
    """
    計算 Gini 係數，用於衡量分布的不均勻性。

    Args:
        values (list or np.ndarray): 一組數值。

    Returns:
        float: Gini 係數 (0 到 1)。
    """
    if len(values) == 0:
        return 0.0
    sorted_values = np.sort(values)
    n = len(values)
    cumsum = np.cumsum(sorted_values)
    return (2 * np.sum((np.arange(1, n + 1)) * sorted_values)) / (n * cumsum[-1]) - (n + 1) / n if cumsum[-1] > 0 else 0

def compute_foreign_currency_repetition_features(foreign_currency_txns, N):
    """
    外幣重複金額特徵
    
    Args:
        foreign_currency_txns (defaultdict): {acct_idx: {currency: [amt1, amt2]}}
        N (int): 總節點數。

    Returns:
        dict: {acct_idx: {feature_name: value}}
    """
    foreign_currency_count = np.zeros(N, dtype=np.float32)
    foreign_duplicate_ratio = np.zeros(N, dtype=np.float32)
    max_foreign_duplicate = np.zeros(N, dtype=np.float32)
    foreign_entropy = np.zeros(N, dtype=np.float32)
    
    for acct_idx, currency_dict in foreign_currency_txns.items():
        if len(currency_dict) == 0:
            continue
        
        foreign_currency_count[acct_idx] = len(currency_dict)
        
        all_foreign_amounts = []
        for currency, amounts in currency_dict.items():
            all_foreign_amounts.extend(amounts)
        
        if len(all_foreign_amounts) > 0:
            rounded_amounts = [round(amt, 2) for amt in all_foreign_amounts]
            from collections import Counter
            amount_counts = Counter(rounded_amounts)
            
            duplicates = [count for count in amount_counts.values() if count > 1]
            
            if duplicates:
                foreign_duplicate_ratio[acct_idx] = len(duplicates) / len(amount_counts)
                max_foreign_duplicate[acct_idx] = max(duplicates)
            
            total = len(rounded_amounts)
            probs = [count / total for count in amount_counts.values()]
            foreign_entropy[acct_idx] = -sum(p * np.log(p + 1e-10) for p in probs)
    
    results = {}
    for idx in range(N):
        results[idx] = {
            'foreign_currency_types': foreign_currency_count[idx],
            'foreign_duplicate_ratio': foreign_duplicate_ratio[idx],
            'max_foreign_duplicate_count': max_foreign_duplicate[idx],
            'foreign_amount_entropy': foreign_entropy[idx],
        }
    return results

# ============ 主要特徵聚合函數 ============
def agg_account_features(trans_df, id2idx, hour_bins=24):
    """
    主要的特徵聚合函數 (v4 版)。

    Args:
        trans_df (pd.DataFrame): 完整的交易資料。
        id2idx (dict): 帳戶 ID 到節點索引的映射。
        hour_bins (int, optional): 時間直方圖的 bin 數量。預設為 24。

    Returns:
        pd.DataFrame: 以帳戶 ID 為索引 (index) 的特徵 DataFrame。
    """
    print(f"開始聚合 {len(id2idx):,} 個帳戶的特徵...")
    
    N = len(id2idx)
    
    # 初始化特徵字典
    features = {
        'out_count': np.zeros(N, dtype=np.float32),
        'in_count': np.zeros(N, dtype=np.float32),
        'out_sum': np.zeros(N, dtype=np.float32),
        'in_sum': np.zeros(N, dtype=np.float32),
        'unique_to': defaultdict(set),
        'unique_from': defaultdict(set),
        'hour_hist': np.zeros((N, hour_bins), dtype=np.float32),
        'night_txn': np.zeros(N, dtype=np.float32),
        'large_99_out': np.zeros(N, dtype=np.float32),
        'ultra_micro_500_out': np.zeros(N, dtype=np.float32),
        'ultra_micro_500_in': np.zeros(N, dtype=np.float32),
        'ultra_micro_out': np.zeros(N, dtype=np.float32),
        'ultra_micro_in': np.zeros(N, dtype=np.float32),
        'micro_out': np.zeros(N, dtype=np.float32),
        'micro_in': np.zeros(N, dtype=np.float32),
        'near_50k_out': np.zeros(N, dtype=np.float32),
        'near_50k_in': np.zeros(N, dtype=np.float32),
        'near_100k_out': np.zeros(N, dtype=np.float32),
        'near_100k_in': np.zeros(N, dtype=np.float32),
        'near_500k_out': np.zeros(N, dtype=np.float32),
        'near_500k_in': np.zeros(N, dtype=np.float32),
        'txn_out': defaultdict(list),
        'txn_in': defaultdict(list),
        'txn_timestamps': defaultdict(list),
        'cash_flows': defaultdict(list),
        'hour_dist': defaultdict(lambda: defaultdict(int)),
        'dow_dist': defaultdict(lambda: defaultdict(int)),
        'out_partners': defaultdict(lambda: defaultdict(int)),
        'in_partners': defaultdict(lambda: defaultdict(int)),
        'amount_counts': defaultdict(lambda: defaultdict(int)),
        'same_minute_ops': defaultdict(list),
        'currency_usage': defaultdict(set),
        'multi_currency_times': defaultdict(set),
        'convergence_sources': defaultdict(lambda: defaultdict(list)),
        'foreign_currency_txns': defaultdict(lambda: defaultdict(list)),
    }

    amounts = trans_df['txn_amt'].dropna()
    p99 = np.percentile(amounts, 99) if len(amounts) > 0 else 100000
    large_threshold = 50000
    print(f"金額閾值: P99={p99:.0f}")
    
    # 第一遍：收集所有交易信息
    print("處理交易...")
    for idx, row in trans_df.iterrows():
        if idx % 500000 == 0:
            print(f"  進度: {idx:,}/{len(trans_df):,}")
        
        fa, ta = str(row['from_acct']), str(row['to_acct'])
        if fa not in id2idx or ta not in id2idx:
            continue
        
        i, j = id2idx[fa], id2idx[ta]
        amt = float(row['txn_amt']) if pd.notnull(row['txn_amt']) else 0.0
        hour, minute = parse_time_features(row.get('txn_time', ''))
        date_num = parse_date(row.get('txn_date', ''))
        currency = str(row.get('currency_type', 'TWD'))
        
        features['out_count'][i] += 1
        features['in_count'][j] += 1
        features['out_sum'][i] += amt
        features['in_sum'][j] += amt
        features['unique_to'][i].add(ta)
        features['unique_from'][j].add(fa)
        features['hour_hist'][i, hour % hour_bins] += 1
        if hour >= 20 or hour <= 8:
            features['night_txn'][i] += 1
        
        if date_num > 0:
            features['txn_out'][i].append((date_num, hour, minute, amt, ta))
            features['txn_in'][j].append((date_num, hour, minute, amt, fa))
            features['txn_timestamps'][i].append((date_num, hour, minute))
            features['cash_flows'][i].append((date_num, hour, minute, -amt))
            features['cash_flows'][j].append((date_num, hour, minute, amt))
            features['hour_dist'][i][hour] += 1
            features['dow_dist'][i][date_num % 7] += 1
            time_minutes = date_num * 1440 + hour * 60 + minute
            features['convergence_sources'][j][time_minutes].append(fa)
        
        if amt >= p99:
            features['large_99_out'][i] += 1
        if 0 < amt <= 500:
            features['ultra_micro_500_out'][i] += 1
            features['ultra_micro_500_in'][j] += 1
        if 0 < amt < 105:
            features['ultra_micro_out'][i] += 1
            features['ultra_micro_in'][j] += 1
        elif 105 <= amt < 500:
            features['micro_out'][i] += 1
            features['micro_in'][j] += 1
        if 47000 <= amt < 51000:
            features['near_50k_out'][i] += 1
            features['near_50k_in'][j] += 1
        elif 95000 <= amt < 110000:
            features['near_100k_out'][i] += 1
            features['near_100k_in'][j] += 1
        elif 410000 <= amt < 505000:
            features['near_500k_out'][i] += 1
            features['near_500k_in'][j] += 1

        features['out_partners'][i][ta] += 1
        features['in_partners'][j][fa] += 1
        rounded_amt = round(amt, -2)
        features['amount_counts'][i][rounded_amt] += 1
        features['currency_usage'][i].add(currency)
        if date_num > 0:
            time_key = (date_num, hour)
            features['multi_currency_times'][i].add((time_key, currency))
        if currency != 'TWD':
            features['foreign_currency_txns'][i][currency].append(amt)
            features['foreign_currency_txns'][j][currency].append(amt)

    # 第二遍：檢測同時操作
    print("檢測同時操作...")
    for acct_idx in range(N):
        if acct_idx not in features['txn_in'] or acct_idx not in features['txn_out']:
            continue
        in_by_time = defaultdict(list)
        for date, hour, minute, amt, source in features['txn_in'][acct_idx]:
            in_by_time[(date, hour, minute)].append(amt)
        out_by_time = defaultdict(list)
        for date, hour, minute, amt, target in features['txn_out'][acct_idx]:
            out_by_time[(date, hour, minute)].append(amt)
        
        common_times = set(in_by_time.keys()) & set(out_by_time.keys())
        for time_key in common_times:
            features['same_minute_ops'][acct_idx].append((time_key, sum(in_by_time[time_key]), sum(out_by_time[time_key])))
    
    # 後處理：計算衍生特徵
    print("計算衍生特徵...")
    behavior_features = compute_behavior_features(features['txn_out'], N, large_threshold)
    temporal_features = compute_temporal_features(features['txn_timestamps'], features['hour_dist'], features['dow_dist'], N)
    velocity_features = compute_velocity_features(features['cash_flows'], N)
    lifecycle_features = compute_lifecycle_features(features['txn_out'], features['txn_in'], N, large_threshold)
    convergence_features = compute_convergence_features(features['convergence_sources'], N)
    test_activation_features = compute_test_activation_features_v2(features['txn_out'], features['txn_in'], features['ultra_micro_out'], features['ultra_micro_in'], N, large_threshold)
    counterparty_features = compute_counterparty_features(features['out_partners'], features['in_partners'], N)
    simultaneous_features = compute_simultaneous_features(features['same_minute_ops'], N)
    amount_features = compute_amount_repetition_features(features['amount_counts'], N)
    currency_features = compute_multi_currency_features(features['currency_usage'], features['multi_currency_times'], N)
    bidirectional_features = compute_bidirectional_features(features['out_partners'], features['in_partners'], N)
    foreign_currency_features = compute_foreign_currency_repetition_features(features['foreign_currency_txns'], N)
    flow_direction_features = compute_flow_direction_features(features['out_count'], features['in_count'], N)
    hour_concentration_features = compute_hour_concentration_features(features['hour_dist'], N)
# 構建 DataFrame
    print("構建特徵 DataFrame...")
    rows = []
    for acct, idx in id2idx.items():
        row = {'acct': acct}
        row['out_count'] = features['out_count'][idx]
        row['in_count'] = features['in_count'][idx]
        row['out_sum'] = features['out_sum'][idx]
        row['in_sum'] = features['in_sum'][idx]
        row['net_flow'] = features['out_sum'][idx] - features['in_sum'][idx]
        row['unique_to'] = len(features['unique_to'][idx])
        row['unique_from'] = len(features['unique_from'][idx])
        
        hist = features['hour_hist'][idx]
        entropy = -np.sum((hist/hist.sum()) * np.log((hist/hist.sum()) + 1e-10)) if hist.sum() > 0 else 0
        total_txn = features['out_count'][idx] + features['in_count'][idx]
        row['night_txn_ratio'] = features['night_txn'][idx] / max(total_txn, 1)
        row['time_entropy'] = entropy
        
        row['large_99_out'] = features['large_99_out'][idx]
        row['ultra_micro_500_out'] = features['ultra_micro_500_out'][idx]
        row['ultra_micro_500_in'] = features['ultra_micro_500_in'][idx]
        row['ultra_micro_out'] = features['ultra_micro_out'][idx]
        row['ultra_micro_in'] = features['ultra_micro_in'][idx]
        row['micro_out'] = features['micro_out'][idx]
        row['micro_in'] = features['micro_in'][idx]
        row['near_50k_out'] = features['near_50k_out'][idx]
        row['near_50k_in'] = features['near_50k_in'][idx]
        row['near_100k_out'] = features['near_100k_out'][idx]
        row['near_100k_in'] = features['near_100k_in'][idx]
        row['near_500k_out'] = features['near_500k_out'][idx]
        row['near_500k_in'] = features['near_500k_in'][idx]
        
        row['avg_out_amount'] = features['out_sum'][idx] / max(features['out_count'][idx], 1)
        row['avg_in_amount'] = features['in_sum'][idx] / max(features['in_count'][idx], 1)
        row['amount_volatility'] = (row['avg_out_amount'] - row['avg_in_amount']) / max(row['avg_out_amount'], 1)

        total_out = features['out_sum'][idx]
        total_in = features['in_sum'][idx]
        total_flow = total_out + total_in
        bias = (total_out - total_in) / total_flow if total_flow > 0 else 0
        row['extreme_inbound'] = 1 if bias < -0.8 else 0
        row['extreme_outbound'] = 1 if bias > 0.8 else 0
        
        row.update(behavior_features[idx])
        row.update(temporal_features[idx])
        row.update(velocity_features[idx])
        row.update(lifecycle_features[idx])
        row.update(convergence_features[idx])
        row.update(test_activation_features[idx])
        row.update(counterparty_features[idx])
        row.update(simultaneous_features[idx])
        row.update(amount_features[idx])
        row.update(currency_features[idx])
        row.update(bidirectional_features[idx])
        row.update(foreign_currency_features[idx])
        row.update(flow_direction_features[idx])
        row.update(hour_concentration_features[idx])

        rows.append(row)
    
    feats_df = pd.DataFrame(rows).set_index('acct')
    print(f"特徵構建完成，形狀: {feats_df.shape}")
    
    return feats_df

# ============ 衍生特徵計算函數 (與 train.py 相同) ============

def compute_behavior_features(txn_out, N, large_threshold):
    """計算行為模式相關特徵。"""
    early_count = np.zeros(N, dtype=np.float32)
    recent_count = np.zeros(N, dtype=np.float32)
    behavior_change_count = np.zeros(N, dtype=np.float32)
    tail_vs_avg_ratio = np.zeros(N, dtype=np.float32)
    for acct_idx, txns in txn_out.items():
        if len(txns) == 0: continue
        txns_sorted = sorted(txns, key=lambda x: x[0])
        split_idx = int(len(txns_sorted) * 0.7)
        early_count[acct_idx] = split_idx
        recent_count[acct_idx] = len(txns_sorted) - split_idx
        behavior_change_count[acct_idx] = recent_count[acct_idx] - early_count[acct_idx]
        last_day = txns_sorted[-1][0]
        last_n_txns = [t for t in txns_sorted if t[0] > last_day - 8]
        if last_n_txns:
            tail_vs_avg_ratio[acct_idx] = len(last_n_txns) / max(len(txns_sorted) / (last_day - txns_sorted[0][0] + 1), 0.1)
    results = {}
    for idx in range(N):
        results[idx] = {'early_txn_count': early_count[idx], 'recent_txn_count': recent_count[idx], 'behavior_change_count': behavior_change_count[idx], 'tail_vs_avg_ratio': tail_vs_avg_ratio[idx]}
    return results

def compute_temporal_features(txn_timestamps, hour_dist, dow_dist, N):
    """計算時間相關特徵。"""
    results = {}
    for idx in range(N):
        timestamps = txn_timestamps.get(idx, [])
        burst_60min = 0
        midnight_count = 0
        if len(timestamps) >= 3:
            timestamps_sorted = sorted(timestamps)
            for i in range(len(timestamps_sorted)):
                count = 1
                start_day, start_hour, start_min = timestamps_sorted[i]
                for j in range(i+1, len(timestamps_sorted)):
                    day, hour, minute = timestamps_sorted[j]
                    time_diff = (day - start_day) * 1440 + (hour - start_hour) * 60 + (minute - start_min)
                    if time_diff <= 60: count += 1
                    else: break
                burst_60min = max(burst_60min, count)
                if start_hour >= 23 or start_hour <= 6: midnight_count += 1
        hour_entropy = 0
        if idx in hour_dist and len(hour_dist[idx]) >= 3:
            total = sum(hour_dist[idx].values())
            probs = [count / total for count in hour_dist[idx].values()]
            hour_entropy = -sum(p * np.log(p + 1e-10) for p in probs)
        dow_entropy = 0
        if idx in dow_dist and len(dow_dist[idx]) >= 2:
            total = sum(dow_dist[idx].values())
            probs = [count / total for count in dow_dist[idx].values()]
            dow_entropy = -sum(p * np.log(p + 1e-10) for p in probs)
        results[idx] = {'burst_60min': burst_60min, 'midnight_burst': midnight_count, 'hour_entropy': hour_entropy, 'dow_entropy': dow_entropy}
    return results

def compute_velocity_features(cash_flows, N):
    """計算資金流速特徵。"""
    fast_turnover_count = np.zeros(N, dtype=np.float32)
    avg_staying_time = np.zeros(N, dtype=np.float32)
    for acct_idx, flows in cash_flows.items():
        if len(flows) < 2: continue
        flows_sorted = sorted(flows, key=lambda x: (x[0], x[1], x[2]))
        time_diffs = []
        balance, last_positive_time = 0, None
        for day, hour, minute, amt in flows_sorted:
            current_time_min = day * 1440 + hour * 60 + minute
            prev_balance = balance
            balance += amt
            if prev_balance > 0 and balance <= 0:
                if last_positive_time is not None:
                    time_diff = current_time_min - last_positive_time
                    time_diffs.append(time_diff)
                    if time_diff < 60: fast_turnover_count[acct_idx] += 1
            if balance > 0 and last_positive_time is None: last_positive_time = current_time_min
            elif balance <= 0: last_positive_time = None
        if time_diffs: avg_staying_time[acct_idx] = np.mean(time_diffs)
    results = {}
    for idx in range(N):
        results[idx] = {'fast_turnover_count': fast_turnover_count[idx], 'avg_staying_time': avg_staying_time[idx]}
    return results

def compute_lifecycle_features(txn_out, txn_in, N, large_threshold):
    """計算帳戶生命週期特徵。"""
    account_lifespan = np.zeros(N, dtype=np.float32)
    single_day_turnover = np.zeros(N, dtype=np.float32)
    txn_density = np.zeros(N, dtype=np.float32)
    short_life_burst = np.zeros(N, dtype=np.float32)
    for acct_idx, txns in txn_out.items():
        if len(txns) == 0: continue
        txns_sorted = sorted(txns, key=lambda x: x[0])
        first_day = txns_sorted[0][0]
        last_day = txns_sorted[-1][0]
        lifespan = last_day - first_day + 1
        account_lifespan[acct_idx] = lifespan
        total_txns = len(txns_sorted)
        if acct_idx in txn_in: total_txns += len(txn_in[acct_idx])
        txn_density[acct_idx] = total_txns / max(lifespan, 1)
        if lifespan <= 10 and total_txns >= 20: short_life_burst[acct_idx] = 1
        elif lifespan <= 5 and total_txns >= 10: short_life_burst[acct_idx] = 1
        day_turnovers = defaultdict(float)
        for date, _, _, amt, _ in txns_sorted: day_turnovers[date] += amt
        if acct_idx in txn_in:
            for date, _, _, amt, _ in txn_in[acct_idx]: day_turnovers[date] += amt
        if day_turnovers: single_day_turnover[acct_idx] = max(day_turnovers.values())
    results = {}
    for idx in range(N):
        results[idx] = {'account_lifespan': account_lifespan[idx], 'single_day_turnover': single_day_turnover[idx], 'txn_density': txn_density[idx], 'short_life_burst': short_life_burst[idx]}
    return results

def compute_convergence_features(convergence_sources, N):
    """計算資金匯集特徵。"""
    convergence_1h = np.zeros(N, dtype=np.float32)
    convergence_6h = np.zeros(N, dtype=np.float32)
    convergence_24h = np.zeros(N, dtype=np.float32)
    convergence_diversity = np.zeros(N, dtype=np.float32)
    for acct_idx, time_sources in convergence_sources.items():
        if len(time_sources) < 2: continue
        sorted_times = sorted(time_sources.keys())
        all_sources = []
        for sources in time_sources.values(): all_sources.extend(sources)
        for window_minutes in [60, 360, 1440]:
            max_unique_sources = 0
            for start_time in sorted_times:
                unique_sources = set()
                for time_key, sources in time_sources.items():
                    if start_time <= time_key <= start_time + window_minutes: unique_sources.update(sources)
                max_unique_sources = max(max_unique_sources, len(unique_sources))
            if window_minutes == 60: convergence_1h[acct_idx] = max_unique_sources
            elif window_minutes == 360: convergence_6h[acct_idx] = max_unique_sources
            else: convergence_24h[acct_idx] = max_unique_sources
        unique_sources_total = len(set(all_sources))
        total_txns = len(all_sources)
        convergence_diversity[acct_idx] = unique_sources_total / total_txns if total_txns > 0 else 0
    results = {}
    for idx in range(N):
        results[idx] = {'convergence_1h': convergence_1h[idx], 'convergence_6h': convergence_6h[idx], 'convergence_24h': convergence_24h[idx], 'convergence_diversity': convergence_diversity[idx]}
    return results

def compute_test_activation_features_v2(txn_out, txn_in, ultra_micro_out, ultra_micro_in, N, large_threshold):
    """計算 "測試-啟用" 特徵 v2。"""
    test_then_large_in = np.zeros(N, dtype=np.float32)
    test_then_large_out = np.zeros(N, dtype=np.float32)
    any_micro_then_large_in = np.zeros(N, dtype=np.float32)
    any_micro_then_large_out = np.zeros(N, dtype=np.float32)
    for acct_idx, txns in txn_in.items():
        if len(txns) < 2: continue
        txns_sorted = sorted(txns, key=lambda x: (x[0], x[1], x[2]))
        first_amt = txns_sorted[0][3]
        if first_amt < 200:
            subsequent_large = [t[3] for t in txns_sorted[1:] if t[3] >= large_threshold]
            if subsequent_large: test_then_large_in[acct_idx] = len(subsequent_large)
        for i, (date, hour, minute, amt, source) in enumerate(txns_sorted):
            if amt < 200:
                subsequent_large = [t[3] for t in txns_sorted[i+1:] if t[3] >= large_threshold]
                if subsequent_large:
                    any_micro_then_large_in[acct_idx] = 1
                    break
    for acct_idx, txns in txn_out.items():
        if len(txns) < 2: continue
        txns_sorted = sorted(txns, key=lambda x: (x[0], x[1], x[2]))
        first_amt = txns_sorted[0][3]
        if first_amt < 200:
            subsequent_large = [t[3] for t in txns_sorted[1:] if t[3] >= large_threshold]
            if subsequent_large: test_then_large_out[acct_idx] = len(subsequent_large)
        for i, (date, hour, minute, amt, target) in enumerate(txns_sorted):
            if amt < 200:
                subsequent_large = [t[3] for t in txns_sorted[i+1:] if t[3] >= large_threshold]
                if subsequent_large:
                    any_micro_then_large_out[acct_idx] = 1
                    break
    results = {}
    for idx in range(N):
        results[idx] = {'test_then_large_in': test_then_large_in[idx], 'test_then_large_out': test_then_large_out[idx], 'any_micro_then_large_in': any_micro_then_large_in[idx], 'any_micro_then_large_out': any_micro_then_large_out[idx]}
    return results
def compute_counterparty_features(out_partners, in_partners, N):
    """計算對手方特徵。"""
    out_gini = np.zeros(N, dtype=np.float32)
    in_gini = np.zeros(N, dtype=np.float32)
    for acct_idx in range(N):
        if acct_idx in out_partners and len(out_partners[acct_idx]) > 0:
            out_gini[acct_idx] = compute_gini(list(out_partners[acct_idx].values()))
        if acct_idx in in_partners and len(in_partners[acct_idx]) > 0:
            in_gini[acct_idx] = compute_gini(list(in_partners[acct_idx].values()))
    results = {}
    for idx in range(N):
        results[idx] = {'out_partner_gini': out_gini[idx], 'in_partner_gini': in_gini[idx]}
    return results

def compute_simultaneous_features(same_minute_ops, N):
    """計算同時操作特徵。"""
    same_minute_count = np.zeros(N, dtype=np.float32)
    same_minute_match_ratio = np.zeros(N, dtype=np.float32)
    for acct_idx, ops in same_minute_ops.items():
        same_minute_count[acct_idx] = len(ops)
        if len(ops) > 0:
            matches = 0
            for time_key, in_amt, out_amt in ops:
                if abs(in_amt - out_amt) / max(in_amt, out_amt, 1) < 0.1: matches += 1
            same_minute_match_ratio[acct_idx] = matches / len(ops)
    results = {}
    for idx in range(N):
        results[idx] = {'same_minute_ops_count': same_minute_count[idx], 'same_minute_match_ratio': same_minute_match_ratio[idx]}
    return results

def compute_amount_repetition_features(amount_counts, N):
    """計算金額重複特徵。"""
    most_common_amount_count = np.zeros(N, dtype=np.float32)
    for acct_idx, amounts in amount_counts.items():
        if len(amounts) > 0:
            most_common_amount_count[acct_idx] = max(list(amounts.values()))
    results = {}
    for idx in range(N):
        results[idx] = {'most_common_amount_count': most_common_amount_count[idx]}
    return results

def compute_multi_currency_features(currency_usage, multi_currency_times, N):
    """計算多幣別特徵。"""
    multi_currency_operations = np.zeros(N, dtype=np.float32)
    for acct_idx in range(N):
        if acct_idx in multi_currency_times:
            time_to_currencies = defaultdict(set)
            for (time_key, currency) in multi_currency_times[acct_idx]:
                time_to_currencies[time_key].add(currency)
            multi_count = sum(1 for currencies in time_to_currencies.values() if len(currencies) > 1)
            multi_currency_operations[acct_idx] = multi_count
    results = {}
    for idx in range(N):
        results[idx] = {'multi_currency_ops': multi_currency_operations[idx]}
    return results

def compute_bidirectional_features(out_partners, in_partners, N):
    """計算雙向流動特徵。"""
    bidirectional_partners = np.zeros(N, dtype=np.float32)
    bidirectional_imbalance = np.zeros(N, dtype=np.float32)
    for acct_idx in range(N):
        if acct_idx not in out_partners or acct_idx not in in_partners: continue
        out_set = set(out_partners[acct_idx].keys())
        in_set = set(in_partners[acct_idx].keys())
        common = out_set & in_set
        high_freq_bidirectional = 0
        for partner in common:
            out_count = out_partners[acct_idx][partner]
            in_count = in_partners[acct_idx][partner]
            if (out_count + in_count) >= 3: high_freq_bidirectional += 1
        bidirectional_partners[acct_idx] = high_freq_bidirectional
        if len(common) > 0:
            imbalances = [abs(out_partners[acct_idx][p] - in_partners[acct_idx][p]) / (out_partners[acct_idx][p] + in_partners[acct_idx][p]) for p in common]
            bidirectional_imbalance[acct_idx] = np.mean(imbalances)
    results = {}
    for idx in range(N):
        results[idx] = {'bidirectional_partners': bidirectional_partners[idx], 'bidirectional_imbalance': bidirectional_imbalance[idx]}
    return results

def compute_flow_direction_features(out_count, in_count, N):
    """計算純入金/純出金特徵。"""
    inbound_only_ratio = np.zeros(N, dtype=np.float32)
    outbound_only_ratio = np.zeros(N, dtype=np.float32)
    for idx in range(N):
        total_txn = out_count[idx] + in_count[idx]
        if total_txn > 0:
            inbound_only_ratio[idx] = in_count[idx] / total_txn
            outbound_only_ratio[idx] = out_count[idx] / total_txn
    results = {}
    for idx in range(N):
        results[idx] = {'inbound_only_ratio': inbound_only_ratio[idx], 'outbound_only_ratio': outbound_only_ratio[idx]}
    return results

def compute_hour_concentration_features(hour_dist, N):
    """計算時段集中度特徵。"""
    peak_hour_concentration = np.zeros(N, dtype=np.float32)
    hour_gini = np.zeros(N, dtype=np.float32)
    for acct_idx, hours in hour_dist.items():
        if len(hours) < 2: continue
        total = sum(hours.values())
        max_hour_count = max(hours.values())
        peak_hour_concentration[acct_idx] = max_hour_count / total
        hour_gini[acct_idx] = compute_gini(sorted(hours.values()))
    results = {}
    for idx in range(N):
        results[idx] = {'peak_hour_concentration': peak_hour_concentration[idx], 'hour_gini': hour_gini[idx]}
    return results

# ============ 特徵後處理 ============

def process_extreme_features(feats_df):
    """
    對特徵 DataFrame 進行後處理，例如對數轉換、標準化熵。

    Args:
        feats_df (pd.DataFrame): 聚合後的特徵 DataFrame。

    Returns:
        pd.DataFrame: 處理（新增 log/norm 欄位）後的 DataFrame。
    """
    print("處理極端特徵...")
    amount_features = ['out_sum', 'in_sum', 'net_flow', 'avg_staying_time', 'single_day_turnover']
    for feat in amount_features:
        if feat in feats_df.columns:
            if feat == 'net_flow':
                feats_df[f'{feat}_log'] = np.sign(feats_df[feat]) * np.log1p(np.abs(feats_df[feat]))
            else:
                feats_df[f'{feat}_log'] = np.log1p(np.maximum(feats_df[feat], 0))
    entropy_features = ['time_entropy', 'hour_entropy', 'dow_entropy']
    max_entropy = max(feats_df[f].max() for f in entropy_features if f in feats_df.columns and not feats_df[f].empty)
    if max_entropy > 0:
        for feat in entropy_features:
            if feat in feats_df.columns:
                feats_df[f'{feat}_norm'] = feats_df[feat] / max_entropy
    print(f"  對數轉換: {len(amount_features)} 個")
    print(f"  標準化熵: {len(entropy_features)} 個")
    return feats_df

def remove_redundant_features(feats_df):
    """
    根據全域 `FEATURES_TO_REMOVE` 列表，從 DataFrame 中移除冗餘特徵。

    Args:
        feats_df (pd.DataFrame): 特徵 DataFrame。

    Returns:
        pd.DataFrame: 移除冗餘特徵後的 DataFrame。
    """
    cols_to_drop = [col for col in FEATURES_TO_REMOVE if col in feats_df.columns]
    if cols_to_drop:
        print(f"移除 {len(cols_to_drop)} 個冗餘特徵")
        feats_df = feats_df.drop(columns=cols_to_drop)
    return feats_df

def compute_graph_features(edge_index, num_nodes):
    """
    計算基本的圖結構特徵 (degree, pagerank_approx)。

    Args:
        edge_index (np.ndarray or torch.Tensor): 邊索引，形狀為 (2, num_edges)。
        num_nodes (int): 圖中的總節點數。

    Returns:
        dict: 包含 'out_degree', 'in_degree', 'pagerank_approx' 等特徵的 dict。
    """
    print("計算圖特徵...")
    out_degree = np.zeros(num_nodes, dtype=np.float32)
    in_degree = np.zeros(num_nodes, dtype=np.float32)
    edge_index_np = edge_index.numpy() if isinstance(edge_index, torch.Tensor) else edge_index
    for i in range(edge_index_np.shape[1]):
        src, dst = edge_index_np[0, i], edge_index_np[1, i]
        out_degree[src] += 1
        in_degree[dst] += 1
    max_degree = max(out_degree.max(), in_degree.max(), 1) # 避免除以零
    out_degree_norm = out_degree / max_degree
    in_degree_norm = in_degree / max_degree
    total_degree = out_degree + in_degree
    pagerank_approx = total_degree / (total_degree.sum() + 1e-8)
    return {'out_degree': out_degree, 'in_degree': in_degree, 'out_degree_norm': out_degree_norm, 'in_degree_norm': in_degree_norm, 'pagerank_approx': pagerank_approx}

def save_feature_analysis(feats_df, y, output_dir):
    """
    儲存特徵分析報告 (特徵標準差、警示/正常帳戶的特徵均值對比)。

    Args:
        feats_df (pd.DataFrame): 包含特徵的 DataFrame (不含標籤)。
        y (pd.Series or np.ndarray): 對應的標籤。
        output_dir (str): 儲存報告的目錄路徑。
    """
    print("\n生成特徵分析報告...")
    feature_cols = [c for c in feats_df.columns if c != 'label']
    if not feature_cols:
        print("警告：特徵分析中沒有找到特徵欄位。")
        return
        
    feature_std = pd.DataFrame({'feature': feature_cols, 'std': feats_df[feature_cols].std().values}).sort_values('std', ascending=False)
    feature_std.to_csv(os.path.join(output_dir, 'feature_std.csv'), index=False)
    
    alert_mask = (y == 1)
    normal_mask = (y == 0)
    
    feature_comparison = pd.DataFrame({
        'feature': feature_cols,
        'alert_mean': feats_df.loc[alert_mask, feature_cols].mean().values if alert_mask.sum() > 0 else 0,
        'normal_mean': feats_df.loc[normal_mask, feature_cols].mean().values if normal_mask.sum() > 0 else 0,
    })
    feature_comparison['diff_ratio'] = ((feature_comparison['alert_mean'] - feature_comparison['normal_mean']) / (feature_comparison['normal_mean'].abs() + 1e-8))
    feature_comparison = feature_comparison.sort_values('diff_ratio', key=abs, ascending=False)
    feature_comparison.to_csv(os.path.join(output_dir, 'feature_comparison.csv'), index=False)
    print(f"特徵分析已儲存: {output_dir}")

# ============ 主執行函數 (Main Function) ============

def main(args):
    """
    主執行函數：載入原始資料，執行完整的特徵工程，並儲存預處理後的檔案。

    Args:
        args (argparse.Namespace): 包含所有命令列參數的物件。
    """
    seed_everything(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    
    print("=" * 80)
    print(f"模式: {'Plan B (完整特徵 + 標籤隔離)' if args.use_planb else '完整標籤訓練'}")
    print("=" * 80)

    # --- 1. 載入資料 ---
    print("載入交易資料...")
    trans_df = pd.read_csv(args.transactions)
    trans_df['from_acct'] = trans_df['from_acct'].astype(str)
    trans_df['to_acct'] = trans_df['to_acct'].astype(str)
    
    print("建立節點索引...")
    id2idx = build_node_index(trans_df)
    print(f"總帳戶數: {len(id2idx):,}")

    # --- 2. 特徵工程 (agg_account_features) ---
    print("=" * 80)
    print("開始特徵工程...")
    print("使用全部交易計算特徵（與預測時保持一致）")
    t_start = time.time()
    feats_df = agg_account_features(trans_df, id2idx, hour_bins=24)
    print(f"聚合耗時: {time.time() - t_start:.1f}秒")
    
    feats_df = process_extreme_features(feats_df)
    feats_df = feats_df.fillna(0.0)

    # --- 3. 處理標籤 (Plan B) ---
    print("=" * 80)
    print("載入標籤與預測清單...")
    alerts_df = pd.read_csv(args.alerts)
    alert_set = set(alerts_df['acct'].astype(str).values)
    print(f"警示帳戶數: {len(alert_set):,}")

    predict_set = set()
    if args.use_planb and args.predicts and os.path.exists(args.predicts):
        predict_df = pd.read_csv(args.predicts)
        predict_set = set(predict_df['acct'].astype(str).values)
        print(f"\n{'='*80}")
        print("Plan B 模式啟用")
        print(f"{'='*80}")
        print(f"策略說明:")
        print(f"  1. 特徵工程：使用【全部交易】計算所有帳戶特徵")
        print(f"  2. 標籤設定：預測帳戶標籤設為 NaN（不參與訓練）")
        print(f"  3. 圖結構：保留預測帳戶的所有邊（GNN 可從鄰居學習）")
        print(f"  4. 訓練mask：預測帳戶不進入訓練集")
        print(f"\n預測帳戶數: {len(predict_set):,}")
        print(f"{'='*80}\n")
    
    if args.use_planb:
        feats_df['label'] = feats_df.index.map(
            lambda x: 1 if x in alert_set else (np.nan if x in predict_set else 0)
        )
        trainable_mask_series = ~feats_df['label'].isna()
        feats_df_trainable = feats_df[trainable_mask_series].copy()
        feats_df_trainable['label'] = feats_df_trainable['label'].astype(int)
    else:
        feats_df['label'] = feats_df.index.map(lambda x: 1 if x in alert_set else 0)
        trainable_mask_series = pd.Series([True] * len(feats_df), index=feats_df.index)
        feats_df_trainable = feats_df.copy()
        
    y_trainable = feats_df_trainable['label']
    print(f"\n可訓練樣本數: {len(y_trainable):,}")
    print(f"正樣本比例: {y_trainable.mean():.4f}")

    # --- 4. 特徵分析 (可選) ---
    save_feature_analysis(feats_df_trainable, y_trainable, args.out_dir)
    
    feats_df['label'] = feats_df['label'].fillna(0) # Fill NaN for saving
    X = feats_df.drop(columns=['label'])
    y = feats_df['label'].astype(int).values
    
    del trans_df, feats_df_trainable # 釋放記憶體
    gc.collect()

    # --- 5. 建立圖結構 ---
    print("=" * 80)
    print("建立圖結構...")
    # 重新載入交易資料以建立圖 (同原始 train.py 邏輯)
    trans_df_graph = pd.read_csv(args.transactions)
    trans_df_graph['from_acct'] = trans_df_graph['from_acct'].astype(str)
    trans_df_graph['to_acct'] = trans_df_graph['to_acct'].astype(str)
    
    src = trans_df_graph['from_acct'].map(id2idx)
    dst = trans_df_graph['to_acct'].map(id2idx)
    mask_valid = (~src.isna()) & (~dst.isna())
    src_np = src[mask_valid].astype(int).values
    dst_np = dst[mask_valid].astype(int).values
    del trans_df_graph # 釋放記憶體
    gc.collect()
    
    edge_index_np = np.vstack([
        np.concatenate([src_np, dst_np]), 
        np.concatenate([dst_np, src_np])
    ])
    print(f"邊數量: {edge_index_np.shape[1]:,}")

    # --- 6. 加入圖特徵並標準化 ---
    print("=" * 80)
    print("計算圖特徵並加入 DataFrame...")
    graph_feats = compute_graph_features(edge_index_np, len(id2idx))

    graph_feat_df = pd.DataFrame({
        'graph_out_degree': graph_feats['out_degree'],
        'in_degree': graph_feats['in_degree'],
        'graph_out_degree_norm': graph_feats['out_degree_norm'],
        'in_degree_norm': graph_feats['in_degree_norm'],
        'graph_pagerank_approx': graph_feats['pagerank_approx']
    }, index=X.index)

    X_with_graph = pd.concat([X, graph_feat_df], axis=1)
    print(f"加入圖特徵後: {X_with_graph.shape[1]} 個特徵")

    X_final = remove_redundant_features(X_with_graph)
    print(f"移除冗餘特徵後: {X_final.shape[1]} 個特徵（最終特徵）")

    full_feature_names = list(X_final.columns)

    print("擬合 RobustScaler 並轉換特徵...")
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_final.values)
    print(f"特徵標準化完成，最終形狀: {X_scaled.shape}")

    # --- 7. 建立訓練/驗證遮罩 ---
    print("=" * 80)
    print("建立訓練/驗證遮罩...")
    trainable_indices = np.where(trainable_mask_series.values)[0]
    y_trainable_arr = y[trainable_indices]
    
    train_idx, val_idx = train_test_split(
        trainable_indices, 
        test_size=args.val_ratio, 
        stratify=y_trainable_arr, 
        random_state=args.seed
    )
    
    train_mask = np.zeros(len(id2idx), dtype=bool)
    val_mask = np.zeros(len(id2idx), dtype=bool)
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    print(f"訓練集遮罩: {train_mask.sum()} (正樣本: {y[train_idx].sum()})")
    print(f"驗證集遮罩: {val_mask.sum()} (正樣本: {y[val_idx].sum()})")

    # --- 8. 儲存所有預處理檔案 ---
    print("=" * 80)
    print(f"開始儲存預處理檔案至: {args.out_dir}")
    
    # 保存 NumPy 陣列
    np.save(os.path.join(args.out_dir, 'X_scaled.npy'), X_scaled)
    print(f"✓ 已儲存 X_scaled.npy (Shape: {X_scaled.shape})")
    
    np.save(os.path.join(args.out_dir, 'y_labels.npy'), y)
    print(f"✓ 已儲存 y_labels.npy (Shape: {y.shape})")
    
    np.save(os.path.join(args.out_dir, 'edge_index.npy'), edge_index_np)
    print(f"✓ 已儲存 edge_index.npy (Shape: {edge_index_np.shape})")
    
    np.save(os.path.join(args.out_dir, 'train_mask.npy'), train_mask)
    print(f"✓ 已儲存 train_mask.npy (Count: {train_mask.sum()})")
    
    np.save(os.path.join(args.out_dir, 'val_mask.npy'), val_mask)
    print(f"✓ 已儲存 val_mask.npy (Count: {val_mask.sum()})")
    
    # 保存 Joblib 和 JSON
    joblib.dump(scaler, os.path.join(args.out_dir, 'scaler.joblib'))
    print("✓ 已儲存 scaler.joblib")
    
    with open(os.path.join(args.out_dir, 'feature_columns.json'), 'w') as f:
        json.dump(full_feature_names, f, indent=4)
    print(f"✓ 已儲存 feature_columns.json (Features: {len(full_feature_names)})")
    
    with open(os.path.join(args.out_dir, 'id2idx.json'), 'w') as f:
        json.dump(id2idx, f, indent=4)
    print(f"✓ 已儲存 id2idx.json (Nodes: {len(id2idx)})")
    
    print("=" * 80)
    print("預處理階段完成！")
    print("=" * 80)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='v4 GraphSAGE 預處理腳本',
        epilog="""
        範例：
        python Preprocessing/preprocessing.py \\
            --transactions Data/acct_transaction.csv \\
            --alerts Data/acct_alert.csv \\
            --predicts Data/acct_predict.csv \\
            --out_dir Preprocessing/processed_data \\
            --use_planb
        """
    )
    
    parser.add_argument('--transactions', type=str, required=True, help='(必要) 交易資料路徑 (acct_transaction.csv)')
    parser.add_argument('--alerts', type=str, required=True, help='(必要) 警示帳戶路徑 (acct_alert.csv)')
    parser.add_argument('--predicts', type=str, default=None, help='(Plan B 選用) 預測帳戶路徑 (acct_predict.csv)')
    parser.add_argument('--out_dir', type=str, required=True, help='(必要) 儲存預處理檔案的輸出目錄')
    
    parser.add_argument('--val_ratio', type=float, default=0.2, help='驗證集比例 (預設: 0.2)')
    parser.add_argument('--seed', type=int, default=42, help='隨機種子 (預設: 42)')
    
    parser.add_argument('--use_planb', action='store_true', help='(選用) 啟用 "Plan B" 標籤隔離模式')
    
    args = parser.parse_args()
    main(args)