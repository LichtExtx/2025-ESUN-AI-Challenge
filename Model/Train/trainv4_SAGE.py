#!/usr/bin/env python3
"""
GraphSAGE (v4) 警示帳戶模型訓練腳本

此腳本負責：
1. 載入原始交易資料、警示帳戶標籤、(可選) 預測帳戶清單。
2. 執行 'v4' 版的特徵工程 (agg_account_features)，包含外幣特徵。
3. 處理 'Plan B' 訓練模式（標籤隔離）。
4. 建立完整的 PyTorch Geometric 圖資料物件 (Data)。
5. 訓練一個 EnhancedGraphSAGEModel (包含 Focal Loss)。
6. 評估模型 (AUPRC, F1) 並找出最佳閾值。
7. 儲存所有模型檔案 (best_model.pth, scaler.joblib, feature_columns.json 等)
   至 --out_dir，以供 'predictv4_SAGE.py' 載入使用。

使用範例:
python trainv4_SAGE.py --transactions Data/acct_transaction.csv --alerts Data/acct_alert.csv --predicts Data/acct_predict.csv --out_dir outputv4_SAGE --lr 0.0005 --hidden 1024 --dropout 0.5 --batch_size 512 --epochs 180 --use_planb
"""

import os
import argparse
import time
import json
import joblib
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.preprocessing import RobustScaler  # 使用 RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, precision_recall_curve, auc
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv, BatchNorm
from torch_geometric.loader import NeighborLoader
import random
import warnings
import gc
warnings.filterwarnings('ignore')

# 設定 PyTorch CUDA 分配策略
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

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

# ============ 輔助函數 ============

class FocalLoss(nn.Module):
    """
    Focal Loss 實現，用於處理類別不平衡問題。

    Attributes:
        alpha (float): 正樣本的權重。
        gamma (float): 聚焦參數，用於調整難易樣本的權重。
        reduction (str): 'mean', 'sum', 或 'none'。
    """
    def __init__(self, alpha=0.75, gamma=2.0, reduction='mean'):
        """
        初始化 FocalLoss。

        Args:
            alpha (float, optional): 正樣本的權重。預設為 0.75。
            gamma (float, optional): 聚焦參數。預設為 2.0。
            reduction (str, optional): loss 的
            
            降維方式。預設為 'mean'。
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        """
        Focal Loss 的前向傳播。

        Args:
            inputs (torch.Tensor): 模型的 Logits 輸出 (未經 sigmoid)。
            targets (torch.Tensor): 真實標籤 (0 或 1)。

        Returns:
            torch.Tensor: 計算出的 loss。
        """
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        
        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss

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
    
    特別檢測 TWD 以外的貨幣是否有重複金額，這是異常行為的強烈信號
    例如：同一帳戶多次使用 USD 5950.0 進行交易

    Args:
        foreign_currency_txns (defaultdict): 
            一個 defaultdict，結構為 {acct_idx: {currency: [amt1, amt2]}}。
        N (int): 總節點數 (總帳戶數)。

    Returns:
        dict: 索引為 acct_idx，值為包含以下特徵的 dict：
            - 'foreign_currency_types': 使用的外幣種類數量。
            - 'foreign_duplicate_ratio': 外幣交易中，重複金額的比例。
            - 'max_foreign_duplicate_count': 外幣交易中，單一金額重複的最大次數。
            - 'foreign_amount_entropy': 外幣交易金額的分布熵 (低熵 = 可疑)。
    """
    foreign_currency_count = np.zeros(N, dtype=np.float32)
    foreign_duplicate_ratio = np.zeros(N, dtype=np.float32)
    max_foreign_duplicate = np.zeros(N, dtype=np.float32)
    foreign_entropy = np.zeros(N, dtype=np.float32)  # 新增：金額分布熵
    
    for acct_idx, currency_dict in foreign_currency_txns.items():
        if len(currency_dict) == 0:
            continue
        
        foreign_currency_count[acct_idx] = len(currency_dict)
        
        all_foreign_amounts = []
        for currency, amounts in currency_dict.items():
            all_foreign_amounts.extend(amounts)
        
        if len(all_foreign_amounts) > 0:
            # 計算重複金額（四捨五入到小數點後2位）
            rounded_amounts = [round(amt, 2) for amt in all_foreign_amounts]
            from collections import Counter
            amount_counts = Counter(rounded_amounts)
            
            duplicates = [count for count in amount_counts.values() if count > 1]
            
            if duplicates:
                foreign_duplicate_ratio[acct_idx] = len(duplicates) / len(amount_counts)
                max_foreign_duplicate[acct_idx] = max(duplicates)
            
            # 計算金額分布熵（低熵 = 金額集中 = 可疑）
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

    此函數執行兩階段聚合：
    1. 第一遍 (Pass 1): 遍歷所有交易，收集基本統計數據和原始事件
       (例如交易列表、時間戳、對手方)。
    2. 第二遍 (Pass 2): 呼叫多個 `compute_...` 輔助函數，
       對收集到的原始事件計算更複雜的衍生特徵 (例如 Gini 係數、熵、比率等)。
    3. 最終將所有特徵彙總到一個 DataFrame 中。

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
        # 基本統計
        'out_count': np.zeros(N, dtype=np.float32),
        'in_count': np.zeros(N, dtype=np.float32),
        'out_sum': np.zeros(N, dtype=np.float32),
        'in_sum': np.zeros(N, dtype=np.float32),
        
        # 網絡
        'unique_to': defaultdict(set),
        'unique_from': defaultdict(set),
        
        # 時間
        'hour_hist': np.zeros((N, hour_bins), dtype=np.float32),
        'night_txn': np.zeros(N, dtype=np.float32),
        
        # 大額
        'large_99_out': np.zeros(N, dtype=np.float32),
        
        # 跨行
        # 'cross_bank_total_out': np.zeros(N, dtype=np.float32),
        # 'cross_bank_total_in': np.zeros(N, dtype=np.float32),
        
        # 分層微額交易
        'ultra_micro_500_out': np.zeros(N, dtype=np.float32),
        'ultra_micro_500_in': np.zeros(N, dtype=np.float32),
        'ultra_micro_out': np.zeros(N, dtype=np.float32),
        'ultra_micro_in': np.zeros(N, dtype=np.float32),
        'micro_out': np.zeros(N, dtype=np.float32),
        'micro_in': np.zeros(N, dtype=np.float32),
        
        # 近門檻
        'near_50k_out': np.zeros(N, dtype=np.float32),
        'near_50k_in': np.zeros(N, dtype=np.float32),
        'near_100k_out': np.zeros(N, dtype=np.float32),
        'near_100k_in': np.zeros(N, dtype=np.float32),
        'near_500k_out': np.zeros(N, dtype=np.float32),
        'near_500k_in': np.zeros(N, dtype=np.float32),

        # 交易記錄（用於後處理）
        'txn_out': defaultdict(list),
        'txn_in': defaultdict(list),
        'txn_timestamps': defaultdict(list),
        'cash_flows': defaultdict(list),
        'hour_dist': defaultdict(lambda: defaultdict(int)),
        'dow_dist': defaultdict(lambda: defaultdict(int)),
        
        # 對手方追蹤
        'out_partners': defaultdict(lambda: defaultdict(int)),
        'in_partners': defaultdict(lambda: defaultdict(int)),
        
        # 金額追蹤
        'amount_counts': defaultdict(lambda: defaultdict(int)),
        
        # 同時操作追蹤
        'same_minute_ops': defaultdict(list),
        
        # 幣別追蹤
        'currency_usage': defaultdict(set),
        'multi_currency_times': defaultdict(set),
        
        # 資金匯集
        'convergence_sources': defaultdict(lambda: defaultdict(list)),

        # 外幣重複金額追蹤
        'foreign_currency_txns': defaultdict(lambda: defaultdict(list)),
    }
    # 計算金額閾值
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
        
        # 基本統計
        features['out_count'][i] += 1
        features['in_count'][j] += 1
        features['out_sum'][i] += amt
        features['in_sum'][j] += amt
        
        # 網絡
        features['unique_to'][i].add(ta)
        features['unique_from'][j].add(fa)
        
        # 時間
        features['hour_hist'][i, hour % hour_bins] += 1
        if hour >= 20 or hour <= 8:
            features['night_txn'][i] += 1
        
        # 交易記錄
        if date_num > 0:
            features['txn_out'][i].append((date_num, hour, minute, amt, ta))
            features['txn_in'][j].append((date_num, hour, minute, amt, fa))
            features['txn_timestamps'][i].append((date_num, hour, minute))
            features['cash_flows'][i].append((date_num, hour, minute, -amt))
            features['cash_flows'][j].append((date_num, hour, minute, amt))
            features['hour_dist'][i][hour] += 1
            features['dow_dist'][i][date_num % 7] += 1
            
            # 資金匯集
            time_minutes = date_num * 1440 + hour * 60 + minute
            features['convergence_sources'][j][time_minutes].append(fa)
        
        # 跨行
        # from_type = str(row.get('from_acct_type', '')).strip()
        # to_type = str(row.get('to_acct_type', '')).strip()
        # if from_type and to_type and from_type != to_type:
        #     features['cross_bank_total_out'][i] += 1
        #     features['cross_bank_total_in'][j] += 1
        
        # 大額
        if amt >= p99:
            features['large_99_out'][i] += 1
        
        if 0 < amt <= 500:
            features['ultra_micro_500_out'][i] += 1
            features['ultra_micro_500_in'][j] += 1

        # 分層微額
        if 0 < amt < 105:
            features['ultra_micro_out'][i] += 1
            features['ultra_micro_in'][j] += 1
        elif 105 <= amt < 500:
            features['micro_out'][i] += 1
            features['micro_in'][j] += 1
        
        # 近門檻
        if 47000 <= amt < 51000:
            features['near_50k_out'][i] += 1
            features['near_50k_in'][j] += 1
        elif 95000 <= amt < 110000:
            features['near_100k_out'][i] += 1
            features['near_100k_in'][j] += 1
        elif 410000 <= amt < 505000:
            features['near_500k_out'][i] += 1
            features['near_500k_in'][j] += 1

        # 對手方追蹤
        features['out_partners'][i][ta] += 1
        features['in_partners'][j][fa] += 1
        
        # 金額追蹤（四捨五入到百位）
        rounded_amt = round(amt, -2)
        features['amount_counts'][i][rounded_amt] += 1
        
        # 幣別追蹤
        features['currency_usage'][i].add(currency)
        if date_num > 0:
            time_key = (date_num, hour)
            features['multi_currency_times'][i].add((time_key, currency))

        # 外幣重複金額追蹤（關鍵：只記錄 TWD 以外的貨幣）
        if currency != 'TWD':
            features['foreign_currency_txns'][i][currency].append(amt)
            features['foreign_currency_txns'][j][currency].append(amt)  # 入金方也記錄

    # 第二遍：檢測同時操作
    print("檢測同時操作...")
    for acct_idx in range(N):
        if acct_idx not in features['txn_in'] or acct_idx not in features['txn_out']:
            continue
        
        # 建立時間到交易的映射
        in_by_time = defaultdict(list)
        for date, hour, minute, amt, source in features['txn_in'][acct_idx]:
            time_key = (date, hour, minute)
            in_by_time[time_key].append(amt)
        
        out_by_time = defaultdict(list)
        for date, hour, minute, amt, target in features['txn_out'][acct_idx]:
            time_key = (date, hour, minute)
            out_by_time[time_key].append(amt)
        
        # 檢查同時操作
        common_times = set(in_by_time.keys()) & set(out_by_time.keys())
        for time_key in common_times:
            in_amt = sum(in_by_time[time_key])
            out_amt = sum(out_by_time[time_key])
            features['same_minute_ops'][acct_idx].append((time_key, in_amt, out_amt))
    
    # 後處理：計算衍生特徵
    print("計算衍生特徵...")
    
    behavior_features = compute_behavior_features(features['txn_out'], N, large_threshold)
    temporal_features = compute_temporal_features(
        features['txn_timestamps'], features['hour_dist'], features['dow_dist'], N
    )
    velocity_features = compute_velocity_features(features['cash_flows'], N)
    lifecycle_features = compute_lifecycle_features(
        features['txn_out'], features['txn_in'], N, large_threshold
    )
    convergence_features = compute_convergence_features(features['convergence_sources'], N)
    
    test_activation_features = compute_test_activation_features_v2(
        features['txn_out'], features['txn_in'], 
        features['ultra_micro_out'], features['ultra_micro_in'],
        N, large_threshold
    )
    
    counterparty_features = compute_counterparty_features(
        features['out_partners'], features['in_partners'], N
    )
    
    simultaneous_features = compute_simultaneous_features(features['same_minute_ops'], N)
    
    amount_features = compute_amount_repetition_features(features['amount_counts'], N)
    
    currency_features = compute_multi_currency_features(
        features['currency_usage'], features['multi_currency_times'], N
    )
    
    bidirectional_features = compute_bidirectional_features(
        features['out_partners'], features['in_partners'], N
    )
    
    # 外幣重複金額特徵（強力特徵）
    foreign_currency_features = compute_foreign_currency_repetition_features(
        features['foreign_currency_txns'], N
    )
     # 【新增】純入金/純出金特徵
    flow_direction_features = compute_flow_direction_features(
        features['out_count'], features['in_count'], N
    )
    
    # 【新增】時段集中度特徵
    hour_concentration_features = compute_hour_concentration_features(
        features['hour_dist'], N
    )
    # 構建 DataFrame
    print("構建特徵 DataFrame...")
    rows = []
    for acct, idx in id2idx.items():
        row = {'acct': acct}
        
        # 基本統計
        row['out_count'] = features['out_count'][idx]
        row['in_count'] = features['in_count'][idx]
        row['out_sum'] = features['out_sum'][idx]
        row['in_sum'] = features['in_sum'][idx]
        row['net_flow'] = features['out_sum'][idx] - features['in_sum'][idx]
        
        # 網絡
        row['unique_to'] = len(features['unique_to'][idx])
        row['unique_from'] = len(features['unique_from'][idx])
        
        # 時間
        hist = features['hour_hist'][idx]
        entropy = -np.sum((hist/hist.sum()) * np.log((hist/hist.sum()) + 1e-10)) if hist.sum() > 0 else 0
        total_txn = features['out_count'][idx] + features['in_count'][idx]
        row['night_txn_ratio'] = features['night_txn'][idx] / max(total_txn, 1)
        row['time_entropy'] = entropy
        
        # 跨行
        # row['cross_bank_total_out'] = features['cross_bank_total_out'][idx]
        # row['cross_bank_total_in'] = features['cross_bank_total_in'][idx]
        
        # 大額
        row['large_99_out'] = features['large_99_out'][idx]
        
        # 分層微額

        # 500元以下小額（法規定義）
        row['ultra_micro_500_out'] = features['ultra_micro_500_out'][idx]
        row['ultra_micro_500_in'] = features['ultra_micro_500_in'][idx]
        row['ultra_micro_out'] = features['ultra_micro_out'][idx]
        row['ultra_micro_in'] = features['ultra_micro_in'][idx]
        row['micro_out'] = features['micro_out'][idx]
        row['micro_in'] = features['micro_in'][idx]
        
        # 近門檻
        row['near_50k_out'] = features['near_50k_out'][idx]
        row['near_50k_in'] = features['near_50k_in'][idx]
        row['near_100k_out'] = features['near_100k_out'][idx]
        row['near_100k_in'] = features['near_100k_in'][idx]
        row['near_500k_out'] = features['near_500k_out'][idx]
        row['near_500k_in'] = features['near_500k_in'][idx]
                                        
        # 密度
        row['avg_out_amount'] = features['out_sum'][idx] / max(features['out_count'][idx], 1)
        row['avg_in_amount'] = features['in_sum'][idx] / max(features['in_count'][idx], 1)
        row['amount_volatility'] = (row['avg_out_amount'] - row['avg_in_amount']) / max(row['avg_out_amount'], 1)

        # 資金流向
        total_out = features['out_sum'][idx]
        total_in = features['in_sum'][idx]
        total_flow = total_out + total_in
        bias = (total_out - total_in) / total_flow if total_flow > 0 else 0
        row['extreme_inbound'] = 1 if bias < -0.8 else 0   # 極端入金
        row['extreme_outbound'] = 1 if bias > 0.8 else 0   # 極端出金
        
        # 合併所有衍生特徵
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
    
    # 驗證特徵完整性
    expected_features = [
        'out_partner_gini', 'in_partner_gini',
        'same_minute_ops_count', 'same_minute_match_ratio',
        'most_common_amount_count', 'multi_currency_ops',
        'bidirectional_partners', 'bidirectional_imbalance',
        'foreign_currency_types', 'foreign_duplicate_ratio', 'max_foreign_duplicate_count', 'foreign_amount_entropy',
        'inbound_only_ratio', 'outbound_only_ratio', 'peak_hour_concentration', 'hour_gini'
    ]
    
    missing = [f for f in expected_features if f not in feats_df.columns]
    if missing:
        print(f"警告：以下特徵未加入 DataFrame: {missing}")
    else:
        print("✓ 所有新增特徵已正確加入")
    
    return feats_df

# ============ 衍生特徵計算函數 ============

def compute_behavior_features(txn_out, N, large_threshold):
    """
    計算行為模式相關特徵 (例如早期 vs 近期交易)。

    Args:
        txn_out (defaultdict): 索引為 acct_idx，值為 (date, hour, minute, amt, target) 的列表。
        N (int): 總節點數。
        large_threshold (float): 大額交易的閾值。

    Returns:
        dict: 索引為 acct_idx，值為包含 'early_txn_count', 'recent_txn_count' 等特徵的 dict。
    """
    early_count = np.zeros(N, dtype=np.float32)
    recent_count = np.zeros(N, dtype=np.float32)
    behavior_change_count = np.zeros(N, dtype=np.float32)
    tail_vs_avg_ratio = np.zeros(N, dtype=np.float32)
    
    for acct_idx, txns in txn_out.items():
        if len(txns) == 0:
            continue
        
        txns_sorted = sorted(txns, key=lambda x: x[0])
        split_idx = int(len(txns_sorted) * 0.7)
        early_count[acct_idx] = split_idx
        recent_count[acct_idx] = len(txns_sorted) - split_idx
        behavior_change_count[acct_idx] = recent_count[acct_idx] - early_count[acct_idx]
        
        # 尾段爆發
        last_day = txns_sorted[-1][0]
        last_n_txns = [t for t in txns_sorted if t[0] > last_day - 8]
        if last_n_txns:
            tail_vs_avg_ratio[acct_idx] = len(last_n_txns) / max(len(txns_sorted) / (last_day - txns_sorted[0][0] + 1), 0.1)
    
    results = {}
    for idx in range(N):
        results[idx] = {
            'early_txn_count': early_count[idx],
            'recent_txn_count': recent_count[idx],
            'behavior_change_count': behavior_change_count[idx],
            'tail_vs_avg_ratio': tail_vs_avg_ratio[idx],
        }
    return results

def compute_temporal_features(txn_timestamps, hour_dist, dow_dist, N):
    """
    計算時間相關特徵 (例如 60 分鐘內爆發、時間熵)。

    Args:
        txn_timestamps (defaultdict): 索引為 acct_idx，值為 (date, hour, minute) 的列表。
        hour_dist (defaultdict): 索引為 acct_idx，值為 {hour: count} 的 dict。
        dow_dist (defaultdict): 索引為 acct_idx，值為 {day_of_week: count} 的 dict。
        N (int): 總節點數。

    Returns:
        dict: 索引為 acct_idx，值為包含 'burst_60min', 'hour_entropy' 等特徵的 dict。
    """
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
                    if time_diff <= 60:
                        count += 1
                    else:
                        break
                burst_60min = max(burst_60min, count)
                if start_hour >= 23 or start_hour <= 6:
                    midnight_count += 1
        
        # 熵
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
        
        results[idx] = {
            'burst_60min': burst_60min,
            'midnight_burst': midnight_count,
            'hour_entropy': hour_entropy,
            'dow_entropy': dow_entropy,
        }
    return results
def compute_velocity_features(cash_flows, N):
    """
    計算資金流速特徵 (例如平均停留時間、快轉次數)。

    Args:
        cash_flows (defaultdict): 索引為 acct_idx，值為 (date, hour, minute, amount) 的列表 (amount 可正可負)。
        N (int): 總節點數。

    Returns:
        dict: 索引為 acct_idx，值為包含 'fast_turnover_count', 'avg_staying_time' 特徵的 dict。
    """
    fast_turnover_count = np.zeros(N, dtype=np.float32)
    avg_staying_time = np.zeros(N, dtype=np.float32)
    
    for acct_idx, flows in cash_flows.items():
        if len(flows) < 2:
            continue
        
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
                    if time_diff < 60:
                        fast_turnover_count[acct_idx] += 1
            
            if balance > 0 and last_positive_time is None:
                last_positive_time = current_time_min
            elif balance <= 0:
                last_positive_time = None
        
        if time_diffs:
            avg_staying_time[acct_idx] = np.mean(time_diffs)
    
    results = {}
    for idx in range(N):
        results[idx] = {
            'fast_turnover_count': fast_turnover_count[idx],
            'avg_staying_time': avg_staying_time[idx],
        }
    return results

def compute_lifecycle_features(txn_out, txn_in, N, large_threshold):
    """
    計算帳戶生命週期特徵 (例如壽命、交易密度)。

    Args:
        txn_out (defaultdict): 出金交易列表。
        txn_in (defaultdict): 入金交易列表。
        N (int): 總節點數。
        large_threshold (float): 大額交易閾值。

    Returns:
        dict: 索引為 acct_idx，值為包含 'account_lifespan', 'txn_density' 等特徵的 dict。
    """
    account_lifespan = np.zeros(N, dtype=np.float32)
    single_day_turnover = np.zeros(N, dtype=np.float32)
    txn_density = np.zeros(N, dtype=np.float32)  # 新增
    short_life_burst = np.zeros(N, dtype=np.float32)  # 新增
    
    for acct_idx, txns in txn_out.items():
        if len(txns) == 0:
            continue
        
        txns_sorted = sorted(txns, key=lambda x: x[0])
        first_day = txns_sorted[0][0]
        last_day = txns_sorted[-1][0]
        lifespan = last_day - first_day + 1
        account_lifespan[acct_idx] = lifespan
        
        # 計算總交易數（包含入金）
        total_txns = len(txns_sorted)
        if acct_idx in txn_in:
            total_txns += len(txn_in[acct_idx])
        
        # 交易密度（關鍵特徵！）
        txn_density[acct_idx] = total_txns / max(lifespan, 1)
        
        # 短壽命爆發指標
        if lifespan <= 10 and total_txns >= 20:
            short_life_burst[acct_idx] = 1
        elif lifespan <= 5 and total_txns >= 10:
            short_life_burst[acct_idx] = 1
        
        # 單日週轉
        day_turnovers = defaultdict(float)
        for date, _, _, amt, _ in txns_sorted:
            day_turnovers[date] += amt
        
        if acct_idx in txn_in:
            for date, _, _, amt, _ in txn_in[acct_idx]:
                day_turnovers[date] += amt
        
        if day_turnovers:
            single_day_turnover[acct_idx] = max(day_turnovers.values())
    
    results = {}
    for idx in range(N):
        results[idx] = {
            'account_lifespan': account_lifespan[idx],
            'single_day_turnover': single_day_turnover[idx],
            'txn_density': txn_density[idx],  # 新增
            'short_life_burst': short_life_burst[idx],  # 新增
        }
    return results

def compute_convergence_features(convergence_sources, N):
    """
    計算資金匯集特徵 (例如 1 小時內最大匯集來源數)。

    Args:
        convergence_sources (defaultdict): 
            索引為 acct_idx，值為 {time_in_minutes: [source_acct1, ...]} 的 dict。
        N (int): 總節點數。

    Returns:
        dict: 索引為 acct_idx，值為包含 'convergence_1h', 'convergence_diversity' 等特徵的 dict。
    """
    convergence_1h = np.zeros(N, dtype=np.float32)
    convergence_6h = np.zeros(N, dtype=np.float32)
    convergence_24h = np.zeros(N, dtype=np.float32)
    convergence_diversity = np.zeros(N, dtype=np.float32)
    
    for acct_idx, time_sources in convergence_sources.items():
        if len(time_sources) < 2:
            continue
        
        sorted_times = sorted(time_sources.keys())
        all_sources = []
        for sources in time_sources.values():
            all_sources.extend(sources)
        
        for window_minutes in [60, 360, 1440]:
            max_unique_sources = 0
            for start_time in sorted_times:
                unique_sources = set()
                for time_key, sources in time_sources.items():
                    if start_time <= time_key <= start_time + window_minutes:
                        unique_sources.update(sources)
                max_unique_sources = max(max_unique_sources, len(unique_sources))
            
            if window_minutes == 60:
                convergence_1h[acct_idx] = max_unique_sources
            elif window_minutes == 360:
                convergence_6h[acct_idx] = max_unique_sources
            else:
                convergence_24h[acct_idx] = max_unique_sources
        
        unique_sources_total = len(set(all_sources))
        total_txns = len(all_sources)
        convergence_diversity[acct_idx] = unique_sources_total / total_txns if total_txns > 0 else 0
    
    results = {}
    for idx in range(N):
        results[idx] = {
            'convergence_1h': convergence_1h[idx],
            'convergence_6h': convergence_6h[idx],
            'convergence_24h': convergence_24h[idx],
            'convergence_diversity': convergence_diversity[idx],
        }
    return results

def compute_test_activation_features_v2(txn_out, txn_in, ultra_micro_out, ultra_micro_in, N, large_threshold):
    """
    計算 "測試-啟用" 特徵 v2 (例如 "小額入金" 後接 "大額交易")。

    Args:
        txn_out (defaultdict): 出金交易列表。
        txn_in (defaultdict): 入金交易列表。
        ultra_micro_out (np.ndarray): 微額出金計數。
        ultra_micro_in (np.ndarray): 微額入金計數。
        N (int): 總節點數。
        large_threshold (float): 大額交易閾值。

    Returns:
        dict: 索引為 acct_idx，值為包含 'test_then_large_in', 'any_micro_then_large_out' 等特徵的 dict。
    """
    test_then_large_in = np.zeros(N, dtype=np.float32)
    test_then_large_out = np.zeros(N, dtype=np.float32)
    any_micro_then_large_in = np.zeros(N, dtype=np.float32)
    any_micro_then_large_out = np.zeros(N, dtype=np.float32)
    
    for acct_idx, txns in txn_in.items():
        if len(txns) < 2:
            continue
        
        txns_sorted = sorted(txns, key=lambda x: (x[0], x[1], x[2]))
        
        # 第一筆是微額
        first_amt = txns_sorted[0][3]
        if first_amt < 200:
            subsequent_large = [t[3] for t in txns_sorted[1:] if t[3] >= large_threshold]
            if subsequent_large:
                test_then_large_in[acct_idx] = len(subsequent_large)
        
        # 任意位置微額
        for i, (date, hour, minute, amt, source) in enumerate(txns_sorted):
            if amt < 200:
                subsequent_large = [t[3] for t in txns_sorted[i+1:] if t[3] >= large_threshold]
                if subsequent_large:
                    any_micro_then_large_in[acct_idx] = 1
                    break
    
    for acct_idx, txns in txn_out.items():
        if len(txns) < 2:
            continue
        
        txns_sorted = sorted(txns, key=lambda x: (x[0], x[1], x[2]))
        first_amt = txns_sorted[0][3]
        
        if first_amt < 200:
            subsequent_large = [t[3] for t in txns_sorted[1:] if t[3] >= large_threshold]
            if subsequent_large:
                test_then_large_out[acct_idx] = len(subsequent_large)
        
        for i, (date, hour, minute, amt, target) in enumerate(txns_sorted):
            if amt < 200:
                subsequent_large = [t[3] for t in txns_sorted[i+1:] if t[3] >= large_threshold]
                if subsequent_large:
                    any_micro_then_large_out[acct_idx] = 1
                    break
    
    results = {}
    for idx in range(N):
        results[idx] = {
            'test_then_large_in': test_then_large_in[idx],
            'test_then_large_out': test_then_large_out[idx],
            'any_micro_then_large_in': any_micro_then_large_in[idx],
            'any_micro_then_large_out': any_micro_then_large_out[idx],
        }
    return results

def compute_counterparty_features(out_partners, in_partners, N):
    """
    計算對手方特徵 (例如 Gini 係數衡量對手方集中度)。

    Args:
        out_partners (defaultdict): 索引為 acct_idx，值為 {target_acct: count} 的 dict。
        in_partners (defaultdict): 索引為 acct_idx，值為 {source_acct: count} 的 dict。
        N (int): 總節點數。

    Returns:
        dict: 索引為 acct_idx，值為包含 'out_partner_gini', 'in_partner_gini' 特徵的 dict。
    """
    out_gini = np.zeros(N, dtype=np.float32)
    in_gini = np.zeros(N, dtype=np.float32)
    
    for acct_idx in range(N):
        # 出金對手方
        if acct_idx in out_partners and len(out_partners[acct_idx]) > 0:
            counts = list(out_partners[acct_idx].values())
            out_gini[acct_idx] = compute_gini(counts)
        
        # 入金對手方
        if acct_idx in in_partners and len(in_partners[acct_idx]) > 0:
            counts = list(in_partners[acct_idx].values())
            in_gini[acct_idx] = compute_gini(counts)
    
    results = {}
    for idx in range(N):
        results[idx] = {
            'out_partner_gini': out_gini[idx],
            'in_partner_gini': in_gini[idx],
        }
    return results

def compute_simultaneous_features(same_minute_ops, N):
    """
    計算同時操作特徵 (例如同一分鐘內同時發生入金和出金)。

    Args:
        same_minute_ops (defaultdict): 
            索引為 acct_idx，值為 (time_key, in_amt, out_amt) 的列表。
        N (int): 總節點數。

    Returns:
        dict: 索引為 acct_idx，值為包含 'same_minute_ops_count', 'same_minute_match_ratio' 特徵的 dict。
    """
    same_minute_count = np.zeros(N, dtype=np.float32)
    same_minute_match_ratio = np.zeros(N, dtype=np.float32)
    
    for acct_idx, ops in same_minute_ops.items():
        same_minute_count[acct_idx] = len(ops)
        
        # 計算金額匹配度（入金≈出金）
        if len(ops) > 0:
            matches = 0
            for time_key, in_amt, out_amt in ops:
                if abs(in_amt - out_amt) / max(in_amt, out_amt, 1) < 0.1:
                    matches += 1
            same_minute_match_ratio[acct_idx] = matches / len(ops)
    
    results = {}
    for idx in range(N):
        results[idx] = {
            'same_minute_ops_count': same_minute_count[idx],
            'same_minute_match_ratio': same_minute_match_ratio[idx],
        }
    return results
def compute_amount_repetition_features(amount_counts, N):
    """
    計算金額重複特徵 (例如最常見金額的出現次數)。

    Args:
        amount_counts (defaultdict): 
            索引為 acct_idx，值為 {rounded_amount: count} 的 dict。
        N (int): 總節點數。

    Returns:
        dict: 索引為 acct_idx，值為包含 'most_common_amount_count' 特徵的 dict。
    """
    most_common_amount_count = np.zeros(N, dtype=np.float32)
    
    for acct_idx, amounts in amount_counts.items():
        if len(amounts) > 0:
            counts = list(amounts.values())
            most_common_amount_count[acct_idx] = max(counts)
    
    results = {}
    for idx in range(N):
        results[idx] = {
            'most_common_amount_count': most_common_amount_count[idx],
        }
    return results
def compute_multi_currency_features(currency_usage, multi_currency_times, N):
    """
    計算多幣別操作特徵 (例如同一小時內使用多種幣別)。

    Args:
        currency_usage (defaultdict): 索引為 acct_idx，值為 {currency_str} 的 set。
        multi_currency_times (defaultdict): 
            索引為 acct_idx，值為 {((date, hour), currency)} 的 set。
        N (int): 總節點數。

    Returns:
        dict: 索引為 acct_idx，值為包含 'multi_currency_ops' 特徵的 dict。
    """
    multi_currency_operations = np.zeros(N, dtype=np.float32)
    
    for acct_idx in range(N):
        if acct_idx in multi_currency_times:
            # 統計同一時間點的多幣別操作
            time_to_currencies = defaultdict(set)
            for (time_key, currency) in multi_currency_times[acct_idx]:
                time_to_currencies[time_key].add(currency)
            
            multi_count = sum(1 for currencies in time_to_currencies.values() if len(currencies) > 1)
            multi_currency_operations[acct_idx] = multi_count
    
    results = {}
    for idx in range(N):
        results[idx] = {
            'multi_currency_ops': multi_currency_operations[idx],
        }
    return results

def compute_bidirectional_features(out_partners, in_partners, N):
    """
    計算雙向流動特徵 (例如與多少對手方同時存在雙向交易)。

    Args:
        out_partners (defaultdict): 出金對手方 {target_acct: count}。
        in_partners (defaultdict): 入金對手方 {source_acct: count}。
        N (int): 總節點數。

    Returns:
        dict: 索引為 acct_idx，值為包含 'bidirectional_partners', 'bidirectional_imbalance' 特徵的 dict。
    """
    bidirectional_partners = np.zeros(N, dtype=np.float32)
    bidirectional_imbalance = np.zeros(N, dtype=np.float32)
    
    for acct_idx in range(N):
        if acct_idx not in out_partners or acct_idx not in in_partners:
            continue
        
        out_set = set(out_partners[acct_idx].keys())
        in_set = set(in_partners[acct_idx].keys())
        common = out_set & in_set
        
        high_freq_bidirectional = 0
        for partner in common:
            out_count = out_partners[acct_idx][partner]
            in_count = in_partners[acct_idx][partner]
            total = out_count + in_count
            if total >= 3:  # 只計算互動3次以上的
                high_freq_bidirectional += 1

        bidirectional_partners[acct_idx] = high_freq_bidirectional
        
        # 計算雙向不平衡度
        if len(common) > 0:
            imbalances = []
            for partner in common:
                out_count = out_partners[acct_idx][partner]
                in_count = in_partners[acct_idx][partner]
                imbalance = abs(out_count - in_count) / (out_count + in_count)
                imbalances.append(imbalance)
            bidirectional_imbalance[acct_idx] = np.mean(imbalances)
    
    results = {}
    for idx in range(N):
        results[idx] = {
            'bidirectional_partners': bidirectional_partners[idx],
            'bidirectional_imbalance': bidirectional_imbalance[idx],
        }
    return results

def compute_flow_direction_features(out_count, in_count, N):
    """
    純入金/純出金特徵
    
    異常模式：
    - 只有入金沒有出金（資金匯集型人頭帳戶）
    - 接近純入金（inbound_ratio > 0.8）

    Args:
        out_count (np.ndarray): 每個帳戶的出金次數。
        in_count (np.ndarray): 每個帳戶的入金次數。
        N (int): 總節點數。

    Returns:
        dict: 索引為 acct_idx，值為包含 'inbound_only_ratio', 'outbound_only_ratio' 特徵的 dict。
    """
    inbound_only_ratio = np.zeros(N, dtype=np.float32)
    outbound_only_ratio = np.zeros(N, dtype=np.float32)
    
    for idx in range(N):
        total_txn = out_count[idx] + in_count[idx]
        if total_txn > 0:
            inbound_only_ratio[idx] = in_count[idx] / total_txn
            outbound_only_ratio[idx] = out_count[idx] / total_txn
    
    results = {}
    for idx in range(N):
        results[idx] = {
            'inbound_only_ratio': inbound_only_ratio[idx],
            'outbound_only_ratio': outbound_only_ratio[idx],
        }
    return results


def compute_hour_concentration_features(hour_dist, N):
    """
    時段集中度特徵
    
    異常模式：
    - 所有交易都集中在固定時段（如早上 9-11 點）
    - 固定時段交易（機器人行為）

    Args:
        hour_dist (defaultdict): 索引為 acct_idx，值為 {hour: count} 的 dict。
        N (int): 總節點數。

    Returns:
        dict: 索引為 acct_idx，值為包含 'peak_hour_concentration', 'hour_gini' 特徵的 dict。
    """
    peak_hour_concentration = np.zeros(N, dtype=np.float32)
    hour_gini = np.zeros(N, dtype=np.float32)
    
    for acct_idx, hours in hour_dist.items():
        if len(hours) < 2:
            continue
        
        total = sum(hours.values())
        
        # 計算最高峰時段的集中度
        max_hour_count = max(hours.values())
        peak_hour_concentration[acct_idx] = max_hour_count / total
        
        # 計算 Gini 係數（衡量分布不均度）
        counts = sorted(hours.values())
        hour_gini[acct_idx] = compute_gini(counts)
    
    results = {}
    for idx in range(N):
        results[idx] = {
            'peak_hour_concentration': peak_hour_concentration[idx],
            'hour_gini': hour_gini[idx],
        }
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
    
    # 對數轉換
    amount_features = ['out_sum', 'in_sum', 'net_flow', 
                       'avg_staying_time', 'single_day_turnover']
    
    for feat in amount_features:
        if feat in feats_df.columns:
            if feat in ['net_flow']:
                feats_df[f'{feat}_log'] = np.sign(feats_df[feat]) * np.log1p(np.abs(feats_df[feat]))
            else:
                feats_df[f'{feat}_log'] = np.log1p(np.maximum(feats_df[feat], 0))
    
    # 標準化熵
    entropy_features = ['time_entropy', 'hour_entropy', 'dow_entropy']
    max_entropy = 0
    for feat in entropy_features:
        if feat in feats_df.columns:
            max_entropy = max(max_entropy, feats_df[feat].max())
    
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
    
    max_degree = max(out_degree.max(), in_degree.max())
    out_degree_norm = out_degree / (max_degree + 1e-8)
    in_degree_norm = in_degree / (max_degree + 1e-8)
    
    total_degree = out_degree + in_degree
    pagerank_approx = total_degree / (total_degree.sum() + 1e-8)
    
    return {
        'out_degree': out_degree,
        'in_degree': in_degree,
        'out_degree_norm': out_degree_norm,
        'in_degree_norm': in_degree_norm,
        'pagerank_approx': pagerank_approx
    }

# ============ 模型定義 ============

class EnhancedGraphSAGEModel(nn.Module):
    """
    增強的 GraphSAGE 模型。

    結構:
    - 多層 SAGEConv (含 skip connection)
    - 多層 MLP 分類器

    Attributes:
        num_layers (int): GraphSAGE 層數。
        dropout (float): Dropout 比例。
        convs (nn.ModuleList): GraphSAGE 卷積層列表。
        bns (nn.ModuleList): BatchNorm 層列表。
        classifier (nn.Sequential): 最終的 MLP 分類器。
    """
    def __init__(self, in_channels, hidden=512, dropout=0.3, num_layers=3):
        """
        初始化 EnhancedGraphSAGEModel。

        Args:
            in_channels (int): 輸入特徵的維度。
            hidden (int, optional): 隱藏層的維度。預設為 512。
            dropout (float, optional): Dropout 比例。預設為 0.3。
            num_layers (int, optional): GraphSAGE 層數。預設為 3。
        """
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        self.convs.append(SAGEConv(in_channels, hidden))
        self.bns.append(BatchNorm(hidden))
        
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden, hidden))
            self.bns.append(BatchNorm(hidden))
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.BatchNorm1d(hidden),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden // 2),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, hidden // 4),
            nn.ReLU(),
            nn.BatchNorm1d(hidden // 4),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden // 4, 1)
        )
        
    def forward(self, x, edge_index):
        """
        模型的前向傳播。

        Args:
            x (torch.Tensor): 節點特徵張量。
            edge_index (torch.Tensor): 邊索引張量。

        Returns:
            torch.Tensor: 模型的 Logits 輸出 (未經 sigmoid)。
        """
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            x_new = conv(x, edge_index)
            x_new = bn(x_new)
            x_new = F.relu(x_new)
            x_new = F.dropout(x_new, p=self.dropout, training=self.training)
            
            if i > 0:
                x = x + x_new  # Skip connection
            else:
                x = x_new
        
        return self.classifier(x).squeeze(-1)
    
def evaluate_with_auprc(model, data, mask, device, batch_size=2048):
    """
    評估模型性能，使用 AUPRC (Precision-Recall 曲線下面積) 和 F1 Score。

    此函數會自動尋找最佳 F1 Score 對應的閾值。

    Args:
        model (nn.Module): 待評估的模型。
        data (torch_geometric.data.Data): 完整的圖資料。
        mask (torch.Tensor): 評估用的遮罩 (例如 val_mask)。
        device (torch.device): 執行評估的裝置。
        batch_size (int, optional): 評估時的批次大小。預設為 2048。

    Returns:
        dict: 包含 'auprc', 'f1', 'precision', 'recall', 'best_threshold' 的指標字典。
    """
    model.eval()
    mask_indices = torch.where(mask)[0]
    if len(mask_indices) == 0:
        return {'auprc': 0.0, 'f1': 0.0, 'precision': 0.0, 'recall': 0.0, 'best_threshold': 0.5}
    
    loader = NeighborLoader(data, num_neighbors=[25, 20, 15], batch_size=batch_size,
                            input_nodes=mask_indices, shuffle=False)
    
    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index)
            probs = torch.sigmoid(logits[:batch.batch_size]).cpu().numpy()
            labels = batch.y[:batch.batch_size].cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels)
    
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    precision_arr, recall_arr, _ = precision_recall_curve(all_labels, all_probs)
    auprc = auc(recall_arr, precision_arr)
    
    best_f1 = 0
    best_thresh = 0.5
    for thresh in np.arange(0.05, 0.6, 0.02):
        y_pred = (all_probs >= thresh).astype(int)
        f1 = f1_score(all_labels, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
    
    y_pred_best = (all_probs >= best_thresh).astype(int)
    prec = precision_score(all_labels, y_pred_best, zero_division=0)
    rec = recall_score(all_labels, y_pred_best, zero_division=0)
    
    return {
        'auprc': auprc,
        'f1': best_f1,
        'precision': prec,
        'recall': rec,
        'best_threshold': best_thresh
    }

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
    
    feature_std = pd.DataFrame({
        'feature': feature_cols,
        'std': feats_df[feature_cols].std().values
    }).sort_values('std', ascending=False)
    feature_std.to_csv(os.path.join(output_dir, 'feature_std.csv'), index=False)
    
    alert_mask = (y == 1)
    normal_mask = (y == 0)
    
    feature_comparison = pd.DataFrame({
        'feature': feature_cols,
        'alert_mean': feats_df.loc[alert_mask, feature_cols].mean().values if alert_mask.sum() > 0 else 0,
        'normal_mean': feats_df.loc[normal_mask, feature_cols].mean().values if normal_mask.sum() > 0 else 0,
    })
    
    feature_comparison['diff_ratio'] = (
        (feature_comparison['alert_mean'] - feature_comparison['normal_mean']) / 
        (feature_comparison['normal_mean'].abs() + 1e-8)
    )
    feature_comparison = feature_comparison.sort_values('diff_ratio', key=abs, ascending=False)
    feature_comparison.to_csv(os.path.join(output_dir, 'feature_comparison.csv'), index=False)
    
    print(f"特徵分析已儲存: {output_dir}")

# ============ 主函數 ============

def main(args):
    """
    訓練腳本的主執行函數。

    解析命令列參數，並依序執行以下步驟：
    1. 設定隨機種子、建立輸出目錄。
    2. 載入資料、建立節點索引。
    3. 執行特徵工程 (agg_account_features)。
    4. 處理標籤 (包含 Plan B 邏輯)。
    5. 建立圖結構、計算圖特徵。
    6. 特徵對齊、標準化、儲存模型檔案 (scaler, feature_columns)。
    7. 劃分訓練/驗證集。
    8. 初始化模型、優化器、Loss (FocalLoss)。
    9. 執行訓練循環 (training loop)，包含早停 (early stopping)。
    10. 儲存最佳模型 (best_model.pth) 和訓練摘要 (training_summary.json)。

    Args:
        args (argparse.Namespace): 包含所有命令列參數的物件。
    """
    seed_everything(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    
    print("=" * 80)
    print(f"模式: {'Plan B (完整特徵 + 標籤隔離)' if args.use_planb else '完整標籤訓練'}")
    print("=" * 80)
    
    # 載入數據
    print("載入交易資料...")
    trans_df = pd.read_csv(args.transactions)
    print(f"交易記錄數: {len(trans_df):,}")
    
    trans_df['from_acct'] = trans_df['from_acct'].astype(str)
    trans_df['to_acct'] = trans_df['to_acct'].astype(str)
    
    print("建立節點索引...")
    id2idx = build_node_index(trans_df)
    print(f"總帳戶數: {len(id2idx):,}")
    
    # Plan B 策略說明
    predict_set = set()
    if args.use_planb and args.predicts and os.path.exists(args.predicts):
        predict_df = pd.read_csv(args.predicts)
        predict_df['acct'] = predict_df['acct'].astype(str)
        predict_set = set(predict_df['acct'].values)
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
    
    # 特徵工程（關鍵：使用全部交易，不過濾）
    print("=" * 80)
    print("開始特徵工程...")
    print("使用全部交易計算特徵（與預測時保持一致）")
    t_start = time.time()
    feats_df = agg_account_features(trans_df, id2idx, hour_bins=24)
    print(f"耗時: {time.time() - t_start:.1f}秒")
    
    feats_df = process_extreme_features(feats_df)
    # feats_df = remove_redundant_features(feats_df)  # 移到後面
    feats_df = feats_df.fillna(0.0)
    
    # 載入標籤
    print("=" * 80)
    print("載入警示帳戶標籤...")
    alerts_df = pd.read_csv(args.alerts)
    alerts_df['acct'] = alerts_df['acct'].astype(str)
    alert_set = set(alerts_df['acct'].values)
    print(f"警示帳戶數: {len(alert_set):,}")
    
    # 標籤分配策略
    if args.use_planb:
        # Plan B：預測帳戶標為 NaN（關鍵：不參與訓練）
        feats_df['label'] = feats_df.index.map(
            lambda x: 1 if x in alert_set else (np.nan if x in predict_set else 0)
        )
        trainable_mask = ~feats_df['label'].isna()
        feats_df_trainable = feats_df[trainable_mask].copy()
        feats_df_trainable['label'] = feats_df_trainable['label'].astype(int)
        
        print(f"\n標籤分配 (Plan B):")
        print(f"  警示帳戶: {len(alert_set):,}")
        print(f"  正常帳戶: {(~feats_df['label'].isna()).sum() - len(alert_set):,}")
        print(f"  預測帳戶: {len(predict_set):,} (標籤=NaN, 不參與訓練)")
    else:
        feats_df['label'] = feats_df.index.map(lambda x: 1 if x in alert_set else 0)
        trainable_mask = pd.Series([True] * len(feats_df), index=feats_df.index)
        feats_df_trainable = feats_df.copy()
        
        print(f"\n標籤分配 (完整模式):")
        print(f"  警示帳戶: {len(alert_set):,}")
        print(f"  正常帳戶: {len(feats_df) - len(alert_set):,}")
    
    y_trainable = feats_df_trainable['label']
    print(f"\n可訓練樣本數: {len(y_trainable):,}")
    print(f"正樣本比例: {y_trainable.mean():.4f}")
    
    # 生成特徵分析
    save_feature_analysis(feats_df_trainable, y_trainable, args.out_dir)
    
    # 準備數據
    feats_df['label'] = feats_df['label'].fillna(0)
    X = feats_df.drop(columns=['label'])
    y = feats_df['label'].astype(int)
    
    print(f"\n特徵維度: {X.shape[1]}")
    print(f"特徵列表已儲存，共 {X.shape[1]} 個特徵")
    
    del trans_df
    gc.collect()
    
    # 先不標準化和保存，移到加入圖特徵並移除冗餘特徵之後
    # 保存 id2idx（這個可以先存）    
    with open(os.path.join(args.out_dir, 'id2idx.json'), 'w') as f:
        json.dump(id2idx, f)
    
    # 建立圖結構（關鍵：使用完整交易）
    print("=" * 80)
    print("建立圖結構...")
    
    trans_df_graph = pd.read_csv(args.transactions)
    trans_df_graph['from_acct'] = trans_df_graph['from_acct'].astype(str)
    trans_df_graph['to_acct'] = trans_df_graph['to_acct'].astype(str)
    
    src = trans_df_graph['from_acct'].map(id2idx)
    dst = trans_df_graph['to_acct'].map(id2idx)
    mask_valid = (~src.isna()) & (~dst.isna())
    src = src[mask_valid].astype(int).values
    dst = dst[mask_valid].astype(int).values
    
    del trans_df_graph
    gc.collect()
    
    edge_index = np.vstack([
        np.concatenate([src, dst]), 
        np.concatenate([dst, src])
    ])
    edge_index_tensor = torch.tensor(edge_index, dtype=torch.long)
    print(f"邊數量: {edge_index_tensor.shape[1]:,}")
    
    if args.use_planb:
        print(f"  → 圖包含預測帳戶的所有邊（GNN 可從鄰居學習特徵）")
    
    print("計算圖特徵並加入 DataFrame...")
    graph_feats = compute_graph_features(edge_index, len(id2idx))

    # 建立圖特徵 DataFrame（使用與 feats_df 相同的索引）
    graph_feat_df = pd.DataFrame({
        'graph_out_degree': graph_feats['out_degree'],
        'graph_in_degree': graph_feats['in_degree'],
        'graph_out_degree_norm': graph_feats['out_degree_norm'],
        'graph_in_degree_norm': graph_feats['in_degree_norm'],
        'graph_pagerank_approx': graph_feats['pagerank_approx']
    }, index=X.index)

    # 合併到原始特徵 DataFrame
    X_with_graph = pd.concat([X, graph_feat_df], axis=1)
    print(f"加入圖特徵後: {X_with_graph.shape[1]} 個特徵")

    # 關鍵：現在才移除冗餘特徵（包括圖特徵如果在 FEATURES_TO_REMOVE 中）
    X_final = remove_redundant_features(X_with_graph)
    print(f"移除冗餘特徵後: {X_final.shape[1]} 個特徵（最終特徵）")

    # 更新特徵列表（這才是真正要保存的）
    full_feature_names = list(X_final.columns)

    # 標準化（使用最終特徵）
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_final.values)
    # 保存標準化器和特徵列表（使用最終特徵）
    joblib.dump(scaler, os.path.join(args.out_dir, 'scaler.joblib'))
    with open(os.path.join(args.out_dir, 'feature_columns.json'), 'w') as f:
        json.dump(full_feature_names, f)  # 保存最終特徵名稱

    print(f"✓ 已保存 {len(full_feature_names)} 個最終特徵")
    print(f"最終特徵維度: {X_scaled.shape[1]}")
    x = torch.tensor(X_scaled, dtype=torch.float)  # 使用 X_scaled 而非 X_scaled_with_graph
    y_tensor = torch.tensor(y.values, dtype=torch.float)
    data = Data(x=x, edge_index=edge_index_tensor, y=y_tensor)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用設備: {device}")
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # 劃分訓練/驗證集
    print("=" * 80)
    print("劃分訓練/驗證集...")
    
    trainable_indices = np.where(trainable_mask.values)[0]
    y_trainable_arr = y.values[trainable_indices]
    
    train_idx, val_idx = train_test_split(
        trainable_indices, 
        test_size=args.val_ratio, 
        stratify=y_trainable_arr, 
        random_state=args.seed
    )
    
    train_mask = torch.zeros(x.size(0), dtype=torch.bool)
    val_mask = torch.zeros(x.size(0), dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    data.train_mask = train_mask
    data.val_mask = val_mask
    
    pos_train = int(y_tensor[train_mask].sum())
    neg_train = int(train_mask.sum()) - pos_train
    print(f"訓練集: {train_mask.sum()}, 正樣本: {pos_train}")
    
    # 初始化模型
    print("=" * 80)
    print("初始化模型...")
    
    model = EnhancedGraphSAGEModel(
        in_channels=x.size(1), 
        hidden=args.hidden,
        dropout=args.dropout,
        num_layers=args.num_layers
    ).to(device)
    
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=args.lr, 
        weight_decay=args.weight_decay
    )
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'max', patience=10, factor=0.5, verbose=True
    )
    
    # Focal Loss
    pos_ratio = pos_train / (pos_train + neg_train)
    if pos_ratio < 0.001:
        focal_alpha, focal_gamma = 0.80, 2.0
    elif pos_ratio < 0.01:
        focal_alpha, focal_gamma = 0.75, 2.0
    else:
        focal_alpha, focal_gamma = min(0.90, 1 - pos_ratio), 2.0
    
    criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
    print(f"Focal Loss: alpha={focal_alpha:.2f}, gamma={focal_gamma}")
    print(f"模型參數量: {sum(p.numel() for p in model.parameters()):,}")
    
    train_loader = NeighborLoader(
        data,
        num_neighbors=[25, 20, 15],
        batch_size=args.batch_size,
        input_nodes=train_mask,
        shuffle=True,
        num_workers=0
    )
    
    # 訓練循環
    print("=" * 80)
    print("開始訓練...")
    best_val_f1 = -1.0
    best_val_auprc = -1.0
    best_metrics = {}
    best_threshold = 0.5
    patience_counter = 0
    
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        
        model.train()
        total_loss = 0
        num_batches = 0
        
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            
            logits = model(batch.x, batch.edge_index)
            loss = criterion(logits[:batch.batch_size], batch.y[:batch.batch_size])
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            del batch, logits, loss
        
        avg_loss = total_loss / max(num_batches, 1)
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        metrics_val = evaluate_with_auprc(model, data, data.val_mask, device, batch_size=2048)
        
        t1 = time.time()
        scheduler.step(metrics_val['f1'])
        current_lr = optimizer.param_groups[0]['lr']
        
        if epoch % args.print_every == 0 or epoch == 1:
            print(f"Epoch {epoch:03d} | Loss: {avg_loss:.6f} | LR: {current_lr:.2e} | "
                  f"F1: {metrics_val['f1']:.4f} | AUPRC: {metrics_val['auprc']:.4f} | "
                  f"P: {metrics_val['precision']:.4f} | R: {metrics_val['recall']:.4f} | "
                  f"T: {metrics_val['best_threshold']:.2f} | Time: {(t1-t0):.1f}s")
        
        if metrics_val['f1'] > best_val_f1 + 1e-6:
            best_val_f1 = metrics_val['f1']
            best_val_auprc = metrics_val['auprc']
            best_metrics = metrics_val.copy()
            best_threshold = metrics_val['best_threshold']
            patience_counter = 0
            
            best_state = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'best_val_f1': best_val_f1,
                'best_val_auprc': best_val_auprc,
                'best_threshold': best_threshold,
                'metrics': best_metrics
            }
            torch.save(best_state, os.path.join(args.out_dir, 'best_model.pth'))
            
            if epoch % args.print_every == 0 or epoch == 1:
                print(f"  → 最佳模型")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"早停！")
                break
    
    # 儲存摘要
    print("=" * 80)
    summary = {
        'mode': 'planb_correct' if args.use_planb else 'full_label',
        'planb_strategy': {
            'feature_engineering': 'use_all_transactions',
            'label_setting': 'predict_accts_nan',
            'graph_structure': 'include_predict_acct_edges',
            'training_mask': 'exclude_predict_accts'
        } if args.use_planb else None,
        'best_val_f1': best_val_f1,
        'best_val_auprc': best_val_auprc,
        'best_threshold': best_threshold,
        'best_metrics': best_metrics,
        'num_features': len(full_feature_names),
        'model_hyperparameters': {
            'hidden': args.hidden,
            'dropout': args.dropout,
            'num_layers': args.num_layers,
            'lr': args.lr,
        },
        'focal_loss': {
            'alpha': focal_alpha,
            'gamma': focal_gamma,
        },
    }
    
    with open(os.path.join(args.out_dir, 'training_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("訓練完成！")
    print(f"最佳 F1: {best_val_f1:.4f} | AUPRC: {best_val_auprc:.4f}")
    print(f"檔案已儲存至: {args.out_dir}")
    
    if args.use_planb:
        print("\n" + "=" * 80)
        print("Plan B 模式總結")
        print("=" * 80)
        print("✓ 特徵工程：使用全部交易計算特徵")
        print("✓ 標籤隔離：預測帳戶不參與訓練")
        print("✓ 圖結構：保留預測帳戶的完整連接")
        print("✓ 預測時：使用相同的特徵計算方式")
        print("=" * 80)
    
    print("=" * 80)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='優化版 GraphSAGE 訓練（v4 - 含外幣特徵）',
        epilog="""
        範例：
        python trainv4_SAGE.py \\
            --transactions Data/acct_transaction.csv \\
            --alerts Data/acct_alert.csv \\
            --predicts Data/acct_predict.csv \\
            --out_dir outputv4_SAGE \\
            --lr 0.0005 --hidden 1024 --dropout 0.5 --batch_size 512 --epochs 180 \\
            --use_planb
        """
    )
    
    parser.add_argument('--transactions', type=str, required=True, help='(必要) 交易資料路徑 (acct_transaction.csv)')
    parser.add_argument('--alerts', type=str, required=True, help='(必要) 警示帳戶路徑 (acct_alert.csv)')
    parser.add_argument('--predicts', type=str, default=None, help='(Plan B 選用) 預測帳戶路徑 (acct_predict.csv)')
    parser.add_argument('--out_dir', type=str, default='outputv4_SAGE', help='輸出目錄 (預設: outputv4_SAGE)')
    
    parser.add_argument('--epochs', type=int, default=200, help='訓練輪數 (預設: 200)')
    parser.add_argument('--lr', type=float, default=5e-4, help='學習率 (預設: 0.0005)')
    parser.add_argument('--hidden', type=int, default=1024, help='隱藏層維度 (預設: 1024)')
    parser.add_argument('--num_layers', type=int, default=3, help='GNN 層數 (預設: 3)')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout 率 (預設: 0.5)')
    parser.add_argument('--weight_decay', type=float, default=5e-6, help='權重衰減 (預設: 5e-6)') #1e-5
    parser.add_argument('--batch_size', type=int, default=512, help='批次大小 (預設: 512)')
    
    parser.add_argument('--val_ratio', type=float, default=0.2, help='驗證集比例 (預設: 0.2)')
    parser.add_argument('--patience', type=int, default=20, help='早停耐心值 (預設: 20)')
    parser.add_argument('--seed', type=int, default=42, help='隨機種子 (預設: 42)')
    parser.add_argument('--print_every', type=int, default=5, help='打印頻率 (預設: 5)')
    
    parser.add_argument('--use_planb', action='store_true', help='(選用) 啟用 Plan B 模式（完整特徵 + 標籤隔離）')
    
    args = parser.parse_args()
    main(args)