#!/usr/bin/env python3
"""
GraphSAGE (v4) 警示帳戶預測腳本 (Inference Script)

此腳本負責載入由 'trainv4_SAGE.py' 訓練好的模型檔案，
並對新的交易資料和預測帳戶清單執行預測。

核心功能：
1. 載入模型檔案 (模型權重、Scaler、特徵列表、帳戶索引)。
2. 從 'trainv4_SAGE.py' 導入特徵工程函數。
3. 完整重現訓練時的特徵工程 (包含外幣特徵、圖特徵)。
4. 嚴格執行特徵對齊。
5. 使用 NeighborLoader 進行小批次 (mini-batch) 推論。
6. 根據閾值產生最終預測結果。

使用範例:
python predictv4_SAGE.py --transactions Data/acct_transaction.csv --model_dir outputv4_SAGE --acct_predict Data/acct_predict.csv --output outputv4_SAGE/output.csv --use_planb --threshold 0.25
"""

import os
import argparse
import json
import joblib
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
import warnings
import gc
warnings.filterwarnings('ignore')

# 從訓練腳本導入必要函數
import sys
try:
    # 這些函數必須與 trainv4_SAGE.py 中定義的完全一致
    from trainv4_SAGE import (
        agg_account_features,
        process_extreme_features,
        EnhancedGraphSAGEModel,
        compute_graph_features,
        remove_redundant_features,
    )
except ImportError:
    print("錯誤：無法導入 'trainv4_SAGE.py'。")
    print("請確保 'trainv4_SAGE.py' 與此 'predictv4_SAGE.py' 腳本位於同一目錄中。")
    sys.exit(1)

# 設定 PyTorch CUDA 分配策略
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

def load_artifacts(model_dir):
    """
    從指定的模型目錄載入所有訓練好的模型檔案 (artifacts)。

    Args:
        model_dir (str): 包含模型檔案 (scaler, model, configs) 的目錄路徑。

    Returns:
        tuple: 包含以下元素的元組：
            - id2idx (dict): 帳戶 ID 對 節點索引 的映射。
            - feature_cols (list): 訓練時使用的特徵欄位列表。
            - scaler (object): 已訓練的 scikit-learn scaler 物件。
            - model_state_dict (dict): PyTorch 模型的 state_dict。
            - best_threshold (float): 訓練時找到的最佳分類閾值。
            - training_mode (str): 訓練時的模式 (例如 'planb_correct', 'full_label')。
            - model_hyperparams (dict): 訓練時的模型超參數。

    Raises:
        FileNotFoundError: 如果找不到 'best_model.pth' 或其他必要檔案。
    """
    print("=" * 80)
    print("載入模型檔案...")
    
    # 載入訓練摘要
    summary_path = os.path.join(model_dir, 'training_summary.json')
    model_hyperparams = {}
    
    if os.path.exists(summary_path):
        with open(summary_path, 'r') as f:
            training_summary = json.load(f)
        training_mode = training_summary.get('mode', 'unknown')
        print(f"訓練模式: {training_mode}")
        
        # 顯示 Plan B 策略
        if training_mode == 'planb_correct' and 'planb_strategy' in training_summary:
            strategy = training_summary['planb_strategy']
            print("\nPlan B 訓練策略:")
            print(f"  特徵工程: {strategy.get('feature_engineering', 'N/A')}")
            print(f"  標籤設定: {strategy.get('label_setting', 'N/A')}")
            print(f"  圖結構: {strategy.get('graph_structure', 'N/A')}")
            print(f"  訓練mask: {strategy.get('training_mask', 'N/A')}")
        
        # 讀取超參數
        model_hyperparams = training_summary.get('model_hyperparameters', {})
        if model_hyperparams:
            print("\n訓練時的模型超參數:")
            for key, value in model_hyperparams.items():
                print(f"  {key}: {value}")
        else:
            print("\n未找到模型超參數，將使用命令列參數")
    else:
        training_mode = 'unknown'
        print("找不到 training_summary.json，無法確定訓練模式")
    
    # 載入帳戶索引
    with open(os.path.join(model_dir, 'id2idx.json'), 'r') as f:
        id2idx = json.load(f)
    print(f"✓ 載入帳戶索引: {len(id2idx):,} 個帳戶")
    
    # 載入特徵欄位（這是訓練後的最終特徵，不含圖特徵）
    with open(os.path.join(model_dir, 'feature_columns.json'), 'r') as f:
        feature_cols = json.load(f)
    print(f"✓ 載入特徵欄位: {len(feature_cols)} 個基礎特徵")
    
    # 載入標準化器
    scaler = joblib.load(os.path.join(model_dir, 'scaler.joblib'))
    print("✓ 載入標準化器")
    
    # 載入模型權重
    best_model_path = os.path.join(model_dir, 'best_model.pth')
    if not os.path.exists(best_model_path):
        raise FileNotFoundError(f"找不到模型檔案: {best_model_path}")
        
    best_checkpoint = torch.load(best_model_path, map_location='cpu')
    model_state_dict = best_checkpoint['model_state_dict']
    print("✓ 載入模型狀態")
    
    # 載入最佳閾值
    best_threshold = best_checkpoint.get('best_threshold', 0.5)
    print(f"✓ 載入最佳閾值: {best_threshold:.3f}")
    
    # 顯示訓練指標
    if 'metrics' in best_checkpoint:
        metrics = best_checkpoint['metrics']
        print(f"\n訓練時的驗證指標:")
        print(f"    F1: {metrics.get('f1', 0):.4f}")
        print(f"    AUPRC: {metrics.get('auprc', 0):.4f}")
        print(f"    Precision: {metrics.get('precision', 0):.4f}")
        print(f"    Recall: {metrics.get('recall', 0):.4f}")

    return id2idx, feature_cols, scaler, model_state_dict, best_threshold, training_mode, model_hyperparams

def build_features_for_predict(trans_df, id2idx_saved, predict_accts=None, use_planb=False):
    """
    為預測資料建立特徵，此流程完全複製 'trainv4_SAGE.py' 中的特徵工程。

    核心策略是使用完整的交易資料來計算特徵，以確保與訓練時的一致性。
    'Plan B' 模式會在此函數中被正確處理，使用全部交易進行計算。

    Args:
        trans_df (pd.DataFrame): 完整的交易資料。
        id2idx_saved (dict): 從模型目錄載入的、訓練時的帳戶索引。
        predict_accts (list, optional): 需要預測的帳戶清單。
        use_planb (bool, optional): 是否啟用 Plan B 模式。預設為 False。

    Returns:
        tuple: 包含以下元素的元組：
            - feats_df (pd.DataFrame): 計算完成的特徵 DataFrame，索引為帳戶 ID。
            - id2idx_updated (dict): 更新後的帳戶索引，包含在預測資料中新出現的帳戶。
    """
    print("=" * 80)
    print("建立預測特徵...")
    
    trans_df['from_acct'] = trans_df['from_acct'].astype(str)
    trans_df['to_acct'] = trans_df['to_acct'].astype(str)
    
    if use_planb:
        print("Plan B 模式：與訓練時一致")
        print("  策略：使用全部交易計算特徵（不過濾）")
        print("  目的：確保預測時的特徵與訓練時完全相同")
    else:
        print("完整模式：使用全部交易")
    
    # 關鍵：不過濾交易（與訓練時一致）
    trans_df_for_features = trans_df.copy()
    
    # 建立更新的帳戶索引（包含新帳戶）
    id2idx_updated = id2idx_saved.copy()
    
    all_trans_accts = set(pd.unique(trans_df_for_features[['from_acct', 'to_acct']].values.ravel()))
    all_trans_accts = set(str(a) for a in all_trans_accts if pd.notnull(a))
    
    existing_accts = set(id2idx_saved.keys())
    new_accounts = all_trans_accts - existing_accts
    
    if new_accounts:
        print(f"發現 {len(new_accounts)} 個新帳戶，將分配新索引")
        max_idx = max(id2idx_saved.values()) if id2idx_saved else -1
        for i, acct in enumerate(sorted(new_accounts), start=max_idx + 1):
            id2idx_updated[acct] = i
    
    # 關鍵：完全複製訓練流程（包含外幣特徵）
    print("步驟1: 聚合帳戶特徵（包含外幣重複金額特徵）...")
    feats_df = agg_account_features(trans_df_for_features, id2idx_updated, hour_bins=24)
    print(f"  → 聚合後特徵數: {feats_df.shape[1]}")
    
    # 驗證外幣特徵是否存在
    foreign_features = ['foreign_currency_types', 'foreign_duplicate_ratio', 
                        'max_foreign_duplicate_count', 'foreign_amount_entropy']
    missing_foreign = [f for f in foreign_features if f not in feats_df.columns]
    if missing_foreign:
        print(f"  ⚠ 警告：缺少外幣特徵: {missing_foreign}")
    else:
        print(f"  ✓ 外幣特徵已正確計算")
    
    print("步驟2: 處理極端特徵（對數轉換、標準化等）...")
    feats_df = process_extreme_features(feats_df)
    print(f"  → 處理後特徵數: {feats_df.shape[1]}")
    
    print("步驟3: 移除冗餘特徵...")
    feats_df = remove_redundant_features(feats_df)
    print(f"  → 移除後特徵數: {feats_df.shape[1]}")
    
    feats_df = feats_df.fillna(0.0)
    
    print(f"✓ 特徵建構完成，最終形狀: {feats_df.shape}")
    return feats_df, id2idx_updated

def verify_feature_alignment(train_features, pred_features, verbose=True):
    """
    驗證訓練時的特徵欄位與預測時產生的特徵欄位是否一致。

    Args:
        train_features (list or set): 訓練時使用的特徵欄位列表。
        pred_features (pd.Index or set): 預測時產生的特徵 DataFrame 的欄位。
        verbose (bool, optional): 是否印出詳細的差異 (缺失/多餘的特徵)。預設為 True。

    Returns:
        bool: 如果特徵完全對齊則返回 True，否則返回 False。
    """
    print("\n" + "=" * 80)
    print("驗證特徵對齊...")
    
    missing_in_pred = set(train_features) - set(pred_features.columns)
    extra_in_pred = set(pred_features.columns) - set(train_features)
    
    all_aligned = True
    
    if missing_in_pred:
        all_aligned = False
        print(f"預測時缺少 {len(missing_in_pred)} 個特徵:")
        if verbose:
            for feat in sorted(list(missing_in_pred))[:20]:
                print(f"    - {feat}")
            if len(missing_in_pred) > 20:
                print(f"    ... 還有 {len(missing_in_pred) - 20} 個")
    
    if extra_in_pred:
        all_aligned = False
        print(f"預測時多出 {len(extra_in_pred)} 個特徵:")
        if verbose:
            for feat in sorted(list(extra_in_pred))[:20]:
                print(f"    - {feat}")
            if len(extra_in_pred) > 20:
                print(f"    ... 還有 {len(extra_in_pred) - 20} 個")
    
    if all_aligned:
        print("✓ 特徵完全對齊")
    
    # 檢查關鍵外幣特徵
    foreign_features = ['foreign_currency_types', 'foreign_duplicate_ratio', 
                        'max_foreign_duplicate_count', 'foreign_amount_entropy']
    missing_foreign = [f for f in foreign_features if f in train_features and f not in pred_features.columns]
    if missing_foreign:
        print(f"⚠ 警告：缺少關鍵外幣特徵: {missing_foreign}")
        all_aligned = False
    
    print("=" * 80 + "\n")
    
    return all_aligned

def ensure_all_accounts_in_features(feats_df, predict_accts, id2idx_updated):
    """
    確保所有需要預測的帳戶都存在於最終的特徵 DataFrame 中。

    對於在 `predict_accts` 中但不在 `feats_df` (可能因為沒有交易紀錄) 的帳戶，
    會為它們添加一筆全為 0 的特徵向量。

    Args:
        feats_df (pd.DataFrame): 目前計算出的特徵 DataFrame。
        predict_accts (list): 需要預測的帳戶清單。
        id2idx_updated (dict): 更新後的帳戶索引。

    Returns:
        tuple: 包含以下元素的元組：
            - feats_df (pd.DataFrame): 補齊零特徵後的特徵 DataFrame。
            - id2idx_updated (dict): 可能再次更新的帳戶索引 (針對 predict_accts 中的新帳戶)。
    """
    print("檢查預測帳戶...")
    
    missing_accts = []
    for acct in predict_accts:
        acct_str = str(acct)
        if acct_str not in feats_df.index:
            missing_accts.append(acct_str)
            if acct_str not in id2idx_updated:
                max_idx = max(id2idx_updated.values()) if id2idx_updated else -1
                id2idx_updated[acct_str] = max_idx + 1
    
    if missing_accts:
        print(f"為 {len(missing_accts)} 個預測專用帳戶添加零特徵")
        zeros_df = pd.DataFrame(
            0.0, 
            index=missing_accts, 
            columns=feats_df.columns
        )
        feats_df = pd.concat([feats_df, zeros_df], axis=0)
    
    # 確保所有帳戶都有特徵
    full_index = sorted(id2idx_updated.keys(), key=lambda a: id2idx_updated[a])
    
    missing_in_feats = [a for a in full_index if a not in feats_df.index]
    if missing_in_feats:
        print(f"為 {len(missing_in_feats)} 個額外帳戶添加零特徵")
        zeros_df = pd.DataFrame(
            0.0, 
            index=missing_in_feats, 
            columns=feats_df.columns
        )
        feats_df = pd.concat([feats_df, zeros_df], axis=0)
    
    feats_df = feats_df.reindex(full_index)
    
    print(f"✓ 最終特徵形狀: {feats_df.shape}")
    return feats_df, id2idx_updated

def predict_mini_batch(model, data, predict_indices, device, batch_size=2048):
    """
    使用 NeighborLoader 進行小批次 (mini-batch) 推論。

    這適用於圖非常大，無法一次性放入 GPU 記憶體的情況。

    Args:
        model (torch.nn.Module): 已載入權重的 GraphSAGE 模型。
        data (torch_geometric.data.Data): 完整的圖資料物件。
        predict_indices (torch.Tensor): 需要預測的節點索引 (node indices)。
        device (torch.device): 執行推論的裝置 (cpu or cuda)。
        batch_size (int, optional): 每個批次的節點數。預設為 2048。

    Returns:
        np.ndarray: 針對 `predict_indices` 中每個節點的預測機率 (sigmoid 輸出)。
    """
    print(f"使用批次大小 {batch_size} 進行預測...")
    
    model.eval()
    
    loader = NeighborLoader(
        data,
        num_neighbors=[30, 20, 15],  # 應與訓練時的 num_neighbors 保持一致
        batch_size=batch_size,
        input_nodes=predict_indices,
        shuffle=False,
        num_workers=0
    )
    
    all_probs = []
    batch_count = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index)
            
            # 取出中心節點的預測結果
            batch_size_actual = batch.batch_size
            logits = logits[:batch_size_actual]
            probs = torch.sigmoid(logits).cpu().numpy()
            
            all_probs.extend(probs)
            batch_count += 1
            
            if batch_count % 10 == 0:
                print(f"    已處理 {len(all_probs):,} / {len(predict_indices):,} 個帳戶")
            
            del batch, logits, probs
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return np.array(all_probs)

def main(args):
    """
    預測腳本的主執行函數。

    解析命令列參數，並依序執行以下步驟：
    1. 載入模型檔案 (load_artifacts)
    2. 載入並建立特徵 (build_features_for_predict)
    3. 特徵對齊與標準化 (verify_feature_alignment, scaler.transform)
    4. 建立圖結構
    5. 載入 GNN 模型
    6. 執行小批次預測 (predict_mini_batch)
    7. 整理並儲存結果

    Args:
        args (argparse.Namespace): 包含所有命令列參數的物件。
    """
    print("=" * 80)
    print("GraphSAGE 預測程序（修正版 - 含外幣特徵）")
    print("=" * 80)
    
    # 載入模型檔案
    id2idx, train_feature_cols, scaler, model_state_dict, best_threshold, training_mode, model_hyperparams = load_artifacts(args.model_dir)
    
    # 使用訓練時的超參數
    if model_hyperparams:
        print("\n" + "=" * 80)
        print("使用訓練時的超參數")
        args.hidden = model_hyperparams.get('hidden', args.hidden)
        args.dropout = model_hyperparams.get('dropout', args.dropout)
        args.num_layers = model_hyperparams.get('num_layers', args.num_layers)
        print(f"  hidden={args.hidden}, dropout={args.dropout}, num_layers={args.num_layers}")
        print("=" * 80)
    
    # 檢查模式一致性
    if training_mode == 'planb_correct' and not args.use_planb:
        print("\n⚠ 警告：訓練時使用 Plan B，但預測時未啟用 --use_planb")
        print("    建議：加上 --use_planb 以保持一致性")
    elif training_mode == 'full_label' and args.use_planb:
        print("\n⚠ 警告：訓練時使用完整標籤，但預測時啟用了 --use_planb")
        print("    建議：移除 --use_planb 以保持一致性")
    
    # 決定使用的閾值
    threshold = args.threshold if args.threshold is not None else best_threshold
    print(f"\n使用閾值: {threshold:.3f}")
    
    # 載入交易資料
    print("=" * 80)
    print("載入交易資料...")
    trans_df = pd.read_csv(args.transactions)
    print(f"✓ 交易記錄數: {len(trans_df):,}")
    
    # 載入預測帳戶清單
    print("=" * 80)
    print("載入預測帳戶清單...")
    predict_df = pd.read_csv(args.acct_predict)
    predict_df['acct'] = predict_df['acct'].astype(str)
    predict_accts = predict_df['acct'].tolist()
    print(f"✓ 需要預測的帳戶數: {len(predict_accts):,}")
    
    # 關鍵修正：建立特徵（完全複製訓練流程，包含外幣特徵）
    feats_df, id2idx_updated = build_features_for_predict(
        trans_df, 
        id2idx,
        predict_accts=predict_accts,
        use_planb=args.use_planb
    )
    
    # 關鍵修正：驗證特徵對齊（包含外幣特徵）
    print("\n" + "=" * 80)
    print("特徵對齊檢查")
    print(f"訓練時特徵數: {len(train_feature_cols)}")
    print(f"預測時特徵數: {feats_df.shape[1]}")
    
    is_aligned = verify_feature_alignment(train_feature_cols, feats_df, verbose=True)
    
    if not is_aligned:
        print("\n特徵不對齊！嘗試修正...")
        
        # 補齊缺失的特徵
        for feat in train_feature_cols:
            if feat not in feats_df.columns:
                print(f"  補充缺失特徵: {feat}")
                feats_df[feat] = 0.0
        
        # 移除多餘的特徵
        extra_features = set(feats_df.columns) - set(train_feature_cols)
        if extra_features:
            print(f"  移除多餘特徵: {len(extra_features)} 個")
            feats_df = feats_df.drop(columns=list(extra_features))
        
        # 按照訓練時的順序排列特徵
        feats_df = feats_df[train_feature_cols]
        print(f"✓ 特徵已對齊，最終維度: {feats_df.shape[1]}")
    
    # 暫存交易資料
    trans_df_backup = trans_df.copy()
    del trans_df
    gc.collect()

    # 先確保所有帳戶都有特徵（在加圖特徵之前）
    feats_df, id2idx_updated = ensure_all_accounts_in_features(feats_df, predict_accts, id2idx_updated)

    # 建立圖結構（使用完整交易）
    print("=" * 80)
    print("建立圖結構...")
    trans_df = trans_df_backup
    del trans_df_backup
    
    src = trans_df['from_acct'].map(id2idx_updated)
    dst = trans_df['to_acct'].map(id2idx_updated)
    
    mask_valid = (~src.isna()) & (~dst.isna())
    src = src[mask_valid].astype(int).values
    dst = dst[mask_valid].astype(int).values
    
    del trans_df
    gc.collect()
    
    # 建立雙向邊
    edge_index = np.vstack([
        np.concatenate([src, dst]), 
        np.concatenate([dst, src])
    ])
    edge_index_tensor = torch.tensor(edge_index, dtype=torch.long)
    print(f"✓ 邊數量: {edge_index_tensor.shape[1]:,}")
    
    # 計算圖特徵（總是計算）
    print("計算圖結構特徵並加入 DataFrame...")
    graph_feats = compute_graph_features(edge_index, len(id2idx_updated))

    # 建立圖特徵 DataFrame
    graph_feat_df = pd.DataFrame({
        'graph_out_degree': graph_feats['out_degree'],
        'graph_in_degree': graph_feats['in_degree'],
        'graph_out_degree_norm': graph_feats['out_degree_norm'],
        'graph_in_degree_norm': graph_feats['in_degree_norm'],
        'graph_pagerank_approx': graph_feats['pagerank_approx']
    }, index=feats_df.index)

    # 合併到特徵 DataFrame
    feats_df_with_graph = pd.concat([feats_df, graph_feat_df], axis=1)
    print(f"加入圖特徵後: {feats_df_with_graph.shape[1]} 個特徵")

    # ⭐ 現在才移除冗餘特徵（與訓練時相同）
    feats_df_final = remove_redundant_features(feats_df_with_graph)
    print(f"移除冗餘特徵後: {feats_df_final.shape[1]} 個特徵")

    # 對齊訓練時的特徵（按順序）
    feats_df_final = feats_df_final[train_feature_cols]
    print(f"✓ 特徵對齊完成: {feats_df_final.shape[1]} 個特徵")

    # 標準化
    X_scaled = scaler.transform(feats_df_final.values)
    X_final = X_scaled  # 最終特徵矩陣

    print(f"  - 最終維度: {X_final.shape[1]}")

    # 驗證維度一致性
    expected_dim = len(train_feature_cols) 
    if X_final.shape[1] != expected_dim:
        print(f"\n⚠ 警告：特徵維度不符！")
        print(f"    訓練時: {expected_dim}")
        print(f"    預測時: {X_final.shape[1]}")
        print(f"    差異: {X_final.shape[1] - expected_dim}")
    else:
        print(f"✓ 特徵維度驗證通過: {X_final.shape[1]}")

    x_tensor = torch.tensor(X_final, dtype=torch.float)
    print(f"✓ 最終特徵張量: {x_tensor.shape}")

    del X_scaled  
    gc.collect()
    
    # 建立圖資料物件
    data = Data(x=x_tensor, edge_index=edge_index_tensor)
    
    # 載入模型
    print("=" * 80)
    print("載入模型...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✓ 使用裝置: {device}")
    
    model = EnhancedGraphSAGEModel(
        in_channels=x_tensor.size(1), 
        hidden=args.hidden, 
        dropout=args.dropout,
        num_layers=args.num_layers
    ).to(device)
    
    model.load_state_dict(model_state_dict)
    model.eval()
    print(f"✓ 模型參數量: {sum(p.numel() for p in model.parameters()):,}")
    
    del model_state_dict
    gc.collect()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # 建立預測索引
    predict_indices = torch.tensor(
        [id2idx_updated[acct] for acct in predict_accts if acct in id2idx_updated],
        dtype=torch.long
    )
    
    # 執行預測
    print("=" * 80)
    print("執行預測...")
    probs_array = predict_mini_batch(model, data, predict_indices, device, batch_size=args.batch_size)
    
    print(f"✓ 預測完成")
    print(f"    機率範圍: [{probs_array.min():.4f}, {probs_array.max():.4f}]")
    print(f"    平均機率: {probs_array.mean():.4f}")
    
    # 清理
    del model, data
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # 整理預測結果
    print("=" * 80)
    print("整理預測結果...")
    idx_to_prob = {idx.item(): prob for idx, prob in zip(predict_indices, probs_array)}
    
    results = []
    prob_stats = []
    for acct in predict_accts:
        prob = idx_to_prob.get(id2idx_updated.get(acct, -1), 0.0)
        label = 1 if prob >= threshold else 0
        
        results.append({
            'acct': acct,
            'label': label,
            'prob': prob
        })
        
        prob_stats.append(prob)
    
    result_df = pd.DataFrame(results)
    
    # 統計資訊
    pred_positive = (result_df['label'] == 1).sum()
    avg_prob = np.mean(prob_stats) if prob_stats else 0.0
    
    print(f"\n預測統計:")
    print(f"    預測為警示帳戶: {pred_positive:,} / {len(predict_accts):,} ({pred_positive/len(predict_accts)*100:.1f}%)")
    print(f"    平均機率: {avg_prob:.4f}")
    print(f"    使用閾值: {threshold:.3f}")
    
    # 儲存結果
    print("=" * 80)
    print("儲存結果...")
    
    # 主要結果
    output_df = result_df[['acct', 'label']].copy()
    output_df.to_csv(args.output, index=False)
    print(f"✓ 主要結果: {args.output}")
    
    # 詳細結果（含機率）
    prob_output = args.output.replace('.csv', '_probs.csv')
    result_df.to_csv(prob_output, index=False)
    print(f"✓ 詳細結果: {prob_output}")
    
    # 機率分布統計
    if prob_stats:
        print("\n機率分布統計:")
        print(f"    最小值: {np.min(prob_stats):.4f}")
        print(f"    25%分位: {np.percentile(prob_stats, 25):.4f}")
        print(f"    50%分位: {np.percentile(prob_stats, 50):.4f}")
        print(f"    75%分位: {np.percentile(prob_stats, 75):.4f}")
        print(f"    95%分位: {np.percentile(prob_stats, 95):.4f}")
        print(f"    99%分位: {np.percentile(prob_stats, 99):.4f}")
        print(f"    最大值: {np.max(prob_stats):.4f}")
    
        # 不同閾值下的統計
        print("\n不同閾值下的預測統計:")
        for t in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            count = sum(1 for p in prob_stats if p >= t)
            print(f"    閾值 {t:.1f}: {count:>6,} 個警示帳戶 ({count/len(prob_stats)*100:>5.1f}%)")
    
    print("=" * 80)
    print("預測完成！")
    print("=" * 80)


if __name__ == '__main__':
    # --- 命令列參數解析 ---
    parser = argparse.ArgumentParser(
        description='GraphSAGE 警示帳戶預測（v4 修正版 - 含外幣特徵）',
        epilog="""
        範例：
        python predictv4_SAGE.py \\
            --transactions Data/acct_transaction.csv \\
            --model_dir outputv4_SAGE \\
            --acct_predict Data/acct_predict.csv \\
            --output outputv4_SAGE/output.csv \\
            --use_planb \\
            --threshold 0.25
        """
    )
    
    # 必要參數
    parser.add_argument('--transactions', type=str, required=True,
                        help='(必要) 交易資料檔案路徑 (acct_transaction.csv)')
    parser.add_argument('--model_dir', type=str, required=True,
                        help='(必要) 模型檔案目錄（即 trainv4_SAGE.py 的 --output_dir）')
    parser.add_argument('--acct_predict', type=str, required=True,
                        help='(必要) 預測帳戶清單檔案路徑 (acct_predict.csv)')
    
    # 輸出參數
    parser.add_argument('--output', type=str, default='output.csv',
                        help='輸出檔案路徑 (預設: output.csv)')
    parser.add_argument('--threshold', type=float, default=None,
                        help='分類閾值（可選）。若不指定，則自動使用模型檔案中的 "best_threshold"')
    
    # 模式參數
    parser.add_argument('--use_planb', action='store_true',
                        help='(選用) 使用 Plan B 模式。此旗標應與訓練時的模式保持一致。')
    
    # 模型參數（會被 training_summary.json 覆蓋）
    parser.add_argument('--hidden', type=int, default=1024,
                        help='隱藏層維度（注意：此參數會被模型目錄中的 training_summary.json 自動覆蓋）')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout率（注意：此參數會被模型目錄中的 training_summary.json 自動覆蓋）')
    parser.add_argument('--num_layers', type=int, default=3,
                        help='GraphSAGE層數（注意：此參數會被模型目錄中的 training_summary.json 自動覆蓋）')
    
    # 批次參數
    parser.add_argument('--batch_size', type=int, default=2048,
                        help='預測時的批次大小 (Batch Size)。(預設: 2048)')
    
    args = parser.parse_args()
    main(args)