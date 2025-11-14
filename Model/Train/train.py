#!/usr/bin/env python3
"""
GraphSAGE (v4) 模型訓練腳本 (第二階段)

此腳本負責執行兩階段流程中的【第二階段：模型訓練】。

 'Preprocessing/preprocessing.py' 已經執行完畢。
此腳本會載入由預處理階段儲存的所有檔案 (特徵矩陣、標籤、
圖結構、遮罩、標準化器等)，然後專注於 GName 模型的訓練、
評估和儲存。

使用範例:
python Model/train.py \
    --data_dir Preprocessing/processed_data \
    --out_dir Model/outputv4_SAGE \
    --lr 0.0005 \
    --hidden 1024 \
    --dropout 0.5 \
    --batch_size 512 \
    --epochs 180
"""

import os
import argparse
import time
import json
import joblib
import numpy as np
import pandas as pd
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

# ============ 輔助函數與模型定義 ============

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
            reduction (str, optional): loss 的降維方式。預設為 'mean'。
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

# ============ 主函數 ============

def main(args):
    """
    訓練腳本的主執行函數 (第二階段)。

    解析命令列參數，並依序執行以下步驟：
    1. 載入由 preprocessing.py 產生的所有檔案。
    2. 將 NumPy 陣列轉換為 PyTorch 張量。
    3. 初始化模型、優化器、Loss (FocalLoss)。
    4. 執行訓練循環 (training loop)，包含早停 (early stopping)。
    5. 儲存最佳模型 (best_model.pth) 和訓練摘要 (training_summary.json)。

    Args:
        args (argparse.Namespace): 包含所有命令列參數的物件。
    """
    seed_everything(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    
    print("=" * 80)
    print("GraphSAGE v4 模型訓練 (第二階段)")
    print("=" * 80)

    # --- 1. 載入預處理資料 ---
    print(f"從 '{args.data_dir}' 載入預處理檔案...")
    
    try:
        X_scaled = np.load(os.path.join(args.data_dir, 'X_scaled.npy'))
        y_labels = np.load(os.path.join(args.data_dir, 'y_labels.npy'))
        edge_index_np = np.load(os.path.join(args.data_dir, 'edge_index.npy'))
        train_mask_np = np.load(os.path.join(args.data_dir, 'train_mask.npy'))
        val_mask_np = np.load(os.path.join(args.data_dir, 'val_mask.npy'))
        
        with open(os.path.join(args.data_dir, 'feature_columns.json'), 'r') as f:
            feature_cols = json.load(f)
        
        # 複製 scaler 和 id2idx 到模型輸出目錄，方便預測時使用
        joblib.copy(
            os.path.join(args.data_dir, 'scaler.joblib'),
            os.path.join(args.out_dir, 'scaler.joblib')
        )
        joblib.copy(
            os.path.join(args.data_dir, 'id2idx.json'),
            os.path.join(args.out_dir, 'id2idx.json')
        )
        joblib.copy(
            os.path.join(args.data_dir, 'feature_columns.json'),
            os.path.join(args.out_dir, 'feature_columns.json')
        )

    except FileNotFoundError as e:
        print(f"錯誤：找不到必要的預處理檔案: {e}")
        print(f"請先執行 'Preprocessing/preprocessing.py' 並指定 '--out_dir {args.data_dir}'")
        sys.exit(1)

    in_channels = len(feature_cols)
    print(f"✓ 載入完成。 特徵數: {in_channels}, 節點數: {len(X_scaled)}")

    # --- 2. 轉換為 PyTorch 張量 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用設備: {device}")

    x = torch.tensor(X_scaled, dtype=torch.float)
    y_tensor = torch.tensor(y_labels, dtype=torch.float)
    edge_index_tensor = torch.tensor(edge_index_np, dtype=torch.long)
    train_mask = torch.tensor(train_mask_np, dtype=torch.bool)
    val_mask = torch.tensor(val_mask_np, dtype=torch.bool)
    
    data = Data(x=x, edge_index=edge_index_tensor, y=y_tensor)
    data.train_mask = train_mask
    data.val_mask = val_mask
    
    del X_scaled, y_labels, edge_index_np, train_mask_np, val_mask_np
    gc.collect()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --- 3. 初始化模型 ---
    print("=" * 80)
    print("初始化模型...")
    
    pos_train = int(y_tensor[train_mask].sum())
    neg_train = int(train_mask.sum()) - pos_train
    print(f"訓練集: {train_mask.sum()}, 正樣本: {pos_train}")
    
    model = EnhancedGraphSAGEModel(
        in_channels=in_channels, 
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
    
    # --- 4. 訓練循環 ---
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
        
        # 評估驗證集
        metrics_val = evaluate_with_auprc(model, data, data.val_mask, device, batch_size=2048)
        
        t1 = time.time()
        scheduler.step(metrics_val['f1'])
        current_lr = optimizer.param_groups[0]['lr']
        
        if epoch % args.print_every == 0 or epoch == 1:
            print(f"Epoch {epoch:03d} | Loss: {avg_loss:.6f} | LR: {current_lr:.2e} | "
                  f"F1: {metrics_val['f1']:.4f} | AUPRC: {metrics_val['auprc']:.4f} | "
                  f"P: {metrics_val['precision']:.4f} | R: {metrics_val['recall']:.4f} | "
                  f"T: {metrics_val['best_threshold']:.2f} | Time: {(t1-t0):.1f}s")
        
        # 檢查是否為最佳模型
        if metrics_val['f1'] > best_val_f1 + 1e-6:
            best_val_f1 = metrics_val['f1']
            best_val_auprc = metrics_val['auprc']
            best_metrics = metrics_val.copy()
            best_threshold = metrics_val['best_threshold']
            patience_counter = 0
            
            # 儲存最佳模型
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
                print(f"  → 最佳模型已儲存至 {args.out_dir}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"早停！連續 {args.patience} 個 epochs 未改善。")
                break
    
    # --- 5. 儲存訓練摘要 ---
    print("=" * 80)
    print("訓練完成！儲存訓練摘要...")
    summary = {
        # 'mode' 和 'planb_strategy' 資訊在預處理階段
        'best_val_f1': best_val_f1,
        'best_val_auprc': best_val_auprc,
        'best_threshold': best_threshold,
        'best_metrics': best_metrics,
        'num_features': in_channels,
        'model_hyperparameters': {
            'hidden': args.hidden,
            'dropout': args.dropout,
            'num_layers': args.num_layers,
            'lr': args.lr,
            'weight_decay': args.weight_decay,
            'batch_size': args.batch_size,
        },
        'focal_loss': {
            'alpha': focal_alpha,
            'gamma': focal_gamma,
        },
    }
    
    with open(os.path.join(args.out_dir, 'training_summary.json'), 'w') as f:
        json.dump(summary, f, indent=4)
    
    print(f"最佳 F1: {best_val_f1:.4f} | AUPRC: {best_val_auprc:.4f}")
    print(f"模型與摘要已儲存至: {args.out_dir}")
    print("=" * 80)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='v4 GraphSAGE 訓練腳本 (第二階段)',
        epilog="""
        範例：
        python Model/train.py \\
            --data_dir Preprocessing/processed_data \\
            --out_dir Model/outputv4_SAGE \\
            --lr 0.0005 --hidden 1024 --dropout 0.5 --batch_size 512 --epochs 180
        """
    )
    
    # 檔案路徑參數
    parser.add_argument('--data_dir', type=str, required=True, 
                        help='(必要) 包含預處理檔案的輸入目錄 (來自 preprocessing.py)')
    parser.add_argument('--out_dir', type=str, required=True, 
                        help='(必要) 儲存最終模型和訓練摘要的輸出目錄')
    
    # 訓練超參數
    parser.add_argument('--epochs', type=int, default=200, help='訓練輪數 (預設: 200)')
    parser.add_argument('--lr', type=float, default=5e-4, help='學習率 (預設: 0.0005)')
    parser.add_argument('--hidden', type=int, default=1024, help='隱藏層維度 (預設: 1024)')
    parser.add_argument('--num_layers', type=int, default=3, help='GNN 層數 (預設: 3)')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout 率 (預設: 0.5)')
    parser.add_argument('--weight_decay', type=float, default=5e-6, help='權重衰減 (預設: 5e-6)')
    parser.add_argument('--batch_size', type=int, default=512, help='批次大小 (預設: 512)')
    
    # 執行控制
    parser.add_argument('--patience', type=int, default=20, help='早停耐心值 (預設: 20)')
    parser.add_argument('--seed', type=int, default=42, help='隨機種子 (預設: 42)')
    parser.add_argument('--print_every', type=int, default=5, help='打印頻率 (預設: 5)')
    
    args = parser.parse_args()
    main(args)