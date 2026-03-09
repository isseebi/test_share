import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import zarr
import json
import os
import math
import matplotlib.pyplot as plt

import random 

# 【修正】引数に margin を追加
def visualize_samples(dataset, pred_horizon, min_dist, margin=10, n=5):
    """データセットからn個のサンプルを抽出して並べてプロットする"""
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1: axes = [axes]
    
    indices = random.sample(range(len(dataset)), n)
    
    for i, data_idx in enumerate(indices):
        # データの取得 (1, PRED_HORIZON, 2)
        sample = dataset[data_idx].unsqueeze(0)
        ax = axes[i]
        
        # --- 始点・終点 ---
        start_pt = sample[0, 0].numpy()
        end_pt = sample[0, -1].numpy()
        
        # --- 曲率ロジック ---
        v1 = sample[:, 1:-1, :] - sample[:, :-2, :]
        v2 = sample[:, 2:, :] - sample[:, 1:-1, :]
        dot = (v1 * v2).sum(dim=-1)
        norms = torch.norm(v1, dim=-1) * torch.norm(v2, dim=-1)
        cos_sim = dot / (norms + 1e-8)
        
        # 【修正】始点・終点付近の曲率を除外（値が小さいほど尖っているため、大きな値でマスクする）
        # cos_simのサイズは (1, PRED_HORIZON - 2)
        cos_sim[:, :margin] = 2.0
        cos_sim[:, -margin:] = 2.0
        
        sorted_indices = torch.argsort(cos_sim, dim=1) 
        idx1_rel = sorted_indices[:, 0]
        dist_to_idx1 = torch.abs(sorted_indices - idx1_rel.unsqueeze(1))
        mask = dist_to_idx1 >= min_dist
        idx2_pos = torch.argmax(mask.int(), dim=1)
        idx2_rel = torch.gather(sorted_indices, 1, idx2_pos.unsqueeze(1)).squeeze(1)
        
        t1, t2 = sorted([idx1_rel.item() + 1, idx2_rel.item() + 1])
        way1_pt = sample[0, t1].numpy()
        way2_pt = sample[0, t2].numpy()
        
        # --- プロット ---
        traj = sample[0].numpy()
        ax.plot(traj[:, 0], traj[:, 1], 'gray', alpha=0.3, label='Traj', marker='o', markersize=1)
        ax.scatter(start_pt[0], start_pt[1], c='green', s=50, label='Start', zorder=5)
        ax.scatter(end_pt[0], end_pt[1], c='red', s=50, label='End', zorder=5)
        ax.scatter(way1_pt[0], way1_pt[1], c='orange', s=40, label='Way1', zorder=5)
        ax.scatter(way2_pt[0], way2_pt[1], c='orange', s=40, label='Way2', zorder=5)
        
        # タイトルに取得した実際のインデックスを表示
        ax.set_title(f"Index: {data_idx}")
        ax.set_aspect('equal')
        ax.grid(True, linestyle='--', alpha=0.6)
        if i == 0: ax.legend()

    plt.tight_layout()
    plt.show()


# ==========================================
# 1. 条件付きネットワーク定義
# ==========================================
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class ConditionalTrajectoryDiffusionNet(nn.Module):
    def __init__(self, seq_len=16, feature_dim=2, time_emb_dim=16, cond_dim=8):
        super().__init__()
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        input_dim = seq_len * feature_dim
        
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 2),
            nn.GELU(),
            nn.Linear(time_emb_dim * 2, time_emb_dim)
        )
        
        # 入力: [軌道(seq_len*2次元) + 時間(16次元) + 条件(8次元)]
        self.mlp = nn.Sequential(
            nn.Linear(input_dim + time_emb_dim + cond_dim, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, input_dim)
        )

    def forward(self, x, time, cond):
        B = x.shape[0]
        x_flat = x.view(B, -1)
        t_emb = self.time_mlp(time)
        x_input = torch.cat([x_flat, t_emb, cond], dim=-1)
        out_flat = self.mlp(x_input)
        return out_flat.view(B, self.seq_len, self.feature_dim)

# ==========================================
# 2. データセット (正規化 & ウィンドウ切り出し)
# ==========================================
class TrajectoryDataset(Dataset):
    def __init__(self, zarr_path, stats_path, pred_horizon=16):
        self.pred_horizon = pred_horizon
        root = zarr.open_group(zarr_path, mode='r')
        self.action_data = root['data/action'][:]
        self.episode_ends = root['meta/episode_ends'][:]
        
        with open(stats_path, 'r') as f:
            stats = json.load(f)
        self.min_val = np.array(stats['min'], dtype=np.float32)
        self.max_val = np.array(stats['max'], dtype=np.float32)
        
        self.valid_indices = self._get_valid_indices()

    def _get_valid_indices(self):
        valid_indices, start_idx = [], 0
        for end_idx in self.episode_ends:
            max_start = end_idx - self.pred_horizon
            if max_start >= start_idx:
                valid_indices.extend(range(start_idx, max_start + 1))
            start_idx = end_idx
        return valid_indices

    def __len__(self): 
        return len(self.valid_indices)

    def __getitem__(self, idx):
        start = self.valid_indices[idx]
        end = start + self.pred_horizon
        chunk = self.action_data[start:end]
        # -1 to 1 normalization
        chunk_norm = 2.0 * (chunk - self.min_val) / (self.max_val - self.min_val + 1e-8) - 1.0
        return torch.tensor(chunk_norm, dtype=torch.float32)

# ==========================================
# 3. 学習ループ
# ==========================================
def train():
    # --- パス設定 ---
    SOURCE_DIR = "/home/isseebi/Desktop/user/Reinforcement_learning/study/OpenManipulation/src/dataset"
    ZARR_PATH = os.path.join(SOURCE_DIR, "diffusion_data.zarr")
    STATS_PATH = os.path.join(SOURCE_DIR, "stats.json")
    CHECKPOINT_DIR = os.path.join(SOURCE_DIR, "checkpoints_cond_sharp")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    # --- ハイパーパラメータ ---
    PRED_HORIZON = 16  # 点の数（シーケンス長）
    BATCH_SIZE = 16
    NUM_EPOCHS = 100
    TIMESTEPS = 100
    MIN_DIST = 20 # 経由点同士の最小距離（ステップ数）
    MARGIN = 10  # 【追加】始点・終点から除外するステップ数
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} | Sequence Length: {PRED_HORIZON}")
    
    # データロード
    dataset = TrajectoryDataset(ZARR_PATH, STATS_PATH, pred_horizon=PRED_HORIZON)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    
    # 【修正】MARGINを渡す
    visualize_samples(dataset, PRED_HORIZON, MIN_DIST, margin=MARGIN, n=20)
    
    # モデル・最適化手法
    model = ConditionalTrajectoryDiffusionNet(seq_len=PRED_HORIZON, cond_dim=8).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    # DDPM スケジュール
    beta = torch.linspace(1e-4, 0.02, TIMESTEPS).to(device)
    alpha = 1.0 - beta
    alpha_bar = torch.cumprod(alpha, dim=0)

    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_loss = 0.0
        for batch in dataloader:
            batch = batch.to(device) # shape: (B, PRED_HORIZON, 2)
            B = batch.shape[0]
            
            # --- 【重要点の抽出ロジック】 ---
            with torch.no_grad():
                # 始点と終点
                start_pt = batch[:, 0, :]   
                end_pt   = batch[:, -1, :]  
                
                v1 = batch[:, 1:-1, :] - batch[:, :-2, :]
                v2 = batch[:, 2:, :] - batch[:, 1:-1, :]
                
                # 曲率計算
                dot = (v1 * v2).sum(dim=-1)
                norms = torch.norm(v1, dim=-1) * torch.norm(v2, dim=-1)
                cos_sim = dot / (norms + 1e-8)
                
                # 【修正】始点と終点の付近を除外
                cos_sim[:, :MARGIN] = 30
                cos_sim[:, -MARGIN:] = 30
                
                # 最も尖った点のインデックス取得
                sorted_indices = torch.argsort(cos_sim, dim=1) 
                idx1_rel = sorted_indices[:, 0]
                
                # 2番目に尖った点（1番目から離れた場所）
                dist_to_idx1 = torch.abs(sorted_indices - idx1_rel.unsqueeze(1))
                mask = dist_to_idx1 >= MIN_DIST
                idx2_pos = torch.argmax(mask.int(), dim=1)
                idx2_rel = torch.gather(sorted_indices, 1, idx2_pos.unsqueeze(1)).squeeze(1)
                
                # 実インデックスに復元して座標を取得
                t1 = torch.min(idx1_rel + 1, idx2_rel + 1)
                t2 = torch.max(idx1_rel + 1, idx2_rel + 1)
                way1_pt = batch[torch.arange(B), t1, :]
                way2_pt = batch[torch.arange(B), t2, :]
                
                # 条件ベクトル [始点, 経由1, 経由2, 終点]
                cond = torch.cat([start_pt, way1_pt, way2_pt, end_pt], dim=-1)
            
            # --- DDPM プロセス ---
            t = torch.randint(0, TIMESTEPS, (B,), device=device).long()
            noise = torch.randn_like(batch)
            a_bar_t = alpha_bar[t].view(-1, 1, 1)
            x_t = torch.sqrt(a_bar_t) * batch + torch.sqrt(1 - a_bar_t) * noise
            
            predicted_noise = model(x_t, t, cond)
            loss = loss_fn(predicted_noise, noise)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        print(f"Epoch {epoch:03d}/{NUM_EPOCHS} | Loss: {epoch_loss / len(dataloader):.5f}")
        
        if epoch % 20 == 0 or epoch == NUM_EPOCHS:
            save_path = os.path.join(CHECKPOINT_DIR, f"model_ep{epoch:03d}.pth")
            torch.save(model.state_dict(), save_path)


if __name__ == "__main__":
    train()