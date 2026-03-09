import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import zarr
import json
import os
import math

# ==========================================
# 1. ネットワーク定義 (シンプルなMLPベース)
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

class SimpleTrajectoryDiffusionNet(nn.Module):
    def __init__(self, seq_len=32, feature_dim=2, time_emb_dim=64):
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
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim + time_emb_dim, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, input_dim)
        )

    def forward(self, x, time):
        # x shape: (B, 32, 2)
        B = x.shape[0]
        x_flat = x.reshape(B, -1) # (B, 64)
        t_emb = self.time_mlp(time) # (B, time_emb_dim)
        
        # 軌道データと時間情報を結合して処理
        x_input = torch.cat([x_flat, t_emb], dim=-1)
        out_flat = self.mlp(x_input)
        
        return out_flat.reshape(B, self.seq_len, self.feature_dim) # (B, 32, 2)

# ==========================================
# 2. データセット (32ステップ抽出 & 正規化)
# ==========================================
class TrajectoryDataset(Dataset):
    def __init__(self, zarr_path, stats_path, pred_horizon=32):
        self.pred_horizon = pred_horizon
        root = zarr.open_group(zarr_path, mode='r')
        self.action_data = root['data/action'][:]
        self.episode_ends = root['meta/episode_ends'][:]
        
        # 正規化のための統計量を読み込む (-1 ~ 1の範囲に変換するため)
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
        
        # Min-Max 正規化: [-1, 1] にスケーリング
        chunk_norm = 2.0 * (chunk - self.min_val) / (self.max_val - self.min_val) - 1.0
        return torch.tensor(chunk_norm, dtype=torch.float32)

# ==========================================
# 3. 学習ループ
# ==========================================
def train():
    # --- 設定 ---
    SOURCE_DIR = "/home/isseebi/Desktop/user/Reinforcement_learning/study/OpenManipulation/src/dataset"
    ZARR_PATH = os.path.join(SOURCE_DIR, "diffusion_data.zarr")
    STATS_PATH = os.path.join(SOURCE_DIR, "stats.json")
    CHECKPOINT_DIR = os.path.join(SOURCE_DIR, "checkpoints")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    PRED_HORIZON = 32
    BATCH_SIZE = 128
    NUM_EPOCHS = 100
    SAVE_INTERVAL = 20 # 20エポックごとに保存
    TIMESTEPS = 100    # 拡散プロセスのステップ数
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = TrajectoryDataset(ZARR_PATH, STATS_PATH, pred_horizon=PRED_HORIZON)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    print(f"データセット長: {len(dataset)} サンプル")

    model = SimpleTrajectoryDiffusionNet(seq_len=PRED_HORIZON).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    # DDPMのノイズスケジュール設定 (Linear)
    beta = torch.linspace(1e-4, 0.02, TIMESTEPS).to(device)
    alpha = 1.0 - beta
    alpha_bar = torch.cumprod(alpha, dim=0)

    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_loss = 0.0
        for batch in dataloader:
            batch = batch.to(device)
            B = batch.shape[0]
            
            # ランダムな時刻 t をサンプリング
            t = torch.randint(0, TIMESTEPS, (B,), device=device).long()
            
            # ガウスノイズを生成
            noise = torch.randn_like(batch).to(device)
            
            # 時刻 t におけるノイズを加えたデータ(x_t)を計算
            a_bar_t = alpha_bar[t].view(-1, 1, 1)
            x_t = torch.sqrt(a_bar_t) * batch + torch.sqrt(1 - a_bar_t) * noise
            
            # モデルが「加えたノイズ」を予測
            predicted_noise = model(x_t, t)
            
            # 損失計算 (予測したノイズと実際のノイズの誤差)
            loss = loss_fn(predicted_noise, noise)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch:03d}/{NUM_EPOCHS} | Loss: {avg_loss:.5f}")
        
        # 定期保存
        if epoch % SAVE_INTERVAL == 0 or epoch == NUM_EPOCHS:
            save_path = os.path.join(CHECKPOINT_DIR, f"model_ep{epoch:03d}.pth")
            torch.save(model.state_dict(), save_path)
            print(f" => モデルを保存しました: {save_path}")

if __name__ == "__main__":
    train()