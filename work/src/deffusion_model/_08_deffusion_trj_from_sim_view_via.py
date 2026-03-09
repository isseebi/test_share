import torch
import torch.nn as nn
import numpy as np
import zarr
import json
import os
import math
import matplotlib.pyplot as plt

# ==========================================
# 1. ネットワーク定義 (学習時と全く同じもの)
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
    def __init__(self, seq_len, feature_dim=2, time_emb_dim=16, cond_dim=8):
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
# 2. 推論用の補助関数
# ==========================================
def get_dataset_info_and_sample(zarr_path, stats_path, sample_idx=0):
    """データセットから最大長、正規化情報、テスト用の条件を取得する"""
    root = zarr.open_group(zarr_path, mode='r')
    action_data = root['data/action'][:]
    episode_ends = root['meta/episode_ends'][:]
    
    with open(stats_path, 'r') as f:
        stats = json.load(f)
    min_val = np.array(stats['min'], dtype=np.float32)
    max_val = np.array(stats['max'], dtype=np.float32)
    
    # 最大長の計算
    episodes = []
    start_idx = 0
    max_len = 0
    for end_idx in episode_ends:
        ep_len = end_idx - start_idx
        episodes.append((start_idx, end_idx))
        if ep_len > max_len:
            max_len = ep_len
        start_idx = end_idx
        
    # テスト用のサンプル軌跡を取得し、パディングと正規化を行う
    start, end = episodes[sample_idx]
    chunk = action_data[start:end]
    chunk_norm = 2.0 * (chunk - min_val) / (max_val - min_val + 1e-8) - 1.0
    chunk_tensor = torch.tensor(chunk_norm, dtype=torch.float32)
    
    pad_len = max_len - (end - start)
    if pad_len > 0:
        last_step = chunk_tensor[-1].unsqueeze(0).repeat(pad_len, 1)
        chunk_tensor = torch.cat([chunk_tensor, last_step], dim=0)
        
    return max_len, min_val, max_val, chunk_tensor

def unnormalize(tensor_data, min_val, max_val):
    """ -1 ~ 1 に正規化されたデータを元のスケールに戻す """
    data = tensor_data.cpu().numpy()
    return (data + 1.0) / 2.0 * (max_val - min_val) + min_val

# ==========================================
# 3. メイン推論プロセス
# ==========================================
def generate_and_visualize():
    # --- パス設定 ---
    SOURCE_DIR = "/home/isseebi/Desktop/user/Reinforcement_learning/study/OpenManipulation/src/dataset"
    ZARR_PATH = os.path.join(SOURCE_DIR, "diffusion_data.zarr")
    STATS_PATH = os.path.join(SOURCE_DIR, "stats.json")
    
    # 【注意】学習済みモデルのパスを、実際に保存されたエポック数のファイル名に合わせて変更してください
    MODEL_PATH = os.path.join(SOURCE_DIR, "checkpoints_cond_sharp", "model_ep100.pth") 
    
    # --- ハイパーパラメータ ---
    TIMESTEPS = 100
    MIN_DIST = 20
    MARGIN = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. データセット情報の取得とテスト用データの準備
    # ここでは例として、インデックス0番のエピソードをテストの基準として抽出します
    SEQ_LEN, min_val, max_val, gt_traj = get_dataset_info_and_sample(ZARR_PATH, STATS_PATH, sample_idx=5)
    
    print(f"Sequence Length: {SEQ_LEN}")
    print("Loading model...")
    
    # 2. モデルのロード
    model = ConditionalTrajectoryDiffusionNet(seq_len=SEQ_LEN, cond_dim=8).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    
    # DDPM スケジュールの設定 (学習時と同じ)
    beta = torch.linspace(1e-4, 0.02, TIMESTEPS).to(device)
    alpha = 1.0 - beta
    alpha_bar = torch.cumprod(alpha, dim=0)
    
    # 3. 条件ベクトルの作成 (推論のヒントとなる情報)
    # 実データ(gt_traj)から、学習時と同じロジックでStart, Way1, Way2, Endを抽出します
    gt_traj_batch = gt_traj.unsqueeze(0).to(device) # shape: (1, SEQ_LEN, 2)
    with torch.no_grad():
        start_pt = gt_traj_batch[:, 0, :]
        end_pt   = gt_traj_batch[:, -1, :]
        
        v1 = gt_traj_batch[:, 1:-1, :] - gt_traj_batch[:, :-2, :]
        v2 = gt_traj_batch[:, 2:, :] - gt_traj_batch[:, 1:-1, :]
        dot = (v1 * v2).sum(dim=-1)
        norms = torch.norm(v1, dim=-1) * torch.norm(v2, dim=-1)
        cos_sim = dot / (norms + 1e-8)
        
        cos_sim[:, :MARGIN] = 30
        cos_sim[:, -MARGIN:] = 30
        
        sorted_indices = torch.argsort(cos_sim, dim=1) 
        idx1_rel = sorted_indices[:, 0]
        dist_to_idx1 = torch.abs(sorted_indices - idx1_rel.unsqueeze(1))
        mask = dist_to_idx1 >= MIN_DIST
        idx2_pos = torch.argmax(mask.int(), dim=1)
        idx2_rel = torch.gather(sorted_indices, 1, idx2_pos.unsqueeze(1)).squeeze(1)
        
        t1 = torch.min(idx1_rel + 1, idx2_rel + 1)
        t2 = torch.max(idx1_rel + 1, idx2_rel + 1)
        way1_pt = gt_traj_batch[torch.arange(1), t1, :]
        way2_pt = gt_traj_batch[torch.arange(1), t2, :]
        
        # モデルに渡す条件ベクトル
        cond = torch.cat([start_pt, way1_pt, way2_pt, end_pt], dim=-1)
    
    # 4. 推論（DDPM サンプリングループ）
    print("Generating trajectory...")
    # 完全にランダムなノイズからスタート
    x = torch.randn((1, SEQ_LEN, 2), device=device)
    
    with torch.no_grad():
        for i in reversed(range(TIMESTEPS)):
            t_tensor = torch.tensor([i], device=device).long()
            
            # モデルによるノイズ予測
            predicted_noise = model(x, t_tensor, cond)
            
            # DDPMの数式に基づくデノイジング処理
            alpha_t = alpha[i]
            alpha_bar_t = alpha_bar[i]
            beta_t = beta[i]
            
            # 式: x_{t-1} = (1 / sqrt(alpha_t)) * (x_t - ((1 - alpha_t) / sqrt(1 - alpha_bar_t)) * predicted_noise)
            x = (1.0 / torch.sqrt(alpha_t)) * (x - ((1.0 - alpha_t) / torch.sqrt(1.0 - alpha_bar_t)) * predicted_noise)
            
            # 最後のステップ以外はランダムノイズ（分散）を足す
            if i > 0:
                noise = torch.randn_like(x)
                x = x + torch.sqrt(beta_t) * noise

    # 5. 正規化の解除と可視化
    generated_traj = unnormalize(x[0], min_val, max_val)
    ground_truth = unnormalize(gt_traj_batch[0], min_val, max_val)
    
    cond_pts = unnormalize(cond[0].view(4, 2), min_val, max_val)
    st_pt, w1_pt, w2_pt, en_pt = cond_pts[0], cond_pts[1], cond_pts[2], cond_pts[3]
    
    # グラフの描画
    plt.figure(figsize=(8, 8))
    # 正解データ（データセットにある実際の軌跡）
    plt.plot(ground_truth[:, 0], ground_truth[:, 1], 'black', linestyle='--', alpha=0.5, label='Ground Truth Trajectory')
    # AIが生成したデータ
    plt.plot(generated_traj[:, 0], generated_traj[:, 1], 'blue', alpha=0.8, linewidth=2, label='Generated Trajectory', marker='.', markersize=2)
    
    # 条件として与えた点
    plt.scatter(st_pt[0], st_pt[1], c='green', s=100, label='Condition: Start', zorder=5)
    plt.scatter(en_pt[0], en_pt[1], c='red', s=100, label='Condition: End', zorder=5)
    plt.scatter(w1_pt[0], w1_pt[1], c='orange', s=80, label='Condition: Way1', zorder=5, marker='*')
    plt.scatter(w2_pt[0], w2_pt[1], c='orange', s=80, label='Condition: Way2', zorder=5, marker='*')
    
    plt.title("Diffusion Model Inference Result")
    plt.axis('equal')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    generate_and_visualize()