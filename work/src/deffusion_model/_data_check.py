# import zarr
# import numpy as np
# import torch
# from torch.utils.data import Dataset, DataLoader
# import os

# class DiffusionTrajectoryDataset(Dataset):
#     def __init__(self, zarr_path, pred_horizon=16):
#         """
#         pred_horizon: 1回に切り出す軌道の長さ（ステップ数）。Diffusion Policyでは16などを使います。
#         """
#         super().__init__()
#         self.pred_horizon = pred_horizon
        
#         # Zarrファイルを読み取り専用(mode='r')で開く
#         print(f"Loading Zarr dataset from: {zarr_path}")
#         root = zarr.open_group(zarr_path, mode='r')
        
#         # データをメモリにロード（データが巨大な場合はメモリに乗せない工夫が必要ですが、今回は乗せます）
#         self.action_data = root['data/action'][:]
#         self.episode_ends = root['meta/episode_ends'][:]
        
#         # エピソードを跨がないような「安全な開始インデックス」を計算
#         self.valid_indices = self._get_valid_indices()

#     def _get_valid_indices(self):
#         valid_indices = []
#         start_idx = 0
        
#         for end_idx in self.episode_ends:
#             # そのエピソードの中で、pred_horizon分の長さを確保できる最後の開始位置
#             max_start = end_idx - self.pred_horizon
            
#             # 開始位置が有効ならリストに追加
#             if max_start >= start_idx:
#                 valid_indices.extend(range(start_idx, max_start + 1))
            
#             # 次のエピソードの開始位置を更新
#             start_idx = end_idx
            
#         return valid_indices

#     def __len__(self):
#         # 切り出せるウィンドウの総数
#         return len(self.valid_indices)

#     def __getitem__(self, idx):
#         # 安全な開始インデックスを取得
#         start = self.valid_indices[idx]
#         end = start + self.pred_horizon
        
#         # 指定された長さ（pred_horizon）の軌道を切り出す
#         action_chunk = self.action_data[start:end]
        
#         # PyTorchのTensorに変換して返す
#         return torch.tensor(action_chunk, dtype=torch.float32)

# # --- 確認用の実行コード ---
# if __name__ == "__main__":
#     SOURCE_DIR = "/home/isseebi/Desktop/user/Reinforcement_learning/study/OpenManipulation/src/dataset"
#     ZARR_PATH = os.path.join(SOURCE_DIR, "diffusion_data.zarr")
    
#     # 軌道の長さを16ステップとしてDatasetを初期化
#     PRED_HORIZON = 16
#     dataset = DiffusionTrajectoryDataset(ZARR_PATH, pred_horizon=PRED_HORIZON)
    
#     print("-" * 40)
#     print(f"データセットの総サンプル数（切り出せる数）: {len(dataset)}")
#     print("-" * 40)
    
#     # DataLoaderを使ってバッチ（まとまり）で取り出す準備
#     # batch_size=4, shuffle=True にすることで、ランダムなエピソード・時刻から4つずつ取り出せます
#     dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
#     # 最初の1バッチだけ取り出して中身を確認
#     for batch in dataloader:
#         print(f"バッチのShape: {batch.shape}")
#         print(" -> (バッチサイズ, 予測ホライズン(ステップ数), 次元数)")
#         print("\n=== バッチ内の1つ目のデータ（16ステップ分の軌道） ===")
#         print(batch[0])  # バッチ内の最初のサンプルを表示
#         break  # 1回でループを抜ける

import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
import json
import zarr

# ※ 前述の TrajectoryDataset クラスが定義されている前提です
# もし別ファイルにしている場合は import してください。

class DiffusionTrajectoryDataset(Dataset):
    def __init__(self, zarr_path, pred_horizon=32):
        super().__init__()
        self.pred_horizon = pred_horizon
        
        print(f"Loading Zarr dataset from: {zarr_path}")
        root = zarr.open_group(zarr_path, mode='r')
        
        self.action_data = root['data/action'][:]
        self.episode_ends = root['meta/episode_ends'][:]
        
        # 修正ポイント：各エピソードの「スタート地点」のみを取得
        self.valid_indices = self._get_start_indices()

    def _get_start_indices(self):
        start_indices = []
        current_start = 0
        
        for end_idx in self.episode_ends:
            # エピソードの長さが pred_horizon (32) 以上あるかチェック
            episode_length = end_idx - current_start
            if episode_length >= self.pred_horizon:
                # エピソードの「本当の開始位置」だけを追加
                start_indices.append(current_start)
            
            # 次のエピソードの開始位置へ更新
            current_start = end_idx
            
        return start_indices

    def __len__(self):
        # サンプル数は「エピソードの数」と一致するようになります
        return len(self.valid_indices)

    def __getitem__(self, idx):
        start = self.valid_indices[idx]
        end = start + self.pred_horizon
        action_chunk = self.action_data[start:end]
        return torch.tensor(action_chunk, dtype=torch.float32)

def visualize_keypoint_extraction():
    # --- 設定 (train関数の設定と合わせてください) ---
    SOURCE_DIR = "/home/isseebi/Desktop/user/Reinforcement_learning/study/OpenManipulation/src/dataset"
    ZARR_PATH = os.path.join(SOURCE_DIR, "diffusion_data.zarr")
    STATS_PATH = os.path.join(SOURCE_DIR, "stats.json")
    PRED_HORIZON = 64
    MIN_DIST = 5

    # 1. データセットの準備
    dataset = DiffusionTrajectoryDataset(ZARR_PATH, pred_horizon=PRED_HORIZON)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True) # 1つだけランダムに取得
    
    # 2. データの取得
    batch = next(iter(dataloader)) # Shape: (1, 32, 2)
    traj = batch[0] # (32, 2)
    print(traj)
    
    # --- 3. 抽出アルゴリズムの実行 (train内と同じロジック) ---
    v1 = traj[1:-1, :] - traj[:-2, :]
    v2 = traj[2:, :] - traj[1:-1, :]
    
    dot = (v1 * v2).sum(dim=-1)
    norms = torch.norm(v1, dim=-1) * torch.norm(v2, dim=-1)
    cos_sim = dot / (norms + 1e-8)
    
    sorted_indices = torch.argsort(cos_sim) # 尖っている順 (0~29)
    
    idx1_rel = sorted_indices[0]
    dist_to_idx1 = torch.abs(sorted_indices - idx1_rel)
    mask = dist_to_idx1 >= MIN_DIST
    idx2_pos = torch.argmax(mask.int())
    idx2_rel = sorted_indices[idx2_pos]
    
    # 実インデックス(0~31)に直す
    idx1 = idx1_rel.item() + 1
    idx2 = idx2_rel.item() + 1
    
    # --- 4. 可視化 ---
    plt.figure(figsize=(8, 6))
    
    # 軌道全体
    traj_np = traj.numpy()
    plt.plot(traj_np[:, 0], traj_np[:, 1], 'gray', label='Full Trajectory', alpha=0.5, marker='.')
    # 始点と終点
    plt.scatter(traj_np[0, 0], traj_np[0, 1], color='blue', s=100, label='Start (0)', zorder=5)
    plt.scatter(traj_np[31, 0], traj_np[31, 1], color='black', s=100, label='End (31)', zorder=5)
    
    # 抽出された尖った点
    plt.scatter(traj_np[idx1, 0], traj_np[idx1, 1], color='red', s=150, marker='X', label=f'Sharp 1 (idx:{idx1})', zorder=6)
    plt.scatter(traj_np[idx2, 0], traj_np[idx2, 1], color='orange', s=150, marker='X', label=f'Sharp 2 (idx:{idx2})', zorder=6)
    
    # 曲率の値を注釈として表示 (上位5件)
    for i in range(5):
        best_idx = sorted_indices[i].item() + 1
        val = cos_sim[sorted_indices[i]].item()
        plt.annotate(f"{val:.2f}", (traj_np[best_idx, 0], traj_np[best_idx, 1]), fontsize=8, alpha=0.7)

    plt.title(f"Keypoint Extraction Check (MIN_DIST={MIN_DIST})")
    plt.xlabel("X (normalized)")
    plt.ylabel("Y (normalized)")
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()

    print(f"抽出されたインデックス: {idx1}, {idx2}")
    print(f"その地点のcos_sim: {cos_sim[idx1_rel].item():.4f}, {cos_sim[idx2_rel].item():.4f}")

if __name__ == "__main__":
    visualize_keypoint_extraction()