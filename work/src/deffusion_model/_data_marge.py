import numpy as np
import os
import glob
import zarr
import json

def convert_to_diffusion_format(input_dir, output_file, stats_file):
    # 1. すべての .npy ファイルを取得
    file_list = sorted(glob.glob(os.path.join(input_dir, "dataset_ep*.npy")))
    if not file_list:
        print("ファイルが見つかりません。ディレクトリを確認してください。")
        return

    all_trajectories = []
    trajectory_lengths = [] # 各エピソードのステップ数を保存するリスト
    print(f"{len(file_list)} 個のエピソードファイルを処理中...")

    # --- 抽出するプロット数の設定 ---
    MAX_PLOTS = 16

    for file_path in file_list:
        data = np.load(file_path) # (num_envs, num_steps, 2)
        num_envs = data.shape[0]
        for i in range(num_envs):
            # 始点から最大64プロットのみを抽出
            traj = data[i, :MAX_PLOTS].astype(np.float32)
            
            all_trajectories.append(traj)
            # 各軌道の長さを記録（常に64以下になる）
            trajectory_lengths.append(len(traj))

    # データを1つの配列に連結
    flat_actions = np.concatenate(all_trajectories, axis=0)
    # 各エピソードの終了インデックス（累積和）
    episode_ends = np.cumsum(trajectory_lengths).astype(np.int64)

    # --- プロット数（ステップ数）の統計計算 ---
    traj_lengths_np = np.array(trajectory_lengths)
    len_min = np.min(traj_lengths_np)
    len_max = np.max(traj_lengths_np)
    len_mean = np.mean(traj_lengths_np)
    len_std = np.std(traj_lengths_np)

    # 2. Zarr形式で保存
    print(f"Zarr形式で保存中: {output_file}")
    root = zarr.open_group(output_file, mode='w')
    
    data_group = root.create_group('data')
    data_group.create_array(name='action', data=flat_actions, chunks=(1000, 2))
    
    meta_group = root.create_group('meta')
    meta_group.create_array(name='episode_ends', data=episode_ends, chunks=(len(episode_ends),))

    # 3. 正規化統計量の保存
    # 64プロットに制限された後のデータで統計を計算
    stats = {
        "min": flat_actions.min(axis=0).tolist(),
        "max": flat_actions.max(axis=0).tolist(),
        "mean": flat_actions.mean(axis=0).tolist(),
        "std": flat_actions.std(axis=0).tolist(),
        "length_stats": {
            "min": int(len_min),
            "max": int(len_max),
            "mean": float(len_mean),
            "std": float(len_std)
        }
    }
    
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=4)

    print("--- 変換・統計解析完了 ---")
    print(f"総エピソード数: {len(all_trajectories)}")
    print(f"総ステップ数: {flat_actions.shape[0]}")
    print("-" * 30)
    print(f"プロット数（ステップ数）の統計 (最大 {MAX_PLOTS} に制限済み):")
    print(f"  最小値: {len_min}")
    print(f"  最大値: {len_max}")
    print(f"  平均値: {len_mean:.2f}")
    print("-" * 30)
    print(f"保存先: {output_file}")

# --- 設定 ---
SOURCE_DIR = "/home/isseebi/Desktop/user/Reinforcement_learning/study/OpenManipulation/src/dataset"
OUTPUT_ZARR = os.path.join(SOURCE_DIR, "diffusion_data.zarr")
STATS_JSON = os.path.join(SOURCE_DIR, "stats.json")

if __name__ == "__main__":
    convert_to_diffusion_format(SOURCE_DIR, OUTPUT_ZARR, STATS_JSON)