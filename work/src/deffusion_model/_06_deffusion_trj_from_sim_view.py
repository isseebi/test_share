import torch
import numpy as np
import matplotlib.pyplot as plt
import json
import os
# train.py で定義したネットワークを読み込むためにインポート
from _05_deffusion_trj_from_sim_train import SimpleTrajectoryDiffusionNet

def generate_trajectory():
    # --- 設定 ---
    SOURCE_DIR = "/home/isseebi/Desktop/user/Reinforcement_learning/study/OpenManipulation/src/dataset"
    STATS_PATH = os.path.join(SOURCE_DIR, "stats.json")
    # 学習したモデルのパスを指定（例として100エポック目のものを指定）
    CHECKPOINT_PATH = os.path.join(SOURCE_DIR, "checkpoints", "model_ep100.pth")
    
    PRED_HORIZON = 32
    TIMESTEPS = 100
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 正規化統計量の読み込み (生成した軌道を元のメートル単位に戻すため)
    with open(STATS_PATH, 'r') as f:
        stats = json.load(f)
    min_val = np.array(stats['min'], dtype=np.float32)
    max_val = np.array(stats['max'], dtype=np.float32)

    # モデルのロード
    model = SimpleTrajectoryDiffusionNet(seq_len=PRED_HORIZON).to(device)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.eval()

    # DDPMのスケジュール再構築
    beta = torch.linspace(1e-4, 0.02, TIMESTEPS).to(device)
    alpha = 1.0 - beta
    alpha_bar = torch.cumprod(alpha, dim=0)

    print("軌道の生成を開始します...")
    
    with torch.no_grad():
        # 1. 完全なランダムノイズからスタート (Batch=1, 32steps, 2dims)
        x = torch.randn((1, PRED_HORIZON, 2)).to(device)
        
        # 2. 逆拡散プロセス (ステップ TIMESTEPS-1 から 0 へ徐々にノイズを除去)
        for i in reversed(range(TIMESTEPS)):
            t = torch.tensor([i], device=device).long()
            
            # ノイズを予測
            predicted_noise = model(x, t)
            
            # DDPMの計算式で少しだけノイズを取り除く
            alpha_t = alpha[t].view(-1, 1, 1)
            a_bar_t = alpha_bar[t].view(-1, 1, 1)
            beta_t = beta[t].view(-1, 1, 1)
            
            if i > 0:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x) # 最後のステップはノイズを足さない
                
            # x_{t-1} を計算
            x = (1 / torch.sqrt(alpha_t)) * (x - ((1 - alpha_t) / torch.sqrt(1 - a_bar_t)) * predicted_noise) + torch.sqrt(beta_t) * noise

    # 3. スケーリングを元に戻す (逆正規化 [-1, 1] -> 元の座標)
    generated_traj_norm = x.cpu().numpy()[0] # (32, 2)
    generated_traj = (generated_traj_norm + 1.0) / 2.0 * (max_val - min_val) + min_val

    # 4. 結果をプロット
    plt.figure(figsize=(8, 8))
    plt.plot(generated_traj[:, 0], generated_traj[:, 1], marker='o', linestyle='-', color='b', label='Generated Path')
    plt.scatter(generated_traj[0, 0], generated_traj[0, 1], color='red', s=100, label='Start', zorder=5)
    plt.scatter(generated_traj[-1, 0], generated_traj[-1, 1], color='green', s=100, label='End', zorder=5)
    
    plt.title("Generated 32-Step Trajectory by Diffusion Model")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.legend()
    plt.grid(True)
    plt.axis('equal') # XとYのスケールを合わせる
    
    save_img = os.path.join(SOURCE_DIR, "generated_result.png")
    plt.savefig(save_img)
    print(f"生成完了！画像を保存しました: {save_img}")
    plt.show()

if __name__ == "__main__":
    generate_trajectory()