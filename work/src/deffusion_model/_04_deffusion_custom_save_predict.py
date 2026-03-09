import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from _03_deffusion_custom_save import PathPlanningDiffusionModel, sample

# 推論時もモデルの構造定義が必要なため、ここにクラス定義を再掲するか、
# 共通モジュールとして import する必要があります。
# (ここでは簡略化のため構造が定義されている前提で進めます)

def load_and_sample(test_conditions):
    # 0. デバイス設定
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    
    # 1. 統計量の読み込み
    stats_dict = torch.load("models/stats.pt", map_location=device)
    stats = (stats_dict['mean'].to(device), stats_dict['std'].to(device))
    
    # 2. モデルのインスタンス化と重みの読み込み
    data_dim = 20 * 2
    model = PathPlanningDiffusionModel(data_dim=data_dim).to(device)
    model.load_state_dict(torch.load("models/diffusion_path_model.pth", map_location=device))
    model.eval() # 推論モードに設定
    
    print(f"モデルを {device} 上に読み込みました。推論を開始します...")
    
    # 3. サンプリング実行
    # test_conditions は [Start_x, Start_y, Mid_x, Mid_y, Goal_x, Goal_y] のリスト
    cond_tensor = torch.tensor(test_conditions, dtype=torch.float32)
    generated = sample(model, cond_tensor, stats, guidance_scale=8.0)
    
    return generated.numpy()

# 実行例
if __name__ == "__main__":
    # 好きな3地点を指定 [Start(x,y), Mid(x,y), Goal(x,y)]
    my_conditions = [
        [0.0, 0.0,  3.14, 2.0,  6.28, 0.0],  # 高い山を描く
        [0.0, -1.0, 3.14, 0.0,  6.28, 1.0],  # 斜めに上がる
    ]
    
    results = load_and_sample(my_conditions)
    
    # 可視化
    plt.figure(figsize=(10, 6))
    for i, traj in enumerate(results):
        plt.plot(traj[:, 0], traj[:, 1], 'o-', label=f'Inference {i+1}')
        pts = np.array(my_conditions[i]).reshape(3, 2)
        plt.scatter(pts[:,0], pts[:,1], marker='*', s=300, color='black', zorder=10)
    
    plt.title("Inference using Saved Diffusion Model")
    plt.grid(True); plt.legend(); plt.show()