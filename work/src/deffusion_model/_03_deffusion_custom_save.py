import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os

# デバイスの設定
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# ==========================================
# 1. データの準備（スタート・中間・ゴールの3点を抽出）
# ==========================================
def generate_3waypoint_trajectories(num_samples=5000, seq_length=20):
    x = np.linspace(0, 2 * np.pi, seq_length)
    trajectories = []
    conditions = [] 
    
    for _ in range(num_samples):
        amp = np.random.uniform(0.5, 2.0)
        shift = np.random.uniform(-0.5, 0.5)
        y = amp * np.sin(x) + shift
        traj = np.stack([x, y], axis=1) # (20, 2)
        
        # 3点を抽出
        start_pt = traj[0]      # スタート (x=0)
        mid_idx = np.random.randint(5, 15)
        mid_pt = traj[mid_idx]  # 経由点 (中間付近)
        goal_pt = traj[-1]     # ゴール (x=2*pi)
        
        # 6次元のベクトルにまとめる [x1, y1, x2, y2, x3, y3]
        cond = np.concatenate([start_pt, mid_pt, goal_pt])
        
        trajectories.append(traj.flatten())
        conditions.append(cond)
    
    trajectories = torch.tensor(np.array(trajectories), dtype=torch.float32)
    conditions = torch.tensor(np.array(conditions), dtype=torch.float32)
    
    traj_mean = trajectories.mean(dim=0)
    traj_std = trajectories.std(dim=0) + 1e-6
    
    return (trajectories - traj_mean) / traj_std, conditions, (traj_mean, traj_std)

# ==========================================
# 2. モデルの定義 (cond_dim=6)
# ==========================================
class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim)
        )
    def forward(self, x):
        return x + self.net(x)

class PathPlanningDiffusionModel(nn.Module):
    def __init__(self, data_dim, cond_dim=6, time_dim=64):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_dim), nn.SiLU(), nn.Linear(time_dim, time_dim)
        )
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_dim, time_dim), nn.SiLU(), nn.Linear(time_dim, time_dim)
        )
        self.input_proj = nn.Linear(data_dim + time_dim + time_dim, 256)
        self.res_blocks = nn.ModuleList([ResBlock(256) for _ in range(6)])
        self.output_proj = nn.Sequential(
            nn.Linear(256, 128), nn.SiLU(), nn.Linear(128, data_dim)
        )

    def forward(self, x, t, c):
        t_emb = self.time_mlp(t.unsqueeze(-1).float() / 100.0)
        c_emb = self.cond_mlp(c)
        h = torch.cat([x, t_emb, c_emb], dim=-1)
        h = self.input_proj(h)
        for block in self.res_blocks:
            h = block(h)
        return self.output_proj(h)

# 拡散設定
num_timesteps = 200
betas = torch.linspace(1e-4, 0.02, num_timesteps).to(device)
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)

# ==========================================
# 4. 学習ループ
# ==========================================
def train():
    data_dim = 20 * 2
    model = PathPlanningDiffusionModel(data_dim=data_dim).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=3e-4)
    
    dataset, condset, stats = generate_3waypoint_trajectories()
    dataset, condset = dataset.to(device), condset.to(device)
    stats = (stats[0].to(device), stats[1].to(device))
    
    batch_size = 256
    epochs = 40000 

    print(f"{device} を使用して、3地点条件付き学習を開始します...")
    for epoch in range(epochs):
        indices = torch.randint(0, len(dataset), (batch_size,))
        x_0, c_0 = dataset[indices], condset[indices]
        
        if np.random.rand() < 0.1: # Classifier-Free Guidance
            c_0 = torch.zeros_like(c_0)
        
        t = torch.randint(0, num_timesteps, (batch_size,), device=device)
        noise = torch.randn_like(x_0)
        alpha_t = alphas_cumprod[t].unsqueeze(-1)
        x_t = torch.sqrt(alpha_t) * x_0 + torch.sqrt(1 - alpha_t) * noise
        
        predicted_noise = model(x_t, t, c_0)
        loss = nn.MSELoss()(predicted_noise, noise)
        
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        
        if (epoch + 1) % 5000 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {loss.item():.6f}")
            
    return model, stats

# ==========================================
# 5. 生成プロセス
# ==========================================
@torch.no_grad()
def sample(model, conditions, stats, guidance_scale=8.0):
    num_samples = conditions.shape[0]
    data_dim = 20 * 2
    x_t = torch.randn((num_samples, data_dim), device=device)
    conditions = conditions.to(device)
    traj_mean, traj_std = stats
    
    for t_step in reversed(range(num_timesteps)):
        t = torch.full((num_samples,), t_step, dtype=torch.long, device=device)
        eps_cond = model(x_t, t, conditions)
        eps_uncond = model(x_t, t, torch.zeros_like(conditions))
        predicted_noise = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
        
        alpha, alpha_cum, beta = alphas[t].unsqueeze(-1), alphas_cumprod[t].unsqueeze(-1), betas[t].unsqueeze(-1)
        noise = torch.randn_like(x_t) if t_step > 0 else torch.zeros_like(x_t)
        x_t = (1 / torch.sqrt(alpha)) * (x_t - ((1 - alpha) / torch.sqrt(1 - alpha_cum)) * predicted_noise)
        x_t = x_t + torch.sqrt(beta) * noise
        
    return (x_t * traj_std + traj_mean).view(num_samples, 20, 2).cpu()

def train_and_save():
    # 学習の実行
    model, stats = train() 
    
    # 保存用ディレクトリの作成
    os.makedirs("models", exist_ok=True)
    
    # 1. モデルの重みを保存
    torch.save(model.state_dict(), "models/diffusion_path_model.pth")
    
    # 2. 統計量（mean, std）を保存
    torch.save({
        'mean': stats[0].cpu(),
        'std': stats[1].cpu()
    }, "models/stats.pt")
    
    print("モデルと統計量を models/ ディレクトリに保存しました。")
# 実行
if __name__ == "__main__":
    train_and_save()

    # test_conditions = torch.tensor([
    #     [0.0, 0.0,  3.14, 1.5,  6.28, 0.0],  # 山なり
    #     [0.0, 0.5,  3.14, -1.0, 6.28, 0.5], # 谷なり
    #     [0.0, -0.5, 2.0, 1.0,   6.28, -0.5]  # 変則的
    # ], dtype=torch.float32)
    
    # generated = sample(trained_model, test_conditions, stats).numpy()
    
    # plt.figure(figsize=(10, 6))
    # for i in range(len(test_conditions)):
    #     plt.plot(generated[i,:,0], generated[i,:,1], 'o-', label=f'Path {i+1}', alpha=0.6)
    #     # 指定した3点をプロット
    #     pts = test_conditions[i].reshape(3, 2)
    #     plt.scatter(pts[:,0], pts[:,1], marker='*', s=300, edgecolors='black', zorder=10)
    
    # plt.title("Start-Mid-Goal Conditional Path Generation")
    # plt.xlabel("X (Time/Sequence)"); plt.ylabel("Y (Position)")
    # plt.grid(True); plt.legend(); plt.show()