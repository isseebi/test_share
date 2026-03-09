import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. データの準備（ダミーの軌道データ）
# ==========================================
def generate_dummy_trajectories(num_samples=1000, seq_length=20):
    """サインカーブのような2次元軌道を生成します"""
    x = np.linspace(0, 2 * np.pi, seq_length)
    trajectories = []
    for _ in range(num_samples):
        # 少しずつ形が違うサインカーブを生成
        amp = np.random.uniform(0.8, 1.2)
        shift = np.random.uniform(-0.2, 0.2)
        y = amp * np.sin(x) + shift
        
        # (seq_length, 2) の座標データを (seq_length * 2) の1次元に平坦化
        traj = np.stack([x, y], axis=1).flatten()
        trajectories.append(traj)
    
    return torch.tensor(np.array(trajectories), dtype=torch.float32)

# ==========================================
# 2. モデルの定義（ノイズを予測するニューラルネット）
# ==========================================
class SimpleDiffusionModel(nn.Module):
    def __init__(self, data_dim, time_dim=32):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )
        
        self.net = nn.Sequential(
            nn.Linear(data_dim + time_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, data_dim) # 出力は入力データ（ノイズ）と同じ次元
        )

    def forward(self, x, t):
        # 時間 t をネットワークが理解しやすい形に変換
        t_emb = self.time_mlp(t.unsqueeze(-1).float())
        # 軌道データ x と 時間 t を結合して入力
        x_input = torch.cat([x, t_emb], dim=-1)
        return self.net(x_input)

# ==========================================
# 3. 拡散プロセスの設定（ノイズのスケジュール）
# ==========================================
# ステップ数（T）を決定
num_timesteps = 100
# ベータ（ノイズの強さ）を徐々に大きくする
betas = torch.linspace(1e-4, 0.02, num_timesteps)
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)

# ==========================================
# 4. 学習ループ
# ==========================================
def train():
    data_dim = 20 * 2 # 20ステップの (x, y) 座標
    model = SimpleDiffusionModel(data_dim=data_dim)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    dataset = generate_dummy_trajectories()
    batch_size = 64
    epochs = 20000

    print("学習を開始します...")
    for epoch in range(epochs):
        # ミニバッチの取得
        indices = torch.randint(0, len(dataset), (batch_size,))
        x_0 = dataset[indices]
        
        # ランダムな時刻 t をサンプリング
        t = torch.randint(0, num_timesteps, (batch_size,))
        
        # ランダムなノイズ（イプシロン）を生成
        noise = torch.randn_like(x_0)
        
        # 時刻 t に合わせて x_0 にノイズを加える (Forward Process)
        alpha_t = alphas_cumprod[t].unsqueeze(-1)
        x_t = torch.sqrt(alpha_t) * x_0 + torch.sqrt(1 - alpha_t) * noise
        
        # モデルにノイズを予測させる
        predicted_noise = model(x_t, t)
        
        # 実際のノイズと予測したノイズの誤差（MSE）を計算
        loss = nn.MSELoss()(predicted_noise, noise)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 200 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {loss.item():.4f}")
            
    return model

# ==========================================
# 5. 生成（サンプリング）プロセス
# ==========================================
@torch.no_grad()
def sample(model, num_samples=5):
    data_dim = 20 * 2
    # 完全にランダムなノイズからスタート
    x_t = torch.randn((num_samples, data_dim))
    
    # 時刻 T から 0 に向かって逆算 (Reverse Process)
    for t_step in reversed(range(num_timesteps)):
        t = torch.full((num_samples,), t_step, dtype=torch.long)
        
        # モデルが予測したノイズ
        predicted_noise = model(x_t, t)
        
        # ノイズを引き算して軌道を復元していく計算式
        alpha = alphas[t].unsqueeze(-1)
        alpha_cum = alphas_cumprod[t].unsqueeze(-1)
        beta = betas[t].unsqueeze(-1)
        
        if t_step > 0:
            noise = torch.randn_like(x_t)
        else:
            noise = torch.zeros_like(x_t) # 最後のステップはノイズを加えない
            
        x_t = (1 / torch.sqrt(alpha)) * (x_t - ((1 - alpha) / torch.sqrt(1 - alpha_cum)) * predicted_noise)
        x_t = x_t + torch.sqrt(beta) * noise
        
    return x_t.view(num_samples, 20, 2) # 元の (サンプル数, ステップ数, 2) の形に戻す

# ==========================================
# 実行と可視化
# ==========================================
if __name__ == "__main__":
    # 学習
    trained_model = train()
    
    # 軌道の生成
    print("軌道を生成しています...")
    generated_trajectories = sample(trained_model, num_samples=3).numpy()
    
    # グラフの描画
    
    plt.figure(figsize=(8, 5))
    for i, traj in enumerate(generated_trajectories):
        # plt.plot(traj[:, 0], traj[:, 1], marker='o', label=f'Generated {i+1}')
        plt.scatter(traj[:, 0], traj[:, 1])
    
    plt.title("Generated Trajectories by Diffusion Model")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid(True)
    plt.show()
