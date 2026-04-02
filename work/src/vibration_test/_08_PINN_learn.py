import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# 乱数シードの固定
torch.manual_seed(123)
np.random.seed(123)

# ==========================================
# 1. 観測データの生成 (振動外乱 + 乱数ノイズ)
# ==========================================
m = 1.0
k_true = 4.0
omega_n = np.sqrt(k_true / m)

# 外乱1: 別の周波数を持つ微小振動 (システム外乱)
A_vib = 0.15
omega_vib = 15.0 

# 外乱2: 観測時のランダムノイズ (測定外乱)
noise_level = 0.05

t_raw = torch.linspace(0, 5, 120).view(-1, 1)

# データ生成
u_clean = torch.cos(omega_n * t_raw) # 本来の挙動
u_dist_vib = A_vib * torch.sin(omega_vib * t_raw) # 振動外乱
noise = noise_level * torch.randn_like(t_raw) # 乱数ノイズ

# 全てを足し合わせたものが「観測データ」
u_data = (u_clean + u_dist_vib + noise).detach()

t_data = t_raw.clone().requires_grad_(True)

# ==========================================
# 2. PINNモデルの定義
# ==========================================
class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        # 未知パラメータ k の初期値を 1.0 に設定
        self.k = nn.Parameter(torch.tensor([1.0]))

    def forward(self, t):
        return self.net(t)

model = PINN()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ==========================================
# 3. 学習ループ
# ==========================================
epochs = 10000
k_history = []

for epoch in range(epochs):
    optimizer.zero_grad()
    
    u_pred = model(t_data)
    
    # --- 1. Data Loss (ノイズまみれのデータへの追従) ---
    loss_data = torch.mean((u_pred - u_data) ** 2)
    
    # --- 2. Physics Loss (振動外乱を物理式で考慮) ---
    du_dt = torch.autograd.grad(u_pred, t_data, torch.ones_like(u_pred), create_graph=True)[0]
    d2u_dt2 = torch.autograd.grad(du_dt, t_data, torch.ones_like(du_dt), create_graph=True)[0]
    
    # 物理式上では「既知の外力」として振動外乱を定義
    # これにより、この振動分はモデル誤差(Loss)としてカウントされなくなる
    f_vib = A_vib * torch.sin(omega_vib * t_data)
    
    # 運動方程式: m*u'' + k*u = f_vib
    physics_residual = m * d2u_dt2 + model.k * u_pred - f_vib
    loss_physics = torch.mean(physics_residual ** 2)
    
    # 合計損失
    loss = loss_data + loss_physics
    loss.backward()
    optimizer.step()
    
    k_history.append(model.k.item())
    
    if epoch % 1000 == 0:
        print(f'Epoch: {epoch:5d} | Loss: {loss.item():.5f} | k_pred: {model.k.item():.4f}')

# ==========================================
# 4. 結果の可視化
# ==========================================
plt.figure(figsize=(14, 5))

# 左グラフ: フィッティングの様子
plt.subplot(1, 2, 1)
plt.scatter(t_raw.numpy(), u_data.numpy(), label='Observed (Vib + Noise)', color='gray', alpha=0.4, s=15)
plt.plot(t_raw.numpy(), (u_clean + u_dist_vib).numpy(), label='True (No Noise)', color='blue', linestyle='--', alpha=0.6)
plt.plot(t_raw.numpy(), model(t_data).detach().numpy(), label='PINN Prediction', color='red', linewidth=2)
plt.xlabel('Time (t)')
plt.ylabel('Displacement (u)')
plt.legend()
plt.title('PINN under Multiple Disturbances')

# 右グラフ: k の収束
plt.subplot(1, 2, 2)
plt.plot(k_history, label='Predicted $k$', color='green')
plt.axhline(y=k_true, color='red', linestyle='--', label='True $k$')
plt.xlabel('Epochs')
plt.ylabel('Spring Constant ($k$)')
plt.ylim(0, 6)
plt.legend()
plt.title('Convergence of $k$')

plt.tight_layout()
plt.show()

print(f"\nFinal Results:")
print(f"True k: {k_true}")
print(f"Estimated k: {model.k.item():.4f}")

# # いかパーティクルフィルタによる推論
# import numpy as np
# import matplotlib.pyplot as plt

# # 乱数シードの固定
# np.random.seed(123)

# # ==========================================
# # 1. 真のデータ生成 (PINNの条件に合わせる)
# # ==========================================
# m = 1.0
# k_true = 4.0
# omega_n = np.sqrt(k_true / m)
# A_vib = 0.15
# omega_vib = 15.0
# noise_level = 0.05

# t_raw = np.linspace(0, 5, 120)
# dt = t_raw[1] - t_raw[0]

# # 真の信号と観測値
# u_clean = np.cos(omega_n * t_raw)
# u_dist_vib = A_vib * np.sin(omega_vib * t_raw)
# u_true = u_clean + u_dist_vib # 物理モデルが追従すべき真の挙動
# u_obs = u_true + noise_level * np.random.randn(len(t_raw))

# # ==========================================
# # 2. パーティクルフィルタの設定
# # ==========================================
# n_particles = 2000

# # 初期状態のサンプリング [u, v, k]
# # kは未知なので広めに分布させる (PINNの初期値1.0付近から開始)
# particles = np.zeros((n_particles, 3))
# particles[:, 0] = u_obs[0] + np.random.normal(0, 0.1, n_particles) # u
# particles[:, 1] = 0.0 + np.random.normal(0, 0.1, n_particles)      # v (速度)
# particles[:, 2] = np.random.uniform(0.0, 6.0, n_particles)        # k (バネ定数)

# weights = np.ones(n_particles) / n_particles

# # 推定値格納用
# u_est = []
# k_est = []

# def system_model(p, t, dt, m, A_vib, omega_vib):
#     """バネ質量系の物理モデルに基づく状態遷移"""
#     u, v, k = p[:, 0], p[:, 1], p[:, 2]
    
#     # 外部振動 f_vib
#     f_vib = A_vib * np.sin(omega_vib * t)
    
#     # 加速度 a = (f_vib - k*u) / m
#     a = (f_vib - k * u) / m
    
#     # 状態更新 (オイラー法) + システムノイズ
#     new_v = v + a * dt + np.random.normal(0, 0.01, n_particles)
#     new_u = u + v * dt + np.random.normal(0, 0.005, n_particles)
#     new_k = k + np.random.normal(0, 0.02, n_particles) # kは定数だが歩行させて探索
    
#     return np.stack([new_u, new_v, new_k], axis=1)

# # ==========================================
# # 3. フィルタリングループ
# # ==========================================
# for i in range(len(t_raw)):
#     # 1. 予測 (Prediction)
#     particles = system_model(particles, t_raw[i], dt, m, A_vib, omega_vib)
    
#     # 2. 重み更新 (Likelihood) - 観測値との差
#     # 観測データは変位 u のみ
#     distances = np.abs(particles[:, 0] - u_obs[i])
#     weights = np.exp(-distances**2 / (2 * noise_level**2))
#     weights += 1e-300 # ゼロ割防止
#     weights /= weights.sum()
    
#     # 3. 推定値の計算 (期待値)
#     u_est.append(np.average(particles[:, 0], weights=weights))
#     k_est.append(np.average(particles[:, 2], weights=weights))
    
#     # 4. リサンプリング (SIR)
#     if 1.0 / np.sum(weights**2) < n_particles / 2:
#         indices = np.random.choice(np.arange(n_particles), size=n_particles, p=weights)
#         particles = particles[indices]
#         weights = np.ones(n_particles) / n_particles

# # ==========================================
# # 4. 結果の可視化
# # ==========================================
# plt.figure(figsize=(14, 5))

# # 左グラフ: 軌道推定
# plt.subplot(1, 2, 1)
# plt.scatter(t_raw, u_obs, label='Observed (Vib + Noise)', color='gray', alpha=0.4, s=15)
# plt.plot(t_raw, u_true, label='True Signal', color='blue', linestyle='--', alpha=0.6)
# plt.plot(t_raw, u_est, label='PF Prediction', color='green', linewidth=2)
# plt.xlabel('Time (t)')
# plt.ylabel('Displacement (u)')
# plt.legend()
# plt.title('Particle Filter Denoising')

# # 右グラフ: k の推定推移
# plt.subplot(1, 2, 2)
# plt.plot(k_est, label='Estimated $k$ (PF)', color='green')
# plt.axhline(y=k_true, color='red', linestyle='--', label='True $k$')
# plt.xlabel('Time Steps')
# plt.ylabel('Spring Constant ($k$)')
# plt.ylim(0, 6)
# plt.legend()
# plt.title('Real-time Estimation of $k$')

# plt.tight_layout()
# plt.show()

# print(f"Final Estimated k: {k_est[-1]:.4f}")