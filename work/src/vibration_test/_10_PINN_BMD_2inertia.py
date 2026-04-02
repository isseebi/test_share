# ニューラルネットワーク、特にフーリエ特徴量を使っているモデルは非常に「器用」です。データ点の間で物理法則を無視して、無理やりデータ点だけをなぞるような**「物理的にありえない波打ち」**を簡単に作れてしまいます。
# 2000点のコロケーションポイントを置くことで、「この2000点すべてにおいて、現在の $\omega$ や $\zeta$ を使って微分方程式が成立しなければならない」という非常に厳しい制約を課したことになります。これにより、ネットワークは物理的に正しい（＝パラメータが真値に近い）解を選ばざるを得なくなったのです。


import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# 乱数シードの固定
torch.manual_seed(123)
np.random.seed(123)

# ==========================================
# 1. 観測データの生成 (2自由度連成振動)
# ==========================================
m1, m2 = 1.0, 0.2
mu = m2 / m1

omega1_true, zeta1_true = 4.0, 0.1
omega2_true, zeta2_true = 10.0, 0.05

k1 = m1 * omega1_true**2
c1 = m1 * 2 * zeta1_true * omega1_true
k2 = m2 * omega2_true**2
c2 = m2 * 2 * zeta2_true * omega2_true

A_vib = 5.0
omega_vib = 15.0 

def system_dynamics(t, y):
    x1, x1_dot, x2, x2_dot = y
    f_ext = A_vib * np.sin(omega_vib * t)
    x1_ddot = (-k1*x1 - c1*x1_dot + k2*(x2 - x1) + c2*(x2_dot - x1_dot) + f_ext) / m1
    x2_ddot = (-k2*(x2 - x1) - c2*(x2_dot - x1_dot)) / m2
    return [x1_dot, x1_ddot, x2_dot, x2_ddot]

t_eval = np.linspace(0, 5, 200)
sol = solve_ivp(system_dynamics, [0, 5], [0.0, 0.0, 0.0, 0.0], t_eval=t_eval, method='RK45')

t_raw = sol.t.reshape(-1, 1)
x1_data = sol.y[0].reshape(-1, 1)
x2_data = sol.y[2].reshape(-1, 1)

t_data = torch.tensor(t_raw, dtype=torch.float32, requires_grad=True)
x1_data_tensor = torch.tensor(x1_data, dtype=torch.float32)
x2_data_tensor = torch.tensor(x2_data, dtype=torch.float32)

class PINN_Fourier(nn.Module):
    def __init__(self):
        super(PINN_Fourier, self).__init__()
        
        # --- 1. フーリエ特徴量の設定 ---
        # 様々な周波数成分を作り出すための固定された重み (学習させない)
        # スケール(sigma)は対象の周波数に合わせて調整 (ここでは 5.0 程度)
        sigma = 5.0
        self.B = nn.Parameter(torch.randn(1, 64) * sigma, requires_grad=False)
        
        # --- 2. モデルの巨大化 ---
        # 入力層: sin と cos のペアになるため、特徴量サイズは 64 * 2 = 128 になる
        self.net = nn.Sequential(
            nn.Linear(128, 256), # ユニット数を256に増強
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 256), # 層を深くする
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 2)
        )
        
        # 推定パラメータ
        self.raw_omega1 = nn.Parameter(torch.tensor([2.0]))
        self.raw_zeta1  = nn.Parameter(torch.tensor([0.01]))
        self.raw_omega2 = nn.Parameter(torch.tensor([5.0]))
        self.raw_zeta2  = nn.Parameter(torch.tensor([0.01]))

    def forward(self, t):
        # tの形状は (N, 1)
        # フーリエ特徴量への射影
        t_proj = 2.0 * np.pi * torch.matmul(t, self.B) # (N, 64)
        
        # sinとcosを結合してネットワークに入力
        t_features = torch.cat([torch.sin(t_proj), torch.cos(t_proj)], dim=-1) # (N, 128)
        
        return self.net(t_features)
    
    @property
    def omega1(self): return torch.abs(self.raw_omega1)
    @property
    def zeta1(self):  return torch.abs(self.raw_zeta1)
    @property
    def omega2(self): return torch.abs(self.raw_omega2)
    @property
    def zeta2(self):  return torch.abs(self.raw_zeta2)

# モデルのインスタンス化をこちらに変更します
model = PINN_Fourier()

optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
# 学習率スケジューラ: 5000エポックごとに学習率を半分にする
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.5)

# ==========================================
# ==========================================
# 3. 学習ループの準備とコロケーションポイントの追加
# ==========================================
epochs = 20000 * 3
history = {'omega1': [], 'zeta1': [], 'omega2': [], 'zeta2': [], 'loss': []}

m1_t = torch.tensor([m1])
mu_t = torch.tensor([mu])

# 【修正点1】Physics Lossを評価するための高密度なコロケーションポイントを生成
t_colloc_raw = np.linspace(0, 5, 2000).reshape(-1, 1)
t_colloc = torch.tensor(t_colloc_raw, dtype=torch.float32, requires_grad=True)

# 【修正点2】ネットワークの重みと、推定パラメータの学習率を分ける
optimizer = torch.optim.Adam([
    {'params': model.net.parameters(), 'lr': 1e-3},
    {'params': [model.raw_omega1, model.raw_zeta1, model.raw_omega2, model.raw_zeta2], 'lr': 5e-3}
])
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.5)

# 【修正点3】Physics Loss のスケールが大きいと予想されるため小さめに設定
lambda_phys = 1e-3  

for epoch in range(epochs):
    optimizer.zero_grad()
    
    # --- 1. Data Loss (観測データ点 t_data で計算) ---
    pred_data = model(t_data)
    x1_pred_data = pred_data[:, 0:1]
    x2_pred_data = pred_data[:, 1:2]
    
    loss_data = torch.mean((x1_pred_data - x1_data_tensor)**2) + torch.mean((x2_pred_data - x2_data_tensor)**2)
    
    # --- 2. Physics Loss (コロケーションポイント t_colloc で計算) ---
    pred_colloc = model(t_colloc)
    x1_pred_colloc = pred_colloc[:, 0:1]
    x2_pred_colloc = pred_colloc[:, 1:2]
    
    dx1_dt = torch.autograd.grad(x1_pred_colloc, t_colloc, torch.ones_like(x1_pred_colloc), create_graph=True)[0]
    d2x1_dt2 = torch.autograd.grad(dx1_dt, t_colloc, torch.ones_like(dx1_dt), create_graph=True)[0]
    
    dx2_dt = torch.autograd.grad(x2_pred_colloc, t_colloc, torch.ones_like(x2_pred_colloc), create_graph=True)[0]
    d2x2_dt2 = torch.autograd.grad(dx2_dt, t_colloc, torch.ones_like(dx2_dt), create_graph=True)[0]
    
    f_vib_colloc = A_vib * torch.sin(omega_vib * t_colloc)
    
    # 絶対値制約されたパラメータを使用
    term_k2 = model.omega2**2 * (x2_pred_colloc - x1_pred_colloc)
    term_c2 = 2 * model.zeta2 * model.omega2 * (dx2_dt - dx1_dt)
    term_k1 = model.omega1**2 * x1_pred_colloc
    term_c1 = 2 * model.zeta1 * model.omega1 * dx1_dt
    
    res2 = d2x2_dt2 + term_c2 + term_k2
    res1 = d2x1_dt2 + term_c1 + term_k1 - mu_t * (term_c2 + term_k2) - f_vib_colloc / m1_t
    
    loss_physics = torch.mean(res1**2) + torch.mean(res2**2)
    
    # バランスをとって合算
    loss = loss_data + lambda_phys * loss_physics
    loss.backward()
    
    optimizer.step()
    scheduler.step()
    
    # 履歴の保存
    history['omega1'].append(model.omega1.item())
    history['zeta1'].append(model.zeta1.item())
    history['omega2'].append(model.omega2.item())
    history['zeta2'].append(model.zeta2.item())
    history['loss'].append(loss.item())
    
    if epoch % 1000 == 0:
        print(f'Epoch:{epoch:5d} | Loss:{loss.item():.5f} (Data:{loss_data.item():.5f}, Phys:{loss_physics.item():.5f}) | '
              f'w1:{model.omega1.item():.2f}, z1:{model.zeta1.item():.3f} | '
              f'w2:{model.omega2.item():.2f}, z2:{model.zeta2.item():.3f}')
# ==========================================
# 4. 結果の可視化
# ==========================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 軌跡のフィッティング
axes[0, 0].scatter(t_raw, x1_data, label='Observed $x_1$ (Ball Screw)', color='gray', s=10, alpha=0.8)
axes[0, 0].plot(t_raw, model(t_data)[:, 0].detach().numpy(), label='PINN $x_1$', color='red')
axes[0, 0].set_title('Displacement $x_1$ (Ball Screw)')
axes[0, 0].legend()

axes[0, 1].scatter(t_raw, x2_data, label='Observed $x_2$ (Nozzle)', color='gray', s=10, alpha=0.8)
axes[0, 1].plot(t_raw, model(t_data)[:, 1].detach().numpy(), label='PINN $x_2$', color='red')
axes[0, 1].set_title('Displacement $x_2$ (Nozzle)')
axes[0, 1].legend()

# 固有振動数の収束
axes[1, 0].plot(history['omega1'], label='Predicted $\omega_1$', color='green')
axes[1, 0].axhline(omega1_true, color='red', linestyle='--', label='True $\omega_1$')
axes[1, 0].plot(history['omega2'], label='Predicted $\omega_2$', color='purple')
axes[1, 0].axhline(omega2_true, color='orange', linestyle='--', label='True $\omega_2$')
axes[1, 0].set_title('Convergence of Natural Frequencies ($\omega_n$)')
axes[1, 0].legend()

# 減衰比の収束
axes[1, 1].plot(history['zeta1'], label='Predicted $\zeta_1$', color='green')
axes[1, 1].axhline(zeta1_true, color='red', linestyle='--', label='True $\zeta_1$')
axes[1, 1].plot(history['zeta2'], label='Predicted $\zeta_2$', color='purple')
axes[1, 1].axhline(zeta2_true, color='orange', linestyle='--', label='True $\zeta_2$')
axes[1, 1].set_ylim([0.0, 0.2]) # y軸を見やすく固定
axes[1, 1].set_title('Convergence of Damping Ratios ($\zeta$)')
axes[1, 1].legend()

plt.tight_layout()
plt.show()

print(f"\nFinal Estimation:")
print(f"[Ball Screw] True omega1: {omega1_true:.4f} | Estimated: {model.omega1.item():.4f}")
print(f"[Ball Screw] True zeta1:  {zeta1_true:.4f} | Estimated: {model.zeta1.item():.4f}")
print(f"[Nozzle]     True omega2: {omega2_true:.4f} | Estimated: {model.omega2.item():.4f}")
print(f"[Nozzle]     True zeta2:  {zeta2_true:.4f} | Estimated: {model.zeta2.item():.4f}")