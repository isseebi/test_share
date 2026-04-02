import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# 乱数シードの固定
torch.manual_seed(123)
np.random.seed(123)

# ==========================================
# 1. 観測データの生成 (2慣性系の数値シミュレーション)
# ==========================================
# 既知の質量
m1 = 1.0  # ボールねじの質量
m2 = 2.0  # 設備の質量

# 推定対象の真のパラメータ
k1_true = 16.0  # ボールねじ側の剛性
c1_true = 0.5   # ボールねじ側の減衰
k2_true = 8.0   # 設備側の剛性
c2_true = 0.2   # 設備側の減衰

A_vib = 2.0
omega_vib = 5.0
noise_level = 0.05

# 微分方程式の定義 (データ生成用)
def two_mass_system(t, y):
    u1, v1, u2, v2 = y
    f = A_vib * np.sin(omega_vib * t)
    du1_dt = v1
    dv1_dt = (f - (c1_true + c2_true)*v1 + c2_true*v2 - (k1_true + k2_true)*u1 + k2_true*u2) / m1
    du2_dt = v2
    dv2_dt = (- c2_true*(v2 - v1) - k2_true*(u2 - u1)) / m2
    return [du1_dt, dv1_dt, du2_dt, dv2_dt]

t_raw = np.linspace(0, 5, 200)
# 初期値: [u1, v1, u2, v2] = [0, 0, 0, 0]
sol = solve_ivp(two_mass_system, [0, 5], [0.0, 0.0, 0.0, 0.0], t_eval=t_raw)

u1_true = torch.tensor(sol.y[0]).view(-1, 1).float()
u2_true = torch.tensor(sol.y[2]).view(-1, 1).float() # 答え合わせ用（学習には使わない）

t_data = torch.tensor(t_raw).view(-1, 1).float().requires_grad_(True)

# 観測データ（u1のみにノイズを付加）
noise = noise_level * torch.randn_like(u1_true)
u1_data = (u1_true + noise).detach().float()

# ==========================================
# 2. PINNモデルの定義
# ==========================================
class PINN_2DOF(nn.Module):
    def __init__(self):
        super(PINN_2DOF, self).__init__()
        # 出力を2次元(u1_pred, u2_pred)にする
        self.net = nn.Sequential(
            nn.Linear(1, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 2) 
        )
        # 推定対象のパラメータ（初期値はわざとズラす）
        self.k1 = nn.Parameter(torch.tensor([10.0]))
        self.c1 = nn.Parameter(torch.tensor([0.1]))
        self.k2 = nn.Parameter(torch.tensor([10.0]))
        self.c2 = nn.Parameter(torch.tensor([0.1]))

    def forward(self, t):
        out = self.net(t)
        u1_pred = out[:, 0:1] # ボールねじの変位
        u2_pred = out[:, 1:2] # 設備の変位（隠れ状態）
        return u1_pred, u2_pred

model = PINN_2DOF()
# パラメータが多いので学習率を少し調整
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ==========================================
# 3. 学習ループ
# ==========================================
epochs = 20000
history = {'k1': [], 'c1': [], 'k2': [], 'c2': [], 'loss': []}

for epoch in range(epochs):
    optimizer.zero_grad()
    
    u1_pred, u2_pred = model(t_data)
    
    # --- 1. Data Loss (観測可能な u1 のみ!) ---
    loss_data = torch.mean((u1_pred - u1_data) ** 2)
    
    # --- 2. Physics Loss (u1とu2の両方の物理法則を計算) ---
    du1_dt = torch.autograd.grad(u1_pred, t_data, torch.ones_like(u1_pred), create_graph=True)[0]
    d2u1_dt2 = torch.autograd.grad(du1_dt, t_data, torch.ones_like(du1_dt), create_graph=True)[0]
    
    du2_dt = torch.autograd.grad(u2_pred, t_data, torch.ones_like(u2_pred), create_graph=True)[0]
    d2u2_dt2 = torch.autograd.grad(du2_dt, t_data, torch.ones_like(du2_dt), create_graph=True)[0]
    
    f_vib = A_vib * torch.sin(omega_vib * t_data)
    
    # 式1の残差 (ボールねじ側)
    res_1 = m1*d2u1_dt2 + (model.c1 + model.c2)*du1_dt - model.c2*du2_dt \
            + (model.k1 + model.k2)*u1_pred - model.k2*u2_pred - f_vib
            
    # 式2の残差 (設備側)
    res_2 = m2*d2u2_dt2 + model.c2*(du2_dt - du1_dt) + model.k2*(u2_pred - u1_pred)
    
    loss_physics = torch.mean(res_1 ** 2) + torch.mean(res_2 ** 2)
    
    # 合計損失
    loss = loss_data + loss_physics
    loss.backward()
    optimizer.step()
    
    # 履歴の保存
    history['k1'].append(model.k1.item())
    history['c1'].append(model.c1.item())
    history['k2'].append(model.k2.item())
    history['c2'].append(model.c2.item())
    history['loss'].append(loss.item())
    
    if epoch % 2000 == 0:
        print(f'Epoch: {epoch:5d} | Loss: {loss.item():.5f} | '
              f'k1:{model.k1.item():.1f}, c1:{model.c1.item():.2f}, '
              f'k2:{model.k2.item():.1f}, c2:{model.c2.item():.2f}')

# ==========================================
# 4. 結果の可視化
# ==========================================
plt.figure(figsize=(18, 5))

# ボールねじ (u1) のフィッティング (観測データあり)
plt.subplot(1, 3, 1)
plt.scatter(t_raw, u1_data.numpy(), label='Observed $u_1$', color='gray', alpha=0.5, s=10)
plt.plot(t_raw, model(t_data)[0].detach().numpy(), label='Predicted $u_1$', color='red')
plt.title('Ball Screw Displacement ($u_1$)')
plt.legend()

# 設備 (u2) の推測結果 (観測データなし!)
plt.subplot(1, 3, 2)
plt.plot(t_raw, u2_true.numpy(), label='True $u_2$ (Hidden)', color='blue', linestyle='--')
plt.plot(t_raw, model(t_data)[1].detach().numpy(), label='Predicted $u_2$', color='orange')
plt.title('Equipment Displacement ($u_2$) - Unobserved!')
plt.legend()

# パラメータ (剛性 k) の収束
plt.subplot(1, 3, 3)
plt.plot(history['k1'], label='Predicted $k_1$', color='green')
plt.axhline(y=k1_true, color='green', linestyle='--', alpha=0.5)
plt.plot(history['k2'], label='Predicted $k_2$', color='purple')
plt.axhline(y=k2_true, color='purple', linestyle='--', alpha=0.5)
plt.title('Convergence of Stiffness ($k$)')
plt.legend()

plt.tight_layout()
plt.show()

print(f"\n--- Final Estimation ---")
print(f"k1 (True: {k1_true:.1f}) -> Estimated: {model.k1.item():.2f}")
print(f"c1 (True: {c1_true:.2f}) -> Estimated: {model.c1.item():.3f}")
print(f"k2 (True: {k2_true:.1f}) -> Estimated: {model.k2.item():.2f}")
print(f"c2 (True: {c2_true:.2f}) -> Estimated: {model.c2.item():.3f}")