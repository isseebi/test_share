import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# 乱数シードの固定
torch.manual_seed(123)
np.random.seed(123)

# ==========================================
# 1. 観測データの生成 (ユーザー指定のパラメータを反映)
# ==========================================
m1, m2 = 1.0, 0.2

# 真の物理パラメータ
k1_true, c1_true = 16.0, 0.50
k2_true, c2_true = 8.0, 0.20

A_vib = 5.0
omega_vib = 15.0 

def system_dynamics(t, y):
    x1, x1_dot, x2, x2_dot = y
    f_ext = A_vib * np.sin(omega_vib * t)
    x1_ddot = (-k1_true*x1 - c1_true*x1_dot + k2_true*(x2 - x1) + c2_true*(x2_dot - x1_dot) + f_ext) / m1
    x2_ddot = (-k2_true*(x2 - x1) - c2_true*(x2_dot - x1_dot)) / m2
    return [x1_dot, x1_ddot, x2_dot, x2_ddot]

t_eval = np.linspace(0, 5, 200)
sol = solve_ivp(system_dynamics, [0, 5], [0.0, 0.0, 0.0, 0.0], t_eval=t_eval, method='RK45')

t_raw = sol.t.reshape(-1, 1)
x1_data = sol.y[0].reshape(-1, 1)
x2_data = sol.y[2].reshape(-1, 1)

t_data = torch.tensor(t_raw, dtype=torch.float32, requires_grad=True)
x1_data_tensor = torch.tensor(x1_data, dtype=torch.float32)

# ==========================================
# 2. モデル定義 (パラメータ初期値の変更)
# ==========================================
class PINN_Fourier(nn.Module):
    def __init__(self):
        super(PINN_Fourier, self).__init__()
        # 前回の修正: sigma の値を 5.0 -> 0.5 に小さくする (これは維持)
        sigma = 0.5
        self.B = nn.Parameter(torch.randn(1, 64) * sigma, requires_grad=False)
        
        self.net_x1 = nn.Sequential(
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 1)
        )
        
        self.net_x2 = nn.Sequential(
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 1)
        )
        
        # 修正: k, c の初期値を、真値から離れつつも少し現実的な値に近づける
        # k1_true=16.0, c1_true=0.5, k2_true=8.0, c2_true=0.2
        self.raw_k1 = nn.Parameter(torch.tensor([10.0]))
        self.raw_c1 = nn.Parameter(torch.tensor([0.1]))
        self.raw_k2 = nn.Parameter(torch.tensor([5.0]))
        self.raw_c2 = nn.Parameter(torch.tensor([0.05]))
        
        # 拘束範囲の設定 (オプション: このままでも良いが、物理的な上限がわかっているなら bounded するのも手です)
        # 今回は property での abs() 拘束を維持します。

    # (以下、forward メソッドと property は元のコードと同じ)
    def forward_x1(self, t):
        t_proj = 2.0 * np.pi * torch.matmul(t, self.B)
        t_features = torch.cat([torch.sin(t_proj), torch.cos(t_proj)], dim=-1)
        return self.net_x1(t_features)

    def forward_x2(self, t):
        t_proj = 2.0 * np.pi * torch.matmul(t, self.B)
        t_features = torch.cat([torch.sin(t_proj), torch.cos(t_proj)], dim=-1)
        return self.net_x2(t_features)
    
    @property
    def k1(self): return torch.abs(self.raw_k1)
    @property
    def c1(self): return torch.abs(self.raw_c1)
    @property
    def k2(self): return torch.abs(self.raw_k2)
    @property
    def c2(self): return torch.abs(self.raw_c2)

model = PINN_Fourier()

m1_t = torch.tensor([m1])
m2_t = torch.tensor([m2])

t_colloc_raw = np.linspace(0, 5, 2000).reshape(-1, 1)
t_colloc = torch.tensor(t_colloc_raw, dtype=torch.float32, requires_grad=True)
t_ic = torch.tensor([[0.0]], dtype=torch.float32, requires_grad=True)

# ==========================================
# 3. 学習ループ (Phase 2 のパラメータ学習率と損失重みの調整)
# ==========================================

# (Phase 1 は元のコードと同じ)
print("--- Phase 1: Pre-training x1 network (Strictly fitting data) ---")
optimizer_phase1 = torch.optim.Adam(model.net_x1.parameters(), lr=5e-3)

for epoch in range(10000):
    optimizer_phase1.zero_grad()
    x1_pred = model.forward_x1(t_data)
    loss_data = torch.mean((x1_pred - x1_data_tensor)**2)
    loss_data.backward()
    optimizer_phase1.step()
    
    if epoch % 2000 == 0:
        print(f"Phase1 Epoch {epoch:5d} | x1 Data Loss: {loss_data.item():.7f}")

# ------------------------------------------
# Phase 2: x1を凍結し、x2とパラメータのみを推論 (修正)
# ------------------------------------------
print("\n--- Phase 2: Joint Training (Discovering Parameters and x2) ---")

for param in model.net_x1.parameters():
    param.requires_grad = False

# x2 のネットワークとパラメータのみをオプティマイザに渡す
# 修正: パラメータ更新の学習率を network (2e-3) より大幅に小さくして、安定させる (2e-3 -> 5e-4)
optimizer_phase2 = torch.optim.Adam([
    {'params': model.net_x2.parameters(), 'lr': 2e-3}, # network (unchanged)
    {'params': [model.raw_k1, model.raw_c1, model.raw_k2, model.raw_c2], 'lr': 5e-4} # parameters (slowed)
])
scheduler = torch.optim.lr_scheduler.StepLR(optimizer_phase2, step_size=10000, gamma=0.5)

epochs_phase2 = 30000
history = {'k1': [], 'c1': [], 'k2': [], 'c2': [], 'loss': []}

for epoch in range(epochs_phase2):
    optimizer_phase2.zero_grad()
    
    # --- IC Loss ---
    x2_ic = model.forward_x2(t_ic)
    dx2_dt_ic = torch.autograd.grad(x2_ic, t_ic, torch.ones_like(x2_ic), create_graph=True)[0]
    loss_ic = torch.mean(x2_ic**2) + torch.mean(dx2_dt_ic**2)
    
    # --- Physics Loss (元のコードと同じ) ---
    x1_pred = model.forward_x1(t_colloc)
    dx1_dt = torch.autograd.grad(x1_pred, t_colloc, torch.ones_like(x1_pred), create_graph=True)[0]
    d2x1_dt2 = torch.autograd.grad(dx1_dt, t_colloc, torch.ones_like(dx1_dt), create_graph=True)[0]
    
    x2_pred = model.forward_x2(t_colloc)
    dx2_dt = torch.autograd.grad(x2_pred, t_colloc, torch.ones_like(x2_pred), create_graph=True)[0]
    d2x2_dt2 = torch.autograd.grad(dx2_dt, t_colloc, torch.ones_like(dx2_dt), create_graph=True)[0]
    
    f_vib_colloc = A_vib * torch.sin(omega_vib * t_colloc)
    
    res_global = m1_t * d2x1_dt2 + m2_t * d2x2_dt2 + model.c1 * dx1_dt + model.k1 * x1_pred - f_vib_colloc
    res_local  = m2_t * d2x2_dt2 + model.c2 * (dx2_dt - dx1_dt) + model.k2 * (x2_pred - x1_pred)
    
    loss_physics = torch.mean(res_global**2) + torch.mean(res_local**2)
    
    # 合算 (スケール調整)
    # 修正: 物理損失が非常に大きくなる傾向があるため、重みを 1e-4 -> 1e-2 に「大きく」する
    # 1e-4 だと物理拘束が弱すぎて、x2 の network がパラメータを無視して収束してしまったため、拘束を強める
    loss = loss_ic + 1e-2 * loss_physics
    loss.backward()
    
    optimizer_phase2.step()
    scheduler.step()
    
    history['k1'].append(model.k1.item())
    history['c1'].append(model.c1.item())
    history['k2'].append(model.k2.item())
    history['c2'].append(model.c2.item())
    history['loss'].append(loss.item())
    
    if epoch % 2000 == 0:
        print(f'Epoch:{epoch:5d} | Loss:{loss.item():.5f} | '
              f'k1:{model.k1.item():.2f}, c1:{model.c1.item():.3f} | '
              f'k2:{model.k2.item():.2f}, c2:{model.c2.item():.3f}')
# ==========================================
# 4. 結果の可視化
# ==========================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].scatter(t_raw, x1_data, label='Observed $x_1$', color='gray', s=10, alpha=0.8)
axes[0, 0].plot(t_raw, model.forward_x1(t_data).detach().numpy(), label='PINN $x_1$', color='red')
axes[0, 0].set_title('Displacement $x_1$ (Observed)')
axes[0, 0].legend()

axes[0, 1].plot(t_raw, x2_data, label='True $x_2$ (Hidden)', color='gray', linestyle='--')
axes[0, 1].plot(t_raw, model.forward_x2(t_data).detach().numpy(), label='Inferred PINN $x_2$', color='blue')
axes[0, 1].set_title('Displacement $x_2$ (Unobserved/Inferred)')
axes[0, 1].legend()

axes[1, 0].plot(history['k1'], label='Predicted $k_1$', color='green')
axes[1, 0].axhline(k1_true, color='red', linestyle='--', label='True $k_1$')
axes[1, 0].plot(history['k2'], label='Predicted $k_2$', color='purple')
axes[1, 0].axhline(k2_true, color='orange', linestyle='--', label='True $k_2$')
axes[1, 0].set_title('Convergence of Stiffness ($k$)')
axes[1, 0].legend()

axes[1, 1].plot(history['c1'], label='Predicted $c_1$', color='green')
axes[1, 1].axhline(c1_true, color='red', linestyle='--', label='True $c_1$')
axes[1, 1].plot(history['c2'], label='Predicted $c_2$', color='purple')
axes[1, 1].axhline(c2_true, color='orange', linestyle='--', label='True $c_2$')
axes[1, 1].set_title('Convergence of Damping ($c$)')
axes[1, 1].legend()

plt.tight_layout()
plt.show()

print(f"\n[Final Results]")
print(f"k1: True {k1_true:.2f} | Est {model.k1.item():.2f} (Err: {abs(k1_true-model.k1.item())/k1_true*100:.1f}%)")
print(f"k2: True {k2_true:.2f} | Est {model.k2.item():.2f} (Err: {abs(k2_true-model.k2.item())/k2_true*100:.1f}%)")
print(f"c1: True {c1_true:.2f} | Est {model.c1.item():.2f} (Err: {abs(c1_true-model.c1.item())/c1_true*100:.1f}%)")
print(f"c2: True {c2_true:.2f} | Est {model.c2.item():.2f} (Err: {abs(c2_true-model.c2.item())/c2_true*100:.1f}%)")