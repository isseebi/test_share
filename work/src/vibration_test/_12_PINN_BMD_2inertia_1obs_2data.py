# 2自由度の2慣性系のモデル

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# 乱数シードの固定
torch.manual_seed(123)
np.random.seed(123)

# ==========================================
# 1. 観測データの生成 (2つの異なる周波数条件)
# ==========================================
m1, m2 = 1.0, 0.2
k1_true, c1_true = 16.0, 0.50
k2_true, c2_true = 8.0, 0.20

A_vib = 5.0
omega_A = 15.0  # 条件A: 元の周波数
omega_B = 22.0  # 条件B: 異なる周波数 (少し高めにする)

def get_system_dynamics(omega):
    def system_dynamics(t, y):
        x1, x1_dot, x2, x2_dot = y
        f_ext = A_vib * np.sin(omega * t)
        x1_ddot = (-k1_true*x1 - c1_true*x1_dot + k2_true*(x2 - x1) + c2_true*(x2_dot - x1_dot) + f_ext) / m1
        x2_ddot = (-k2_true*(x2 - x1) - c2_true*(x2_dot - x1_dot)) / m2
        return [x1_dot, x1_ddot, x2_dot, x2_ddot]
    return system_dynamics

t_eval = np.linspace(0, 5, 200)

# 条件Aのデータ生成
sol_A = solve_ivp(get_system_dynamics(omega_A), [0, 5], [0.0, 0.0, 0.0, 0.0], t_eval=t_eval, method='RK45')
x1_data_A = torch.tensor(sol_A.y[0].reshape(-1, 1), dtype=torch.float32)
x2_data_A = sol_A.y[2].reshape(-1, 1) # 比較用の隠れデータ

# 条件Bのデータ生成
sol_B = solve_ivp(get_system_dynamics(omega_B), [0, 5], [0.0, 0.0, 0.0, 0.0], t_eval=t_eval, method='RK45')
x1_data_B = torch.tensor(sol_B.y[0].reshape(-1, 1), dtype=torch.float32)
x2_data_B = sol_B.y[2].reshape(-1, 1) # 比較用の隠れデータ

t_data = torch.tensor(t_eval.reshape(-1, 1), dtype=torch.float32, requires_grad=True)

# ==========================================
# 2. モデル定義 (マルチデータ対応)
# ==========================================
class PINN_MultiData(nn.Module):
    def __init__(self):
        super(PINN_MultiData, self).__init__()
        self.B = nn.Parameter(torch.randn(1, 64) * 0.5, requires_grad=False)
        
        # 条件A用のネットワーク
        self.net_x1_A = self._build_net()
        self.net_x2_A = self._build_net()
        
        # 条件B用のネットワーク
        self.net_x1_B = self._build_net()
        self.net_x2_B = self._build_net()
        
        # 共通の物理パラメータ (ここが両方のデータから制約を受ける)
        self.raw_k1 = nn.Parameter(torch.tensor([10.0]))
        self.raw_c1 = nn.Parameter(torch.tensor([0.1]))
        self.raw_k2 = nn.Parameter(torch.tensor([5.0]))
        self.raw_c2 = nn.Parameter(torch.tensor([0.05]))
        
    def _build_net(self):
        return nn.Sequential(
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 1)
        )

    def _get_features(self, t):
        t_proj = 2.0 * np.pi * torch.matmul(t, self.B)
        return torch.cat([torch.sin(t_proj), torch.cos(t_proj)], dim=-1)

    def forward_A(self, t):
        feat = self._get_features(t)
        return self.net_x1_A(feat), self.net_x2_A(feat)

    def forward_B(self, t):
        feat = self._get_features(t)
        return self.net_x1_B(feat), self.net_x2_B(feat)
    
    @property
    def k1(self): return torch.abs(self.raw_k1)
    @property
    def c1(self): return torch.abs(self.raw_c1)
    @property
    def k2(self): return torch.abs(self.raw_k2)
    @property
    def c2(self): return torch.abs(self.raw_c2)

model = PINN_MultiData()

m1_t = torch.tensor([m1])
m2_t = torch.tensor([m2])

t_colloc = torch.tensor(np.linspace(0, 5, 2000).reshape(-1, 1), dtype=torch.float32, requires_grad=True)
t_ic = torch.tensor([[0.0]], dtype=torch.float32, requires_grad=True)

# ==========================================
# 3. 学習ループ (同時学習)
# ==========================================
print("--- Phase 1: Pre-training x1 networks (Both A and B) ---")
# まずは x1 をデータに馴染ませる
optimizer_phase1 = torch.optim.Adam(
    list(model.net_x1_A.parameters()) + list(model.net_x1_B.parameters()), 
    lr=5e-3
)

for epoch in range(5000):
    optimizer_phase1.zero_grad()
    x1_pred_A, _ = model.forward_A(t_data)
    x1_pred_B, _ = model.forward_B(t_data)
    
    loss_data = torch.mean((x1_pred_A - x1_data_A)**2) + torch.mean((x1_pred_B - x1_data_B)**2)
    loss_data.backward()
    optimizer_phase1.step()
    
    if epoch % 1000 == 0:
        print(f"Phase1 Epoch {epoch:4d} | Data Loss: {loss_data.item():.7f}")

print("\n--- Phase 2: Joint Training (Discovering Parameters and x2) ---")
# 修正: x1 は「凍結」せず、学習率を下げて全体を最適化する
optimizer_phase2 = torch.optim.Adam([
    {'params': model.net_x1_A.parameters(), 'lr': 1e-4}, # 微調整
    {'params': model.net_x1_B.parameters(), 'lr': 1e-4}, # 微調整
    {'params': model.net_x2_A.parameters(), 'lr': 2e-3},
    {'params': model.net_x2_B.parameters(), 'lr': 2e-3},
    {'params': [model.raw_k1, model.raw_c1, model.raw_k2, model.raw_c2], 'lr': 5e-4}
])
scheduler = torch.optim.lr_scheduler.StepLR(optimizer_phase2, step_size=10000, gamma=0.5)

epochs_phase2 = 30000
history = {'k1': [], 'c1': [], 'k2': [], 'c2': [], 'loss': []}

for epoch in range(epochs_phase2):
    optimizer_phase2.zero_grad()
    
    # --- 1. Data Loss (観測データへの適合) ---
    x1_pred_data_A, _ = model.forward_A(t_data)
    x1_pred_data_B, _ = model.forward_B(t_data)
    loss_data = torch.mean((x1_pred_data_A - x1_data_A)**2) + torch.mean((x1_pred_data_B - x1_data_B)**2)
    
    # --- 2. IC Loss (初期条件: x=0, v=0) ---
    _, x2_ic_A = model.forward_A(t_ic)
    _, x2_ic_B = model.forward_B(t_ic)
    dx2_dt_ic_A = torch.autograd.grad(x2_ic_A, t_ic, torch.ones_like(x2_ic_A), create_graph=True)[0]
    dx2_dt_ic_B = torch.autograd.grad(x2_ic_B, t_ic, torch.ones_like(x2_ic_B), create_graph=True)[0]
    
    loss_ic = torch.mean(x2_ic_A**2) + torch.mean(dx2_dt_ic_A**2) + \
              torch.mean(x2_ic_B**2) + torch.mean(dx2_dt_ic_B**2)
    
    # --- 3. Physics Loss (方程式の残差) ---
    def compute_physics_loss(forward_fn, omega):
        x1_p, x2_p = forward_fn(t_colloc)
        dx1_dt = torch.autograd.grad(x1_p, t_colloc, torch.ones_like(x1_p), create_graph=True)[0]
        d2x1_dt2 = torch.autograd.grad(dx1_dt, t_colloc, torch.ones_like(dx1_dt), create_graph=True)[0]
        
        dx2_dt = torch.autograd.grad(x2_p, t_colloc, torch.ones_like(x2_p), create_graph=True)[0]
        d2x2_dt2 = torch.autograd.grad(dx2_dt, t_colloc, torch.ones_like(dx2_dt), create_graph=True)[0]
        
        f_vib = A_vib * torch.sin(omega * t_colloc)
        
        res_global = m1_t * d2x1_dt2 + m2_t * d2x2_dt2 + model.c1 * dx1_dt + model.k1 * x1_p - f_vib
        res_local  = m2_t * d2x2_dt2 + model.c2 * (dx2_dt - dx1_dt) + model.k2 * (x2_p - x1_p)
        
        return torch.mean(res_global**2) + torch.mean(res_local**2)

    loss_phys_A = compute_physics_loss(model.forward_A, omega_A)
    loss_phys_B = compute_physics_loss(model.forward_B, omega_B)
    loss_physics = loss_phys_A + loss_phys_B
    
    # 合算 (データ損失の比重を重く保つのがコツ)
    loss = 10.0 * loss_data + loss_ic + 1e-2 * loss_physics
    loss.backward()
    
    optimizer_phase2.step()
    scheduler.step()
    
    history['k1'].append(model.k1.item())
    history['c1'].append(model.c1.item())
    history['k2'].append(model.k2.item())
    history['c2'].append(model.c2.item())
    history['loss'].append(loss.item())
    
    if epoch % 2000 == 0:
        print(f'Epoch:{epoch:5d} | Total Loss:{loss.item():.5f} | '
              f'k1:{model.k1.item():.2f}, c1:{model.c1.item():.3f} | '
              f'k2:{model.k2.item():.2f}, c2:{model.c2.item():.3f}')

# ==========================================
# 4. 結果の可視化
# ==========================================
fig, axes = plt.subplots(3, 2, figsize=(14, 12))

# --- 条件Aのプロット ---
x1_pred_A, x2_pred_A = model.forward_A(t_data)
axes[0, 0].scatter(t_eval, x1_data_A.numpy(), label='Obs $x_1$ (A)', color='gray', s=10)
axes[0, 0].plot(t_eval, x1_pred_A.detach().numpy(), label='PINN $x_1$ (A)', color='red')
axes[0, 0].set_title('Condition A: $x_1$')
axes[0, 0].legend()

axes[0, 1].plot(t_eval, x2_data_A, label='True $x_2$ (A)', color='gray', linestyle='--')
axes[0, 1].plot(t_eval, x2_pred_A.detach().numpy(), label='PINN $x_2$ (A)', color='blue')
axes[0, 1].set_title('Condition A: $x_2$ (Hidden)')
axes[0, 1].legend()

# --- 条件Bのプロット ---
x1_pred_B, x2_pred_B = model.forward_B(t_data)
axes[1, 0].scatter(t_eval, x1_data_B.numpy(), label='Obs $x_1$ (B)', color='gray', s=10)
axes[1, 0].plot(t_eval, x1_pred_B.detach().numpy(), label='PINN $x_1$ (B)', color='red')
axes[1, 0].set_title('Condition B: $x_1$')
axes[1, 0].legend()

axes[1, 1].plot(t_eval, x2_data_B, label='True $x_2$ (B)', color='gray', linestyle='--')
axes[1, 1].plot(t_eval, x2_pred_B.detach().numpy(), label='PINN $x_2$ (B)', color='blue')
axes[1, 1].set_title('Condition B: $x_2$ (Hidden)')
axes[1, 1].legend()

# --- パラメータ推移 ---
axes[2, 0].plot(history['k1'], label='Predicted $k_1$', color='green')
axes[2, 0].axhline(k1_true, color='red', linestyle='--', label='True $k_1$')
axes[2, 0].plot(history['k2'], label='Predicted $k_2$', color='purple')
axes[2, 0].axhline(k2_true, color='orange', linestyle='--', label='True $k_2$')
axes[2, 0].set_title('Convergence of Stiffness ($k$)')
axes[2, 0].legend()

axes[2, 1].plot(history['c1'], label='Predicted $c_1$', color='green')
axes[2, 1].axhline(c1_true, color='red', linestyle='--', label='True $c_1$')
axes[2, 1].plot(history['c2'], label='Predicted $c_2$', color='purple')
axes[2, 1].axhline(c2_true, color='orange', linestyle='--', label='True $c_2$')
axes[2, 1].set_title('Convergence of Damping ($c$)')
axes[2, 1].legend()

plt.tight_layout()
plt.show()

print(f"\n[Final Results]")
print(f"k1: True {k1_true:.2f} | Est {model.k1.item():.2f} (Err: {abs(k1_true-model.k1.item())/k1_true*100:.1f}%)")
print(f"k2: True {k2_true:.2f} | Est {model.k2.item():.2f} (Err: {abs(k2_true-model.k2.item())/k2_true*100:.1f}%)")
print(f"c1: True {c1_true:.2f} | Est {model.c1.item():.2f} (Err: {abs(c1_true-model.c1.item())/c1_true*100:.1f}%)")
print(f"c2: True {c2_true:.2f} | Est {model.c2.item():.2f} (Err: {abs(c2_true-model.c2.item())/c2_true*100:.1f}%)")