import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from vibe_gen import VibrationSimulator

# =========================================================
# 1. シミュレーターの実行とデータ取得
# =========================================================
sim_cases = 'high_freq_vibration'
img_name = f"PINN_param_{sim_cases}.png"
sim_instance = VibrationSimulator(mode=sim_cases, with_compensation=True)

sim_data = sim_instance.run()
param = sim_instance.get_parameters()
print("True Parameters:", param)

t = sim_data['t']
xs_shape = sim_data['xs']
xn_shape = sim_data['xn']

wn_real = param['wn']
zeta_real = param['zeta']
shaper_wn_setting = param['shaper_wn']
shaper_zeta_setting = param['shaper_zeta']
dt_sim = param['dt']

# たわみ量 (d = xn - xs) の計算
deflection_shape = xn_shape - xs_shape

# ステージ xs の加速度を数値微分で事前計算
dxs_dt = np.gradient(xs_shape, dt_sim)
d2xs_dt2 = np.gradient(dxs_dt, dt_sim)

# =========================================================
# 2. PINN学習データの準備
# =========================================================
skip = 10
t_train = torch.tensor(t[::skip], dtype=torch.float32).view(-1, 1).requires_grad_(True)
d_train = torch.tensor(deflection_shape[::skip], dtype=torch.float32).view(-1, 1)
d2xs_dt2_train = torch.tensor(d2xs_dt2[::skip], dtype=torch.float32).view(-1, 1)

print(f"Number of training samples: {len(t_train)}")

# =========================================================
# 3. PINNモデルの定義
# =========================================================
class ParameterEstimationPINN(nn.Module):
    def __init__(self, init_wn, init_zeta):
        super().__init__()
        # 【改善】活性化関数を SiLU (Swish) に変更。高次微分の伝播が滑らかになり収束が早まります。
        self.net = nn.Sequential(
            nn.Linear(1, 64), nn.SiLU(),
            nn.Linear(64, 64), nn.SiLU(),
            nn.Linear(64, 64), nn.SiLU(),
            nn.Linear(64, 1)
        )
        self.wn = nn.Parameter(torch.tensor([init_wn], dtype=torch.float32))
        self.zeta = nn.Parameter(torch.tensor([init_zeta], dtype=torch.float32))

    def forward(self, t):
        return self.net(t)

def compute_grad(y, x):
    return torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y), 
                               create_graph=True, retain_graph=True)[0]

# =========================================================
# 4. 学習ループ (Adam -> L-BFGS のハイブリッド)
# =========================================================
pinn = ParameterEstimationPINN(init_wn=shaper_wn_setting, init_zeta=shaper_zeta_setting)

# 【改善】エポック数を劇的に削減
adam_epochs = 100000
lbfgs_epochs = 1000 # L-BFGSは内部で複数回最適化を行うため、少ないエポック数で十分です

optimizer_adam = optim.Adam(pinn.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer_adam, 'min', patience=1000, factor=0.5)

history = {"wn": [], "zeta": [], "loss": []}
print(f"Target: wn={wn_real}, zeta={zeta_real}")

# --- Phase 1: Adamによる大域的な探索 ---
print("--- Starting Adam Optimization ---")
for epoch in range(adam_epochs):
    optimizer_adam.zero_grad()
    
    d_pred = pinn(t_train)
    loss_data = torch.mean((d_pred - d_train)**2)
    
    dd_dt = compute_grad(d_pred, t_train)
    d2d_dt2 = compute_grad(dd_dt, t_train)
    
    # 【改善】物理法則を標準形で実装 (分母に変数をおかないことで勾配を安定化)
    residual = d2d_dt2 + 2 * pinn.zeta * pinn.wn * dd_dt + (pinn.wn**2) * d_pred + d2xs_dt2_train
    loss_physics = torch.mean(residual**2)
    
    # 【改善】loss_physics のスケールが大きくなりすぎるのを防ぐための重み付け
    # wn^2 が掛かっているため、物理損失のオーダーをデータ損失のオーダーに近づけます
    lambda_phys = 1e-6 
    loss = loss_data + lambda_phys * loss_physics
    
    loss.backward()
    optimizer_adam.step()
    scheduler.step(loss)
    
    history["wn"].append(pinn.wn.item())
    history["zeta"].append(pinn.zeta.item())
    history["loss"].append(loss.item())
    
    if (epoch + 1) % 2000 == 0:
        print(f"Adam Epoch {epoch+1:5d} | Loss: {loss.item():.2e} | wn: {pinn.wn.item():.2f} | zeta: {pinn.zeta.item():.4f}")

# --- Phase 2: L-BFGSによる高精度な局所最適化 ---
print("--- Switching to L-BFGS Optimization ---")
optimizer_lbfgs = optim.LBFGS(pinn.parameters(), lr=1.0, max_iter=20, tolerance_grad=1e-7, tolerance_change=1e-9, history_size=50)

def closure():
    optimizer_lbfgs.zero_grad()
    d_pred = pinn(t_train)
    loss_data = torch.mean((d_pred - d_train)**2)
    
    dd_dt = compute_grad(d_pred, t_train)
    d2d_dt2 = compute_grad(dd_dt, t_train)
    
    residual = d2d_dt2 + 2 * pinn.zeta * pinn.wn * dd_dt + (pinn.wn**2) * d_pred + d2xs_dt2_train
    loss_physics = torch.mean(residual**2)
    
    loss = loss_data + 1e-6 * loss_physics
    loss.backward()
    
    # グラフ描画用に履歴を保存
    history["wn"].append(pinn.wn.item())
    history["zeta"].append(pinn.zeta.item())
    history["loss"].append(loss.item())
    return loss

for epoch in range(lbfgs_epochs):
    loss = optimizer_lbfgs.step(closure)
    
    if (epoch + 1) % 100 == 0:
        print(f"L-BFGS Epoch {epoch+1:4d} | Loss: {loss.item():.2e} | wn: {pinn.wn.item():.2f} | zeta: {pinn.zeta.item():.4f}")

wn_est = pinn.wn.item()
zeta_est = pinn.zeta.item()

# =========================================================
# 5. 推定結果の可視化
# =========================================================
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history["wn"], label='Estimated wn', color='blue')
plt.axhline(wn_real, color='black', linestyle='--', label='True wn')
plt.title('Convergence of wn')
plt.xlabel('Iterations (Adam + L-BFGS)')
plt.legend(); plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history["zeta"], label='Estimated zeta', color='orange')
plt.axhline(zeta_real, color='black', linestyle='--', label='True zeta')
plt.title('Convergence of zeta')
plt.xlabel('Iterations (Adam + L-BFGS)')
plt.legend(); plt.grid(True)
plt.tight_layout()
plt.savefig(img_name, dpi=300)
plt.show()

print(f"Final Estimation -> wn: {wn_est:.4f}, zeta: {zeta_est:.4f}")

# #####学習結果で振動確認
sim2 = VibrationSimulator(
        mode=sim_cases, 
        with_compensation=True, 
        shaper_wn=wn_est,    
        shaper_zeta=zeta_est  
    )
sim2.run()
sim2.plot()