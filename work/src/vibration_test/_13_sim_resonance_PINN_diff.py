import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

##変更点
# ノズルの振動を原点からの変位ではなく、スライダとの相対変位（たわみ量）を予測するように変更
# 物理損失の式も、たわみ量の微分方程に基づく形に修正

# =========================================================
# 1. シミュレーション部分 (データ生成)
# =========================================================
def simulate_slider_nozzle_control(wn, zeta, Kp, Kd, target_pos, shaper_wn, shaper_zeta, time_max=3.0, dt=0.001):
    t = np.arange(0, time_max, dt)
    n_steps = len(t)
    M, m = 1.0, 0.1
    k = m * wn**2
    c = 2 * m * zeta * wn
    
    use_shaping = shaper_wn > 0
    if use_shaping:
        wd = shaper_wn * np.sqrt(1 - shaper_zeta**2)
        K_val = np.exp(-shaper_zeta * np.pi / np.sqrt(1 - shaper_zeta**2))
        A1, A2, t2 = 1 / (1 + K_val), K_val / (1 + K_val), np.pi / wd
    else:
        A1, A2, t2 = 1.0, 0.0, 0.0

    xs, vs, xn, vn = np.zeros(n_steps), np.zeros(n_steps), np.zeros(n_steps), np.zeros(n_steps)
    
    for i in range(1, n_steps):
        curr_time = i * dt
        curr_target = target_pos * (A1 + A2) if curr_time >= t2 else target_pos * A1 if use_shaping else target_pos
        
        F_ctrl = Kp * (curr_target - xs[i-1]) - Kd * vs[i-1]
        F_interaction = k * (xs[i-1] - xn[i-1]) + c * (vs[i-1] - vn[i-1])
        
        a_s = (F_ctrl - F_interaction) / M
        a_n = F_interaction / m
        
        vs[i] = vs[i-1] + a_s * dt
        xs[i] = xs[i-1] + vs[i] * dt
        vn[i] = vn[i-1] + a_n * dt
        xn[i] = xn[i-1] + vn[i] * dt
        
    return t, xs, xn

# パラメータ設定
wn_real, zeta_real = 30.0, 0.05
shaper_wn_setting, shaper_zeta_setting = 20.0, 0.1 # 意図的にズラした初期想定

# データ生成（シェーピングが不完全で振動しているケース）
t, xs_shape, xn_shape = simulate_slider_nozzle_control(
    wn_real, zeta_real, 80.0, 5.0, 1.0, 
    shaper_wn=shaper_wn_setting, shaper_zeta=shaper_zeta_setting)

# たわみ量 (d = xn - xs) の計算
deflection_shape = xn_shape - xs_shape

# =========================================================
# 2. PINN学習データの準備
# =========================================================
skip = 10
t_train = torch.tensor(t[::skip], dtype=torch.float32).view(-1, 1).requires_grad_(True)
d_train = torch.tensor(deflection_shape[::skip], dtype=torch.float32).view(-1, 1) # たわみ量を教師データに
xn_train = torch.tensor(xn_shape[::skip], dtype=torch.float32).view(-1, 1)

# =========================================================
# 3. PINNモデルの定義
# =========================================================
class ParameterEstimationPINN(nn.Module):
    def __init__(self, init_wn, init_zeta):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 2)  # 出力: [d_pred, xn_pred]
        )
        self.wn = nn.Parameter(torch.tensor([init_wn], dtype=torch.float32))
        self.zeta = nn.Parameter(torch.tensor([init_zeta], dtype=torch.float32))

    def forward(self, t):
        return self.net(t)

def compute_grad(y, x):
    return torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y), 
                               create_graph=True, retain_graph=True)[0]

# =========================================================
# 4. 学習ループ
# =========================================================
pinn = ParameterEstimationPINN(init_wn=shaper_wn_setting, init_zeta=shaper_zeta_setting)
optimizer = optim.Adam(pinn.parameters(), lr=1e-3)

epochs = 20000
history = {"wn": [], "zeta": [], "loss": []}

print(f"Target: wn={wn_real}, zeta={zeta_real}")

for epoch in range(epochs):
    optimizer.zero_grad()
    
    preds = pinn(t_train)
    d_pred = preds[:, 0:1]   # たわみの予測
    xn_pred = preds[:, 1:2]  # ノズル絶対位置の予測
    
    # --- (A) Data Loss ---
    loss_data = torch.mean((d_pred - d_train)**2) + torch.mean((xn_pred - xn_train)**2)
    
    # --- (B) Physics Loss ---
    # 時間微分
    dd_dt = compute_grad(d_pred, t_train)   # たわみの速度
    dxn_dt = compute_grad(xn_pred, t_train) # ノズルの速度
    ddxn_dt2 = compute_grad(dxn_dt, t_train) # ノズルの加速度
    
    # 物理法則: xn'' + (wn^2)*d + (2*zeta*wn)*d' = 0
    # 数値的安定のため wn^2 で割った形式
    wn_safe = pinn.wn + 1e-6
    residual = (ddxn_dt2 / (wn_safe**2)) + d_pred + (2 * pinn.zeta / wn_safe) * dd_dt
    loss_physics = torch.mean(residual**2)
    
    # 合計損失
    loss = loss_data + loss_physics
    loss.backward()
    optimizer.step()
    
    history["wn"].append(pinn.wn.item())
    history["zeta"].append(pinn.zeta.item())
    history["loss"].append(loss.item())
    
    if (epoch + 1) % 2000 == 0:
        print(f"Epoch {epoch+1:5d} | Loss: {loss.item():.2e} | wn: {pinn.wn.item():.2f} | zeta: {pinn.zeta.item():.4f}")

# =========================================================
# 5. 推定結果の可視化
# =========================================================
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history["wn"], label='Estimated wn', color='blue')
plt.axhline(wn_real, color='black', linestyle='--', label='True wn')
plt.title('Convergence of wn')
plt.legend(); plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history["zeta"], label='Estimated zeta', color='orange')
plt.axhline(zeta_real, color='black', linestyle='--', label='True zeta')
plt.title('Convergence of zeta')
plt.legend(); plt.grid(True)
plt.show()