import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from vibe_gen import VibrationSimulator
import time

# {'t': t, 'xn': xn, 'xs': xs, 'u': u, 'deflection': xn - xs}
# {'mode': 'low_freq_vibration', 'name': 'Low-Frequency Vibration Mode', 'M': 1.0, 'm': 0.1, 'k': 1000.0, 'c': 0.6, 'dt': 0.001, 'time_max': 1.0, 'wn': 100.0, 'zeta': 0.03, 'Kp': 150.0, 'Kd': 10.0, 'dist_type': 'none', 'dist_amp': 0.0, 'use_shaping': True, 'shaper_wn': 30.0, 'shaper_zeta': 0.03}

###### シミューレーターの実行開始
# sim_cases = [
#     "low_freq_vibration",
#     "high_freq_vibration",
#     "white_noise_model",
#     "pulse_wave_model",
#     "custom_equation_model"
# ]
sim_cases = 'pulse_wave_model'
img_name = f"PINN_param_{sim_cases}.png"
sim_instance = VibrationSimulator(mode=sim_cases, with_compensation=True)

sim_data = sim_instance.run()
param = sim_instance.get_parameters()
print(param)
# sim_instance.plot()

plt.plot(sim_data['t'],sim_data['deflection'])
plt.show()

t = sim_data['t']
xs_shape = sim_data['xs']
xn_shape = sim_data['xn']

wn_real = param['wn']
zeta_real = param['zeta']
shaper_wn_setting = param['shaper_wn']
shaper_zeta_setting = param['shaper_zeta']
dt_sim=param['dt']
###### シミューレーターの実行終了

# たわみ量 (d = xn - xs) の計算
deflection_shape = xn_shape - xs_shape

# =========================================================
# 2. PINN学習データの準備
# =========================================================
skip = 10
t_train = torch.tensor(t[::skip], dtype=torch.float32).view(-1, 1).requires_grad_(True)
d_train = torch.tensor(deflection_shape[::skip], dtype=torch.float32).view(-1, 1) # たわみ量を教師データに
xn_train = torch.tensor(xn_shape[::skip], dtype=torch.float32).view(-1, 1)
print(len(t_train), len(d_train), len(xn_train))
print("サンプルデータ数")
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

epochs = 300000
history = {"wn": [], "zeta": [], "loss": []}

print(f"Target: wn={wn_real}, zeta={zeta_real}")
start_time = time.time()

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

wn_est = pinn.wn.item()
zeta_est = pinn.zeta.item()

end_time = time.time()
# 経過時間を計算
elapsed_time = end_time - start_time
print(f"かかった時間: {elapsed_time:.2f} 秒")
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
plt.savefig(img_name, dpi=300)
plt.show()

print(wn_est, zeta_est)
# #####学習結果で振動確認
sim2 = VibrationSimulator(
        mode=sim_cases, 
        with_compensation=True, 
        shaper_wn=wn_est,    
        shaper_zeta=zeta_est  
    )
sim2.run()
sim2.plot()