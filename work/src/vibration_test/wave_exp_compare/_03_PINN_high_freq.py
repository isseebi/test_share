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
skip = 1
t_train = torch.tensor(t[::skip], dtype=torch.float32).view(-1, 1).requires_grad_(True)
d_train = torch.tensor(deflection_shape[::skip], dtype=torch.float32).view(-1, 1) # たわみ量を教師データに
xn_train = torch.tensor(xn_shape[::skip], dtype=torch.float32).view(-1, 1)
print(len(t_train), len(d_train), len(xn_train))
print("サンプルデータ数")

# =========================================================
# 3. PINNモデルの定義 (高周波対応版: Fourier Features導入)
# =========================================================
class ParameterEstimationPINN(nn.Module):
    def __init__(self, init_wn, init_zeta):
        super().__init__()
        
        # --- ネットワーク定義（前回と同じ） ---
        self.sigma = 10.0
        self.B = nn.Parameter(torch.randn(1, 32) * self.sigma, requires_grad=False)
        self.net = nn.Sequential(
            nn.Linear(64, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 2)
        )
        
        # --- 修正: スケール因子の導入 ---
        # 対象となる値のおおよそのオーダー（桁）を設定します
        self.wn_scale = 100.0  
        self.zeta_scale = 0.1
        
        # ネットワークが直接学習するパラメータは 1.0 前後になるように初期化
        self.wn_hat = nn.Parameter(torch.tensor([init_wn / self.wn_scale], dtype=torch.float32))
        self.zeta_hat = nn.Parameter(torch.tensor([init_zeta / self.zeta_scale], dtype=torch.float32))

    # propertyデコレータを使って、外から pinn.wn と呼んだらスケールが戻るようにする
    @property
    def wn(self):
        return self.wn_hat * self.wn_scale
        
    @property
    def zeta(self):
        return self.zeta_hat * self.zeta_scale

    def forward(self, t):
        v = 2.0 * torch.pi * t @ self.B
        t_encoded = torch.cat([torch.cos(v), torch.sin(v)], dim=-1)
        return self.net(t_encoded)
def compute_grad(y, x):
    return torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y), 
                               create_graph=True, retain_graph=True)[0]

# =========================================================
# 4. 学習ループ
# =========================================================
pinn = ParameterEstimationPINN(init_wn=shaper_wn_setting, init_zeta=shaper_zeta_setting)

# 学習率を少し上げ、スケジューラを導入
optimizer = optim.Adam(pinn.parameters(), lr=2e-3)

# エポック数は劇的に減らせる可能性があります（まずは元の半分程度で試してみてください）
epochs = 140000 
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

history = {"wn": [], "zeta": [], "loss": []}

print(f"Target: wn={wn_real}, zeta={zeta_real}")
start_time = time.time()

# Physics Loss と Data Loss のバランスを調整する係数
# 高周波の場合、Physics Lossが大きくなりすぎるのを防ぐために小さく設定することがあります
lambda_physics = 1.0 

for epoch in range(epochs):
    optimizer.zero_grad()
    
    preds = pinn(t_train)
    d_pred = preds[:, 0:1]   # たわみの予測
    xn_pred = preds[:, 1:2]  # ノズル絶対位置の予測
    
    # --- (A) Data Loss ---
    loss_data = torch.mean((d_pred - d_train)**2) + torch.mean((xn_pred - xn_train)**2)
    
    # --- (B) Physics Loss ---
    # 時間微分
    dd_dt = compute_grad(d_pred, t_train)
    dxn_dt = compute_grad(xn_pred, t_train)
    ddxn_dt2 = compute_grad(dxn_dt, t_train)
    
    # 物理法則: xn'' + (wn^2)*d + (2*zeta*wn)*d' = 0
    wn_safe = pinn.wn + 1e-6
    residual = (ddxn_dt2 / (wn_safe**2)) + d_pred + (2 * pinn.zeta / wn_safe) * dd_dt
    loss_physics = torch.mean(residual**2)
    
    # 合計損失
    loss = loss_data + lambda_physics * loss_physics
    loss.backward()
    
    # 勾配の爆発を防ぐためのクリッピング（高周波対応の保険）
    torch.nn.utils.clip_grad_norm_(pinn.parameters(), max_norm=1.0)
    
    optimizer.step()
    scheduler.step() # スケジューラの更新
    
    history["wn"].append(pinn.wn.item())
    history["zeta"].append(pinn.zeta.item())
    history["loss"].append(loss.item())
    
    if (epoch + 1) % 2000 == 0:
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch+1:5d} | Loss: {loss.item():.2e} | wn: {pinn.wn.item():.2f} | zeta: {pinn.zeta.item():.4f} | lr: {current_lr:.1e}")

wn_est = pinn.wn.item()
zeta_est = pinn.zeta.item()

end_time = time.time()
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