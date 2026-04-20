import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from vibe_gen import VibrationSimulator
import time

# =========================================================
# 1. シミュレーターの実行とデータ取得
# =========================================================
sim_cases = 'pulse_wave_model'
img_name = f"PINN_param_{sim_cases}.png"
sim_instance = VibrationSimulator(mode=sim_cases, with_compensation=True)

sim_data = sim_instance.run()
param = sim_instance.get_parameters()
print("--- シミュレータの真のパラメータ ---")
print(param)

t = sim_data['t']
xs_shape = sim_data['xs']
xn_shape = sim_data['xn']

wn_real = param['wn']
zeta_real = param['zeta']
shaper_wn_setting = param['shaper_wn']
shaper_zeta_setting = param['shaper_zeta']
dt_sim = param['dt']

deflection_shape = xn_shape - xs_shape

# データの事前平滑化 (移動平均)
def moving_average(x, w=5):
    return np.convolve(x, np.ones(w), 'valid') / w

window = 5
t_smooth = t[window-1:]
d_smooth = moving_average(deflection_shape, w=window)
xn_smooth = moving_average(xn_shape, w=window)

# =========================================================
# 2. PINN学習データの準備
# =========================================================
skip = 10
t_train = torch.tensor(t_smooth[::skip], dtype=torch.float32).view(-1, 1).requires_grad_(True)
d_train = torch.tensor(d_smooth[::skip], dtype=torch.float32).view(-1, 1)
xn_train = torch.tensor(xn_smooth[::skip], dtype=torch.float32).view(-1, 1)

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
            nn.Linear(64, 2)  
        )
        self.wn_scale = 100.0
        self.zeta_scale = 0.01
        
        init_wn_raw = np.log(np.exp(init_wn / self.wn_scale) - 1.0)
        init_zeta_raw = np.log(np.exp(init_zeta / self.zeta_scale) - 1.0)
        self.wn_raw = nn.Parameter(torch.tensor([init_wn_raw], dtype=torch.float32))
        self.zeta_raw = nn.Parameter(torch.tensor([init_zeta_raw], dtype=torch.float32))

    @property
    def wn(self):
        return torch.nn.functional.softplus(self.wn_raw) * self.wn_scale

    @property
    def zeta(self):
        return torch.nn.functional.softplus(self.zeta_raw) * self.zeta_scale

    def forward(self, t):
        return self.net(t)

def compute_grad(y, x):
    return torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y), 
                               create_graph=True, retain_graph=True)[0]

# =========================================================
# 4. 学習ループ (Adamのみ + 収束改善版)
# =========================================================
pinn = ParameterEstimationPINN(init_wn=shaper_wn_setting, init_zeta=shaper_zeta_setting)

history = {"wn": [], "zeta": [], "loss": []}
start_time = time.time()

target_lambda_phys = 0.1 
adam_epochs = 100000  

# ★改善点1: パラメータごとに学習率を分割 (zetaの学習率を他より高く設定)
optimizer_adam = optim.Adam([
    {'params': pinn.net.parameters(), 'lr': 1e-3},
    {'params': [pinn.wn_raw], 'lr': 1e-3},
    {'params': [pinn.zeta_raw], 'lr': 5e-3}  # zetaの勾配は小さくなりやすいため5倍に設定
])

# ★改善点2: コサインアニーリングで学習後半に学習率を滑らかに下げる
# これにより、L-BFGSのような終盤の微小な収束（Fine-tuning）を実現します
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer_adam, T_max=adam_epochs, eta_min=1e-5)

criterion_data = nn.HuberLoss(delta=0.1)

print(f"\n--- Adam Optimizer Started ({adam_epochs} epochs) ---")
for epoch in range(adam_epochs):
    optimizer_adam.zero_grad()
    preds = pinn(t_train)
    d_pred = preds[:, 0:1]   
    xn_pred = preds[:, 1:2]  
    
    loss_data = criterion_data(d_pred, d_train) + criterion_data(xn_pred, xn_train)
    
    # 後半にかけて物理ロスの重みを 0 から target_lambda_phys まで滑らかに上げる
    if epoch < (adam_epochs // 2):
        lambda_phys = 0.0
    else:
        ratio = (epoch - (adam_epochs // 2)) / (adam_epochs // 2)
        lambda_phys = target_lambda_phys * ratio
    
    if lambda_phys > 0:
        dd_dt = compute_grad(d_pred, t_train)   
        dxn_dt = compute_grad(xn_pred, t_train) 
        ddxn_dt2 = compute_grad(dxn_dt, t_train) 
        
        residual = (ddxn_dt2 / (pinn.wn**2)) + d_pred + (2 * pinn.zeta / pinn.wn) * dd_dt
        loss_physics = torch.mean(residual**2)
    else:
        loss_physics = 0.0
        
    loss = loss_data + lambda_phys * loss_physics
    loss.backward()
    
    optimizer_adam.step()
    scheduler.step() # ★毎エポック学習率を更新
    
    history["wn"].append(pinn.wn.item())
    history["zeta"].append(pinn.zeta.item())
    history["loss"].append(loss.item())
    
    if (epoch + 1) % 5000 == 0:
        # 現在の zeta の学習率を取得して表示（スケジューラが効いているか確認用）
        current_zeta_lr = optimizer_adam.param_groups[2]['lr']
        print(f"Epoch {epoch+1:5d} | Loss: {loss.item():.2e} | wn: {pinn.wn.item():.2f} | zeta: {pinn.zeta.item():.4f} | zeta_LR: {current_zeta_lr:.1e}")
        
end_time = time.time()
print(f"\n学習完了！ かかった時間: {end_time - start_time:.2f} 秒")

wn_est = pinn.wn.item()
zeta_est = pinn.zeta.item()

# =========================================================
# 5. 推定結果の表示・保存と誤差率の表示
# =========================================================
error_wn = abs(wn_est - wn_real) / wn_real * 100
error_zeta = abs(zeta_est - zeta_real) / zeta_real * 100

print("\n=========================================")
print(f"【最終推定結果】")
print(f"Target (真値) : wn = {wn_real:.4f}, zeta = {zeta_real:.4f}")
print(f"Estimated(推定): wn = {wn_est:.4f}, zeta = {zeta_est:.4f}")
print(f"誤差率 (wn)   : {error_wn:.2f} %")
print(f"誤差率 (zeta) : {error_zeta:.2f} %")
print("=========================================\n")

# 学習曲線の可視化と画像の保存
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
plt.tight_layout()

plt.savefig(img_name, dpi=300)
print(f"学習プロットを保存しました: {img_name}")
plt.show()

# ##### 学習結果で振動確認
print("学習したパラメータを使用してシミュレータを実行します...")
sim2 = VibrationSimulator(
        mode=sim_cases, 
        with_compensation=True, 
        shaper_wn=wn_est,    
        shaper_zeta=zeta_est  
    )
sim2.run()
sim2.plot()