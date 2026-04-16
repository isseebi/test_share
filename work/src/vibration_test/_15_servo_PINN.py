import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# 乱数シードの固定
torch.manual_seed(42)
np.random.seed(42)

# ==========================================
# 1. 物理法則を満たすダミーデータの生成
# ==========================================
N_points = 500
t_data = np.linspace(0, 2, N_points).reshape(-1, 1)

# 【真の物理パラメータ】
true_m = 2.0  # 質量
true_c = 0.5  # 減衰係数
true_k = 10.0 # 剛性

w_b = 2 * np.pi * 0.5
w_p = 2 * np.pi * 1.5

xb_data = 0.1 * np.sin(w_b * t_data)
dxb_data = 0.1 * w_b * np.cos(w_b * t_data)

x_rel = 0.2 * np.sin(w_p * t_data)
dx_rel = 0.2 * w_p * np.cos(w_p * t_data)
ddx_rel = -0.2 * w_p**2 * np.sin(w_p * t_data)

xp_data = xb_data + x_rel
ddxp_data = -0.1 * w_b**2 * np.sin(w_b * t_data) + ddx_rel

tau_data = true_m * ddxp_data + true_c * dx_rel + true_k * x_rel

# データのテンソル化
t_tensor = torch.tensor(t_data, dtype=torch.float32, requires_grad=True)
tau_tensor = torch.tensor(tau_data, dtype=torch.float32)
xp_tensor = torch.tensor(xp_data, dtype=torch.float32)
xb_tensor = torch.tensor(xb_data, dtype=torch.float32)

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
            nn.Linear(64, 2)
        )
        
        # 物理パラメータの初期値（意図的に1.0からスタート）
        self.m = nn.Parameter(torch.tensor([1.0]))
        self.c = nn.Parameter(torch.tensor([1.0]))
        self.k = nn.Parameter(torch.tensor([1.0]))

    def forward(self, t):
        return self.net(t)

model = PINN()

# ==========================================
# 3. 学習の設定と履歴保存用のリスト
# ==========================================
optimizer = torch.optim.Adam([
    {'params': model.net.parameters(), 'lr': 1e-3},
    {'params': [model.m, model.c, model.k], 'lr': 1e-2}
])

epochs = 8000

# ★推移を記録するためのリスト
history_loss = []
history_m = []
history_c = []
history_k = []
history_epochs = []

# ==========================================
# 4. 学習ループ
# ==========================================
print("--- 学習開始 ---")
for epoch in range(epochs):
    optimizer.zero_grad()
    
    pred = model(t_tensor)
    xp_pred = pred[:, 0:1]
    xb_pred = pred[:, 1:2]
    
    dxp_dt = torch.autograd.grad(xp_pred, t_tensor, grad_outputs=torch.ones_like(xp_pred), create_graph=True)[0]
    ddxp_dt2 = torch.autograd.grad(dxp_dt, t_tensor, grad_outputs=torch.ones_like(dxp_dt), create_graph=True)[0]
    
    dxb_dt = torch.autograd.grad(xb_pred, t_tensor, grad_outputs=torch.ones_like(xb_pred), create_graph=True)[0]

    data_loss = torch.mean((xp_pred - xp_tensor)**2) + torch.mean((xb_pred - xb_tensor)**2)
    
    physics_residual = model.m * ddxp_dt2 + model.c * (dxp_dt - dxb_dt) + model.k * (xp_pred - xb_pred) - tau_tensor
    physics_loss = torch.mean(physics_residual**2)
    
    total_loss = data_loss + 1e-3 * physics_loss 
    
    total_loss.backward()
    optimizer.step()
    
    # ★10エポックごとに履歴を記録
    if epoch % 10 == 0:
        history_epochs.append(epoch)
        history_loss.append(total_loss.item())
        history_m.append(model.m.item())
        history_c.append(model.c.item())
        history_k.append(model.k.item())
    
    if epoch % 1000 == 0:
        print(f"Epoch {epoch:4d} | Total Loss: {total_loss.item():.5f} | 推定 -> m: {model.m.item():.3f}, c: {model.c.item():.3f}, k: {model.k.item():.3f}")

print("--- 学習完了 ---\n")

# ==========================================
# 5. 結果の確認とグラフ描画
# ==========================================
print(f"最終推定結果: m = {model.m.item():.3f}, c = {model.c.item():.3f}, k = {model.k.item():.3f}")

# 波形の予測結果を取得
model.eval()
with torch.no_grad():
    pred_eval = model(t_tensor)
    xp_eval = pred_eval[:, 0].numpy()

# グラフの描画設定 (3つのプロットウィンドウを表示します)
fig = plt.figure(figsize=(15, 10))

# --- ① 波形の予測フィット結果 ---
plt.subplot(2, 2, 1)
plt.plot(t_data, xp_data, label="Real Data ($x_p$)", color='gray', linestyle='dashed', linewidth=2)
plt.plot(t_data, xp_eval, label="PINN Prediction ($\hat{x}_p$)", color='red', alpha=0.7)
plt.title("Target Displacement: Real vs PINN")
plt.xlabel("Time [s]")
plt.ylabel("Displacement [m]")
plt.legend()
plt.grid(True)

# --- ② Lossの推移（対数グラフ） ---
plt.subplot(2, 2, 2)
plt.plot(history_epochs, history_loss, color='black')
plt.yscale('log')
plt.title("Total Loss History")
plt.xlabel("Epoch")
plt.ylabel("Loss (Log Scale)")
plt.grid(True, which="both", ls="--")

# --- ③ 各物理パラメータの学習推移 ---
plt.subplot(2, 1, 2)
plt.plot(history_epochs, history_m, label='Estimated m', color='blue')
plt.axhline(y=true_m, color='blue', linestyle='dotted', label='True m (2.0)')

plt.plot(history_epochs, history_c, label='Estimated c', color='green')
plt.axhline(y=true_c, color='green', linestyle='dotted', label='True c (0.5)')

plt.plot(history_epochs, history_k, label='Estimated k', color='orange')
plt.axhline(y=true_k, color='orange', linestyle='dotted', label='True k (10.0)')

plt.title("Physical Parameters Convergence History")
plt.xlabel("Epoch")
plt.ylabel("Parameter Value")
plt.legend(loc='center right')
plt.grid(True)

plt.tight_layout()
plt.show()