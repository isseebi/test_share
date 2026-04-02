import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. 教師データ生成（シミュレーションによる「実験データ」の作成）
# ==========================================
# AIには隠しておく「真の物理特性」
k_true, c_true = 2000.0, 5.0  

def generate_synthetic_data():
    # 物理定数の設定（3つの重りがつながったシステム）
    m1, m2, m3 = 1.0, 1.0, 5.0
    k3, c3 = 20000.0, 20.0
    dt_sim = 0.001 
    t_sim = np.arange(0, 4.0, dt_sim)
    
    # 入力信号：0.5秒から1.5秒の間だけ力を加える（ガクンと動かす）
    u_sim = np.zeros(len(t_sim))
    u_sim[int(0.5 / dt_sim):int(1.0 / dt_sim)] = 10.0
    u_sim[int(1.0 / dt_sim):int(1.5 / dt_sim)] = -10.0
    
    # 初期状態（位置・速度）
    xr, vr, xt, vt, xb, vb = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    xr_log, xt_log, xb_log = [], [], []

    # オイラー法による数値シミュレーション（物理法則に従って時間を進める）
    for i in range(len(t_sim)):
        # 相対変位と相対速度の計算
        dx = (xb + xr) - xt; dv = (vb + vr) - vt
        # バネの復元力とダンパーの抵抗力（これがAIに当てさせたい部分）
        F_spring = k_true * dx + c_true * dv
        
        # 各質点にかかる加速度を計算 (F = ma => a = F/m)
        ar = (u_sim[i] - F_spring) / m1
        at = F_spring / m2
        ab = (-F_spring - k3 * xb - c3 * vb) / m3
        
        # 速度と位置を更新
        vr += ar * dt_sim; xr += vr * dt_sim
        vt += at * dt_sim; xt += vt * dt_sim
        vb += ab * dt_sim; xb += vb * dt_sim
        
        xr_log.append(xr); xt_log.append(xt); xb_log.append(xb)
        
    # 全データから10個飛ばしでサンプリング（粗い観測データを模倣）
    t_out = t_sim[::10]
    x_out = np.stack([xr_log[::10], xt_log[::10], xb_log[::10]], axis=1)
    return torch.tensor(t_out, dtype=torch.float32).view(-1, 1), torch.tensor(x_out, dtype=torch.float32)

# 実験データの準備
t_data, x_data = generate_synthetic_data()

# ==========================================
# 2. PINN（物理情報ニューラルネットワーク）の定義
# ==========================================
class ThreeMassPINN(nn.Module):
    def __init__(self):
        super().__init__()
        # 4層の全結合ネットワーク：時間を入力して3つの位置(x1, x2, x3)を出力する
        self.net = nn.Sequential(
            nn.Linear(1, 128), nn.GELU(), 
            nn.Linear(128, 128), nn.GELU(), 
            nn.Linear(128, 128), nn.GELU(), 
            nn.Linear(128, 3)
        )
        # 推定したい物理パラメータ（勾配計算の対象にするため nn.Parameter で定義）
        # 初期値は適当な値（k=800, c=15など）からスタート
        self.k_raw = nn.Parameter(torch.tensor([0.8]))
        self.c_raw = nn.Parameter(torch.tensor([1.5]))

    def forward(self, t): 
        # ニューラルネットワークの出力。少し小さめの値になるようスケーリング
        return self.net(t) * 0.1 

    # 推定値を分かりやすいスケールに変換して取得するプロパティ
    @property
    def k_est(self): return torch.abs(self.k_raw) * 1000.0
    @property
    def c_est(self): return torch.abs(self.c_raw) * 10.0

model = ThreeMassPINN()
k_init, c_init = model.k_est.item(), model.c_est.item()

# ==========================================
# 3. 学習プロセス
# ==========================================

# --- Step 0: 事前学習 (Pre-training) ---
# 物理は無視して、とりあえずニューラルネットを実験データの波形に近づける
print("--- Step 0: Pre-training (波形のフィッティング) ---")
optimizer_pre = optim.Adam(model.net.parameters(), lr=1e-3)
for epoch in range(2000):
    optimizer_pre.zero_grad()
    x_pred = model(t_data)
    loss_data = torch.mean((x_pred - x_data)**2) # 予測と実測の差
    loss_data.backward()
    optimizer_pre.step()

# --- Step 1: Adamによる同時学習 ---
# ネットワークの重みと、物理パラメータ(k, c)を同時に更新していく
print("--- Step 1: Adam (データと物理法則の両立) ---")
optimizer_adam = optim.Adam([
    {'params': model.net.parameters(), 'lr': 2e-4}, 
    {'params': [model.k_raw], 'lr': 1e-3},
    {'params': [model.c_raw], 'lr': 1e-3}
])
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer_adam, T_max=5000, eta_min=1e-5)

for epoch in range(5000):
    optimizer_adam.zero_grad()
    
    # 1. データロス：観測されている時刻での予測精度
    x_pred_data = model(t_data)
    loss_data = torch.mean((x_pred_data - x_data)**2)
    
    # 2. 物理ロス：任意の時刻(t_ph)において運動方程式が成立しているか？
    t_ph = torch.linspace(0, 4.0, 4000).view(-1, 1).requires_grad_(True)
    x_pred_ph = model(t_ph)
    xr, xt, xb = x_pred_ph[:, 0:1], x_pred_ph[:, 1:2], x_pred_ph[:, 2:3]
    
    # 自動微分を使って、位置(x)から速度(v)と加速度(a)を算出
    vt = torch.autograd.grad(xt, t_ph, torch.ones_like(xt), create_graph=True)[0]
    at = torch.autograd.grad(vt, t_ph, torch.ones_like(vt), create_graph=True)[0]
    vr = torch.autograd.grad(xr, t_ph, torch.ones_like(xr), create_graph=True)[0]
    vb = torch.autograd.grad(xb, t_ph, torch.ones_like(xb), create_graph=True)[0]
    
    # 推定中の k, c を使って物理方程式の「矛盾」を計算
    f_spring = model.k_est * (xb + xr - xt) + model.c_est * (vb + vr - vt)
    # 本来なら at (加速度) = f_spring (力/質量) になるはず。そのズレを最小化する。
    loss_physics = torch.mean((1.0 * at - f_spring)**2)
    
    # 合計損失：データへの適合を重視しつつ、物理法則も守らせる
    total_loss = 1e5 * loss_data + 1e-4 * loss_physics
    total_loss.backward()
    optimizer_adam.step()
    scheduler.step()

# --- Step 2: L-BFGSによる精密仕上げ ---
# 非常に収束性の高いアルゴリズムで、パラメータを真値に追い込む
print("--- Step 2: L-BFGS (最終調整) ---")
optimizer_lbfgs = optim.LBFGS(model.parameters(), max_iter=2000, line_search_fn="strong_wolfe")

def closure():
    optimizer_lbfgs.zero_grad()
    x_pred_data = model(t_data)
    loss_data = torch.mean((x_pred_data - x_data)**2)
    t_ph = torch.linspace(0, 4.0, 4000).view(-1, 1).requires_grad_(True)
    x_pred_ph = model(t_ph)
    xr, xt, xb = x_pred_ph[:, 0:1], x_pred_ph[:, 1:2], x_pred_ph[:, 2:3]
    vt = torch.autograd.grad(xt, t_ph, torch.ones_like(xt), create_graph=True)[0]
    at = torch.autograd.grad(vt, t_ph, torch.ones_like(vt), create_graph=True)[0]
    vr = torch.autograd.grad(xr, t_ph, torch.ones_like(xr), create_graph=True)[0]
    vb = torch.autograd.grad(xb, t_ph, torch.ones_like(xb), create_graph=True)[0]
    f_spring = model.k_est * (xb + xr - xt) + model.c_est * (vb + vr - vt)
    loss_physics = torch.mean((1.0 * at - f_spring)**2)
    total_loss = 1e5 * loss_data + 1e-4 * loss_physics
    total_loss.backward()
    return total_loss

optimizer_lbfgs.step(closure)
k_final, c_final = model.k_est.item(), model.c_est.item()

# ==========================================
# 4. 推定されたパラメータを使って制御を試す（インプット整形）
# ==========================================


def generate_ff_input(k_val, c_val, t_sim, dt):
    # 推定したk, cからシステムの固有振動数と減衰比を計算
    m_eq = 0.5 
    omega_n = np.sqrt(k_val / m_eq)
    zeta = c_val / (2 * np.sqrt(k_val * m_eq))
    
    # ZVシェイパー：1回目の入力と、半周期遅らせた2回目の入力を合成して振動を消す
    dt_delay = np.pi / (omega_n * np.sqrt(1 - zeta**2)) # 半周期の計算
    K = np.exp(-zeta * np.pi / np.sqrt(1 - zeta**2))    # 減衰を考慮した振幅比
    A1, A2 = 1.0 / (1.0 + K), K / (1.0 + K)
    
    u_base = np.zeros_like(t_sim)
    u_base[(t_sim >= 0.5) & (t_sim < 1.0)] = 10.0
    u_base[(t_sim >= 1.0) & (t_sim < 1.5)] = -10.0
    
    u_shaped = np.zeros_like(t_sim)
    delay_idx = int(dt_delay / dt)
    for i in range(len(t_sim)):
        val = A1 * u_base[i]
        if i >= delay_idx: val += A2 * u_base[i - delay_idx]
        u_shaped[i] = val
    return u_shaped

def run_control_simulation(k_est, c_est):
    # 推定されたパラメータで設計した入力を使って、実際のシステムを動かしてみる
    # (内部では k_true, c_true の本物の物理法則が働いている)
    m1, m2, m3 = 1.0, 1.0, 5.0
    k3, c3 = 20000.0, 20.0
    dt_sim = 0.001 
    t_sim = np.arange(0, 4.0, dt_sim)
    u_ff = generate_ff_input(k_est, c_est, t_sim, dt_sim)
    
    xr, vr, xt, vt, xb, vb = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    xt_log = []
    for i in range(len(t_sim)):
        dx = (xb + xr) - xt; dv = (vb + vr) - vt
        F_spring = k_true * dx + c_true * dv 
        ar = (u_ff[i] - F_spring) / m1; at = F_spring / m2; ab = (-F_spring - k3 * xb - c3 * vb) / m3
        vr += ar * dt_sim; xr += vr * dt_sim; vt += at * dt_sim; xt += vt * dt_sim; vb += ab * dt_sim; xb += vb * dt_sim
        xt_log.append(xt)
    return t_sim, np.array(xt_log)

# 学習前、学習後、理想（真値を知っている場合）の3パターンをシミュレーション
t_sim, xt_init_sim = run_control_simulation(k_init, c_init)
t_sim, xt_final_sim = run_control_simulation(k_final, c_final)
t_sim, xt_true_sim = run_control_simulation(k_true, c_true)

# ==========================================
# 5. 結果のグラフ表示
# ==========================================


# [Image of 3-mass spring-damper system]


plt.figure(figsize=(18, 5))

# グラフ1：全体の動き（AIの学習によって制御が改善したか）
plt.subplot(1, 3, 1)
plt.plot(t_sim, xt_true_sim, 'r--', label='Ideal (True Param)')
plt.plot(t_sim, xt_init_sim, 'g-', label='Before Training')
plt.plot(t_sim, xt_final_sim, 'b-', label='After Training')
plt.title("Overall Trajectory")
plt.legend()

# グラフ2：残された振動（ここがピタッと止まればパラメータ推定が成功している証拠）
plt.subplot(1, 3, 2)
mask = (t_sim >= 1.4) & (t_sim <= 3.5)
plt.plot(t_sim[mask], xt_true_sim[mask], 'r--')
plt.plot(t_sim[mask], xt_init_sim[mask], 'g-')
plt.plot(t_sim[mask], xt_final_sim[mask], 'b-')
plt.title("Zoom on Residual Vibration")

# グラフ3：AIが当てたパラメータの正解度
plt.subplot(1, 3, 3)
params = ['Stiffness (k)', 'Damping (c)']
plt.bar(np.arange(2)-0.2, [k_true, c_true], 0.4, label='True', color='gray')
plt.bar(np.arange(2)+0.2, [k_final, c_final], 0.4, label='AI Estimated', color='blue')
plt.xticks(np.arange(2), params)
plt.title("Parameter Accuracy")
plt.legend()

plt.tight_layout()
plt.show()