import numpy as np
import matplotlib.pyplot as plt

def simulate_slider_nozzle_control(wn, zeta, Kp, Kd, target_pos, shaper_wn, shaper_zeta, time_max=3.0, dt=0.001):
    """
    スライダのPD制御と、それに伴うノズルの慣性振動シミュレーション
    (ZVインプットシェーピング機能付き)
    """
    t = np.arange(0, time_max, dt)
    n_steps = len(t)
    
    # 物理パラメータ
    M = 1.0  # スライダの質量
    m = 0.1  # ノズル（先端負荷）の質量
    
    # ノズルとスライダを繋ぐ剛性と減衰
    k = m * wn**2
    c = 2 * m * zeta * wn

    # ----------------------------------------------------
    # ZV (Zero Vibration) インプットシェーパーの設計
    # ----------------------------------------------------
    use_shaping = shaper_wn > 0
    if use_shaping:
        # 減衰固有振動数 wd
        wd = shaper_wn * np.sqrt(1 - shaper_zeta**2)
        # インパルスの振幅比を決定する係数 K
        K_val = np.exp(-shaper_zeta * np.pi / np.sqrt(1 - shaper_zeta**2))
        
        A1 = 1 / (1 + K_val)     # 1回目のステップの振幅割合
        A2 = K_val / (1 + K_val) # 2回目のステップの振幅割合
        t2 = np.pi / wd          # 2回目のステップを入れるタイミング（半周期後）
    else:
        A1, A2, t2 = 1.0, 0.0, 0.0

    # 状態変数の初期化
    xs = np.zeros(n_steps) # スライダ位置
    vs = np.zeros(n_steps) # スライダ速度
    xn = np.zeros(n_steps) # ノズル位置
    vn = np.zeros(n_steps) # ノズル速度
    u = np.zeros(n_steps)  # モーターの制御入力(推力)
    
    target_shaped = np.zeros(n_steps) # シェーピングされた目標軌道
    
    for i in range(1, n_steps):
        current_time = i * dt
        
        # インプットシェーピングによる目標値の階段状変化
        if use_shaping:
            if current_time < t2:
                curr_target = target_pos * A1
            else:
                curr_target = target_pos * (A1 + A2) # 最終的に target_pos に到達
        else:
            curr_target = target_pos
            
        target_shaped[i] = curr_target

        # 現在の状態
        xs_curr = xs[i-1]
        vs_curr = vs[i-1]
        xn_curr = xn[i-1]
        vn_curr = vn[i-1]
        
        # PD制御器による推力計算 (シェーピングされた目標位置を使用)
        F_ctrl = Kp * (curr_target - xs_curr) - Kd * vs_curr
        u[i] = F_ctrl
        
        # スライダとノズル間に働く力 (バネ・ダンパによる相互作用)
        F_interaction = k * (xs_curr - xn_curr) + c * (vs_curr - vn_curr)
        
        # 運動方程式に基づく加速度の計算
        a_s = (F_ctrl - F_interaction) / M
        a_n = F_interaction / m
        
        # 状態の更新 (オイラー法)
        vs[i] = vs_curr + a_s * dt
        xs[i] = xs_curr + vs[i] * dt
        
        vn[i] = vn_curr + a_n * dt
        xn[i] = xn_curr + vn[i] * dt
        
    target_shaped[0] = target_pos * A1 if use_shaping else target_pos

    return t, xs, xn, u, target_shaped

# プラント（実際の物理モデル）のパラメータ
wn_real = 30.0   
zeta_real = 0.05 

# =========================================================
# 【ユーザー設定エリア】色々変更して挙動を比較してください
# =========================================================
Kp_setting = 80.0
Kd_setting = 5.0
target = 1.0

# --- インプットシェーパー（ZVシェーパー）の設定 ---
# 制御側が想定する対象の「固有振動数」と「減衰率」を指定します。

# 【正解のパラメータ】(実際の物理モデルと完全に一致させた場合)
# 以下のコメントアウトを外し、テスト用パラメータを上書きすると
# 理論上、ノズルの残留振動が最も綺麗に打ち消されます。
# shaper_wn_setting = 30.0
# shaper_zeta_setting = 0.05

# 【テスト用パラメータ】(想定がズレている場合をテスト)
# わざと実際の値(30.0, 0.05)からズラすことで、ロバスト性を確認できます。
shaper_wn_setting = 20.0
shaper_zeta_setting = 0.1
# =========================================================

# 1. シェーピングなし（比較用：shaper_wn=0で無効化）
t, xs_base, xn_base, u_base, target_base = simulate_slider_nozzle_control(
    wn_real, zeta_real, Kp_setting, Kd_setting, target, 
    shaper_wn=0, shaper_zeta=0)

# 2. シェーピングあり
t, xs_shape, xn_shape, u_shape, target_shape = simulate_slider_nozzle_control(
    wn_real, zeta_real, Kp_setting, Kd_setting, target, 
    shaper_wn=shaper_wn_setting, shaper_zeta=shaper_zeta_setting)

# -----------------
# 結果のプロット
# -----------------
plt.figure(figsize=(10, 10))

# 1. 絶対位置のグラフ（シェーピングあり/なしのノズル位置比較）
plt.subplot(3, 1, 1)
plt.plot(t, target_shape, label='Shaped Target (Input)', color='green', linestyle=':', linewidth=2)
plt.plot(t, xn_base, label='Nozzle (No Shaping)', color='gray', alpha=0.5)
plt.plot(t, xn_shape, label='Nozzle (With Shaping)', color='red')
plt.axhline(target, color='black', linestyle='--', alpha=0.5, label='Final Target')
plt.title('Nozzle Position: No Shaping vs With Shaping')
plt.ylabel('Absolute Position')
plt.legend()
plt.grid(True)

# 2. ノズルの揺れ（スライダからの相対変位）
plt.subplot(3, 1, 2)
deflection_base = xn_base - xs_base
deflection_shape = xn_shape - xs_shape
plt.plot(t, deflection_base, label='Deflection (No Shaping)', color='gray', alpha=0.5)
plt.plot(t, deflection_shape, label='Deflection (With Shaping)', color='orange')
plt.ylabel('Deflection (xn - xs)')
plt.legend()
plt.grid(True)

# 3. 制御入力のグラフ
plt.subplot(3, 1, 3)
plt.plot(t, u_base, label='Control Effort (No Shaping)', color='gray', alpha=0.5)
plt.plot(t, u_shape, label='Control Effort (With Shaping)', color='blue')
plt.xlabel('Time [s]')
plt.ylabel('Force')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# =========================================================
# 1. 学習データの準備（不適切なシェーピング時の振動データを使用）
# =========================================================
# 学習を高速化するため、データを間引いて使用（例: 10ステップごと）
skip = 10
t_train = torch.tensor(t[::skip], dtype=torch.float32).view(-1, 1)
xs_train = torch.tensor(xs_shape[::skip], dtype=torch.float32).view(-1, 1)
xn_train = torch.tensor(xn_shape[::skip], dtype=torch.float32).view(-1, 1)

# PyTorchの自動微分を使うためにrequires_gradをTrueにする
t_train.requires_grad_(True)

# =========================================================
# 2. PINNモデルの定義
# =========================================================
class ParameterEstimationPINN(nn.Module):
    def __init__(self, init_wn, init_zeta):
        super().__init__()
        # 時間tを入力とし、スライダ位置xsとノズル位置xnを出力するネットワーク
        self.net = nn.Sequential(
            nn.Linear(1, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 2)  # 出力: [xs, xn]
        )
        
        # 推定したい物理パラメータ（初期値は「ズレた想定値」をセット）
        self.wn = nn.Parameter(torch.tensor([init_wn], dtype=torch.float32))
        self.zeta = nn.Parameter(torch.tensor([init_zeta], dtype=torch.float32))

    def forward(self, t):
        return self.net(t)

# 勾配計算のヘルパー関数
def compute_grad(y, x):
    return torch.autograd.grad(
        y, x, 
        grad_outputs=torch.ones_like(y), 
        create_graph=True, 
        retain_graph=True
    )[0]

# =========================================================
# 3. 学習ループ
# =========================================================
# モデルの初期化（意図的にズラした設定値 20.0, 0.1 からスタートさせる）
pinn = ParameterEstimationPINN(init_wn=shaper_wn_setting, init_zeta=shaper_zeta_setting)

# Optimizerの設定（Adamを使用）
optimizer = optim.Adam(pinn.parameters(), lr=1e-3)

epochs = 20000
history_wn = []
history_zeta = []
history_loss = []

print("--- PINN Training Started ---")
print(f"Target: wn={wn_real}, zeta={zeta_real}")
print(f"Initial (Wrong): wn={pinn.wn.item():.2f}, zeta={pinn.zeta.item():.2f}")

for epoch in range(epochs):
    optimizer.zero_grad()
    
    # ネットワークの予測値
    preds = pinn(t_train)
    xs_pred = preds[:, 0:1]
    xn_pred = preds[:, 1:2]
    
    # -----------------------------
    # (A) Data Loss（観測データとの誤差）
    # -----------------------------
    loss_data = torch.mean((xs_pred - xs_train)**2) + torch.mean((xn_pred - xn_train)**2)
    
    # -----------------------------
    # (B) Physics Loss（運動方程式の残差）
    # -----------------------------
    # 1階微分（速度）
    dxs_dt = compute_grad(xs_pred, t_train)
    dxn_dt = compute_grad(xn_pred, t_train)
    
    # 2階微分（加速度）
    ddxn_dt2 = compute_grad(dxn_dt, t_train)
    
    # 正規化された運動方程式の残差（勾配爆発を防ぐため wn^2 で割る形にする）
    # ddxn / wn^2 - (xs - xn) - (2*zeta/wn) * (dxs - dxn) = 0
    wn_safe = pinn.wn + 1e-6 # ゼロ除算防止
    residual = (ddxn_dt2 / (wn_safe**2)) - (xs_pred - xn_pred) - (2 * pinn.zeta / wn_safe) * (dxs_dt - dxn_dt)
    
    loss_physics = torch.mean(residual**2)
    
    # -----------------------------
    # Total Loss
    # -----------------------------
    # Data LossとPhysics Lossのバランスを取る（ここではシンプルに足し合わせる）
    loss = loss_data + loss_physics
    
    loss.backward()
    optimizer.step()
    
    # 記録
    history_wn.append(pinn.wn.item())
    history_zeta.append(pinn.zeta.item())
    history_loss.append(loss.item())
    
    if (epoch + 1) % 1000 == 0:
        print(f"Epoch {epoch+1:4d} | Loss: {loss.item():.4e} | Estimated wn: {pinn.wn.item():.2f} | Estimated zeta: {pinn.zeta.item():.4f}")

print("--- Training Finished ---")

# =========================================================
# 4. 推定結果のプロット
# =========================================================
plt.figure(figsize=(12, 5))

# パラメータ推移のプロット (wn)
plt.subplot(1, 2, 1)
plt.plot(history_wn, label='Estimated wn', color='blue')
plt.axhline(wn_real, color='black', linestyle='--', label='True wn (30.0)')
plt.axhline(shaper_wn_setting, color='red', linestyle=':', label='Initial Guess (20.0)')
plt.title('Convergence of wn')
plt.xlabel('Epochs')
plt.ylabel('wn')
plt.legend()
plt.grid(True)

# パラメータ推移のプロット (zeta)
plt.subplot(1, 2, 2)
plt.plot(history_zeta, label='Estimated zeta', color='orange')
plt.axhline(zeta_real, color='black', linestyle='--', label='True zeta (0.05)')
plt.axhline(shaper_zeta_setting, color='red', linestyle=':', label='Initial Guess (0.1)')
plt.title('Convergence of zeta')
plt.xlabel('Epochs')
plt.ylabel('zeta')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()