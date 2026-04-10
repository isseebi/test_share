import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

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
# 【ユーザー設定エリア】
# =========================================================
Kp_setting = 80.0
Kd_setting = 5.0
target = 1.0

# 【テスト用パラメータ】(想定がズレている場合をテスト)
shaper_wn_setting = 20.0
shaper_zeta_setting = 0.1
# =========================================================

# 1. シェーピングなし（比較用）
t, xs_base, xn_base, u_base, target_base = simulate_slider_nozzle_control(
    wn_real, zeta_real, Kp_setting, Kd_setting, target, 
    shaper_wn=0, shaper_zeta=0)

# 2. シェーピングあり
dt_sim = t[1] - t[0]
t, xs_shape, xn_shape, u_shape, target_shape = simulate_slider_nozzle_control(
    wn_real, zeta_real, Kp_setting, Kd_setting, target, 
    shaper_wn=shaper_wn_setting, shaper_zeta=shaper_zeta_setting, dt=dt_sim)

# -----------------
# 結果のプロット (シミュレーション)
# -----------------
plt.figure(figsize=(10, 10))

plt.subplot(3, 1, 1)
plt.plot(t, target_shape, label='Shaped Target (Input)', color='green', linestyle=':', linewidth=2)
plt.plot(t, xn_base, label='Nozzle (No Shaping)', color='gray', alpha=0.5)
plt.plot(t, xn_shape, label='Nozzle (With Shaping)', color='red')
plt.axhline(target, color='black', linestyle='--', alpha=0.5, label='Final Target')
plt.title('Nozzle Position: No Shaping vs With Shaping')
plt.ylabel('Absolute Position')
plt.legend()
plt.grid(True)

plt.subplot(3, 1, 2)
deflection_base = xn_base - xs_base
deflection_shape = xn_shape - xs_shape
plt.plot(t, deflection_base, label='Deflection (No Shaping)', color='gray', alpha=0.5)
plt.plot(t, deflection_shape, label='Deflection (With Shaping)', color='orange')
plt.ylabel('Deflection (xn - xs)')
plt.legend()
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(t, u_base, label='Control Effort (No Shaping)', color='gray', alpha=0.5)
plt.plot(t, u_shape, label='Control Effort (With Shaping)', color='blue')
plt.xlabel('Time [s]')
plt.ylabel('Force')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# =========================================================
# 【推論フェーズ】非線形最小二乗法によるパラメータ推定
# =========================================================

# 1. データの準備と数値微分
# ノズルの運動方程式: ddxn = wn^2 * (xs - xn) + 2 * zeta * wn * (dxs - dxn)
# この方程式を満たすように、データから速度と加速度を計算します。
dxs = np.gradient(xs_shape, dt_sim)
dxn = np.gradient(xn_shape, dt_sim)
ddxn = np.gradient(dxn, dt_sim)

# データの間引き (計算量削減のため)
skip = 10
xs_data = xs_shape[::skip]
xn_data = xn_shape[::skip]

## ノイズの付加 (現実のセンサーを模倣するため)
np.random.seed(42)
noise_level = 0.005 # 位置に対するノイズ（0.5%程度。現実のセンサーではごく普通）
xs_data = xs_data + np.random.normal(0, noise_level, len(xs_data))
xn_data = xn_data + np.random.normal(0, noise_level, len(xn_data))

dxs_data = dxs[::skip]
dxn_data = dxn[::skip]
ddxn_data = ddxn[::skip]

# 探索履歴を保存するリスト
history_wn = []
history_zeta = []

# 2. 残差関数の定義
def physics_residuals(params, xs, xn, dxs, dxn, ddxn):
    wn_est, zeta_est = params
    
    # 探索の軌跡を記録 (ヤコビアン計算の微小変化も含まれます)
    history_wn.append(wn_est)
    history_zeta.append(zeta_est)
    
    # モデルから計算される理論上の加速度
    model_ddxn = (wn_est**2) * (xs - xn) + (2 * zeta_est * wn_est) * (dxs - dxn)
    
    # 実際の加速度との差分（残差）を返す
    return ddxn - model_ddxn

# 3. 最適化の実行
print("\n--- Least Squares Estimation Started ---")
print(f"Target (True) : wn={wn_real:.2f}, zeta={zeta_real:.4f}")

initial_guess = [shaper_wn_setting, shaper_zeta_setting]
print(f"Initial Guess : wn={initial_guess[0]:.2f}, zeta={initial_guess[1]:.4f}")

# 物理的な制約 (固有振動数と減衰率は0以上)
bounds = ([0.0, 0.0], [np.inf, np.inf])

# 非線形最小二乗法のソルバーを実行
result = least_squares(
    physics_residuals, 
    initial_guess, 
    bounds=bounds,
    args=(xs_data, xn_data, dxs_data, dxn_data, ddxn_data)
)

wn_est, zeta_est = result.x

print("--- Estimation Finished ---")
print(f"Estimated     : wn={wn_est:.4f}, zeta={zeta_est:.4f}")
print(f"Optimization Status: {result.message}")

# 4. 探索プロセスのプロット
# （非線形最小二乗法は数ステップ〜数十ステップで一瞬で収束します）
plt.figure(figsize=(12, 5))

# wnの探索軌跡
plt.subplot(1, 2, 1)
plt.plot(history_wn, label='Search Path (wn)', color='blue', marker='.')
plt.axhline(wn_real, color='black', linestyle='--', label='True wn (30.0)')
plt.axhline(shaper_wn_setting, color='red', linestyle=':', label='Initial Guess (20.0)')
plt.title('Optimization Path of wn')
plt.xlabel('Function Evaluations')
plt.ylabel('wn')
plt.legend()
plt.grid(True)

# zetaの探索軌跡
plt.subplot(1, 2, 2)
plt.plot(history_zeta, label='Search Path (zeta)', color='orange', marker='.')
plt.axhline(zeta_real, color='black', linestyle='--', label='True zeta (0.05)')
plt.axhline(shaper_zeta_setting, color='red', linestyle=':', label='Initial Guess (0.1)')
plt.title('Optimization Path of zeta')
plt.xlabel('Function Evaluations')
plt.ylabel('zeta')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()