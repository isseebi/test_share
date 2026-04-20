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
    
    k = m * wn**2
    c = 2 * m * zeta * wn

    # ZV インプットシェーパーの設計
    use_shaping = shaper_wn > 0
    if use_shaping:
        wd = shaper_wn * np.sqrt(1 - shaper_zeta**2)
        K_val = np.exp(-shaper_zeta * np.pi / np.sqrt(1 - shaper_zeta**2))
        A1 = 1 / (1 + K_val)
        A2 = K_val / (1 + K_val)
        t2 = np.pi / wd
    else:
        A1, A2, t2 = 1.0, 0.0, 0.0

    xs = np.zeros(n_steps)
    vs = np.zeros(n_steps)
    xn = np.zeros(n_steps)
    vn = np.zeros(n_steps)
    u = np.zeros(n_steps)  
    target_shaped = np.zeros(n_steps)
    
    for i in range(1, n_steps):
        current_time = i * dt
        if use_shaping:
            if current_time < t2:
                curr_target = target_pos * A1
            else:
                curr_target = target_pos * (A1 + A2)
        else:
            curr_target = target_pos
            
        target_shaped[i] = curr_target

        xs_curr, vs_curr = xs[i-1], vs[i-1]
        xn_curr, vn_curr = xn[i-1], vn[i-1]
        
        F_ctrl = Kp * (curr_target - xs_curr) - Kd * vs_curr
        u[i] = F_ctrl
        
        F_interaction = k * (xs_curr - xn_curr) + c * (vs_curr - vn_curr)
        
        a_s = (F_ctrl - F_interaction) / M
        a_n = F_interaction / m
        
        vs[i] = vs_curr + a_s * dt
        xs[i] = xs_curr + vs[i] * dt
        vn[i] = vn_curr + a_n * dt
        xn[i] = xn_curr + vn[i] * dt
        
    target_shaped[0] = target_pos * A1 if use_shaping else target_pos

    return t, xs, xn, u, target_shaped

# プラント（実際の物理モデル）のパラメータ
wn_real = 80.0   
zeta_real = 0.05 

# ユーザー設定エリア
Kp_setting = 80.0
Kd_setting = 5.0
target = 1.0

# テスト用パラメータ (想定がズレている場合)
shaper_wn_setting = 20.0
shaper_zeta_setting = 0.1

# シェーピングありのシミュレーションを実行
t, xs_shape, xn_shape, u_shape, target_shape = simulate_slider_nozzle_control(
    wn_real, zeta_real, Kp_setting, Kd_setting, target, 
    shaper_wn=shaper_wn_setting, shaper_zeta=shaper_zeta_setting)


# =========================================================
# 拡張カルマンフィルタ (EKF) によるパラメータ推定
# =========================================================
print("--- EKF Training Started ---")

# 時間刻みとステップ数
dt_ekf = t[1] - t[0]
n_steps = len(t)

# 既知の物理パラメータ（システムの質量は分かっている前提）
M = 1.0
m = 0.1

# 状態ベクトル X = [xs, vs, xn, vn, wn, zeta]^T
# 初期状態の設定
X = np.zeros(6)
X[0] = xs_shape[0]           # xs
X[1] = 0.0                   # vs
X[2] = xn_shape[0]           # xn
X[3] = 0.0                   # vn
X[4] = shaper_wn_setting     # wnの初期推測値 (20.0)
X[5] = shaper_zeta_setting   # zetaの初期推測値 (0.1)

# 誤差共分散行列 P の初期化（初期推測の不確実性）
P = np.diag([1e-6, 1e-6, 1e-6, 1e-6, 10.0, 0.1])

# プロセスノイズ共分散 Q
# パラメータ(wn, zeta)に小さなノイズを乗せることで、ゲインを維持し収束を促す
Q = np.diag([1e-8, 1e-8, 1e-8, 1e-8, 1e-2, 1e-6])

# 観測ノイズ共分散 R
# xs と xn が観測可能とする
R = np.diag([1e-4, 1e-4])

# 観測行列 H (状態ベクトルのうち、xs[0]とxn[2]を観測)
H = np.zeros((2, 6))
H[0, 0] = 1.0
H[1, 2] = 1.0

I = np.eye(6)

history_wn = []
history_zeta = []

for i in range(1, n_steps):
    # -----------------------------
    # 1. 予測ステップ (Predict)
    # -----------------------------
    xs_prev, vs_prev, xn_prev, vn_prev, wn_prev, zeta_prev = X
    u_prev = u_shape[i-1] # 制御入力は既知とする
    
    dx = xs_prev - xn_prev
    dv = vs_prev - vn_prev
    
    # 非線形運動方程式による状態予測 (オイラー法)
    a_s = (u_prev - (m * wn_prev**2 * dx + 2 * m * zeta_prev * wn_prev * dv)) / M
    a_n = (m * wn_prev**2 * dx + 2 * m * zeta_prev * wn_prev * dv) / m
    
    X_pred = np.zeros(6)
    X_pred[0] = xs_prev + vs_prev * dt_ekf
    X_pred[1] = vs_prev + a_s * dt_ekf
    X_pred[2] = xn_prev + vn_prev * dt_ekf
    X_pred[3] = vn_prev + a_n * dt_ekf
    X_pred[4] = wn_prev  # パラメータは一定として予測
    X_pred[5] = zeta_prev

    # ヤコビアン A (システム行列の偏微分) の計算
    A = np.eye(6)
    
    # ∂f1 / ∂X
    A[0, 1] = dt_ekf
    
    # ∂f2 / ∂X (スライダ速度)
    A[1, 0] = -dt_ekf * (m * wn_prev**2) / M
    A[1, 1] = 1.0 - dt_ekf * (2 * m * zeta_prev * wn_prev) / M
    A[1, 2] = dt_ekf * (m * wn_prev**2) / M
    A[1, 3] = dt_ekf * (2 * m * zeta_prev * wn_prev) / M
    A[1, 4] = -dt_ekf * (m / M) * (2 * wn_prev * dx + 2 * zeta_prev * dv)
    A[1, 5] = -dt_ekf * (m / M) * (2 * wn_prev * dv)
    
    # ∂f3 / ∂X
    A[2, 3] = dt_ekf
    
    # ∂f4 / ∂X (ノズル速度)
    A[3, 0] = dt_ekf * (wn_prev**2)
    A[3, 1] = dt_ekf * (2 * zeta_prev * wn_prev)
    A[3, 2] = -dt_ekf * (wn_prev**2)
    A[3, 3] = 1.0 - dt_ekf * (2 * zeta_prev * wn_prev)
    A[3, 4] = dt_ekf * (2 * wn_prev * dx + 2 * zeta_prev * dv)
    A[3, 5] = dt_ekf * (2 * wn_prev * dv)
    
    # 誤差共分散の予測
    P_pred = A @ P @ A.T + Q
    
    # -----------------------------
    # 2. 更新ステップ (Update)
    # -----------------------------
    # 現在の観測データ
    Z = np.array([xs_shape[i], xn_shape[i]])
    
    # イノベーション (観測残差)
    y_tilde = Z - H @ X_pred
    
    # イノベーション共分散
    S = H @ P_pred @ H.T + R
    
    # カルマンゲイン
    K = P_pred @ H.T @ np.linalg.inv(S)
    
    # 状態の更新
    X = X_pred + K @ y_tilde
    
    # 誤差共分散の更新
    P = (I - K @ H) @ P_pred
    
    # 記録
    history_wn.append(X[4])
    history_zeta.append(X[5])

print("--- EKF Training Finished ---")

# =========================================================
# 推定結果のプロット
# =========================================================
plt.figure(figsize=(12, 5))
t_ekf = t[1:] # 記録したステップ数に合わせる

# パラメータ推移のプロット (wn)
plt.subplot(1, 2, 1)
plt.plot(t_ekf, history_wn, label='Estimated wn', color='blue')
plt.axhline(wn_real, color='black', linestyle='--', label='True wn (30.0)')
plt.axhline(shaper_wn_setting, color='red', linestyle=':', label='Initial Guess (20.0)')
plt.title('EKF Convergence of wn')
plt.xlabel('Time [s]')
plt.ylabel('wn')
plt.legend()
plt.grid(True)

# パラメータ推移のプロット (zeta)
plt.subplot(1, 2, 2)
plt.plot(t_ekf, history_zeta, label='Estimated zeta', color='orange')
plt.axhline(zeta_real, color='black', linestyle='--', label='True zeta (0.05)')
plt.axhline(shaper_zeta_setting, color='red', linestyle=':', label='Initial Guess (0.1)')
plt.title('EKF Convergence of zeta')
plt.xlabel('Time [s]')
plt.ylabel('zeta')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()