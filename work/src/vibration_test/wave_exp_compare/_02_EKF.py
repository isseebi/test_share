import numpy as np
import matplotlib.pyplot as plt
from vibe_gen import VibrationSimulator

###### シミューレーターの実行開始
# sim_cases = [
#     "low_freq_vibration",
#     "high_freq_vibration",
#     "white_noise_model",
#     "pulse_wave_model",
#     "custom_equation_model"
# ]

# 1. シミュレーションデータの取得
sim_cases = 'custom_equation_model'
img_name = f"EKF_param_{sim_cases}.png"
sim_instance = VibrationSimulator(mode=sim_cases, with_compensation=True)
sim_data = sim_instance.run()
params_real = sim_instance.get_parameters()

t = sim_data['t']
xs = sim_data['xs'] # 入力（指令値）
xn_obs = sim_data['xn'] # 観測値（ノズル位置）
dt = params_real['dt']

# 2. 拡張カルマンフィルタ (EKF) の設定
# 状態ベクトル z = [xn, dxn, wn, zeta]
# 初期値の設定
z = np.array([
    xn_obs[0],                # xn: 初期位置
    0.0,                      # dxn: 初期速度
    params_real['shaper_wn'], # wn: 推定初期値 (シェーパーの設定値を利用)
    params_real['shaper_zeta']# zeta: 推定初期値
]).reshape(-1, 1)

# 共分散行列 P (推定の不確かさ)
P = np.diag([0.1, 0.1, 100.0, 0.01])

# プロセスノイズ Q (モデルの不完全さ)/大きくすると変化に敏感になる
Q = np.diag([1e-7, 1e-7, 1e-2, 1e-6])

# 観測ノイズ R (センサ誤差)/外乱からの影響度
R = np.array([[1e-4]])

# 履歴保存用
history_z = []

# 3. EKFループ
for i in range(len(t)):
    # --- 予測ステップ ---
    xn, dxn, wn, zeta = z.flatten()
    current_xs = xs[i]
    
    # 非線形状態方程式 f(z, u)
    # ddxn = wn^2 * (xs - xn) - 2 * zeta * wn * dxn (入力微分dxsは0と仮定、または既知なら追加)
    ddxn = (wn**2) * (current_xs - xn) - 2 * zeta * wn * dxn
    
    # 状態の更新 (オイラー積分)
    z_pred = z.copy()
    z_pred[0, 0] += dxn * dt
    z_pred[1, 0] += ddxn * dt
    # wn, zeta は一定（予測ステップでは変化なし）

    # ヤコビアン行列 F = df/dz
    F = np.array([
        [1, dt, 0, 0],
        [-(wn**2) * dt, 1 - 2 * zeta * wn * dt, (2 * wn * (current_xs - xn) - 2 * zeta * dxn) * dt, -2 * wn * dxn * dt],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    
    P_pred = F @ P @ F.T + Q

    # --- 更新ステップ ---
    # 観測行列 H = dh/dz (xnのみを観測していると仮定)
    H = np.array([[1, 0, 0, 0]])
    
    # 観測残差 (イノベーション)
    y = xn_obs[i] - (H @ z_pred)
    
    # カルマンゲイン K
    S = H @ P_pred @ H.T + R
    K = P_pred @ H.T @ np.linalg.inv(S)
    
    # 状態と共分散の更新
    z = z_pred + K @ y
    P = (np.eye(4) - K @ H) @ P_pred
    
    history_z.append(z.flatten())

history_z = np.array(history_z)

# 4. 結果の可視化
wn_est_final = history_z[-1, 2]
zeta_est_final = history_z[-1, 3]

print(f"--- EKF Estimation Finished ---")
print(f"True Values: wn={params_real['wn']:.4f}, zeta={params_real['zeta']:.4f}")
print(f"Estimated  : wn={wn_est_final:.4f}, zeta={zeta_est_final:.4f}")

fig, ax = plt.subplots(2, 1, figsize=(10, 8))

ax[0].plot(t, history_z[:, 2], label='EKF Estimated wn', color='blue')
ax[0].axhline(params_real['wn'], color='black', linestyle='--', label='True wn')
ax[0].set_ylabel('Natural Frequency [rad/s]')
ax[0].legend()
ax[0].grid(True)

ax[1].plot(t, history_z[:, 3], label='EKF Estimated zeta', color='orange')
ax[1].axhline(params_real['zeta'], color='black', linestyle='--', label='True zeta')
ax[1].set_ylabel('Damping Ratio')
ax[1].set_xlabel('Time [s]')
ax[1].legend()
ax[1].grid(True)
plt.savefig(img_name, dpi=300)

plt.tight_layout()
plt.show()

# 推定結果を用いた検証シミュレーション
sim_final = VibrationSimulator(
    mode=sim_cases, 
    with_compensation=True, 
    shaper_wn=wn_est_final,    
    shaper_zeta=zeta_est_final
)
sim_final.run()
sim_final.plot()