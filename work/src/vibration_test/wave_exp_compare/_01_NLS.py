import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from vibe_gen import VibrationSimulator

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
sim_cases = 'custom_equation_model'
sim_instance = VibrationSimulator(mode=sim_cases, with_compensation=True)

sim_xn = sim_instance.run()
param = sim_instance.get_parameters()
print(param)
# sim_instance.plot()

plt.plot(sim_xn['t'],sim_xn['deflection'])
plt.show()


xs_shape = sim_xn['xs']
xn_shape = sim_xn['xn']

wn_real = param['wn']
zeta_real = param['zeta']
shaper_wn_setting = param['shaper_wn']
shaper_zeta_setting = param['shaper_zeta']
dt_sim=param['dt']
###### シミューレーターの実行終了

# ----------
# # 1. データの準備と数値微分
# # ノズルの運動方程式: ddxn = wn^2 * (xs - xn) + 2 * zeta * wn * (dxs - dxn)
# # この方程式を満たすように、データから速度と加速度を計算します。
dxs = np.gradient(xs_shape, dt_sim)
dxn = np.gradient(xn_shape, dt_sim)
ddxn = np.gradient(dxn, dt_sim)

# # データの間引き (計算量削減のため)
skip = 1
xs_data = xs_shape[::skip]
xn_data = xn_shape[::skip]

## ノイズの付加 (現実のセンサーを模倣するため)
# np.random.seed(42)
# noise_level = 0.005 # 位置に対するノイズ（0.5%程度。現実のセンサーではごく普通）
# xs_data = xs_data + np.random.normal(0, noise_level, len(xs_data))
# xn_data = xn_data + np.random.normal(0, noise_level, len(xn_data))

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

plt.savefig("NLS_param.png", dpi=300)
plt.tight_layout()
plt.show()

#####学習結果で振動確認
sim2 = VibrationSimulator(
        mode=sim_cases, 
        with_compensation=True, 
        shaper_wn=wn_est,    
        shaper_zeta=zeta_est  
    )
sim2.run()
sim2.plot()