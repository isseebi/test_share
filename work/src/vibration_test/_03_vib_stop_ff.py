import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# 1. システム定数
m = 1.0     # 質量
k = 100.0   # ばね定数
c = 1.0     # 減衰
v_total = 2.0  # 合計初速度

# 2. 理想的なパラメータの計算
omega_n = np.sqrt(k / m)
zeta = c / (2 * np.sqrt(m * k))
omega_d = omega_n * np.sqrt(1 - zeta**2)
td = np.pi / omega_d # 2回目のパルスのタイミング

# 理論上のゲイン比
K = np.exp(-zeta * np.pi / np.sqrt(1 - zeta**2))
a1_ideal = 1 / (1 + K)
a2_ideal = K / (1 + K)

# 不適切なゲインの設定 (例: 2回目のゲインを50%増やす)
a1_bad = a1_ideal
a2_bad = a2_ideal * 1.5 +4

# 3. 物理モデル
def system_dynamics(state, t, m, k, c):
    x, v = state
    return [v, (-k * x - c * v) / m]

t_span = np.linspace(0, 3, 1000)

def simulate_shaping(a1, a2, td_val):
    # 1回目のパルス
    res1 = odeint(system_dynamics, [0.0, a1 * v_total], t_span, args=(m, k, c))
    
    # 2回目のパルス追加タイミング
    idx_td = np.searchsorted(t_span, td_val)
    state_at_td = res1[idx_td].copy()
    state_at_td[1] += a2 * v_total # 2回目の入力を加算
    
    # td以降を再計算
    t_span_post = t_span[idx_td:] - t_span[idx_td]
    res2 = odeint(system_dynamics, state_at_td, t_span_post, args=(m, k, c))
    
    res_combined = np.zeros_like(res1)
    res_combined[:idx_td] = res1[:idx_td]
    res_combined[idx_td:] = res2
    return res_combined

# 実行
res_ideal = simulate_shaping(a1_ideal, a2_ideal, td)
res_bad_gain = simulate_shaping(a1_bad, a2_bad, td)

# 4. 可視化
plt.figure(figsize=(10, 6))
plt.plot(t_span, res_ideal[:, 0], 'b-', label='Ideal FF Gain (Suppressed)', linewidth=2)
plt.plot(t_span, res_bad_gain[:, 0], 'g-', label='Improper FF Gain (Vibration Remains)', linewidth=2)
plt.axvline(td, color='gray', linestyle=':', label='2nd Pulse Timing')

plt.title('Comparison: Ideal vs Improper Feedforward Gain')
plt.xlabel('Time [s]')
plt.ylabel('Displacement [m]')
plt.legend()
plt.grid(True)
plt.show()

import numpy as np
import matplotlib.pyplot as plt

# 1. 前回のシミュレーションからパラメータを引き継ぐ（もしこのコード単体で動かす場合は定義が必要）
try:
    _ = td, a1_ideal, a2_ideal, a1_bad, a2_bad, v_total
except NameError:
    # パラメータが未定義の場合のための再定義（前回のコードと同じ内容）
    m = 1.0; k = 100.0; c = 1.0; v_total = 2.0
    omega_n = np.sqrt(k / m)
    zeta = c / (2 * np.sqrt(m * k))
    omega_d = omega_n * np.sqrt(1 - zeta**2)
    td = np.pi / omega_d
    K = np.exp(-zeta * np.pi / np.sqrt(1 - zeta**2))
    a1_ideal = 1 / (1 + K)
    a2_ideal = K / (1 + K)
    a1_bad = a1_ideal
    a2_bad = a2_ideal * 1.5 

# 2. 入力パルスを時間軸上に作成（可視化用に短い幅を持たせる）
t_input = np.linspace(0, 3, 1000)
pulse_width_visual = 0.02 # 可視化用のパルス幅（秒）

def create_pulse_train(t_span, a1, a2, td_val, total_v):
    # パルス幅に相当するインデックス数を計算
    dt = t_span[1] - t_span[0]
    num_idx = int(pulse_width_visual / dt)
    
    pulses = np.zeros_like(t_span)
    
    # 1回目のパルス (t=0)
    # 物理的には「速度」を与えるが、ここでは「トルク/加速度」として積分がtotal_vになるように作成
    # 簡略化のため、t=0からnum_idxまでの矩形パルスとする
    if num_idx > 0:
        pulses[:num_idx] = (a1 * total_v) / pulse_width_visual

    # 2回目のパルス (t=td)
    idx_td = np.searchsorted(t_span, td_val)
    if 0 <= idx_td < len(t_span) - num_idx:
        pulses[idx_td : idx_td + num_idx] = (a2 * total_v) / pulse_width_visual
        
    return pulses

# パルス列の生成
pulses_ideal = create_pulse_train(t_input, a1_ideal, a2_ideal, td, v_total)
pulses_bad = create_pulse_train(t_input, a1_bad, a2_bad, td, v_total)

# 3. 可視化
plt.figure(figsize=(10, 5))

# 理想パルス
plt.subplot(2, 1, 1)
plt.plot(t_input, pulses_ideal, 'b-', label='Ideal Input Pulses (ZV)', linewidth=2)
plt.axvline(td, color='gray', linestyle=':', label='2nd Pulse Timing')
plt.title('Motor Torque Input (Acceleration Pulses)')
plt.ylabel('Acceleration [m/s^2]')
plt.grid(True)
plt.legend()
plt.ylim(0, max(max(pulses_ideal), max(pulses_bad)) * 1.1)

# 不適切パルス
plt.subplot(2, 1, 2)
plt.plot(t_input, pulses_bad, 'g-', label='Improper Gain (a2 is excessive)', linewidth=2)
plt.axvline(td, color='gray', linestyle=':')
plt.xlabel('Time [s]')
plt.ylabel('Acceleration [m/s^2]')
plt.grid(True)
plt.legend()
plt.ylim(0, max(max(pulses_ideal), max(pulses_bad)) * 1.1)

plt.tight_layout()
plt.show()

print(f"Ideal a1 (at t=0): {a1_ideal:.3f}")
print(f"Ideal a2 (at t=td): {a2_ideal:.3f}")
print(f"Improper a2 (at t=td): {a2_bad:.3f}")