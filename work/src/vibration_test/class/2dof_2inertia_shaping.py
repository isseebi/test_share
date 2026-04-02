import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# パラメータ設定
mt = 7.0     # 移動体全体の質量
mb = 50.0    # 土台質量
Kb = 20000.0 # 土台の大地とのバネ定数
Cb = 200.0   # 土台の大地との粘性係数

# 固有角振動数の計算
wn = np.sqrt(Kb / (mt + mb))
Td = np.pi / wn
shaper = [(0.5, 0.0), (0.5, Td)]
print(f"[2DOF 2Inertia] wn: {wn:.2f} rad/s, Delay Td: {Td:.4f} s")

# 制御ゲイン (追従誤差を極力減らすため、高いゲインとFFを設定)
Kp = 800000.0
Kd = 8000.0

# 台形速度軌道プロファイル
target_pos = 1.0
max_vel = 0.5
accel = 1.0

ta = max_vel / accel
tc = target_pos / max_vel - ta
td = ta
tend_base = ta + tc + td

def get_trajectory(t):
    if t <= 0:
        return 0.0, 0.0, 0.0
    elif t < ta:
        v_ref = accel * t
        x_ref = 0.5 * accel * t**2
        a_ref = accel
    elif t < ta + tc:
        v_ref = max_vel
        x_ref = 0.5 * accel * ta**2 + max_vel * (t - ta)
        a_ref = 0.0
    elif t < tend_base:
        dt = t - (ta + tc)
        v_ref = max_vel - accel * dt
        x_ref = 0.5 * accel * ta**2 + max_vel * tc + max_vel * dt - 0.5 * accel * dt**2
        a_ref = -accel
    else:
        v_ref = 0.0
        x_ref = target_pos
        a_ref = 0.0
    return x_ref, v_ref, a_ref

def get_shaped_trajectory(t):
    x_sum, v_sum, a_sum = 0.0, 0.0, 0.0
    for A, delay in shaper:
        if t >= delay:
            x, v, a = get_trajectory(t - delay)
            x_sum += A * x
            v_sum += A * v
            a_sum += A * a
    return x_sum, v_sum, a_sum

def equations_unshaped(t, y):
    xb, vb, xr, vr = y
    x_ref, v_ref, a_ref = get_trajectory(t)
    F = mt * a_ref + Kp * (x_ref - xr) + Kd * (v_ref - vr)
    ab = (-F - Kb * xb - Cb * vb) / mb
    ar = F / mt - ab
    return [vb, ab, vr, ar]

def equations_shaped(t, y):
    xb, vb, xr, vr = y
    x_ref, v_ref, a_ref = get_shaped_trajectory(t)
    F = mt * a_ref + Kp * (x_ref - xr) + Kd * (v_ref - vr)
    ab = (-F - Kb * xb - Cb * vb) / mb
    ar = F / mt - ab
    return [vb, ab, vr, ar]

# シミュレーション実行
sim_time = tend_base + Td + 1.0
t_eval = np.linspace(0, sim_time, 2000)
y0 = [0, 0, 0, 0]

sol_unshaped = solve_ivp(equations_unshaped, (0, sim_time), y0, t_eval=t_eval, method='RK45')
sol_shaped = solve_ivp(equations_shaped, (0, sim_time), y0, t_eval=t_eval, method='RK45')

# 軌道
xref_unshaped = [get_trajectory(t)[0] for t in sol_unshaped.t]
xref_shaped = [get_shaped_trajectory(t)[0] for t in sol_shaped.t]

plt.figure(figsize=(10, 10))
plt.subplot(3, 1, 1)
plt.plot(sol_unshaped.t, xref_unshaped, 'k--', label='Ref Unshaped')
plt.plot(sol_shaped.t, xref_shaped, 'b--', label='Ref Shaped')
plt.plot(sol_unshaped.t, sol_unshaped.y[2], 'r', alpha=0.5, label='xr (Relative Pos) Unshaped')
plt.plot(sol_shaped.t, sol_shaped.y[2], 'g', label='xr (Relative Pos) Shaped')
plt.ylabel('Relative Position [m]')
plt.title('2 DOF 2 Inertia: Input Shaping Comparison (Tight Tracking)')
plt.grid(True)
plt.legend()

plt.subplot(3, 1, 2)
# 土台の振動 xb を比較
plt.plot(sol_unshaped.t, sol_unshaped.y[0], 'r', alpha=0.5, label='Base Pos (xb) Unshaped')
plt.plot(sol_shaped.t, sol_shaped.y[0], 'g', label='Base Pos (xb) Shaped')
plt.ylabel('Base Position [m]')
plt.grid(True)
plt.legend()

plt.subplot(3, 1, 3)
# 到達後のZoomアップ
plt.plot(sol_unshaped.t, sol_unshaped.y[0], 'r', alpha=0.5, label='Base Pos (xb) Unshaped')
plt.plot(sol_shaped.t, sol_shaped.y[0], 'g', label='Base Pos (xb) Shaped')
plt.xlim([tend_base - 0.2, sim_time])
plt.xlabel('Time [s]')
plt.ylabel('Base Pos (Zoom) [m]')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig('2dof_2inertia_shaping_result.png')
print("Saved 2dof_2inertia_shaping_result.png")
