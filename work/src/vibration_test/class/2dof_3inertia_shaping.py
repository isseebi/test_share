import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.linalg import eig

# パラメータ設定
m1 = 2.0    # モータ質量
m2 = 5.0    # テーブル質量
m3 = 1.0    # 手先質量
K1 = 8000.0 # モータ-テーブル間設定
C1 = 20.0   
K2 = 1000.0 # テーブル-手先間
C2 = 5.0    

# 固有角振動数の計算
M = np.diag([m1, m2, m3])
K_mat = np.array([
    [K1, -K1, 0],
    [-K1, K1+K2, -K2],
    [0, -K2, K2]
])
eigenvalues, _ = eig(K_mat, M)
wns = np.sqrt(np.sort(np.real(eigenvalues[eigenvalues > 1e-5])))
wn1, wn2 = wns[0], wns[1]

print(f"[2DOF 3Inertia] Mode 1: wn = {wn1:.2f} rad/s")
print(f"[2DOF 3Inertia] Mode 2: wn = {wn2:.2f} rad/s")

# 2モード統合ZVシェーパの作成
Td1 = np.pi / wn1
Td2 = np.pi / wn2
shaper_2mode = []
for A1, t1 in [(0.5, 0.0), (0.5, Td1)]:
    for A2, t2 in [(0.5, 0.0), (0.5, Td2)]:
        shaper_2mode.append((A1 * A2, t1 + t2))

shaper_2mode.sort(key=lambda x: x[1])

# 制御ゲイン (追従誤差を極力なくす)
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
    for A, delay in shaper_2mode:
        if t >= delay:
            x, v, a = get_trajectory(t - delay)
            x_sum += A * x
            v_sum += A * v
            a_sum += A * a
    return x_sum, v_sum, a_sum

def equations_unshaped(t, y):
    x1, v1, x2, v2, x3, v3 = y
    x_ref, v_ref, a_ref = get_trajectory(t)
    F = (m1 + m2 + m3) * a_ref + Kp * (x_ref - x1) + Kd * (v_ref - v1)
    f_spring1 = K1 * (x1 - x2) + C1 * (v1 - v2)
    f_spring2 = K2 * (x2 - x3) + C2 * (v2 - v3)
    a1 = (F - f_spring1) / m1
    a2 = (f_spring1 - f_spring2) / m2
    a3 = f_spring2 / m3
    return [v1, a1, v2, a2, v3, a3]

def equations_shaped(t, y):
    x1, v1, x2, v2, x3, v3 = y
    x_ref, v_ref, a_ref = get_shaped_trajectory(t)
    F = (m1 + m2 + m3) * a_ref + Kp * (x_ref - x1) + Kd * (v_ref - v1)
    f_spring1 = K1 * (x1 - x2) + C1 * (v1 - v2)
    f_spring2 = K2 * (x2 - x3) + C2 * (v2 - v3)
    a1 = (F - f_spring1) / m1
    a2 = (f_spring1 - f_spring2) / m2
    a3 = f_spring2 / m3
    return [v1, a1, v2, a2, v3, a3]

# シミュレーション実行
sim_time = tend_base + Td1 + Td2 + 1.0
t_eval = np.linspace(0, sim_time, 2500)
y0 = [0, 0, 0, 0, 0, 0]

sol_unshaped = solve_ivp(equations_unshaped, (0, sim_time), y0, t_eval=t_eval, method='RK45')
sol_shaped = solve_ivp(equations_shaped, (0, sim_time), y0, t_eval=t_eval, method='RK45')

# グラフ描画
xref_unshaped = [get_trajectory(t)[0] for t in sol_unshaped.t]
xref_shaped = [get_shaped_trajectory(t)[0] for t in sol_shaped.t]

plt.figure(figsize=(10, 12))
plt.subplot(4, 1, 1)
plt.plot(sol_unshaped.t, xref_unshaped, 'k--', label='Ref Unshaped')
plt.plot(sol_shaped.t, xref_shaped, 'b--', label='Ref Shaped')
plt.plot(sol_unshaped.t, sol_unshaped.y[0], 'r', alpha=0.5, label='Motor (x1) Unshaped')
plt.plot(sol_shaped.t, sol_shaped.y[0], 'g', label='Motor (x1) Shaped')
plt.ylabel('Position [m]')
plt.title('2 DOF 3 Inertia: Shaping Comparison (Tight Tracking)')
plt.grid(True)
plt.legend()

plt.subplot(4, 1, 2)
def_unshaped1 = sol_unshaped.y[0] - sol_unshaped.y[2]
def_shaped1 = sol_shaped.y[0] - sol_shaped.y[2]
plt.plot(sol_unshaped.t, def_unshaped1, 'r', alpha=0.5, label='M-T def_Unshaped (x1-x2)')
plt.plot(sol_shaped.t, def_shaped1, 'g', label='M-T def_Shaped (x1-x2)')
plt.ylabel('Deflection M-T [m]')
plt.grid(True)
plt.legend()

plt.subplot(4, 1, 3)
def_unshaped2 = sol_unshaped.y[2] - sol_unshaped.y[4]
def_shaped2 = sol_shaped.y[2] - sol_shaped.y[4]
plt.plot(sol_unshaped.t, def_unshaped2, 'r', alpha=0.5, label='T-H def_Unshaped (x2-x3)')
plt.plot(sol_shaped.t, def_shaped2, 'g', label='T-H def_Shaped (x2-x3)')
plt.ylabel('Deflection T-H [m]')
plt.grid(True)
plt.legend()

plt.subplot(4, 1, 4)
# 到達後のZoomアップ
plt.plot(sol_unshaped.t, def_unshaped1, 'r', alpha=0.5, label='M-T def_Unshaped')
plt.plot(sol_shaped.t, def_shaped1, 'g', label='M-T def_Shaped')
plt.plot(sol_unshaped.t, def_unshaped2, 'k', alpha=0.5, linestyle=':', label='T-H def_Unshaped')
plt.plot(sol_shaped.t, def_shaped2, 'c', linestyle=':', label='T-H def_Shaped')
plt.xlim([tend_base - 0.2, sim_time])
plt.xlabel('Time [s]')
plt.ylabel('Deflection (Zoom) [m]')
plt.grid(True)
plt.legend(loc='best')

plt.tight_layout()
plt.savefig('2dof_3inertia_shaping_result.png')
print("Saved 2dof_3inertia_shaping_result.png")
