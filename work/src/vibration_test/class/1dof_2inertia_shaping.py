import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# パラメータ設定
m1 = 2.0   # モータ等価質量
m2 = 5.0   # ボールねじ(テーブル)質量
K = 5000.0 # バネ定数
C = 10.0   # 粘性減衰係数

# 固有角振動数の計算 (減衰無視の簡易ZVシェーパ用)
wn = np.sqrt(K * (1.0/m1 + 1.0/m2))
Td = np.pi / wn
shaper = [(0.5, 0.0), (0.5, Td)]
print(f"[1DOF 2Inertia] wn: {wn:.2f} rad/s, Delay Td: {Td:.4f} s")

# 制御ゲイン (追従誤差を極力無くすために非常に高く設定し、FF項を追加)
Kp = 500000.0
Kd = 5000.0

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
    x1, v1, x2, v2 = y
    x_ref, v_ref, a_ref = get_trajectory(t)
    # フィードフォワード制御 + 高ゲインPD制御
    F = (m1 + m2) * a_ref + Kp * (x_ref - x1) + Kd * (v_ref - v1)
    spring_force = K * (x1 - x2) + C * (v1 - v2)
    a1 = (F - spring_force) / m1
    a2 = spring_force / m2
    return [v1, a1, v2, a2]

def equations_shaped(t, y):
    x1, v1, x2, v2 = y
    x_ref, v_ref, a_ref = get_shaped_trajectory(t)
    F = (m1 + m2) * a_ref + Kp * (x_ref - x1) + Kd * (v_ref - v1)
    spring_force = K * (x1 - x2) + C * (v1 - v2)
    a1 = (F - spring_force) / m1
    a2 = spring_force / m2
    return [v1, a1, v2, a2]

# シミュレーション実行
sim_time = tend_base + Td + 1.0
t_eval = np.linspace(0, sim_time, 2000)
y0 = [0, 0, 0, 0]

sol_unshaped = solve_ivp(equations_unshaped, (0, sim_time), y0, t_eval=t_eval, method='RK45')
sol_shaped = solve_ivp(equations_shaped, (0, sim_time), y0, t_eval=t_eval, method='RK45')

# 軌道計算
xref_unshaped = [get_trajectory(t)[0] for t in sol_unshaped.t]
xref_shaped = [get_shaped_trajectory(t)[0] for t in sol_shaped.t]

# グラフ描画
plt.figure(figsize=(10, 10))
plt.subplot(3, 1, 1)
plt.plot(sol_unshaped.t, xref_unshaped, 'k--', label='Ref Unshaped')
plt.plot(sol_shaped.t, xref_shaped, 'b--', label='Ref Shaped')
plt.plot(sol_unshaped.t, sol_unshaped.y[0], 'r', alpha=0.5, label='Motor (x1) Unshaped')
plt.plot(sol_shaped.t, sol_shaped.y[0], 'g', label='Motor (x1) Shaped')
plt.ylabel('Position [m]')
plt.title('1 DOF 2 Inertia: Tightly Tracking Motor & Shaping Comparison')
plt.grid(True)
plt.legend()

plt.subplot(3, 1, 2)
# 残留振動（質点1と質点2の相対変位を見ることで内部の揺れを確認）
deflection_unshaped = sol_unshaped.y[0] - sol_unshaped.y[2]
deflection_shaped = sol_shaped.y[0] - sol_shaped.y[2]
plt.plot(sol_unshaped.t, deflection_unshaped, 'r', alpha=0.5, label='Deflection Unshaped (x1 - x2)')
plt.plot(sol_shaped.t, deflection_shaped, 'g', label='Deflection Shaped (x1 - x2)')
plt.ylabel('Deflection [m]')
plt.grid(True)
plt.legend()

plt.subplot(3, 1, 3)
# 到達後のZoomアップ
plt.plot(sol_unshaped.t, deflection_unshaped, 'r', alpha=0.5, label='Deflection Unshaped (x1 - x2)')
plt.plot(sol_shaped.t, deflection_shaped, 'g', label='Deflection Shaped (x1 - x2)')
plt.xlim([tend_base - 0.2, sim_time])
plt.xlabel('Time [s]')
plt.ylabel('Deflection (Zoom) [m]')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig('1dof_2inertia_shaping_result.png')
print("Saved 1dof_2inertia_shaping_result.png")
