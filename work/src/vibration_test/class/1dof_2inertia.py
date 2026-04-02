import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# パラメータ設定
m1 = 2.0   # モータ等価質量
m2 = 5.0   # ボールねじ(テーブル)質量
K = 5000.0 # バネ定数
C = 10.0   # 粘性減衰係数

# 制御ゲイン
Kp = 1500.0
Kd = 100.0

# 台形速度軌道プロファイル
target_pos = 1.0
max_vel = 0.5
accel = 1.0

ta = max_vel / accel
tc = target_pos / max_vel - ta
td = ta
tend = ta + tc + td

def get_trajectory(t):
    if t < ta:
        v_ref = accel * t
        x_ref = 0.5 * accel * t**2
    elif t < ta + tc:
        v_ref = max_vel
        x_ref = 0.5 * accel * ta**2 + max_vel * (t - ta)
    elif t < tend:
        dt = t - (ta + tc)
        v_ref = max_vel - accel * dt
        x_ref = 0.5 * accel * ta**2 + max_vel * tc + max_vel * dt - 0.5 * accel * dt**2
    else:
        v_ref = 0.0
        x_ref = target_pos
    return x_ref, v_ref

def equations(t, y):
    # y = [x1, v1, x2, v2]
    x1, v1, x2, v2 = y
    
    x_ref, v_ref = get_trajectory(t)
    
    # PD制御による推力
    F = Kp * (x_ref - x1) + Kd * (v_ref - v1)
    
    # 運動方程式
    spring_force = K * (x1 - x2) + C * (v1 - v2)
    
    a1 = (F - spring_force) / m1
    a2 = spring_force / m2
    
    return [v1, a1, v2, a2]

# シミュレーション実行
t_span = (0, tend + 1.0)
t_eval = np.linspace(t_span[0], t_span[1], 1000)
y0 = [0, 0, 0, 0]

sol = solve_ivp(equations, t_span, y0, t_eval=t_eval, method='RK45')

# 結果のプロット
x_refs = [get_trajectory(t)[0] for t in sol.t]

plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(sol.t, x_refs, 'k--', label='Reference')
plt.plot(sol.t, sol.y[0], label='Motor (x1)')
plt.plot(sol.t, sol.y[2], label='Table (x2)')
plt.ylabel('Position [m]')
plt.title('1 DOF 2 Inertia System')
plt.grid(True)
plt.legend()

plt.subplot(2, 1, 2)
# 相対変位のプロット
plt.plot(sol.t, sol.y[0] - sol.y[2], label='Relative Deflection (x1 - x2)', color='red')
plt.xlabel('Time [s]')
plt.ylabel('Deflection [m]')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig('1dof_2inertia_result.png')
print("Saved 1dof_2inertia_result.png")
