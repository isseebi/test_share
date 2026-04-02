import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# パラメータ設定
m1 = 2.0    # モータ質量
m2 = 5.0    # テーブル質量
m3 = 1.0    # 手先質量
K1 = 8000.0 # モータ-テーブル間バネ定数
C1 = 20.0   # モータ-テーブル間粘性係数
K2 = 1000.0 # テーブル-手先間バネ定数
C2 = 5.0    # テーブル-手先間粘性係数

# 制御ゲイン
Kp = 2000.0
Kd = 150.0

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
    # y = [x1, v1, x2, v2, x3, v3]
    x1, v1, x2, v2, x3, v3 = y
    
    x_ref, v_ref = get_trajectory(t)
    
    F = Kp * (x_ref - x1) + Kd * (v_ref - v1)
    
    f_spring1 = K1 * (x1 - x2) + C1 * (v1 - v2)
    f_spring2 = K2 * (x2 - x3) + C2 * (v2 - v3)
    
    a1 = (F - f_spring1) / m1
    a2 = (f_spring1 - f_spring2) / m2
    a3 = f_spring2 / m3
    
    return [v1, a1, v2, a2, v3, a3]

# シミュレーション実行
t_span = (0, tend + 1.0)
t_eval = np.linspace(t_span[0], t_span[1], 1500)
y0 = [0, 0, 0, 0, 0, 0]

sol = solve_ivp(equations, t_span, y0, t_eval=t_eval, method='RK45')

# 結果のプロット
x_refs = [get_trajectory(t)[0] for t in sol.t]

plt.figure(figsize=(10, 8))
plt.subplot(2, 1, 1)
plt.plot(sol.t, x_refs, 'k--', label='Reference')
plt.plot(sol.t, sol.y[0], label='Motor (x1)')
plt.plot(sol.t, sol.y[2], label='Table (x2)')
plt.plot(sol.t, sol.y[4], label='End Effector (x3)')
plt.ylabel('Position [m]')
plt.title('2 DOF 3 Inertia System')
plt.grid(True)
plt.legend()

plt.subplot(2, 1, 2)
# 変位差のプロット
plt.plot(sol.t, sol.y[0] - sol.y[2], label='Deflection (Motor-Table)')
plt.plot(sol.t, sol.y[2] - sol.y[4], label='Deflection (Table-Hand)')
plt.xlabel('Time [s]')
plt.ylabel('Deflection [m]')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig('2dof_3inertia_result.png')
print("Saved 2dof_3inertia_result.png")
