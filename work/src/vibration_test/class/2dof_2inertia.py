import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# パラメータ設定
mt = 7.0     # 移動体(ボールねじ全体)質量
mb = 50.0    # 土台質量
Kb = 20000.0 # 土台の大地とのバネ定数
Cb = 200.0   # 土台の大地との粘性係数

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
    # y = [xb, vb, xr, vr]  # xb:土台, xr:土台からの相対移動位置
    xb, vb, xr, vr = y
    
    x_ref, v_ref = get_trajectory(t)
    
    # 制御力
    F = Kp * (x_ref - xr) + Kd * (v_ref - vr)
    
    # 運動方程式
    # 全体の重心系の移動反力が土台にかかる
    # mt * (ab + ar) = F
    # mb * ab = -F - Kb*xb - Cb*vb
    
    ab = (-F - Kb * xb - Cb * vb) / mb
    ar = F / mt - ab
    
    return [vb, ab, vr, ar]

# シミュレーション実行
t_span = (0, tend + 1.0)
t_eval = np.linspace(t_span[0], t_span[1], 1000)
y0 = [0, 0, 0, 0]

sol = solve_ivp(equations, t_span, y0, t_eval=t_eval, method='RK45')

# 結果のプロット
x_refs = [get_trajectory(t)[0] for t in sol.t]
x_abs  = sol.y[0] + sol.y[2] # 絶対位置

plt.figure(figsize=(10, 8))
plt.subplot(3, 1, 1)
plt.plot(sol.t, x_refs, 'k--', label='Reference (Relative)')
plt.plot(sol.t, sol.y[2], label='Moving Body (xr)')
plt.ylabel('Relative Pos [m]')
plt.title('2 DOF 2 Inertia System (Moving Body and Base)')
plt.grid(True)
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(sol.t, sol.y[0], label='Base Position (xb)', color='orange')
plt.ylabel('Base Pos [m]')
plt.grid(True)
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(sol.t, x_abs, label='Absolute Position (xb + xr)', color='green')
plt.ylabel('Abs Pos [m]')
plt.xlabel('Time [s]')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig('2dof_2inertia_result.png')
print("Saved 2dof_2inertia_result.png")
