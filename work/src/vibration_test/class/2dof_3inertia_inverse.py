import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# パラメータ設定
m1 = 2.0    # モータ質量
m2 = 5.0    # テーブル質量
m3 = 1.0    # 手先質量
K1 = 8000.0 # モータ-テーブル間設定
C1 = 20.0   
K2 = 1000.0 # テーブル-手先間
C2 = 5.0    

# 制御ゲイン (モータの追従を高める)
Kp = 800000.0
Kd = 8000.0

# 5次多項式軌道プロファイルパラメータ
target_pos = 1.0
T = 2.0
sim_time = T + 2.0

def get_ideal_trajectory(t):
    """
    5次多項式による理想的な参照軌道
    """
    if t <= 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    elif t >= T:
        return target_pos, 0.0, 0.0, 0.0, 0.0
    
    tau = t / T
    x = target_pos * (10*tau**3 - 15*tau**4 + 6*tau**5)
    v = (target_pos / T) * (30*tau**2 - 60*tau**3 + 30*tau**4)
    a = (target_pos / T**2) * (60*tau - 180*tau**2 + 120*tau**3)
    j = (target_pos / T**3) * (60 - 360*tau + 360*tau**2)
    s = (target_pos / T**4) * (-360 + 720*tau)
    return x, v, a, j, s

def get_inverse_trajectory(t):
    """
    2自由度3慣性系の逆モデル
    手先位置(x3)をx_idealに一致させるためのモータ位置(x1)指令を逆算する
    C1, C2を無視した剛性のみでの逆モデル近似
    """
    x_id, v_id, a_id, j_id, s_id = get_ideal_trajectory(t)
    
    # x1 = x_id + ((m3/K2) + (m2+m3)/K1) * a_id + (m2*m3 / (K1*K2)) * s_id
    coef_a = (m3 / K2) + (m2 + m3) / K1
    coef_s = (m2 * m3) / (K1 * K2)
    
    x_inv = x_id + coef_a * a_id + coef_s * s_id
    
    # 速度指令については近似的にスナップの微分（ここでは0として省略）を使わず計算可能範囲で
    v_inv = v_id + coef_a * j_id
    
    # 加速度指令も近似
    a_inv = a_id
    
    return x_inv, v_inv, a_inv

def equations_uncompensated(t, y):
    x1, v1, x2, v2, x3, v3 = y
    x_ref, v_ref, a_ref, _, _ = get_ideal_trajectory(t)
    F = (m1 + m2 + m3) * a_ref + Kp * (x_ref - x1) + Kd * (v_ref - v1)
    f_spring1 = K1 * (x1 - x2) + C1 * (v1 - v2)
    f_spring2 = K2 * (x2 - x3) + C2 * (v2 - v3)
    a1 = (F - f_spring1) / m1
    a2 = (f_spring1 - f_spring2) / m2
    a3 = f_spring2 / m3
    return [v1, a1, v2, a2, v3, a3]

def equations_inverse(t, y):
    x1, v1, x2, v2, x3, v3 = y
    x_ref, v_ref, a_ref = get_inverse_trajectory(t)
    F = (m1 + m2 + m3) * a_ref + Kp * (x_ref - x1) + Kd * (v_ref - v1)
    f_spring1 = K1 * (x1 - x2) + C1 * (v1 - v2)
    f_spring2 = K2 * (x2 - x3) + C2 * (v2 - v3)
    a1 = (F - f_spring1) / m1
    a2 = (f_spring1 - f_spring2) / m2
    a3 = f_spring2 / m3
    return [v1, a1, v2, a2, v3, a3]

# シミュレーション実行
t_eval = np.linspace(0, sim_time, 3500)
y0 = [0, 0, 0, 0, 0, 0]

sol_uncomp = solve_ivp(equations_uncompensated, (0, sim_time), y0, t_eval=t_eval, method='RK45')
sol_inv = solve_ivp(equations_inverse, (0, sim_time), y0, t_eval=t_eval, method='RK45')

# グラフ描画
xid_all = [get_ideal_trajectory(t)[0] for t in sol_uncomp.t]
xref_inv_all = [get_inverse_trajectory(t)[0] for t in sol_inv.t]

plt.figure(figsize=(10, 10))
plt.subplot(3, 1, 1)
plt.plot(sol_uncomp.t, xid_all, 'k--', label='Ideal Trajectory')
plt.plot(sol_inv.t, xref_inv_all, 'b:', label='Inverse Motor Command')
plt.plot(sol_uncomp.t, sol_uncomp.y[4], 'r', alpha=0.5, label='Hand (x3) Uncompensated')
plt.plot(sol_inv.t, sol_inv.y[4], 'g', label='Hand (x3) Compensated')
plt.ylabel('Position [m]')
plt.title('2 DOF 3 Inertia: Inverse Model Compensation')
plt.grid(True)
plt.legend()

plt.subplot(3, 1, 2)
err_uncomp = sol_uncomp.y[4] - xid_all
err_inv = sol_inv.y[4] - xid_all
plt.plot(sol_uncomp.t, err_uncomp, 'r', alpha=0.5, label='Tracking Err Uncomp (Hand x3)')
plt.plot(sol_inv.t, err_inv, 'g', label='Tracking Err Comp (Hand x3)')
plt.ylabel('Tracking Error [m]')
plt.grid(True)
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(sol_uncomp.t, err_uncomp, 'r', alpha=0.5, label='Tracking Err Uncomp')
plt.plot(sol_inv.t, err_inv, 'g', label='Tracking Err Comp')
plt.xlim([T - 0.2, sim_time])
plt.xlabel('Time [s]')
plt.ylabel('Tracking Err (Zoom) [m]')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig('2dof_3inertia_inverse_result.png')
print("Saved 2dof_3inertia_inverse_result.png")
