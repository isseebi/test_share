import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# パラメータ設定
m1 = 2.0   # モータ等価質量
m2 = 5.0   # ボールねじ(テーブル)質量
K = 5000.0 # バネ定数
C = 10.0   # 粘性減衰係数

# 制御ゲイン (追従誤差を極力無くすために高く設定)
Kp = 500000.0
Kd = 5000.0

# 5次多項式軌道プロファイルパラメータ
target_pos = 1.0
T = 2.0 # 移動時間全体 (2秒)
sim_time = T + 2.0

def get_ideal_trajectory(t):
    """
    5次多項式による理想的な参照軌道を生成
    戻り値: x, v, a, j(jerk), s(snap)
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
    逆モデルによるモータ指令軌道の算出
    x_inv = x_ideal + (m2 / K) * a_ideal
    """
    x_id, v_id, a_id, j_id, s_id = get_ideal_trajectory(t)
    
    # 簡略化した逆モデル(バネ反力を相殺)
    x_inv = x_id + (m2 / K) * a_id
    v_inv = v_id + (m2 / K) * j_id
    a_inv = a_id + (m2 / K) * s_id
    return x_inv, v_inv, a_inv

def equations_uncompensated(t, y):
    x1, v1, x2, v2 = y
    x_ref, v_ref, a_ref, _, _ = get_ideal_trajectory(t)
    # 逆モデルなし（目標軌道をそのまま指令値とする）
    F = (m1 + m2) * a_ref + Kp * (x_ref - x1) + Kd * (v_ref - v1)
    spring_force = K * (x1 - x2) + C * (v1 - v2)
    a1 = (F - spring_force) / m1
    a2 = spring_force / m2
    return [v1, a1, v2, a2]

def equations_inverse(t, y):
    x1, v1, x2, v2 = y
    # 逆モデル補償あり（計算された補償軌道を指令値とする）
    x_ref, v_ref, a_ref = get_inverse_trajectory(t)
    F = (m1 + m2) * a_ref + Kp * (x_ref - x1) + Kd * (v_ref - v1)
    spring_force = K * (x1 - x2) + C * (v1 - v2)
    a1 = (F - spring_force) / m1
    a2 = spring_force / m2
    return [v1, a1, v2, a2]

# シミュレーション実行
t_eval = np.linspace(0, sim_time, 3000)
y0 = [0, 0, 0, 0]

sol_uncomp = solve_ivp(equations_uncompensated, (0, sim_time), y0, t_eval=t_eval, method='RK45')
sol_inv = solve_ivp(equations_inverse, (0, sim_time), y0, t_eval=t_eval, method='RK45')

# グラフ描画
xid_all = [get_ideal_trajectory(t)[0] for t in sol_uncomp.t]
xref_inv_all = [get_inverse_trajectory(t)[0] for t in sol_inv.t]

plt.figure(figsize=(10, 10))
plt.subplot(3, 1, 1)
plt.plot(sol_uncomp.t, xid_all, 'k--', label='Ideal Trajectory')
plt.plot(sol_inv.t, xref_inv_all, 'b:', label='Inverse Motor Command')
plt.plot(sol_uncomp.t, sol_uncomp.y[2], 'r', alpha=0.5, label='Table Uncompensated')
plt.plot(sol_inv.t, sol_inv.y[2], 'g', label='Table Compensated')
plt.ylabel('Position [m]')
plt.title('1 DOF 2 Inertia: Inverse Model Compensation')
plt.grid(True)
plt.legend()

plt.subplot(3, 1, 2)
# 誤差 (理想軌道と実際のモデルのテーブル位置のズレ)
err_uncomp = sol_uncomp.y[2] - xid_all
err_inv = sol_inv.y[2] - xid_all
plt.plot(sol_uncomp.t, err_uncomp, 'r', alpha=0.5, label='Tracking Err Uncompensated')
plt.plot(sol_inv.t, err_inv, 'g', label='Tracking Err Compensated')
plt.ylabel('Tracking Error [m]')
plt.grid(True)
plt.legend()

plt.subplot(3, 1, 3)
# 到達後のZoomアップ
plt.plot(sol_uncomp.t, err_uncomp, 'r', alpha=0.5, label='Tracking Err Uncomp')
plt.plot(sol_inv.t, err_inv, 'g', label='Tracking Err Comp')
plt.xlim([T - 0.2, sim_time])
plt.xlabel('Time [s]')
plt.ylabel('Tracking Err (Zoom) [m]')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig('1dof_2inertia_inverse_result.png')
print("Saved 1dof_2inertia_inverse_result.png")
