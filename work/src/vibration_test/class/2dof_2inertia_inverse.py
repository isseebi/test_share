import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import scipy.signal as signal
from scipy.interpolate import interp1d

# パラメータ設定
mt = 7.0     # 移動体全体の質量
mb = 50.0    # 土台質量
Kb = 20000.0 # 土台の大地とのバネ定数
Cb = 200.0   # 土台の大地との粘性係数

# 制御ゲイン
Kp = 800000.0
Kd = 8000.0

# 5次多項式軌道プロファイルパラメータ
target_pos = 1.0
T = 2.0
sim_time = T + 2.0

def get_ideal_trajectory_array(times):
    x_arr, v_arr, a_arr = np.zeros_like(times), np.zeros_like(times), np.zeros_like(times)
    for i, t in enumerate(times):
        if t <= 0:
            pass
        elif t >= T:
            x_arr[i] = target_pos
        else:
            tau = t / T
            x_arr[i] = target_pos * (10*tau**3 - 15*tau**4 + 6*tau**5)
            v_arr[i] = (target_pos / T) * (30*tau**2 - 60*tau**3 + 30*tau**4)
            a_arr[i] = (target_pos / T**2) * (60*tau - 180*tau**2 + 120*tau**3)
    return x_arr, v_arr, a_arr

def get_ideal_trajectory_scalar(t):
    if t <= 0: return 0.0, 0.0, 0.0
    elif t >= T: return target_pos, 0.0, 0.0
    tau = t / T
    x = target_pos * (10*tau**3 - 15*tau**4 + 6*tau**5)
    v = (target_pos / T) * (30*tau**2 - 60*tau**3 + 30*tau**4)
    a = (target_pos / T**2) * (60*tau - 180*tau**2 + 120*tau**3)
    return x, v, a

# オフラインで土台の「予想される揺れ xb(t)」を計算しておく
t_sim_arr = np.linspace(0, sim_time, 5000)
x_id_arr, v_id_arr, a_id_arr = get_ideal_trajectory_array(t_sim_arr)

# mb * ab + Cb * vb + Kb * xb = -mt * a_id
sys = signal.TransferFunction([-mt], [mb, Cb, Kb])
_, xb_id_arr, _ = signal.lsim(sys, U=a_id_arr, T=t_sim_arr)

# 微分による vb, abの推定
dt = t_sim_arr[1] - t_sim_arr[0]
vb_id_arr = np.gradient(xb_id_arr, dt)
ab_id_arr = np.gradient(vb_id_arr, dt)

xb_interp = interp1d(t_sim_arr, xb_id_arr, fill_value=0.0, bounds_error=False)
vb_interp = interp1d(t_sim_arr, vb_id_arr, fill_value=0.0, bounds_error=False)
ab_interp = interp1d(t_sim_arr, ab_id_arr, fill_value=0.0, bounds_error=False)

def get_inverse_trajectory(t):
    x_id, v_id, a_id = get_ideal_trajectory_scalar(t)
    xb = xb_interp(t)
    vb = vb_interp(t)
    ab = ab_interp(t)
    
    # 絶対位置が x_id になるための相対位置 x_ref の逆算: x_abs = xb + x_ref = x_id => x_ref = x_id - xb
    x_inv = x_id - xb
    v_inv = v_id - vb
    a_inv = a_id - ab
    return float(x_inv), float(v_inv), float(a_inv)

def equations_uncompensated(t, y):
    xb, vb, xr, vr = y
    x_ref, v_ref, a_ref = get_ideal_trajectory_scalar(t)
    F = mt * a_ref + Kp * (x_ref - xr) + Kd * (v_ref - vr)
    ab = (-F - Kb * xb - Cb * vb) / mb
    ar = F / mt - ab
    return [vb, ab, vr, ar]

def equations_inverse(t, y):
    xb, vb, xr, vr = y
    x_ref, v_ref, a_ref = get_inverse_trajectory(t)
    F = mt * a_ref + Kp * (x_ref - xr) + Kd * (v_ref - vr)
    ab = (-F - Kb * xb - Cb * vb) / mb
    ar = F / mt - ab
    return [vb, ab, vr, ar]

# シミュレーション実行
t_eval = np.linspace(0, sim_time, 3000)
y0 = [0, 0, 0, 0]

sol_uncomp = solve_ivp(equations_uncompensated, (0, sim_time), y0, t_eval=t_eval, method='RK45')
sol_inv = solve_ivp(equations_inverse, (0, sim_time), y0, t_eval=t_eval, method='RK45')

# グラフ描画
xid_all = [get_ideal_trajectory_scalar(t)[0] for t in sol_uncomp.t]
xabs_uncomp = sol_uncomp.y[0] + sol_uncomp.y[2]
xabs_inv = sol_inv.y[0] + sol_inv.y[2]

plt.figure(figsize=(10, 10))
plt.subplot(3, 1, 1)
plt.plot(sol_uncomp.t, xid_all, 'k--', label='Ideal Trajectory (Abs Pos)')
plt.plot(sol_uncomp.t, xabs_uncomp, 'r', alpha=0.5, label='Abs Pos Uncompensated')
plt.plot(sol_inv.t, xabs_inv, 'g', label='Abs Pos Compensated')
plt.ylabel('Position [m]')
plt.title('2 DOF 2 Inertia: Inverse Model Compensation')
plt.grid(True)
plt.legend()

plt.subplot(3, 1, 2)
# 絶対位置トラッキングエラー
err_uncomp = xabs_uncomp - xid_all
err_inv = xabs_inv - xid_all
plt.plot(sol_uncomp.t, err_uncomp, 'r', alpha=0.5, label='Abs Pos Err Uncomp')
plt.plot(sol_inv.t, err_inv, 'g', label='Abs Pos Err Comp')
plt.ylabel('Abs Pos Error [m]')
plt.grid(True)
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(sol_uncomp.t, err_uncomp, 'r', alpha=0.5, label='Abs Pos Err Uncomp')
plt.plot(sol_inv.t, err_inv, 'g', label='Abs Pos Err Comp')
plt.xlim([T - 0.2, sim_time])
plt.xlabel('Time [s]')
plt.ylabel('Pos Error (Zoom) [m]')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig('2dof_2inertia_inverse_result.png')
print("Saved 2dof_2inertia_inverse_result.png")
