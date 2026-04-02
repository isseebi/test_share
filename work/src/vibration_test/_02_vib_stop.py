import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# 1. 物理モデルの定義
def system_dynamics(state, t, m, k, c):
    x, v = state
    # 運動方程式: m*a + c*v + k*x = 0  => a = (-k*x - c*v) / m
    dxdt = v
    dvdt = (-k * x - c * v) / m
    return [dxdt, dvdt]

def simulate(m, k, c, v0, label):
    t = np.linspace(0, 5, 1000)
    # 初期状態: [変位0 (停止した瞬間の位置), 速度v0 (急停止直前の速度 = 慣性)]
    initial_state = [0.0, v0]
    
    sol = odeint(system_dynamics, initial_state, t, args=(m, k, c))
    return t, sol[:, 0], label

# 共通の初速度（急停止前の走行速度）
v_start = 2.0 

# 2. パラメータを振って比較
results = [
    # 標準設定
    simulate(m=1.0, k=100, c=2.0, v0=v_start, label='Standard'),
    # 質量が大きい（慣性が強く、大きく飛び出す）
    simulate(m=3.0, k=100, c=2.0, v0=v_start, label='High Mass (Heavy)'),
    # バネが強い（戻す力が強く、振幅が小さく周期が短い）
    simulate(m=1.0, k=400, c=2.0, v0=v_start, label='High Stiffness'),
    # 減衰が強い（ブレーキが強く、すぐに収束する）
    simulate(m=1.0, k=100, c=15.0, v0=v_start, label='High Damping')
]

# 3. 可視化
plt.figure(figsize=(12, 7))
for t, x, label in results:
    plt.plot(t, x, label=label, linewidth=2)

plt.title(f'Inertial Vibration after Sudden Stop (Initial Velocity: {v_start}m/s)', fontsize=14)
plt.xlabel('Time [s]', fontsize=12)
plt.ylabel('Displacement [m] (Forward Displacement)', fontsize=12)
plt.axhline(0, color='black', linestyle='--', alpha=0.5)
plt.grid(True, which='both', linestyle=':', alpha=0.7)
plt.legend()
plt.show()