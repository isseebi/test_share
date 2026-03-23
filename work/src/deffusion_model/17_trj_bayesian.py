import numpy as np
import matplotlib.pyplot as plt
import optuna
from traject_gen import CustomPathGenerator

# ==============================================================================
# パラメータ設定
# ==============================================================================
DT = 1.0 / 60.0
OMEGA_N = 3.0
ZETA = 1.0
ARM_LENGTH = 0.5
BOB_RADIUS = 0.05
K_FF = (ARM_LENGTH + BOB_RADIUS) / 9.81
TARGET_PASS_POINT = np.array([1.0, 2.0]) # 本来通りたい経由点

# ==============================================================================
# 軌道シミュレーション関数（フィルタのダイナミクスのみ）
# ==============================================================================
def simulate_trajectory(waypoints):
    path_gen = CustomPathGenerator(step_size=0.02)
    tx_list_raw, ty_list_raw = path_gen.generate(waypoints)
    
    # リストをコピーして、後で延長できるようにする
    tx_list = list(tx_list_raw)
    ty_list = list(ty_list_raw)
    
    x_r, y_r = 0.0, 0.0
    v_r_x, v_r_y = 0.0, 0.0
    
    cmd_x_list, cmd_y_list = [], []
    
    # 1. 参照軌道がある間のシミュレーション
    for tx, ty in zip(tx_list, ty_list):
        # X軸のフィルタ
        a_r_x = (OMEGA_N**2) * (tx - x_r) - 2 * ZETA * OMEGA_N * v_r_x
        v_r_x += a_r_x * DT
        x_r += v_r_x * DT
        cmd_x = x_r + K_FF * a_r_x
        
        # Y軸のフィルタ
        a_r_y = (OMEGA_N**2) * (ty - y_r) - 2 * ZETA * OMEGA_N * v_r_y
        v_r_y += a_r_y * DT
        y_r += v_r_y * DT
        cmd_y = y_r + K_FF * a_r_y
        
        cmd_x_list.append(cmd_x)
        cmd_y_list.append(cmd_y)

    # 2. ゴール地点で静止して、フィルタ出力が収束するまで延長
    goal_x, goal_y = tx_list[-1], ty_list[-1]
    
    # 最大で5秒間(DT*300)延長するか、十分近づくまでループ
    for _ in range(300): 
        # 現在の誤差を計算
        dist_to_goal = np.sqrt((cmd_x - goal_x)**2 + (cmd_y - goal_y)**2)
        velocity = np.sqrt(v_r_x**2 + v_r_y**2)
        
        # 誤差が1mm以下、かつ速度が十分小さくなったら終了
        if dist_to_goal < 0.001 and velocity < 0.001:
            break
            
        # 目標値をゴールに固定してフィルタを計算
        a_r_x = (OMEGA_N**2) * (goal_x - x_r) - 2 * ZETA * OMEGA_N * v_r_x
        v_r_x += a_r_x * DT
        x_r += v_r_x * DT
        cmd_x = x_r + K_FF * a_r_x
        
        a_r_y = (OMEGA_N**2) * (goal_y - y_r) - 2 * ZETA * OMEGA_N * v_r_y
        v_r_y += a_r_y * DT
        y_r += v_r_y * DT
        cmd_y = y_r + K_FF * a_r_y
        
        cmd_x_list.append(cmd_x)
        cmd_y_list.append(cmd_y)
        tx_list.append(goal_x) # プロット用に参照値も延長
        ty_list.append(goal_y)
        
    return tx_list, ty_list, cmd_x_list, cmd_y_list
# ==============================================================================
# 経由点の最適化 (Optuna)
# ==============================================================================
def objective(trial):
    # 本来の[1.0, 2.0]より外側を探索範囲とする
    wy_x = trial.suggest_float("wy_x", 0.8, 2.5)
    wy_y = trial.suggest_float("wy_y", 2.0, 4.0)
    
    waypoints = [[0.0, 0.0], [wy_x, wy_y], [1.0, 0.0]]
    _, _, cmd_x_list, cmd_y_list = simulate_trajectory(waypoints)
    
    # [1.0, 2.0] との最短距離を計算
    trajectory_points = np.column_stack((cmd_x_list, cmd_y_list))
    distances = np.linalg.norm(trajectory_points - TARGET_PASS_POINT, axis=1)
    min_dist = np.min(distances)
    
    # ==========================================================================
    # 制約: 軌道のy座標がTARGET_PASS_POINT[1] (2.0) を超えないようにする
    # ==========================================================================
    max_y = np.max(cmd_y_list)
    if max_y > TARGET_PASS_POINT[1]:
        # 超過した量に対して大きなペナルティを課す (重み付けとして1000を掛ける)
        penalty = (max_y - TARGET_PASS_POINT[1]) * 1000.0
        return min_dist + penalty
    
    return min_dist

print("=== ベイズ最適化による経由点の探索を開始 ===")
optuna.logging.set_verbosity(optuna.logging.WARNING)
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=500)

best_x = study.best_params["wy_x"]
best_y = study.best_params["wy_y"]
print(f"最適化完了: 調整後の経由点 -> [{best_x:.3f}, {best_y:.3f}]")
print(f"目標点 [1.0, 2.0] との最短誤差: {study.best_value:.4f} m\n")

# ==============================================================================
# 最適化結果を使った最終シミュレーションとプロット
# ==============================================================================
best_waypoints = [[0.0, 0.0], [best_x, best_y], [1.0, 0.0]]
tx_list, ty_list, cmd_x_list, cmd_y_list = simulate_trajectory(best_waypoints)

# --- テキストファイルへの出力 ---
txt_filename = "optimized_trajectory.txt"
# numpyのcolumn_stackでxとyを結合し、savetxtで一気に保存
trajectory_data = np.column_stack((cmd_x_list, cmd_y_list))
header = "X_Position, Y_Position"
np.savetxt(txt_filename, trajectory_data, delimiter=',', header=header, fmt='%.6f', comments='')
print(f"軌道データを保存しました: {txt_filename}")

plt.figure(figsize=(10, 8))

# 軌道のプロット
plt.plot(tx_list, ty_list, 'r--', label="Generated Path (to Optimized Waypoint)")
plt.plot(cmd_x_list, cmd_y_list, 'b-', linewidth=2, label="Filtered Trajectory (Command)")

# 重要なポイントのプロット
plt.plot(0.0, 0.0, 'go', markersize=10, label="START [0.0, 0.0]")
plt.plot(1.0, 0.0, 'rx', markersize=10, label="END [1.0, 0.0]")
plt.plot(TARGET_PASS_POINT[0], TARGET_PASS_POINT[1], 'm*', markersize=15, label="Target Pass Point [1.0, 2.0]")
plt.plot(best_x, best_y, 'co', markersize=8, label=f"Optimized Waypoint [{best_x:.2f}, {best_y:.2f}]")

# 軌道の最大y座標のラインを可視化（確認用）
max_y_actual = np.max(cmd_y_list)
plt.axhline(y=2.0, color='gray', linestyle=':', label="Constraint Line (Y=2.0)")
plt.plot([cmd_x_list[np.argmax(cmd_y_list)]], [max_y_actual], 'ks', label=f"Max Trajectory Y [{max_y_actual:.2f}]")

# 一定間隔でマーカーを打つ
plot_interval = 10
plt.scatter(
    cmd_x_list[::plot_interval], 
    cmd_y_list[::plot_interval], 
    color='blue', marker='o', s=20, alpha=0.5
)

plt.title("Trajectory Optimization Result (Filter Only)")
plt.xlabel("X Position (m)")
plt.ylabel("Y Position (m)")
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.tight_layout()

save_path = "optimized_trajectory.png"
plt.savefig(save_path)
print(f"グラフを保存しました: {save_path}")
plt.show()