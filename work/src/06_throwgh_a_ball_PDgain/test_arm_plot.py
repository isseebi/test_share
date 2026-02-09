import os
import sys
import numpy as np
import time
import matplotlib.pyplot as plt

# Isaac Sim Path (ユーザー環境に合わせて変更してください)
ISAAC_SIM_PATH = "/home/isseebi/Desktop/user/isaac-sim-standalone-5.0.0-linux-x86_64"

def setup_environment():
    """Sets up the environment variables required for Isaac Sim."""
    env_changed = False
    new_env = os.environ.copy()
    
    # 必要な環境変数の設定（省略なしで記述）
    if new_env.get("OMNI_KIT_ACCEPT_EULA") != "YES":
        new_env["OMNI_KIT_ACCEPT_EULA"] = "YES"
        env_changed = True
    if new_env.get("ISAAC_PATH") != ISAAC_SIM_PATH:
        new_env["ISAAC_PATH"] = ISAAC_SIM_PATH
        env_changed = True
    if new_env.get("EXP_PATH") != f"{ISAAC_SIM_PATH}/apps":
        new_env["EXP_PATH"] = f"{ISAAC_SIM_PATH}/apps"
        env_changed = True
    if new_env.get("CARB_APP_PATH") != f"{ISAAC_SIM_PATH}/kit":
        new_env["CARB_APP_PATH"] = f"{ISAAC_SIM_PATH}/kit"
        env_changed = True
    if new_env.get("LD_PRELOAD") != f"{ISAAC_SIM_PATH}/kit/libcarb.so":
        new_env["LD_PRELOAD"] = f"{ISAAC_SIM_PATH}/kit/libcarb.so"
        env_changed = True
    
    kit_lib_path = os.path.join(ISAAC_SIM_PATH, "kit")
    ld_library_path = new_env.get("LD_LIBRARY_PATH", "")
    if kit_lib_path not in ld_library_path:
        new_env["LD_LIBRARY_PATH"] = f"{kit_lib_path}:{ld_library_path}" if ld_library_path else kit_lib_path
        env_changed = True
    
    python_packages = os.path.join(ISAAC_SIM_PATH, "python_packages")
    python_path = new_env.get("PYTHONPATH", "")
    if python_packages not in python_path:
        new_env["PYTHONPATH"] = f"{python_packages}:{python_path}" if python_path else python_packages
        env_changed = True
    
    kit_site_packages = os.path.join(ISAAC_SIM_PATH, "kit", "python", "lib", "python3.11", "site-packages")
    if kit_site_packages not in new_env["PYTHONPATH"]:
         new_env["PYTHONPATH"] = f"{kit_site_packages}:{new_env['PYTHONPATH']}"
         env_changed = True

    if env_changed:
        print("Configuring environment and restarting script...")
        os.execve(sys.executable, [sys.executable] + sys.argv, new_env)

setup_environment()

from isaacsim import SimulationApp
# 描画を見たい場合は headless=False に設定
simulation_app = SimulationApp({"headless": False})

# 定義された環境クラスをインポート (ファイル名が arm_env.py であると仮定)
from arm_env import ArmThrowVecEnv
from stable_baselines3 import PPO

# --- 行列演算用関数 (Forward Kinematics用) ---
def get_rotation_z(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

def get_rotation_y(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0, s, 0], [0, 1, 0, 0], [-s, 0, c, 0], [0, 0, 0, 1]])

def get_translation(x, y, z):
    return np.array([[1, 0, 0, x], [0, 1, 0, y], [0, 0, 1, z], [0, 0, 0, 1]])

def calculate_fk(q):
    """現在の関節角度 q=[q1, q2, q3] から手先位置(X, Z)を計算"""
    q1, q2, q3 = q
    # Base -> J1 -> J2 -> J3 -> EE
    T_w_j1 = get_translation(0, 0, 0.05) @ get_rotation_z(q1)
    T_j1_j2 = get_translation(0, 0, 0.5) @ get_rotation_y(q2)
    T_j2_j3 = get_translation(0, 0, 0.5) @ get_rotation_y(q3)
    T_j3_ee = get_translation(0, 0, 0.4)
    T_total = T_w_j1 @ T_j1_j2 @ T_j2_j3 @ T_j3_ee
    pos = T_total[:3, 3]
    return pos[0], pos[2] # X, Z

def main():
    try:
        print("Creating env...")
        URDF_PATH = "/home/isseebi/Desktop/user/Reinforcement_learning/study/isaacsim/simple_3dof_arm.urdf"
        NUM_ENVS = 1
        
        env = ArmThrowVecEnv(num_envs=NUM_ENVS, urdf_path=URDF_PATH, headless=False)
        print("Env created.")
        
        print("Loading model...")
        # モデルパスは実際のファイル名に合わせてください
        model = PPO.load("arm_model_10000000_steps")
        print("Model loaded.")
        
        obs = env.reset()
        
        # --- データ記録用リスト ---
        robot_pos_history = [] 
        command_history = []
        ball_pos_history = []  # ボール位置履歴用
        
        max_safety_steps = 1000 
        current_step = 0

        print("Starting visualization...")
        
        # 最初のフレームのボール位置も記録
        # obsの構造: [joint_pos(3), joint_vel(3), ball_pos(3)]
        ball_pos_history.append(obs[0, 6:9])

        while simulation_app.is_running():
            # 1. ロボット関節角度 (Index 0-2)
            current_angles = obs[0, :3]  
            robot_pos_history.append(current_angles)

            # 2. AIアクション
            action, _states = model.predict(obs, deterministic=True)
            command_history.append(action[0])

            # 3. ステップ実行
            obs, rewards, dones, info = env.step(action)
            current_step += 1
            
            # --- 重要: ボール位置の取得 (Index 6-8) ---
            # 環境定義により obs の最後3要素がボール位置(x,y,z)です
            current_ball_pos = obs[0, 6:9]
            ball_pos_history.append(current_ball_pos)

            if dones[0]:
                print(f"Episode finished at step {current_step}.")
                break
            
            if current_step >= max_safety_steps:
                print("Max safety steps reached.")
                break

        # --- グラフ描画 ---
        if len(robot_pos_history) > 0:
            print("Processing data for plotting...")
            
            # 1. ロボット手先軌道の計算 (FK)
            pos_data = np.array(robot_pos_history)
            robot_x = []
            robot_z = []
            for q in pos_data:
                rx, rz = calculate_fk(q)
                robot_x.append(rx)
                robot_z.append(rz)
            
            # 2. ボール軌道の取得
            ball_data = np.array(ball_pos_history)
            ball_x = ball_data[:, 0]
            ball_z = ball_data[:, 2] # Z軸

            # 3. プロット作成
            plt.figure(figsize=(10, 8))
            
            # ロボットの軌跡 (時間経過で色変化)
            steps = np.arange(len(robot_x))
            sc = plt.scatter(robot_x, robot_z, c=steps, cmap='Blues', s=30, label='Robot Hand (Time)', edgecolors='none')
            
            # ボールの軌跡 (赤色で描画)
            plt.plot(ball_x, ball_z, color='red', linewidth=2, linestyle='-', label='Ball Trajectory', alpha=0.7)
            
            # 特徴点プロット
            plt.plot(robot_x[0], robot_z[0], 'bx', markersize=10, markeredgewidth=2, label='Robot Start')
            plt.plot(robot_x[-1], robot_z[-1], 'bo', markersize=10, label='Robot End')
            
            plt.scatter(ball_x[0], ball_z[0], color='red', marker='x', s=80, label='Ball Start', zorder=5)
            plt.scatter(ball_x[-1], ball_z[-1], color='red', marker='*', s=150, label='Ball End', zorder=5)

            # グラフ装飾
            plt.title('Robot End-Effector and Ball Trajectories (XZ Plane)')
            plt.xlabel('Global X Position (m)')
            plt.ylabel('Global Z Position (m)')
            plt.axis('equal') 
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend(loc='upper right')
            
            # 保存
            filename = "trajectory_xz_with_ball.png"
            plt.savefig(filename)
            print(f"Plot saved to {filename}")
            plt.close()

            # --- アクションの推移も念のため保存 ---
            plt.figure(figsize=(10, 4))
            plt.plot(command_history)
            plt.title("Action History")
            plt.savefig("action_history.png")
            plt.close()

            print('ボール落下地点')
            print(ball_x[-1], ball_z[-1])

        else:
            print("No data collected.")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'env' in locals() and env is not None:
            env.close()
        if 'simulation_app' in locals() and simulation_app is not None:
            simulation_app.close()

if __name__ == "__main__":
    main()