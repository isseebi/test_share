import argparse
from isaaclab.app import AppLauncher

# Parse args
parser = argparse.ArgumentParser(description="Play Magician Throwing Task")
parser.add_argument("--headless", action="store_true", default=False, help="Run in headless mode")
args = parser.parse_args()

# Launch App
app_launcher = AppLauncher(headless=args.headless)
simulation_app = app_launcher.app

import torch
import sys
import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize

# Isaac Lab 関連のインポート
sys.path.append("/home/isseebi/Desktop/user/IsaacLab/source/isaaclab_tasks")
from isaaclab_tasks.manager_based.manipulation.magisian.throw_env_cfg import ThrowEnvCfg
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_rl.sb3 import Sb3VecEnvWrapper

def main():
    # Model Path
    model_path = "magician_throw_p2_1638400_steps_success"
    env_stats_path = "magician_throw_vecnormalize.pkl"
    output_file = "action_and_trajectory_log.txt" 
    
    if not os.path.exists(model_path + ".zip"):
        print(f"Model file {model_path}.zip not found!")
        return

    # Create Environment
    env_cfg = ThrowEnvCfg()
    env_cfg.scene.num_envs = 1 
    
    # --- 【追加】カメラの初期位置と注視点の設定 ---
    # eye: カメラの座標 [x, y, z]
    # lookat: カメラが向く方向（ターゲット座標） [x, y, z]
    env_cfg.viewer.eye = [1.5, 1.5, 1.5]    # ロボットを斜め上から見る位置
    env_cfg.viewer.lookat = [0.0, 0.0, 0.5] # ロボットのベース付近
    # -------------------------------------------

    env = ManagerBasedRLEnv(cfg=env_cfg)
    
    # ロボットとエンドエフェクタの情報を取得
    robot = env.unwrapped.scene["robot"]
    ee_link_name = "magician_link_gripper_core" 
    ee_link_idx, _ = robot.find_bodies(ee_link_name)
    
    env_sb3 = Sb3VecEnvWrapper(env)
    
    # Load Normalization Stats
    if os.path.exists(env_stats_path):
        env_sb3 = VecNormalize.load(env_stats_path, env_sb3)
        env_sb3.training = False 
        env_sb3.norm_reward = False 
    
    # Load Model
    model = PPO.load(model_path, env=env_sb3)
    
    # Reset
    obs = env_sb3.reset()
    
    print(f"Starting play loop... Logging to '{output_file}'")
    print("TIP: Use Right-Click + WASD to move the camera in the Viewport.")
    
    with open(output_file, "w") as f:
        step_count = 0
        while simulation_app.is_running():
            # 推論
            action, _states = model.predict(obs, deterministic=True)
            
            # 手先座標の取得
            ee_pos = robot.data.body_pos_w[0, ee_link_idx[0]].cpu().numpy()
            ee_x, ee_y, ee_z = ee_pos
            
            # アクションの取得
            raw_action = action[0] 
            action_str = ", ".join(map(str, raw_action.tolist()))
            
            # 保存
            f.write(f"{step_count},{ee_x},{ee_y},{ee_z},{action_str}\n")
            
            # ステップ実行（ここでカメラの描画も更新されます）
            obs, rewards, dones, infos = env_sb3.step(action)
            step_count += 1
            
            # カメラをロボットに追従させたい場合は以下を有効化（オプション）
            # env.unwrapped.sim.set_camera_view(eye=ee_pos + np.array([0.5, 0.5, 0.5]), target=ee_pos)

            if dones.any():
                print(f"Episode finished after {step_count} steps.")
                # エピソード終了時に自動リセットしたくない場合は break せず reset()
                # obs = env_sb3.reset()                
                break
        
    env_sb3.close()
    simulation_app.close()
    print("Done.")

if __name__ == "__main__":
    main()