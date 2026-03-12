
"""
Play script for Magician Throwing Task.
Loads the trained PPO model and runs it in the environment.
"""

import argparse
from isaaclab.app import AppLauncher

# Parse args
parser = argparse.ArgumentParser(description="Play Magician Throwing Task")
parser.add_argument("--headless", action="store_true", default=False, help="Run in headless mode")
args = parser.parse_args()

# Launch App
# Force headless=False for play usually, unless specified
app_launcher = AppLauncher(headless=args.headless)
simulation_app = app_launcher.app

import torch
import sys
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize

# Add path to finding the config
sys.path.append("/home/isseebi/Desktop/user/IsaacLab/source/isaaclab_tasks")
from isaaclab_tasks.manager_based.manipulation.magisian.throw_env_cfg import ThrowEnvCfg
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_rl.sb3 import Sb3VecEnvWrapper

def main():
    # Model Path
    model_path = "magician_throw_p2_2818048_steps"
    env_stats_path = "magician_throw_vecnormalize.pkl"
    
    if not os.path.exists(model_path + ".zip"):
        print(f"Model file {model_path}.zip not found! Have you trained the model?")
        return

    # Create Environment
    env_cfg = ThrowEnvCfg()
    env_cfg.scene.num_envs = 1 if not args.headless else 100
    
    env = ManagerBasedRLEnv(cfg=env_cfg)
    env = Sb3VecEnvWrapper(env)
    
    # Load Normalization Stats
    if os.path.exists(env_stats_path):
        print(f"Loading environment stats from {env_stats_path}...")
        env = VecNormalize.load(env_stats_path, env)
        env.training = False # Don't update stats during play
        env.norm_reward = False # Don't normalize rewards for display
    else:
        print("Warning: VecNormalize stats not found. Running without normalization (performance might be poor).")
    
    # Load Model
    print(f"Loading model from {model_path}...")
    model = PPO.load(model_path, env=env)
    
    # Reset
    obs = env.reset()
    
    print("Starting play loop...")
    
    while simulation_app.is_running():
        # Predict Action
        action, _states = model.predict(obs, deterministic=True)
        print(f"Predicted Action: {action}")
        
        # Step
        obs, rewards, dones, infos = env.step(action)
        
    env.close()
    simulation_app.close()

if __name__ == "__main__":
    main()
