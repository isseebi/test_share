
import argparse
import os
import sys

# Add Isaac Lab path if needed, or assume environment is set up
from isaaclab.app import AppLauncher

# Parse args for headless
parser = argparse.ArgumentParser(description="Train A1 with Isaac Lab")
parser.add_argument("--headless", action="store_true", default=False, help="Run in headless mode")
args = parser.parse_args()

# Launch App
# app_launcher = AppLauncher(headless=args.headless)
# simulation_app = app_launcher.app

# Import Isaac Lab and other libs
import gymnasium as gym
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_rl.sb3 import Sb3VecEnvWrapper

from a1_lab_cfg import A1WalkEnvCfg

def main():
    # Create Environment
    env_cfg = A1WalkEnvCfg()
    env_cfg.scene.num_envs = 4096 if args.headless else 16 # Adjust based on mode
    
    env = ManagerBasedRLEnv(cfg=env_cfg)
    
    # Wrap for SB3
    env = Sb3VecEnvWrapper(env)
    
    # Normalize
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    
    # PPO
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log="./tensorboard_logs_a1_lab/",
        learning_rate=3e-4,
        n_steps=24,
        batch_size=4096,
        n_epochs=5,
        ent_coef=0.01,
        device="cuda"
    )
    
    print("Starting training...")
    checkpoint_callback = CheckpointCallback(save_freq=1000, save_path='./logs_a1_lab/', name_prefix='a1_lab_model')
    model.learn(total_timesteps=30000000, callback=checkpoint_callback)
    
    model.save("a1_lab_ppo")
    env.save("a1_lab_vecnormalize.pkl")
    
    env.close()
    # simulation_app.close()

if __name__ == "__main__":
    main()
