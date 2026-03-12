
import argparse
import os
import sys

# Add Isaac Lab path if needed, or assume environment is set up
from isaaclab.app import AppLauncher

# Parse args for headless
parser = argparse.ArgumentParser(description="Train Magician Throwing with Isaac Lab")
parser.add_argument("--headless", action="store_true", default=False, help="Run in headless mode")
args = parser.parse_args()

# Launch App
app_launcher = AppLauncher(headless=args.headless)
simulation_app = app_launcher.app

# Import Isaac Lab and other libs
import gymnasium as gym
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_rl.sb3 import Sb3VecEnvWrapper

# Import our custom config
# Note: We are importing from the file path we just created. 
# We might need to adjust python path or move the file to a registered package if direct import fails.
# But for now, let's assume we can import it if we add the path.
sys.path.append("/home/isseebi/Desktop/user/IsaacLab/source/isaaclab_tasks")
from isaaclab_tasks.manager_based.manipulation.magisian.throw_env_cfg import ThrowEnvCfg

def main():
    # Create Environment
    env_cfg = ThrowEnvCfg()
    # env_cfg.scene.num_envs = 4096 if args.headless else 4
    env_cfg.scene.num_envs = 8192 if args.headless else 4

    env = ManagerBasedRLEnv(cfg=env_cfg)
    
    # Wrap for SB3
    env = Sb3VecEnvWrapper(env)
    
    # Normalize
    # Clip observations heavily as physics can be unstable
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    
    policy_kwargs = dict(
        activation_fn=torch.nn.ReLU,
        # net_arch=dict(pi=[256, 128, 64], vf=[256, 128, 64])
        net_arch=dict(pi=[256, 256], vf=[256, 256])
    )
    
    # PPO
    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log="./tensorboard_logs_magician_throw/",
        learning_rate=3e-4,
        n_steps=64, # Short rollout for continuous tasks? Or longer? default is usually 2048 for locomotion.
        # For manipulation, maybe similar. A1 uses 24.
        batch_size=8192,
        n_epochs=5,
        ent_coef=0.01,
        device="cuda"
    )
    
    print("Starting training...")
    checkpoint_callback = CheckpointCallback(save_freq= 200000 // env_cfg.scene.num_envs, save_path='./logs_magician_throw/', name_prefix='magician_throw_model')
    try:
        model.learn(total_timesteps=10000000, callback=checkpoint_callback)
    except KeyboardInterrupt:
        print("Training interrupted.")
    
    model.save("magician_throw_ppo")
    env.save("magician_throw_vecnormalize.pkl")
    
    env.close()
    simulation_app.close()

if __name__ == "__main__":
    main()
