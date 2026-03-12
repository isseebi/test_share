
import argparse
import os
import sys

# Add Isaac Lab path
from isaaclab.app import AppLauncher

# Parse args
parser = argparse.ArgumentParser(description="Curriculum Training for Magician Throwing")
parser.add_argument("--headless", action="store_true", default=False, help="Run in headless mode")
parser.add_argument("--phase", type=int, default=1, choices=[1, 2], help="Training Phase: 1 (Force Hold), 2 (Release)")
parser.add_argument("--ckpt", type=str, default=None, help="Path to model checkpoint (zip) for Phase 2")
parser.add_argument("--venv", type=str, default=None, help="Path to VecNormalize inputs (pkl) for Phase 2")
args = parser.parse_args()

# Launch App
app_launcher = AppLauncher(headless=args.headless)
simulation_app = app_launcher.app

# Import Isaac Lab and others
import gymnasium as gym
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_rl.sb3 import Sb3VecEnvWrapper

# Add path to tasks if not installed as package
sys.path.append(os.getcwd()) # Ensure current dir is in path
# Assuming the script is run from the directory containing 'isaaclab_tasks' or parent
# Adjust path as necessary based on user workspace. 
# User workspace: /home/isseebi/Desktop/user/IsaacLab/source/isaaclab_tasks
sys.path.append("/home/isseebi/Desktop/user/IsaacLab/source/isaaclab_tasks")

from isaaclab_tasks.manager_based.manipulation.magisian.throw_env_cfg import ThrowEnvCfg

def main():
    print(f"--- Starting Curriculum Training Phase {args.phase} ---")

    # 1. Configure Environment
    env_cfg = ThrowEnvCfg()
    env_cfg.scene.num_envs = 8192 if args.headless else 4
    
    # Curriculum Logic
    if args.phase == 1:
        print("[Phase 1] FORCE HOLD ENABLED. Robot will learn to generate velocity without dropping.")
        # Enable Force Hold
        env_cfg.actions.gripper_action.force_hold = True
        log_dir = "./logs_curriculum/phase1/"
        tensorboard_dir = "./tensorboard_logs_curriculum/phase1/"
    elif args.phase == 2:
        print("[Phase 2] FORCE HOLD DISABLED. Robot must learn WHEN to release.")
        # Disable Force Hold (Allow Release)
        env_cfg.actions.gripper_action.force_hold = False
        log_dir = "./logs_curriculum/phase2/"
        tensorboard_dir = "./tensorboard_logs_curriculum/phase2/"
        
        if not args.ckpt:
            raise ValueError("Phase 2 requires --ckpt argument to load Phase 1 model!")

    # 2. Create Environment
    env = ManagerBasedRLEnv(cfg=env_cfg)
    env = Sb3VecEnvWrapper(env)
    
    # 3. Normalize
    if args.phase == 2 and args.venv:
        print(f"Loading VecNormalize stats from {args.venv}...")
        env = VecNormalize.load(args.venv, env)
        env.training = True # Ensure training mode
        env.norm_reward = True
    else:
        # New normalization for Phase 1 or if no venv provided
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # 4. Create or Load Model
    policy_kwargs = dict(
        activation_fn=torch.nn.ReLU,
        net_arch=dict(pi=[256, 256], vf=[256, 256])
    )
    
    if args.phase == 1:
        # Create New Model
        model = PPO(
            "MlpPolicy",
            env,
            policy_kwargs=policy_kwargs,
            verbose=1,
            tensorboard_log=tensorboard_dir,
            learning_rate=3e-4,
            n_steps=64,
            batch_size=8192,
            n_epochs=5,
            ent_coef=0.01,
            device="cuda"
        )
    elif args.phase == 2:
        # Load Existing Model
        print(f"Loading model from {args.ckpt}...")
        model = PPO.load(
            args.ckpt,
            env=env,
            print_system_info=True,
            tensorboard_log=tensorboard_dir, # Log to new dir
            # Optional: Adjust Learning Rate for fine-tuning?
            learning_rate=1e-4, 
            # Force kwargs if needed, usually passed from saved model but we can override
            device="cuda"
        )
        
    # 5. Train
    checkpoint_callback = CheckpointCallback(
        save_freq=200000 // env_cfg.scene.num_envs, 
        save_path=log_dir, 
        name_prefix=f'magician_throw_p{args.phase}'
    )
    
    print(f"Training started for Phase {args.phase}...")
    try:
        total_timesteps = 1000000 if args.phase == 1 else 5000000 # Shorter for fine-tuning?
        model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback, reset_num_timesteps=(args.phase==1))
    except KeyboardInterrupt:
        print("Training interrupted.")

    # 6. Save Final
    model_save_path = f"magician_throw_p{args.phase}_final"
    venv_save_path = f"magician_throw_p{args.phase}_vecnormalize.pkl"
    
    model.save(model_save_path)
    env.save(venv_save_path)
    print(f"Saved model to {model_save_path}")
    print(f"Saved stats to {venv_save_path}")

    env.close()
    simulation_app.close()

if __name__ == "__main__":
    main()
