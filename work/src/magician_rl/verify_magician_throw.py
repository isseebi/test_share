
"""
Verification script for Magician Throwing Task.
This script launches the environment and runs a loop where the robot:
1. Starts with the object attached.
2. Moves randomly (mimicking a throw).
3. Releases the suction after a few seconds.
"""

import argparse
from isaaclab.app import AppLauncher

# Parse args
parser = argparse.ArgumentParser(description="Verify Magician Throwing Task")
parser.add_argument("--headless", action="store_true", default=False, help="Run in headless mode")
args = parser.parse_args()

# Launch App
app_launcher = AppLauncher(headless=args.headless)
simulation_app = app_launcher.app

import torch
import sys
import os

# Add path to finding the config
sys.path.append("/home/isseebi/Desktop/user/IsaacLab/source/isaaclab_tasks")
from isaaclab_tasks.manager_based.manipulation.magisian.throw_env_cfg import ThrowEnvCfg
from isaaclab.envs import ManagerBasedRLEnv

def main():
    # Create Environment
    env_cfg = ThrowEnvCfg()
    env_cfg.scene.num_envs = 4 if not args.headless else 100
    
    env = ManagerBasedRLEnv(cfg=env_cfg)
    
    # Reset
    obs, _ = env.reset()
    
    print("Starting verification loop...")
    print("Robot should start with object ATTACHED.")
    print("Action > 0: Keep suction ON. Action <= 0: Release suction.")
    
    step_count = 0
    while simulation_app.is_running():
        # Actions: 3 joints + 1 suction
        # We want to move the arm vigorously and then release.
        
        # 1. Joint Actions (Random but trying to move fast)
        # Scale is 0.5 in actions config? No, it's MagicianMimicAction.
        # MagicianMimicAction inherits form JointPositionAction? 
        # Let's check actions.py... yes, it uses JointPositionAction machinery.
        # So we send target positions for the 3 active joints.
        
        # Create random actions
        actions = torch.rand((env.num_envs, 4), device=env.device) * 2.0 - 1.0
        
        # 2. Suction Control
        # First 100 steps: KEEP ATTACHED (Value > 0)
        # After 100 steps: RELEASE (Value <= 0)
        
        if step_count % 200 < 50:
             # Prepare to throw (move to some start pose?)
             pass
        elif step_count % 200 < 100:
             # Throwing motion!
             pass
        
        # Simple Logic:
        # Suction is the LAST action dimension (dim 3, since 0,1,2 are joints)
        # Wait, let's verify ActionCfg in throw_env_cfg.py
        # arm_action (3 joints)
        # gripper_action (1 dim)
        # The environment concatenates them?
        # ManagerBasedRLEnv processes actions based on the order in ActionCfg?
        # Usually it's a dictionary if raw, but if wrapped it might be flat.
        # ManagerBasedRLEnv expects a dictionary of actions matching the keys in ActionCfg.
        
        # Ah, ManagerBasedRLEnv.step(action) takes a torch.Tensor if it's a flat space, 
        # OR a dict if the action space is Dict?
        # Let's assume it handles flattened if we passed a tensor, OR we construct a dict.
        # Ideally, we construct a dict to be safe.
        
        action_dict = {}
        
        # Arm: 3 joints. Range usually [-1, 1] scaled.
        # Let's actuate them with sin waves to make it swing.
        time_factor = torch.tensor(step_count * 0.05, device=env.device)
        joint_targets = torch.zeros((env.num_envs, 3), device=env.device)
        joint_targets[:, 0] = torch.sin(time_factor) * 0.5 # Base rotation
        joint_targets[:, 1] = torch.cos(time_factor) * 0.5 # Shoulder
        joint_targets[:, 2] = torch.sin(time_factor) * 0.5 # Elbow
        
        # Concatenate actions: [arm_action (3), gripper_action (1)]
        suction_cmd = torch.ones((env.num_envs, 1), device=env.device)
        if (step_count % 100) > 80:
            suction_cmd[:] = -1.0 # Release
            if step_count % 10 == 0:
                print(f"Step {step_count}: RELEASING!")
        else:
            suction_cmd[:] = 1.0 # Keep attached

        action_tensor = torch.cat([joint_targets, suction_cmd], dim=-1)
        
        # Step
        obs, rew, terminated, truncated, extras = env.step(action_tensor)
        
        # Reset if needed (automatically handled by env, but good to know)
        if step_count % 200 == 0:
             print(f"Step {step_count}: Resetting environment cycle (logic only, env handles its own reset)")
        
        # Log object height to see if it drops
        # object is at env.scene["object"]
        # accessing scene via env might be tricky if env is wrapped.
        # But ManagerBasedRLEnv has .scene
        if step_count % 10 == 0:
            obj_height = env.scene["object"].data.root_pos_w[0, 2].item()
            print(f"Step {step_count}: Object Height: {obj_height:.4f}")

        step_count += 1
        
    env.close()
    simulation_app.close()

if __name__ == "__main__":
    main()
