
import argparse
import sys
import os
import torch

# Import AppLauncher first!
try:
    from isaaclab.app import AppLauncher
except ImportError as e:
    print(f"Failed to import AppLauncher: {e}")
    sys.exit(1)

# Create the parser
parser = argparse.ArgumentParser(description="Verify Isaac Lab Simulation")
parser.add_argument("--headless", action="store_true", default=True, help="Run in headless mode")
args = parser.parse_args()

# Launch the App
try:
    app_launcher = AppLauncher(headless=args.headless)
    simulation_app = app_launcher.app
    print("Simulation App launched successfully!")
except Exception as e:
    print(f"Failed to launch Simulation App: {e}")
    sys.exit(1)

# Now import other things that might depend on the app/sim being running
try:
    import isaaclab.sim as sim_utils
    from isaaclab.envs import ManagerBasedRLEnv
    print("Successfully imported isaaclab modules.")
    
    # Define a minimal config inline to verify Isaac Lab itself
    from isaaclab.envs import ManagerBasedRLEnvCfg
    from isaaclab.scene import InteractiveSceneCfg
    from isaaclab.managers import ObservationGroupCfg, EventTermCfg
    from isaaclab.utils import configclass
    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg

    from isaaclab.assets import ArticulationCfg
    from isaaclab.actuators import ImplicitActuatorCfg

    import isaaclab.envs.mdp as mdp

    @configclass
    class MinimalEnvCfg(ManagerBasedRLEnvCfg):
        # Scene settings
        scene = InteractiveSceneCfg(num_envs=1, env_spacing=2.0)
        
        # Basic Settings
        @configclass
        class ObservationsCfg:
            policy = ObservationGroupCfg(concatenate_terms=False) # Empty observations
        observations = ObservationsCfg()
        
        # Actions
        @configclass
        class ActionsCfg:
            joint_pos = mdp.JointPositionActionCfg(
                asset_name="robot",
                joint_names=[".*"],
                scale=1.0,
                offset=0.0,
            )
        actions = ActionsCfg()
        
        events = {}
        
        # Required fields
        decimation = 2
        episode_length_s = 5.0
        
        # Dummy rewards and terminations
        @configclass
        class RewardsCfg:
            pass
        rewards = RewardsCfg()
        
        @configclass
        class TerminationsCfg:
            pass
        terminations = TerminationsCfg()
        
        def __post_init__(self):
            # Add a simple ground plane
            self.scene.ground = AssetBaseCfg(
                prim_path="/World/defaultGroundPlane",
                spawn=sim_utils.GroundPlaneCfg(),
            )
            
            # Add a robot using pre-defined config
            from isaaclab_assets.robots.unitree import UNITREE_A1_CFG
            self.scene.robot = UNITREE_A1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    try:
        print("Creating minimal environment...")
        env_cfg = MinimalEnvCfg()
        env = ManagerBasedRLEnv(cfg=env_cfg)
        print("Environment created successfully.")
        
        print("Resetting environment...")
        obs, _ = env.reset()
        print("Environment reset successfully.")
        
        print("Stepping environment...")
        # Create zero action
        action = torch.zeros(env.num_envs, env.action_space.shape[1], device=env.device)
        obs, rew, terminated, truncated, info = env.step(action)
        print("Environment stepped successfully.")
        
        env.close()
        print("Environment closed.")
        
    except Exception as e:
        print(f"Failed to run environment loop: {e}")
        import traceback
        traceback.print_exc()
        simulation_app.close()
        sys.exit(1)

except Exception as e:
    print(f"An error occurred in the main block: {e}")
    simulation_app.close()
    sys.exit(1)

print("Isaac Lab verification script completed successfully.")
simulation_app.close()
