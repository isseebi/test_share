
import os
import sys
import numpy as np

# Isaac Sim Path
ISAAC_SIM_PATH = "/home/isseebi/Desktop/user/isaac-sim-standalone-5.0.0-linux-x86_64"

def setup_environment():
    """Sets up the environment variables required for Isaac Sim and restarts the script if needed."""
    env_changed = False
    new_env = os.environ.copy()

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

# Start Simulation
# headless=True for training
simulation_app = SimulationApp({"headless": True})

from a1_walk_env import A1WalkVecEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import VecNormalize

def main():
    try:
        print("Creating env...")
        URDF_PATH = "/home/isseebi/Desktop/user/Reinforcement_learning/study/isaacsim/a1.urdf"
        NUM_ENVS = 2048
        # NUM_ENVS = 2 # Reduced for UI stability
        TOTAL_TIMESTEPS = 30000000 

        env = A1WalkVecEnv(num_envs=NUM_ENVS, urdf_path=URDF_PATH, headless=True)
        # Add Normalization
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

        print("Env created.")
        
        print("Init PPO...")
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log="./tensorboard_logs_a1_walk/",
            learning_rate=3e-4,
            n_steps=24,          # Shorter rollouts for faster updates
            batch_size=4096,        
            n_epochs=5,          
            ent_coef=0.01,       # Reduced entropy
            clip_range=0.2,
            gae_lambda=0.95,
            gamma=0.99,
            device="cuda"
        )
        print("PPO initialized.")
        
        print("Starting learn...")
        checkpoint_callback = CheckpointCallback(save_freq=1000, save_path='./logs_a1_walk/', name_prefix='a1_walk_model')
        model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=checkpoint_callback)
        print("Learn finished.")
        
        model.save("a1_walk_ppo")
        env.save("a1_walk_vecnormalize.pkl") # Save normalization stats
        print("Model saved.")
        
        env.close()
        simulation_app.close()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
