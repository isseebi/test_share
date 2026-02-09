
import os
import sys
import numpy as np
import time

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
# headless=False for testing/visualization
simulation_app = SimulationApp({"headless": False})

from walk_env import WalkVecEnv
from stable_baselines3 import PPO

def main():
    try:
        print("Creating env...")
        URDF_PATH = "/home/isseebi/Desktop/user/Reinforcement_learning/study/isaacsim/simple_walk_robot.urdf"
        NUM_ENVS = 1 # Small number for visualization
        
        env = WalkVecEnv(num_envs=NUM_ENVS, urdf_path=URDF_PATH, headless=False)
        
        # Load Normalization Stats
        from stable_baselines3.common.vec_env import VecNormalize
        # We need to wrap the env to load stats, but we don't want to update them during test
        # Note: VecNormalize requires a dummy venv if we just want to load? 
        # Actually we can just wrap it and load.
        env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0)
        try:
            env = VecNormalize.load("walk_vecnormalize.pkl", env) # Load into the env
            env.training = False # Disable update
            env.norm_reward = False
        except:
            print("Warning: Could not load normalization stats. Running without normalization.")

        print("Env created.")
        
        print("Loading model...")
        # Load Model
        model_path = "walk_model_28672000_steps"
        if os.path.exists(model_path + ".zip"):
            model = PPO.load(model_path, env=env)
            print("Model loaded.")
        else:
            model = None
            print("No model found, running random actions.")
        
        print("Starting simulation...")
        obs = env.reset()
        
        while simulation_app.is_running():
            if model:
                action, _states = model.predict(obs, deterministic=True)
            else:
                action = np.random.uniform(-1, 1, (NUM_ENVS, env.action_space.shape[0]))
                
            obs, rewards, dones, info = env.step(action)
            
            # Optional: Print rewards or info
            # print(f"Reward: {rewards[0]}")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        env.close()
        simulation_app.close()

if __name__ == "__main__":
    main()
