
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
# headless=False for visualization
simulation_app = SimulationApp({"headless": False})

from a1_walk_env import A1WalkVecEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize

def main():
    try:
        print("Creating env...")
        URDF_PATH = "/home/isseebi/Desktop/user/Reinforcement_learning/study/isaacsim/a1.urdf"
        # Use fewer envs for testing visualization
        NUM_ENVS = 1

        env = A1WalkVecEnv(num_envs=NUM_ENVS, urdf_path=URDF_PATH, headless=False)
        
        # Load Normalization Stats
        # Note: We must load the stats to ensure observations are normalized correctly
        # However, VecNormalize.load requires the venv to be passed or created inside.
        # But we already created env.
        # Correct way:
        # env = VecNormalize.load("a1_walk_vecnormalize.pkl", env)
        # But if the file doesn't exist (not trained yet), we should handle it.
        
        model_path = "a1_walk_model_28672000_steps"
        stats_path = "a1_walk_vecnormalize.pkl"
        
        if os.path.exists(stats_path):
            print(f"Loading normalization stats from {stats_path}")
            env = VecNormalize.load(stats_path, env)
            env.training = False # Don't update stats during test
            env.norm_reward = False # Don't normalize rewards during test
        else:
            print("Warning: Normalization stats not found. Running without normalization (might perform poorly).")
            # Wrap with VecNormalize just to match structure if needed, or just use raw env if model was trained without?
            # The training script uses VecNormalize, so we should expect it.
            # If we don't have stats, we can't really test properly if the model expects normalized inputs.
            # But let's proceed.
            pass

        print("Env created.")
        
        if os.path.exists(model_path + ".zip"):
            print(f"Loading model from {model_path}")
            model = PPO.load(model_path, env=env)
            print("Model loaded.")
        else:
            print("Model not found. Running with random actions.")
            model = None

        print("Starting simulation loop...")
        
        obs = env.reset()
        
        while simulation_app.is_running():
            if model:
                action, _ = model.predict(obs, deterministic=True)
            else:
                action = [env.action_space.sample() for _ in range(NUM_ENVS)]
            
            obs, reward, done, info = env.step(action)
            
            # Optional: Slow down for visualization
            # time.sleep(0.016) 
            
        env.close()
        simulation_app.close()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
