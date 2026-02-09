
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
simulation_app = SimulationApp({"headless": True})

from arm_env import ArmThrowVecEnv

try:
    print("Creating env...")
    URDF_PATH = "/home/isseebi/Desktop/user/Reinforcement_learning/study/isaacsim/simple_3dof_arm.urdf"
    env = ArmThrowVecEnv(num_envs=1, urdf_path=URDF_PATH, headless=True)
    print("Env created.")
    
    print("Resetting env...")
    obs = env.reset()
    print(f"Observation shape: {obs.shape}")
    
    expected_obs_dim = 15
    if obs.shape[1] != expected_obs_dim:
        print(f"ERROR: Expected observation dimension {expected_obs_dim}, got {obs.shape[1]}")
    else:
        print("Observation dimension correct.")
        
    print("Stepping env...")
    # Action space: 9
    action = np.zeros((1, 9))
    obs, rewards, dones, infos = env.step(action)
    print("Step successful.")
    print(f"Observation after step: {obs}")
    
    env.close()
    simulation_app.close()
    print("Verification Passed.")
except Exception as e:
    print(f"Verification Failed: {e}")
    import traceback
    traceback.print_exc()
