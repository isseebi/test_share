import os
import sys

# Isaac Sim Path
ISAAC_SIM_PATH = "/home/isseebi/Desktop/user/isaac-sim-standalone-5.0.0-linux-x86_64"

def setup_environment():
    """Sets up the environment variables required for Isaac Sim and restarts the script if needed."""
    env_changed = False
    new_env = os.environ.copy()

    # 1. OMNI_KIT_ACCEPT_EULA
    if new_env.get("OMNI_KIT_ACCEPT_EULA") != "YES":
        new_env["OMNI_KIT_ACCEPT_EULA"] = "YES"
        env_changed = True

    # 2. LD_LIBRARY_PATH
    # Isaac Sim requires its libraries to be in LD_LIBRARY_PATH
    kit_lib_path = os.path.join(ISAAC_SIM_PATH, "kit")
    ld_library_path = new_env.get("LD_LIBRARY_PATH", "")
    if kit_lib_path not in ld_library_path:
        # Prepend to ensure priority
        new_env["LD_LIBRARY_PATH"] = f"{kit_lib_path}:{ld_library_path}" if ld_library_path else kit_lib_path
        env_changed = True

    # 3. PYTHONPATH
    # Ensure python_packages is in PYTHONPATH to import isaacsim
    python_packages = os.path.join(ISAAC_SIM_PATH, "python_packages")
    python_path = new_env.get("PYTHONPATH", "")
    if python_packages not in python_path:
        new_env["PYTHONPATH"] = f"{python_packages}:{python_path}" if python_path else python_packages
        env_changed = True
        
    # Also add kit site-packages just in case, though isaacsim package should handle it
    kit_site_packages = os.path.join(ISAAC_SIM_PATH, "kit", "python", "lib", "python3.11", "site-packages")
    if kit_site_packages not in new_env["PYTHONPATH"]:
         new_env["PYTHONPATH"] = f"{kit_site_packages}:{new_env['PYTHONPATH']}"
         env_changed = True

    if env_changed:
        print("Configuring environment and restarting script...")
        # Use execve to replace the current process
        os.execve(sys.executable, [sys.executable] + sys.argv, new_env)

setup_environment()

from isaacsim import SimulationApp

# Start Simulation
simulation_app = SimulationApp({"headless": False})

import omni.isaac.core.utils.prims as prim_utils
from omni.isaac.core.robots import Robot
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.motion_generation import LulaKinematicsSolver
from omni.isaac.core.articulations import Articulation
from omni.isaac.core import World
import numpy as np
import carb

# Settings
URDF_PATH = "/home/isseebi/Desktop/user/Reinforcement_learning/study/isaacsim/simple_3dof_arm.urdf"
TARGET_POSITION = np.array([0.3, 0.0, 0.5])

def main():
    # Initialize World
    world = World()
    
    # Add Ground Plane
    world.scene.add_default_ground_plane()
    
    # Add Lighting
    from pxr import UsdLux, Sdf
    stage = world.stage
    light_prim = UsdLux.DistantLight.Define(stage, Sdf.Path("/World/defaultLight"))
    light_prim.CreateIntensityAttr(1000)
    light_prim.CreateAngleAttr(0.53)
    
    # Load URDF
    from omni.isaac.core.utils.extensions import enable_extension
    try:
        enable_extension("isaacsim.asset.importer.urdf")
    except Exception as e:
        print(f"Warning: Could not enable isaacsim.asset.importer.urdf: {e}")
    
    import omni.kit.commands
    
    # Import URDF
    status, import_config = omni.kit.commands.execute("URDFCreateImportConfig")
    import_config.merge_fixed_joints = False
    import_config.convex_decomp = False
    import_config.fix_base = True
    import_config.make_default_prim = True
    
    omni.kit.commands.execute(
        "URDFParseAndImportFile",
        urdf_path=URDF_PATH,
        import_config=import_config,
    )
    
    # Create Robot instance
    robot_prim_path = "/simple_3dof_arm"
    robot = Articulation(prim_path=robot_prim_path, name="my_robot")
    world.scene.add(robot)
    
    # Spawn Ball
    from omni.isaac.core.objects import DynamicSphere
    ball = DynamicSphere(
        prim_path="/World/ball",
        name="ball",
        position=np.array([0.0, 0.0, 1.5]), # Slightly above the cup (robot is ~1.4m tall extended)
        radius=0.04,
        color=np.array([0.0, 1.0, 0.0]),
        mass=0.1
    )
    world.scene.add(ball)

    # Reset the world
    world.reset()
    
    # Set initial joint positions (straight up)
    # The robot has 3 revolute joints. 0,0,0 should be straight up if defined correctly.
    # Based on URDF:
    # joint1: base_link -> link1 (z-axis)
    # joint2: link1 -> link2 (y-axis)
    # joint3: link2 -> link3 (y-axis)
    # link1 is 0.5m, link2 is 0.5m, link3 is 0.4m.
    # 0,0,0 should be vertical.
    robot.set_joint_positions(np.array([0.0, 0.0, 0.0]))
    
    print("Robot initialized. Waiting for ball to settle...")

    step_count = 0
    state = "WAITING" # WAITING, MOVING, DONE
    
    while simulation_app.is_running():
        world.step(render=True)
        
        if not world.is_playing():
            continue
            
        step_count += 1
        
        if state == "WAITING":
            # Wait for 200 steps (approx 3-4 seconds at 60Hz)
            if step_count > 200:
                print("Ball settled. Moving robot...")
                state = "MOVING"
                
                # Set target to a random position to drop the ball
                # Tilt joint 2 and 3
                target_joints = np.array([1.5, 1.0, 1.0]) 
                action = ArticulationAction(joint_positions=target_joints)
                robot.apply_action(action)
                
        elif state == "MOVING":
            # Keep applying the action (PD control needs constant target)
            target_joints = np.array([1.5, 1.0, 1.0])
            action = ArticulationAction(joint_positions=target_joints)
            robot.apply_action(action)
            
            if step_count > 500:
                print("Simulation finished.")
                state = "DONE"
                # Optional: exit or just keep running
                # break

    simulation_app.close()

if __name__ == "__main__":
    main()
