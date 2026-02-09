import os
import sys
import traceback  # エラー詳細を表示するためのモジュール

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
    kit_lib_path = os.path.join(ISAAC_SIM_PATH, "kit")
    ld_library_path = new_env.get("LD_LIBRARY_PATH", "")
    if kit_lib_path not in ld_library_path:
        new_env["LD_LIBRARY_PATH"] = f"{kit_lib_path}:{ld_library_path}" if ld_library_path else kit_lib_path
        env_changed = True

    # 3. PYTHONPATH
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
# "headless": False にすることでGUIを表示します
simulation_app = SimulationApp({"headless": False})

# --- ここからメイン処理 ---
try:
    # 互換性警告が出ますが、Isaac Sim 5.0でもomni.isaac.coreは動作します
    import omni.isaac.core.utils.prims as prim_utils
    from omni.isaac.core.robots import Robot
    from omni.isaac.core.utils.types import ArticulationAction
    from omni.isaac.core.articulations import Articulation
    from omni.isaac.core import World
    import omni.kit.commands
    from pxr import UsdLux, Sdf, Gf
    import numpy as np
    
    # 【重要】ここに保存したa1.urdfの絶対パスを指定してください
    URDF_PATH = "/home/isseebi/Desktop/user/Reinforcement_learning/study/isaacsim/a1.urdf" 

    def main():
        print("initializing World...")
        world = World()
        world.scene.add_default_ground_plane()
        
        # Lighting
        stage = world.stage
        light_prim = UsdLux.DistantLight.Define(stage, Sdf.Path("/World/defaultLight"))
        light_prim.CreateIntensityAttr(1000)
        light_prim.CreateAngleAttr(0.53)
        
        # Enable Extension
        from omni.isaac.core.utils.extensions import enable_extension
        try:
            # Isaac Sim 5.0の新しいURDFインポータ
            enable_extension("isaacsim.asset.importer.urdf")
        except Exception as e:
            print(f"Warning (Extension): {e}")
        
        # Create Import Config
        print("Creating Import Config...")
        status, import_config = omni.kit.commands.execute("URDFCreateImportConfig")
        if not status:
            raise RuntimeError("Failed to create URDF Import Config")

        import_config.merge_fixed_joints = False
        import_config.convex_decomp = False
        import_config.fix_base = False
        import_config.make_default_prim = True
        import_config.create_physics_scene = True
        # import_config.set_search_paths(...) は削除済み

        print(f"Importing URDF from: {URDF_PATH}")
        if not os.path.exists(URDF_PATH):
            raise FileNotFoundError(f"URDF file not found at: {URDF_PATH}")

        omni.kit.commands.execute(
            "URDFParseAndImportFile",
            urdf_path=URDF_PATH,
            import_config=import_config,
        )
        
        # Create Robot Articulation
        print("Creating Robot Articulation...")
        robot_prim_path = "/a1"
        robot = Articulation(prim_path=robot_prim_path, name="a1_robot")
        world.scene.add(robot)

        print("Resetting World...")
        world.reset()
        
        robot.set_world_pose(position=np.array([0.0, 0.0, 0.5]))
        
        dof = robot.num_dof
        print(f"Robot Degrees of Freedom: {dof}")
        
        if dof > 0:
            kp = 100.0
            kd = 1.0
            robot.get_articulation_controller().set_gains(
                kps=np.full(dof, kp), 
                kds=np.full(dof, kd)
            )

        print("Simulation Loop Starting...")
        step_count = 0
        stand_target = np.array([0.0, 0.9, -1.8] * 4) # 仮のターゲット
        
        while simulation_app.is_running():
            world.step(render=True)
            
            if not world.is_playing():
                continue
                
            step_count += 1
            if step_count > 100 and dof == 12:
                action = ArticulationAction(joint_positions=stand_target)
                robot.apply_action(action)

    if __name__ == "__main__":
        main()

except Exception as e:
    # エラー発生時にここに来ます
    print("\n" + "="*60)
    print("!!! ERROR OCCURRED !!!")
    print("="*60)
    # エラーの詳細（スタックトレース）を表示
    traceback.print_exc()
    print("="*60 + "\n")

finally:
    # エラーがあってもなくても、最後にここで止めてウィンドウを閉じないようにする
    print("Simulation finished or crashed.")
    input("Press Enter to close the application...")
    simulation_app.close()