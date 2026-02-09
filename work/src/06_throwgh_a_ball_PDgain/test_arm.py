import os
import sys
import numpy as np
import time
import matplotlib.pyplot as plt

# Isaac Sim Path (ユーザー環境に合わせて変更してください)
ISAAC_SIM_PATH = "/home/isseebi/Desktop/user/isaac-sim-standalone-5.0.0-linux-x86_64"
# URDF Path (ユーザー環境に合わせて変更してください)
URDF_PATH = "/home/isseebi/Desktop/user/Reinforcement_learning/study/isaacsim/simple_3dof_arm.urdf"
# 学習済みモデルのパス
MODEL_PATH = "arm_model_20000000_steps"

def setup_environment():
    """Sets up the environment variables required for Isaac Sim."""
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

# --- SimulationApp Start ---
from isaacsim import SimulationApp
# 描画を見たい場合は headless=False に設定
simulation_app = SimulationApp({"headless": False})

# --- Imports after SimulationApp ---
import torch
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3 import PPO

from omni.isaac.core import World
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.prims import RigidPrimView
from omni.isaac.core.utils.prims import define_prim
import omni.kit.commands
from omni.isaac.core.prims import XFormPrim

# --- Environment Definition ---
class ArmThrowVecEnv(VecEnv):
    def __init__(self, num_envs, urdf_path, headless=False):
        # Action: 3 joint positions + 3 Kp + 3 Kd = 9
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(9,), dtype=np.float32)
        
        # Observation: 3 joint pos + 3 joint vel + 3 ball pos = 9
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32)
        
        super().__init__(num_envs, self.observation_space, self.action_space)
        
        self.urdf_path = urdf_path
        self.headless = headless
        self._setup_world()
        
        self.current_kps = torch.ones((self.num_envs, 3), device="cuda:0") * 1000.0 
        self.current_kds = torch.ones((self.num_envs, 3), device="cuda:0") * 100.0

    def _setup_world(self):
        self.world = World(backend="torch", device="cuda:0")
        self.world.scene.add_default_ground_plane()
        
        omni.kit.commands.execute(
            "CreatePrim",
            prim_path="/World/DistantLight",
            prim_type="DistantLight",
            attributes={"inputs:intensity": 1000, "inputs:angle": 0.0}
        )
        XFormPrim("/World/DistantLight").set_world_pose(orientation=np.array([0.92388, 0.38268, 0.0, 0.0])) 

        from omni.isaac.core.utils.extensions import enable_extension
        enable_extension("isaacsim.asset.importer.urdf")
        
        define_prim("/World/envs")
        template_path = "/World/template"
        define_prim(template_path)
        
        self._import_urdf(template_path) 
        
        from omni.isaac.core.objects import DynamicSphere
        ball_prim_path = f"{template_path}/ball"
        DynamicSphere(
            prim_path=ball_prim_path,
            name="ball",
            position=np.array([0.0, 0.0, 1.5]),
            radius=0.04,
            color=np.array([0.0, 1.0, 0.0]),
            mass=0.5
        )
        
        self.env_pos = []
        spacing = 2.0
        num_cols = int(np.sqrt(self.num_envs))
        
        for i in range(self.num_envs):
            env_path = f"/World/envs/env_{i}"
            omni.kit.commands.execute("CopyPrim", path_from=template_path, path_to=env_path)
            row = i // num_cols
            col = i % num_cols
            pos = np.array([row * spacing, col * spacing, 0.0])
            self.env_pos.append(pos)
            XFormPrim(env_path).set_world_pose(position=pos)
            
        self.env_pos = np.array(self.env_pos)
        omni.kit.commands.execute("DeletePrims", paths=[template_path])
        
        self.robots = ArticulationView(prim_paths_expr="/World/envs/env_*/simple_3dof_arm", name="robots_view")
        self.world.scene.add(self.robots)
        self.balls = RigidPrimView(prim_paths_expr="/World/envs/env_*/ball", name="balls_view")
        self.world.scene.add(self.balls)
        
        self.world.reset()
        self.initial_ball_pos_global = self.env_pos + np.array([0.0, 0.0, 1.5])
        self.max_steps = 200
        self.step_counts = torch.zeros(self.num_envs, device="cuda:0")
        
    def _import_urdf(self, prim_path):
        status, import_config = omni.kit.commands.execute("URDFCreateImportConfig")
        import_config.merge_fixed_joints = False
        import_config.convex_decomp = False
        import_config.fix_base = True
        import_config.make_default_prim = True
        
        result = omni.kit.commands.execute(
            "URDFParseAndImportFile",
            urdf_path=self.urdf_path,
            import_config=import_config,
        )
        if result[0]:
            omni.kit.commands.execute("MovePrim", path_from="/simple_3dof_arm", path_to=f"{prim_path}/simple_3dof_arm")
        
    def step_async(self, actions):
        actions_tensor = torch.tensor(actions, device="cuda:0", dtype=torch.float32)
        self.pos_targets = actions_tensor[:, 0:3] * np.pi
        self.current_kps = (actions_tensor[:, 3:6] + 1.0) / 2.0 * 10000.0
        self.current_kds = (actions_tensor[:, 6:9] + 1.0) / 2.0 * 1000.0
        
    def step_wait(self):
        self.robots.set_joint_position_targets(self.pos_targets)
        self.robots.set_gains(kps=self.current_kps, kds=self.current_kds)
        
        self.world.step(render=not self.headless)
        self.step_counts += 1
        
        joint_pos = self.robots.get_joint_positions()
        joint_vel = self.robots.get_joint_velocities()
        ball_pos, _ = self.balls.get_world_poses()
        
        obs = torch.cat([joint_pos, joint_vel, ball_pos], dim=1)
        
        rewards = torch.zeros(self.num_envs, device="cuda:0")
        dones = torch.zeros(self.num_envs, device="cuda:0", dtype=torch.bool)
        infos = [{} for _ in range(self.num_envs)]
        
        ball_z = ball_pos[:, 2]
        has_landed = ball_z < 0.05
        landed_indices = torch.where(has_landed)[0]
        
        if len(landed_indices) > 0:
            dones[landed_indices] = True
            
        timeout_indices = torch.where(self.step_counts >= self.max_steps)[0]
        if len(timeout_indices) > 0:
            dones[timeout_indices] = True
            
        done_indices = torch.where(dones)[0]
        if len(done_indices) > 0:
            self._reset_idx(done_indices)
            
        return obs.cpu().numpy(), rewards.cpu().numpy(), dones.cpu().numpy(), infos
        
    def _reset_idx(self, env_ids):
        positions = torch.zeros((len(env_ids), 3), device="cuda:0")
        self.robots.set_joint_positions(positions, indices=env_ids)
        self.robots.set_joint_velocities(positions, indices=env_ids)
        self.robots.set_joint_position_targets(positions, indices=env_ids)
        
        default_kp = 1000.0
        default_kd = 100.0
        self.current_kps[env_ids] = default_kp
        self.current_kds[env_ids] = default_kd
        
        self.robots.set_gains(
            kps=torch.ones((len(env_ids), 3), device="cuda:0") * default_kp,
            kds=torch.ones((len(env_ids), 3), device="cuda:0") * default_kd,
            indices=env_ids
        )
        
        indices = env_ids.cpu().numpy()
        reset_pos = self.initial_ball_pos_global[indices]
        self.balls.set_world_poses(positions=torch.tensor(reset_pos, device="cuda:0", dtype=torch.float32), indices=env_ids)
        self.balls.set_velocities(torch.zeros((len(env_ids), 6), device="cuda:0"), indices=env_ids)
        self.balls.set_masses(torch.ones((len(env_ids)), device="cuda:0") * 0.5, indices=env_ids)
        self.step_counts[env_ids] = 0
        
    def reset(self):
        self._reset_idx(torch.arange(self.num_envs, device="cuda:0"))
        joint_pos = self.robots.get_joint_positions()
        joint_vel = self.robots.get_joint_velocities()
        ball_pos = self.balls.get_world_poses()[0]
        obs = torch.cat([joint_pos, joint_vel, ball_pos], dim=1)
        return obs.cpu().numpy()
        
    def close(self):
        self.world.stop()

    # --- 以下の4つのメソッドを追加しました ---
    def get_attr(self, attr_name, indices=None):
        return [getattr(self, attr_name) for _ in range(self.num_envs)]

    def set_attr(self, attr_name, value, indices=None):
        setattr(self, attr_name, value)

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        pass

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * self.num_envs

# --- Helper Functions ---
def get_rotation_z(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

def get_rotation_y(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0, s, 0], [0, 1, 0, 0], [-s, 0, c, 0], [0, 0, 0, 1]])

def get_translation(x, y, z):
    return np.array([[1, 0, 0, x], [0, 1, 0, y], [0, 0, 1, z], [0, 0, 0, 1]])

def calculate_fk(q):
    q1, q2, q3 = q
    # Base -> J1 -> J2 -> J3 -> EE
    T_w_j1 = get_translation(0, 0, 0.05) @ get_rotation_z(q1)
    T_j1_j2 = get_translation(0, 0, 0.5) @ get_rotation_y(q2)
    T_j2_j3 = get_translation(0, 0, 0.5) @ get_rotation_y(q3)
    T_j3_ee = get_translation(0, 0, 0.4)
    T_total = T_w_j1 @ T_j1_j2 @ T_j2_j3 @ T_j3_ee
    pos = T_total[:3, 3]
    return pos[0], pos[2] # X, Z

def main():
    try:
        print("Creating env...")
        NUM_ENVS = 1
        
        env = ArmThrowVecEnv(num_envs=NUM_ENVS, urdf_path=URDF_PATH, headless=False)
        print("Env created.")
        
        print(f"Loading model from {MODEL_PATH} ...")
        # モデルが古い観測空間(15次元)で学習されている場合は再学習が必要です
        model = PPO.load(MODEL_PATH)
        print("Model loaded.")
        
        obs = env.reset()
        
        robot_pos_history = [] 
        command_history = []
        ball_pos_history = []
        
        max_safety_steps = 1000 
        current_step = 0

        print("Starting visualization...")
        
        ball_pos_history.append(obs[0, 6:9])

        while simulation_app.is_running():
            current_angles = obs[0, :3]  
            robot_pos_history.append(current_angles)

            action, _states = model.predict(obs, deterministic=True)
            command_history.append(action[0])

            obs, rewards, dones, info = env.step(action)
            current_step += 1
            
            current_ball_pos = obs[0, 6:9]
            ball_pos_history.append(current_ball_pos)

            if dones[0]:
                print(f"Episode finished at step {current_step}.")
                break
            
            if current_step >= max_safety_steps:
                print("Max safety steps reached.")
                break

        if len(robot_pos_history) > 0:
            print("Processing data for plotting...")
            
            pos_data = np.array(robot_pos_history)
            robot_x = []
            robot_z = []
            for q in pos_data:
                rx, rz = calculate_fk(q)
                robot_x.append(rx)
                robot_z.append(rz)
            
            ball_data = np.array(ball_pos_history)
            ball_x = ball_data[:, 0]
            ball_z = ball_data[:, 2]

            plt.figure(figsize=(10, 8))
            
            steps = np.arange(len(robot_x))
            plt.scatter(robot_x, robot_z, c=steps, cmap='Blues', s=30, label='Robot Hand (Time)', edgecolors='none')
            plt.plot(ball_x, ball_z, color='red', linewidth=2, linestyle='-', label='Ball Trajectory', alpha=0.7)
            
            plt.plot(robot_x[0], robot_z[0], 'bx', markersize=10, markeredgewidth=2, label='Robot Start')
            plt.plot(robot_x[-1], robot_z[-1], 'bo', markersize=10, label='Robot End')
            
            plt.scatter(ball_x[0], ball_z[0], color='red', marker='x', s=80, label='Ball Start', zorder=5)
            plt.scatter(ball_x[-1], ball_z[-1], color='red', marker='*', s=150, label='Ball End', zorder=5)

            plt.title('Robot End-Effector and Ball Trajectories (XZ Plane)')
            plt.xlabel('Global X Position (m)')
            plt.ylabel('Global Z Position (m)')
            plt.axis('equal') 
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend(loc='upper right')
            
            filename = "trajectory_xz_with_ball.png"
            plt.savefig(filename)
            print(f"Plot saved to {filename}")
            plt.close()

            plt.figure(figsize=(12, 6))
            cmd_hist = np.array(command_history)
            # plt.plot(cmd_hist[:, 0:3], label='Target Pos', linestyle='-')
            plt.plot(cmd_hist[:, 3:6], label='Kp Action', linestyle='--')
            plt.plot(cmd_hist[:, 6:9], label='Kd Action', linestyle=':')
            plt.title("Action History (Raw Output)")
            plt.legend()
            plt.savefig("action_history.png")
            plt.close()

            print('ボール落下地点')
            print(f"X: {ball_x[-1]:.4f}, Z: {ball_z[-1]:.4f}")
            print('ゲイン')
            print(cmd_hist[:, 3:6])

        else:
            print("No data collected.")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'env' in locals() and env is not None:
            env.close()
        if 'simulation_app' in locals() and simulation_app is not None:
            simulation_app.close()

if __name__ == "__main__":
    main()