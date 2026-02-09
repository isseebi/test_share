import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv

from omni.isaac.core import World
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.prims import RigidPrimView
from omni.isaac.cloner import Cloner
from omni.isaac.core.utils.prims import define_prim
import omni.kit.commands
from omni.isaac.core.prims import XFormPrim

class ArmThrowVecEnv(VecEnv):
    def __init__(self, num_envs, urdf_path, headless=False):
        # Action: 3 joint positions
        # 変更点: 以前は Kp, Kd を含めて shape=(9,) でしたが、位置のみにするため shape=(3,) に変更
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        
        # Observation: 3 joint pos + 3 joint vel + 3 ball pos = 9 
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32)
        
        super().__init__(num_envs, self.observation_space, self.action_space)
        
        self.urdf_path = urdf_path
        self.headless = headless
        self._setup_world()
        
        # 変更点: ゲインを固定値として初期化 (既定の値として reset 時と同じ値を使用)
        self.default_kp = 10000.0
        self.default_kd = 500.0
        self.current_kps = torch.ones((self.num_envs, 3), device="cuda:0") * self.default_kp
        self.current_kds = torch.ones((self.num_envs, 3), device="cuda:0") * self.default_kd

    def _setup_world(self):
        self.world = World(backend="torch", device="cuda:0")
        self.world.scene.add_default_ground_plane()
        
        omni.kit.commands.execute(
            "CreatePrim",
            prim_path="/World/DistantLight",
            prim_type="DistantLight",
            attributes={
                "inputs:intensity": 1000, 
                "inputs:angle": 0.0,      
            }
        )
        light_prim = XFormPrim("/World/DistantLight")
        light_prim.set_world_pose(orientation=np.array([0.92388, 0.38268, 0.0, 0.0])) 

        from omni.isaac.core.utils.extensions import enable_extension
        enable_extension("isaacsim.asset.importer.urdf")
        
        define_prim("/World/envs")
        template_path = "/World/template"
        define_prim(template_path)
        
        # 1. Load Robot URDF
        self._import_urdf(template_path) 
        
        # 2. Create Ball in template
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
        
        # 3. Manual Cloning
        self.env_pos = []
        spacing = 2.0
        num_cols = int(np.sqrt(self.num_envs))
        
        for i in range(self.num_envs):
            env_path = f"/World/envs/env_{i}"
            omni.kit.commands.execute("CopyPrim", path_from=template_path, path_to=env_path)
            
            row = i // num_cols
            col = i % num_cols
            x = row * spacing
            y = col * spacing
            pos = np.array([x, y, 0.0])
            self.env_pos.append(pos)
            
            env_prim = XFormPrim(env_path)
            env_prim.set_world_pose(position=pos)
            
        self.env_pos = np.array(self.env_pos)

        omni.kit.commands.execute("DeletePrims", paths=[template_path])
        
        # 4. Create Views
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
        else:
            print("URDF Import failed")
        
    def step_async(self, actions):
        # actions: [pos_targets(3)] 
        # 変更点: アクションには関節位置のみが含まれるようになりました
        actions_tensor = torch.tensor(actions, device="cuda:0", dtype=torch.float32)
        
        # 1. Position Targets
        # アクション全体が関節角度ターゲットになります
        self.pos_targets = actions_tensor * np.pi
        
        
    def step_wait(self):
        # Apply actions
        self.robots.set_joint_position_targets(self.pos_targets)
        
        # 既定の固定ゲインを適用
        self.robots.set_gains(kps=self.current_kps, kds=self.current_kds)
        
        self.world.step(render=not self.headless)
        self.step_counts += 1
        
        joint_pos = self.robots.get_joint_positions()
        joint_vel = self.robots.get_joint_velocities()
        ball_pos, _ = self.balls.get_world_poses()
        
        # --- 修正点1: ボールの速度を取得 ---
        # get_velocities() は (num_envs, 6) を返し、最初の3つが線速度(x, y, z)
        ball_velocities = self.balls.get_velocities()
        ball_vel_x = ball_velocities[:, 0]
        
        obs = torch.cat([joint_pos, joint_vel, ball_pos], dim=1)
        
        rewards = torch.zeros(self.num_envs, device="cuda:0")
        dones = torch.zeros(self.num_envs, device="cuda:0", dtype=torch.bool)
        infos = [{} for _ in range(self.num_envs)]
        
        # --- 修正点2: 毎ステップの報酬に「x軸方向の速度」を加算 ---
        # 係数(0.1など)を掛けることで、最終的な飛距離報酬とのバランスを調整します
        vel_reward_weight = 0.1
        rewards += vel_reward_weight * ball_vel_x

        # Reward: Ball Throwing Distance
        ball_z = ball_pos[:, 2]
        has_landed = ball_z < 0.05
                
        env_origins = torch.tensor(self.env_pos, device="cuda:0", dtype=torch.float32)
        relative_ball_pos = ball_pos - env_origins
        
        landed_indices = torch.where(has_landed)[0]
        if len(landed_indices) > 0:
            # 1. 飛距離報酬 (X軸方向)
            dist_x = relative_ball_pos[landed_indices, 0]
            
            # 着地時は飛距離を大きく加算
            rewards[landed_indices] += dist_x 
            dones[landed_indices] = True
            
        timeout_indices = torch.where(self.step_counts >= self.max_steps)[0]
        if len(timeout_indices) > 0:
            dones[timeout_indices] = True

        # --- リセット処理 ---
        done_indices = torch.where(dones)[0]
        if len(done_indices) > 0:
            self._reset_idx(done_indices)
            
        return obs.cpu().numpy(), rewards.cpu().numpy(), dones.cpu().numpy(), infos
        
    def _reset_idx(self, env_ids):
        positions = torch.zeros((len(env_ids), 3), device="cuda:0")
        self.robots.set_joint_positions(positions, indices=env_ids)
        self.robots.set_joint_velocities(positions, indices=env_ids)
        self.robots.set_joint_position_targets(positions, indices=env_ids)
        
        # 変更点: 固定ゲインを再適用（アクションによって変更されないため一定に保たれます）
        self.current_kps[env_ids] = self.default_kp
        self.current_kds[env_ids] = self.default_kd
        
        self.robots.set_gains(
            kps=self.current_kps[env_ids],
            kds=self.current_kds[env_ids],
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

    def get_attr(self, attr_name, indices=None):
        return [getattr(self, attr_name) for _ in range(self.num_envs)]

    def set_attr(self, attr_name, value, indices=None):
        setattr(self, attr_name, value)

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        pass

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * self.num_envs
    
    def seed(self, seed=None):
        pass