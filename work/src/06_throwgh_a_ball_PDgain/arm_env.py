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
        # Action: 3 joint positions + 3 Kp + 3 Kd = 9 (変更なし)
        # Limits approx -1 to 1 (will be scaled)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(9,), dtype=np.float32)
        
        # Observation: 3 joint pos + 3 joint vel + 3 ball pos = 9 
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32)
        
        super().__init__(num_envs, self.observation_space, self.action_space)
        
        self.urdf_path = urdf_path
        self.headless = headless
        self._setup_world()
        
        # Initialize Kp and Kd storage for actions (observationには使いませんが、内部計算に必要)
        self.current_kps = torch.ones((self.num_envs, 3), device="cuda:0") * 1000.0 
        self.current_kds = torch.ones((self.num_envs, 3), device="cuda:0") * 100.0

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
        # actions: [pos_targets(3), kp_raw(3), kd_raw(3)]
        actions_tensor = torch.tensor(actions, device="cuda:0", dtype=torch.float32)
        
        # 1. Position Targets
        self.pos_targets = actions_tensor[:, 0:3] * np.pi
        
        # 2. Kp (アクションとしては残っているため計算して更新する)
        self.current_kps = (actions_tensor[:, 3:6] + 1.0) / 2.0 * 10000.0
        
        # 3. Kd (アクションとしては残っているため計算して更新する)
        self.current_kds = (actions_tensor[:, 6:9] + 1.0) / 2.0 * 1000.0
        
    def step_wait(self):
        # --- 前半の処理（アクション適用とデータ取得）はそのまま ---
        self.robots.set_joint_position_targets(self.pos_targets)
        self.robots.set_gains(kps=self.current_kps, kds=self.current_kds)
        
        self.world.step(render=not self.headless)
        self.step_counts += 1
        
        joint_pos = self.robots.get_joint_positions()
        joint_vel = self.robots.get_joint_velocities()
        ball_pos, _ = self.balls.get_world_poses()
        ball_velocities = self.balls.get_velocities() # ボールの速度を取得
        
        obs = torch.cat([joint_pos, joint_vel, ball_pos], dim=1)
        
        rewards = torch.zeros(self.num_envs, device="cuda:0")
        dones = torch.zeros(self.num_envs, device="cuda:0", dtype=torch.bool)
        
        ball_z = ball_pos[:, 2]
        ball_vel_x = ball_velocities[:, 0] # 前方向の速度
        
        env_origins = torch.tensor(self.env_pos, device="cuda:0", dtype=torch.float32)
        relative_ball_pos = ball_pos - env_origins
        dist_x = relative_ball_pos[:, 0]

        # --- 報酬の設計 ---

        # 1. 生存報酬 / タイムペナルティ (早く投げさせるため)
        rewards -= 0.01 

        # 2. 飛行中の報酬 (Shaping Reward)
        # ボールが一定以上の高さにあり、かつ前方に速度を持っている場合に加点
        # これにより「地面に置く」よりも「空中に放り出す」方が得になる
        # is_flying = ball_z > 0.2
        # rewards += torch.where(is_flying, ball_vel_x * 0.1, torch.zeros_like(rewards))

        # 3. 着地時の大きな報酬
        has_landed = ball_z < 0.05
        landed_indices = torch.where(has_landed)[0]
        
        if len(landed_indices) > 0:
            # 着地時の飛距離を評価（2乗にすることで、遠くへ飛ばすほど加速度的に報酬が増える）
            # さらに、着地時に「高い位置から落ちてきたか」を考慮するとより投げやすくなる
            final_dist = dist_x[landed_indices]
            rewards[landed_indices] += final_dist * 10.0 
            dones[landed_indices] = True

        # タイムアウト処理 (200ステップ超えたら終了)
        timeout_indices = torch.where(self.step_counts >= self.max_steps)[0]
        if len(timeout_indices) > 0:
            dones[timeout_indices] = True

        # --- 後半の処理（リセット等）はそのまま ---
        done_indices = torch.where(dones)[0]
        if len(done_indices) > 0:
            self._reset_idx(done_indices)
            
        return obs.cpu().numpy(), rewards.cpu().numpy(), dones.cpu().numpy(), [{}] * self.num_envs
        
    def _reset_idx(self, env_ids):
        positions = torch.zeros((len(env_ids), 3), device="cuda:0")
        self.robots.set_joint_positions(positions, indices=env_ids)
        self.robots.set_joint_velocities(positions, indices=env_ids)
        self.robots.set_joint_position_targets(positions, indices=env_ids)
        
        default_kp = 5000.0
        default_kd = 500.0
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
        
        # Observation: [joint_pos(3), joint_vel(3), ball_pos(3)] = 9
        # 変更点: kp, kd を除外
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