import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv

from omni.isaac.core import World
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.utils.prims import define_prim
import omni.kit.commands
from omni.isaac.core.prims import XFormPrim
from omni.isaac.core.utils.rotations import quat_to_euler_angles

from omni.isaac.core.prims import RigidPrimView

class A1WalkVecEnv(VecEnv):
    def __init__(self, num_envs, urdf_path, headless=False):
        # Action: 12 joint positions (4 legs * 3 joints)
        self.num_joints = 12
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.num_joints,), dtype=np.float32)
        
        # Observation: 
        # Base Pos (3) + Base Quat (4) + Base Lin Vel (3) + Base Ang Vel (3) + Joint Pos (12) + Joint Vel (12) + Foot Contact (4) + Proj Gravity (3) + Phase (2) = 46
        self.single_obs_dim = 3 + 4 + 3 + 3 + self.num_joints + self.num_joints + 4 + 3 + 2
        self.history_len = 3
        self.obs_dim = self.single_obs_dim * self.history_len
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)
        
        super().__init__(num_envs, self.observation_space, self.action_space)
        
        self.urdf_path = urdf_path
        self.headless = headless
        self._setup_world()
        
        # Default Pose for A1 (Stand)
        # Isaac Sim DOF Order: [FL_hip, FR_hip, RL_hip, RR_hip, FL_thigh, FR_thigh, RL_thigh, RR_thigh, FL_calf, FR_calf, RL_calf, RR_calf]
        # Hip: 0.0, Thigh: 0.9, Calf: -1.8
        self.default_dof_pos = torch.tensor(
            [0.0] * 4 + [0.9] * 4 + [-1.8] * 4, device="cuda:0", dtype=torch.float32
        )

        self.default_kp = 100.0 # Adjusted for A1 (from control_a1.py)
        self.default_kd = 1.0   # Adjusted for A1 (from control_a1.py)

        self.current_kps = torch.ones((self.num_envs, self.num_joints), device="cuda:0") * self.default_kp
        self.current_kds = torch.ones((self.num_envs, self.num_joints), device="cuda:0") * self.default_kd
        
        # Gait Parameters
        self.gait_phase = torch.zeros(self.num_envs, device="cuda:0")
        self.gait_freq = 1.5 
        self.dt = 1.0 / 60.0 
        
        # History Buffer
        self.obs_history = torch.zeros((self.num_envs, self.obs_dim), device="cuda:0")
        
        # Action Smoothing
        self.actions_filtered = torch.zeros((self.num_envs, self.num_joints), device="cuda:0")
        self.prev_actions_filtered = torch.zeros((self.num_envs, self.num_joints), device="cuda:0")
        self.action_alpha = 0.8 # Smoothing factor (0.8 * new + 0.2 * old)

        # Domain Randomization
        self.push_interval = 100 # Steps between pushes
        self.last_push_step = 0
        
    def _setup_world(self):
        self.world = World(backend="torch", device="cuda:0")
        physx = self.world.get_physics_context()
        physx.set_gpu_found_lost_aggregate_pairs_capacity(20000) 
        physx.set_gpu_total_aggregate_pairs_capacity(20000)

        self.world.scene.add_default_ground_plane()
        
        # --- Friction ---
        from omni.isaac.core.materials import PhysicsMaterial
        from pxr import UsdShade
        
        self.high_friction_material = PhysicsMaterial(
            prim_path="/World/Physics_Materials/HighFriction",
            name="high_friction",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0
        )
        
        ground_prim = self.world.stage.GetPrimAtPath("/World/defaultGroundPlane")
        mat_prim = self.world.stage.GetPrimAtPath("/World/Physics_Materials/HighFriction")
        material = UsdShade.Material(mat_prim)
        UsdShade.MaterialBindingAPI(ground_prim).Bind(material, UsdShade.Tokens.strongerThanDescendants)
        
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
        
        # --- Bind Friction to Robot Feet ---
        # A1 feet names: FL_foot, FR_foot, RL_foot, RR_foot
        feet_names = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
        
        # 2. Manual Cloning
        self.env_pos = []
        spacing = 3.0
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
        
        # 3. Create Views
        # Robot path: /World/envs/env_*/a1
        self.robots = ArticulationView(prim_paths_expr="/World/envs/env_*/a1", name="robots_view")
        self.world.scene.add(self.robots)
        
        # Feet path: /World/envs/env_*/a1/.*_foot
        # Using regex to catch all feet
        self.feet = RigidPrimView(prim_paths_expr="/World/envs/env_*/a1/.*foot", name="feet_view", reset_xform_properties=False, track_contact_forces=True)
        self.world.scene.add(self.feet)
        
        self.world.reset()
        
        self.max_steps = 500
        self.step_counts = torch.zeros(self.num_envs, device="cuda:0")


    def _import_urdf(self, prim_path):
        status, import_config = omni.kit.commands.execute("URDFCreateImportConfig")
        import_config.merge_fixed_joints = False
        import_config.convex_decomp = False
        import_config.fix_base = False 
        import_config.make_default_prim = True
        
        result = omni.kit.commands.execute(
            "URDFParseAndImportFile",
            urdf_path=self.urdf_path,
            import_config=import_config,
        )
        
        if result[0]:
            # The robot is imported at /a1 (as per URDF name)
            # We need to move it to prim_path/a1
            omni.kit.commands.execute("MovePrim", path_from="/a1", path_to=f"{prim_path}/a1")
        else:
            print("URDF Import failed")
        
    def step_async(self, actions):
        # actions: [joint_pos_targets(12)] 
        actions_tensor = torch.tensor(actions, device="cuda:0", dtype=torch.float32)
        
        # Action Smoothing (EMA)
        self.actions_filtered = self.action_alpha * actions_tensor + (1.0 - self.action_alpha) * self.prev_actions_filtered
        self.prev_actions_filtered = self.actions_filtered.clone()
        
        # Residual Control
        self.pos_targets = self.default_dof_pos + self.actions_filtered * 0.5
        
    def _quat_to_euler_torch(self, quat):
        w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = torch.atan2(sinr_cosp, cosr_cosp)
        sinp = 2 * (w * y - z * x)
        pitch = torch.where(torch.abs(sinp) >= 1, torch.sign(sinp) * np.pi / 2, torch.asin(sinp))
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = torch.atan2(siny_cosp, cosy_cosp)
        return roll, pitch, yaw

    def _rotate_vec_by_quat_torch(self, vec, quat):
        w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        vx, vy, vz = vec[:, 0], vec[:, 1], vec[:, 2]
        cx = y * vz - z * vy
        cy = z * vx - x * vz
        cz = x * vy - y * vx
        t2x = cx + w * vx
        t2y = cy + w * vy
        t2z = cz + w * vz
        ccx = y * t2z - z * t2y
        ccy = z * t2x - x * t2z
        ccz = x * t2y - y * t2x
        rx = vx + 2 * ccx
        ry = vy + 2 * ccy
        rz = vz + 2 * ccz
        return torch.stack([rx, ry, rz], dim=1)

    def step_wait(self):
        # --- 1. Apply Actions ---
        self.robots.set_joint_position_targets(self.pos_targets)
        
        # --- Domain Randomization: Push ---
        if self.step_counts[0] % self.push_interval == 0:
             root_vels = self.robots.get_velocities()
             root_vels[:, :2] += (torch.rand((self.num_envs, 2), device="cuda:0") - 0.5) * 1.0 
             self.robots.set_velocities(root_vels)

        
        # --- 2. Step Simulation ---
        self.world.step(render=not self.headless)
        self.step_counts += 1
        
        # Update Gait Phase
        self.gait_phase = (self.gait_phase + self.dt * self.gait_freq) % 1.0
        
        # --- 3. Get Observations ---
        obs_single = self._compute_obs()
        
        # Update History
        self.obs_history = torch.roll(self.obs_history, shifts=-self.single_obs_dim, dims=1)
        self.obs_history[:, -self.single_obs_dim:] = obs_single
        
        # --- 4. Rewards ---
        rewards = self._compute_rewards()
        
        # --- 5. Reset Logic ---
        root_pos, _ = self.robots.get_world_poses()
        is_fallen = root_pos[:, 2] < 0.15
        time_out = self.step_counts >= self.max_steps
        dones = is_fallen | time_out
        
        rewards[is_fallen] -= 10.0 
        
        done_indices = torch.where(dones)[0]
        if len(done_indices) > 0:
            self._reset_idx(done_indices)
        
        infos = [{} for _ in range(self.num_envs)]
            
        return self.obs_history.cpu().numpy(), rewards.cpu().numpy(), dones.cpu().numpy(), infos
        
    def _compute_obs(self):
        joint_pos = self.robots.get_joint_positions()
        joint_vel = self.robots.get_joint_velocities()
        root_pos, root_rot = self.robots.get_world_poses()
        root_vels = self.robots.get_velocities()
        
        forces = self.feet.get_net_contact_forces()
        if forces is None:
            forces = torch.zeros((self.num_envs, 4, 3), device="cuda:0")
        else:
            forces = forces.view(self.num_envs, 4, 3)
        in_contact = (torch.norm(forces, dim=2) > 0.01).float()
        
        gravity_vec = torch.tensor([0.0, 0.0, -1.0], device="cuda:0").repeat(self.num_envs, 1)
        w, x, y, z = root_rot[:, 0], root_rot[:, 1], root_rot[:, 2], root_rot[:, 3]
        inv_rot = torch.stack([w, -x, -y, -z], dim=1)
        projected_gravity = self._rotate_vec_by_quat_torch(gravity_vec, inv_rot)
        
        base_lin_vel = self._rotate_vec_by_quat_torch(root_vels[:, :3], inv_rot)
        base_ang_vel = self._rotate_vec_by_quat_torch(root_vels[:, 3:], inv_rot)
        
        phase_sin = torch.sin(2 * np.pi * self.gait_phase).unsqueeze(1)
        phase_cos = torch.cos(2 * np.pi * self.gait_phase).unsqueeze(1)
        
        obs = torch.cat([root_pos, root_rot, base_lin_vel, base_ang_vel, joint_pos, joint_vel, in_contact, projected_gravity, phase_sin, phase_cos], dim=1)
        return obs

    def _compute_rewards(self):
        rewards = torch.zeros(self.num_envs, device="cuda:0")
        
        root_pos, root_rot = self.robots.get_world_poses()
        root_vels = self.robots.get_velocities()
        
        w, x, y, z = root_rot[:, 0], root_rot[:, 1], root_rot[:, 2], root_rot[:, 3]
        inv_rot = torch.stack([w, -x, -y, -z], dim=1)
        base_lin_vel = self._rotate_vec_by_quat_torch(root_vels[:, :3], inv_rot)
        base_ang_vel = self._rotate_vec_by_quat_torch(root_vels[:, 3:], inv_rot)
        
        gravity_vec = torch.tensor([0.0, 0.0, -1.0], device="cuda:0").repeat(self.num_envs, 1)
        projected_gravity = self._rotate_vec_by_quat_torch(gravity_vec, inv_rot)

        # 1. Linear Velocity Tracking (Forward)
        target_vel = 0.3
        lin_vel_error = torch.square(base_lin_vel[:, 0] - target_vel)
        rewards += 2.0 * torch.exp(-lin_vel_error / 0.25)
        
        # 2. Angular Velocity Penalty (Keep straight)
        ang_vel_error = torch.square(base_ang_vel[:, 2]) # Yaw rate
        rewards -= 0.5 * ang_vel_error
        
        # 3. Base Height
        target_height = 0.30
        height_error = torch.square(root_pos[:, 2] - target_height)
        rewards += 1.0 * torch.exp(-height_error / 0.01)
        
        # 4. Gravity Alignment (Upright)
        rewards -= 5.0 * torch.sum(torch.square(projected_gravity[:, :2]), dim=1)
        
        # 5. Survival Reward
        rewards += 0.1
        
        # 6. Foot Clearance (Reward for lifting feet)
        forces = self.feet.get_net_contact_forces()
        if forces is None: forces = torch.zeros((self.num_envs, 4, 3), device="cuda:0")
        else: forces = forces.view(self.num_envs, 4, 3)
        in_contact = (torch.norm(forces, dim=2) > 0.01).float()
        
        num_feet_in_air = 4.0 - torch.sum(in_contact, dim=1)
        rewards += 0.1 * num_feet_in_air
        
        # 7. Action Smoothness
        rewards -= 0.05 * torch.sum(torch.square(self.actions_filtered - self.prev_actions_filtered), dim=1)
        
        # 8. Joint Regularization
        joint_pos = self.robots.get_joint_positions()
        rewards -= 0.05 * torch.sum(torch.square(joint_pos - self.default_dof_pos), dim=1)

        return rewards
        
    def _reset_idx(self, env_ids):
        num_resets = len(env_ids)
        
        j_pos = (torch.rand((num_resets, self.num_joints), device="cuda:0") * 0.2) - 0.1 + self.default_dof_pos[0] # Add default pose offset
        # Actually, better to just randomize around default pose
        j_pos = self.default_dof_pos.repeat(num_resets, 1) + (torch.rand((num_resets, self.num_joints), device="cuda:0") * 0.2 - 0.1)

        j_vel = torch.zeros((num_resets, self.num_joints), device="cuda:0")
        
        self.robots.set_joint_positions(j_pos, indices=env_ids)
        self.robots.set_joint_velocities(j_vel, indices=env_ids)
        self.robots.set_joint_position_targets(j_pos, indices=env_ids)
        
        indices = env_ids.cpu().numpy()
        env_origins = self.env_pos[indices]
        
        root_pos = torch.tensor(env_origins, device="cuda:0", dtype=torch.float32)
        root_pos[:, 2] = 0.35 # Slightly higher for A1 start
        
        root_rot = torch.zeros((num_resets, 4), device="cuda:0")
        root_rot[:, 0] = 1.0 
        
        root_vel = torch.zeros((num_resets, 6), device="cuda:0")
        
        self.robots.set_world_poses(positions=root_pos, orientations=root_rot, indices=env_ids)
        self.robots.set_velocities(root_vel, indices=env_ids)
        
        self.step_counts[env_ids] = 0
        
        self.prev_actions_filtered[env_ids] = 0.0
        self.actions_filtered[env_ids] = 0.0
        
        if not hasattr(self, 'gait_phase'): self.gait_phase = torch.zeros(self.num_envs, device="cuda:0")
        self.gait_phase[env_ids] = torch.rand(num_resets, device="cuda:0")
        
        obs_single = self._compute_obs()
        
        reset_obs = obs_single[env_ids] 
        reset_history = reset_obs.repeat(1, self.history_len) 
        self.obs_history[env_ids] = reset_history
        
    def reset(self):
        self._reset_idx(torch.arange(self.num_envs, device="cuda:0"))
        return self.obs_history.cpu().numpy()
        
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
