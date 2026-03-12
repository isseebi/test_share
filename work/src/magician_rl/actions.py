from __future__ import annotations
import torch
import numpy as np
from pxr import Usd, UsdGeom, UsdPhysics, Gf
import isaaclab.utils.math as math_utils
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.envs import ManagerBasedEnv
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.utils import configclass
from isaaclab.assets import Articulation

# =============================================================================
# Magician Mimic Action
# =============================================================================
class MagicianMimicAction(JointPositionAction):
    """3つのアクティブ関節を操作し、2つのMimic関節を自動連動させるアクション"""
    def __init__(self, cfg: MagicianMimicActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._mimic_1_id, _ = self._asset.find_joints("magician_joint_mimic_1")
        self._mimic_2_id, _ = self._asset.find_joints("magician_joint_mimic_2")
        self._joint_2_idx = 1 
        self._joint_3_idx = 2 
        
        device = self._asset.device
        mimic_ids_list = []
        if len(self._mimic_1_id) > 0: mimic_ids_list.append(self._mimic_1_id[0])
        if len(self._mimic_2_id) > 0: mimic_ids_list.append(self._mimic_2_id[0])
        self._mimic_ids = torch.tensor(mimic_ids_list, device=device) if mimic_ids_list else None

    def apply_actions(self):
        super().apply_actions()
        if self._mimic_ids is not None:
            q_2_target = self.processed_actions[:, self._joint_2_idx]
            q_3_target = self.processed_actions[:, self._joint_3_idx]
            mimic_targets = torch.stack([-q_2_target, -q_3_target], dim=1)
            self._asset.set_joint_position_target(mimic_targets, joint_ids=self._mimic_ids)

            # B. Enforce State (Kinematic Reset to remove physics drift)
            # Find source joint indices in the full robot asset
            # self._joint_ids contains [idx_j1, idx_j2, idx_j3] corresponding to our action group
            source_j2_idx = self._joint_ids[1]
            source_j3_idx = self._joint_ids[2]
            
            # Read current pos/vel of source joints
            # Note: We use the simulation state from the physics buffer
            cur_pos = self._asset.data.joint_pos 
            cur_vel = self._asset.data.joint_vel
            
            q2_pos = cur_pos[:, source_j2_idx]
            q3_pos = cur_pos[:, source_j3_idx]
            q2_vel = cur_vel[:, source_j2_idx]
            q3_vel = cur_vel[:, source_j3_idx]
            
            # Compute mimic state: q_mimic = -q_source
            mimic_pos = torch.stack([-q2_pos, -q3_pos], dim=1)
            mimic_vel = torch.stack([-q2_vel, -q3_vel], dim=1)
            
            # Write to sim (Teleport mimic joints to match source)
            # This ensures geometric consistency is enforced at the beginning of every step
            self._asset.write_joint_state_to_sim(
                position=mimic_pos, 
                velocity=mimic_vel, 
                joint_ids=self._mimic_ids,
                env_ids=None # All envs
            )

@configclass
class MagicianMimicActionCfg(JointPositionActionCfg):
    class_type = MagicianMimicAction


# =============================================================================
# Magician Suction Action (Debug修正版)
# =============================================================================
import torch
import numpy as np
from pxr import Usd, UsdGeom, UsdPhysics, Gf
import isaaclab.utils.math as math_utils
from isaaclab.managers import ActionTerm
from isaaclab.envs import ManagerBasedEnv

class MagicianSuctionAction(ActionTerm):
    """
    Magicianの吸着アクションクラス (修正版)
    物理エンジン(Fabric/PhysX)のリアルタイム座標を取得して判定を行います。
    """
    def __init__(self, cfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._asset = env.scene[cfg.asset_name]
        self._target_object = env.scene[cfg.target_object_name]
        self.stage = env.scene.stage
        
        # --- 1. リンクインデックスの特定 ---
        # "magician_link_4" (手先リンク) のボディインデックスを取得
        link_name = "magician_link_4"
        link_ids, _ = self._asset.find_bodies(link_name)
        if len(link_ids) == 0:
            raise ValueError(f"Body '{link_name}' not found in robot asset!")
        self._ee_link_idx = link_ids[0]

        # --- 2. ターゲット名の抽出 (USDパス構築用) ---
        raw_prim_path = self._target_object.cfg.prim_path
        self._target_prim_name = raw_prim_path.split("/")[-1] # 例: "Object"

        # --- 3. データの準備 ---
        self.is_attached = torch.zeros(self._env.num_envs, dtype=torch.bool, device=self._asset.device)
        self._raw_actions = torch.zeros((self._env.num_envs, 1), device=self._asset.device)
        self._processed_actions = torch.zeros((self._env.num_envs, 1), device=self._asset.device)
        
        # オフセットをTensor化 (全環境分複製)
        # shape: (num_envs, 3)
        self._offset = torch.tensor(cfg.offset, device=self._asset.device).unsqueeze(0).repeat(self._env.num_envs, 1)

        self._debug_step_count = 0
        self.joint_paths = []
        self.sphere_prims = []

        self._setup_end_effectors()

    def _setup_end_effectors(self):
        """USD上に可視化用の赤い球体と、ジョイント用のパスを準備します"""
        for i in range(self._env.num_envs):
            env_path = f"/World/envs/env_{i}"
            robot_path = f"{env_path}/Robot"
            target_link_path = f"{robot_path}/magician_link_4"
            
            sphere_path = f"{target_link_path}/TipContactSphere"
            
            if not self.stage.GetPrimAtPath(sphere_path).IsValid():
                sphere_geom = UsdGeom.Sphere.Define(self.stage, sphere_path)
                sphere_geom.GetRadiusAttr().Set(self.cfg.radius)
                sphere_geom.GetDisplayColorAttr().Set([(1.0, 0.0, 0.0)])
                sphere_geom.AddTranslateOp().Set(Gf.Vec3d(*self.cfg.offset))
                # UsdPhysics.CollisionAPI.Apply(sphere_geom.GetPrim()) <--- 削除: これが爆発の原因
            
            self.sphere_prims.append(self.stage.GetPrimAtPath(sphere_path))
            self.joint_paths.append(f"{sphere_path}/AdhocFixedJoint")

    def apply_actions(self):
        self._debug_step_count += 1
        
        # --- A. 物理データからのリアルタイム座標計算 (Torch一括処理) ---
        
        # 1. ロボット手先(link_4)の位置と回転を取得
        # shape: (num_envs, 3), (num_envs, 4)
        ee_pos_w = self._asset.data.body_pos_w[:, self._ee_link_idx]
        ee_quat_w = self._asset.data.body_quat_w[:, self._ee_link_idx]
        
        # 2. オフセットを回転させて加算 -> 吸盤のワールド座標
        # sphere_pos = link_pos + (link_rot * offset_local)
        sphere_pos_w = ee_pos_w + math_utils.quat_apply(ee_quat_w, self._offset)
        
        # 3. ターゲットオブジェクトの座標を取得
        target_pos_w = self._target_object.data.root_pos_w
        
        # 4. 距離計算
        dists = torch.norm(sphere_pos_w - target_pos_w, dim=-1)

        # --- B. アクション入力判定 ---
        # 0.0 より大きければ吸着ONとみなす
        # suction_cmds = (self._processed_actions.squeeze(-1) > 0.0)
        
        # User requested to FORCE HOLD to learn velocity first.
        if self.cfg.force_hold:
             suction_cmds = torch.ones_like(self._processed_actions.squeeze(-1), dtype=torch.bool)
        else:
             suction_cmds = (self._processed_actions.squeeze(-1) > 0.0)

        # --- C. データのCPU転送 (for Loop処理) ---
        cmds_cpu = suction_cmds.cpu().numpy()
        attached_cpu = self.is_attached.cpu().numpy()
        dists_cpu = dists.cpu().numpy()
        sphere_pos_cpu = sphere_pos_w.cpu().numpy()

        # --- D. リアルタイムログ出力 (Env 0) ---
        # これで動いている座標が見えるはずです
        # print(f"[Step {self._debug_step_count}] Suction Pos: x={sphere_pos_cpu[0][0]:.4f}, y={sphere_pos_cpu[0][1]:.4f}, z={sphere_pos_cpu[0][2]:.4f}")

        # --- E. 個別環境のロジック処理 ---
        for i in range(self._env.num_envs):
            cmd = cmds_cpu[i]
            att = attached_cpu[i]
            dist = dists_cpu[i]

            # デバッグ (60ステップに1回詳細表示)
            # if i == 0 and self._debug_step_count % 60 == 0:
            #     print(f"--- [DEBUG] Env 0 Dist: {dist:.4f} (Thresh: {self.cfg.threshold}) | Cmd: {cmd} | Att: {att}")

            if cmd and not att:
                # 吸着コマンドON & まだ吸着していない
                if dist < self.cfg.threshold:
                    # print(f"[DEBUG Action] ATTACHING in env {i} (Dist: {dist:.4f})")
                    
                    # USDパスを構築してJoint作成
                    target_prim_path = f"/World/envs/env_{i}/{self._target_prim_name}"
                    target_prim = self.stage.GetPrimAtPath(target_prim_path)
                    
                    if target_prim.IsValid():
                        self._create_joint(i, target_prim)
                        self.is_attached[i] = True
                    else:
                        print(f"[Error] Target prim not found at {target_prim_path}")

            elif not cmd and att:
                # 吸着コマンドOFF & 吸着中 -> 解放
                # print(f"[DEBUG Action] DETACHING in env {i}")
                self._remove_joint(i)
                self.is_attached[i] = False

    def _create_joint(self, env_idx, target_prim):
        try:
            # --- 相対位置計算 (前回と同じ) ---
            ee_pos_w = self._asset.data.body_pos_w[env_idx, self._ee_link_idx]
            ee_quat_w = self._asset.data.body_quat_w[env_idx, self._ee_link_idx]
            
            offset_vec = self._offset[env_idx]
            sphere_pos_w = ee_pos_w + math_utils.quat_apply(ee_quat_w, offset_vec)
            sphere_quat_w = ee_quat_w.clone()

            target_pos_w = self._target_object.data.root_pos_w[env_idx]
            target_quat_w = self._target_object.data.root_quat_w[env_idx]

            local_pos, local_rot = math_utils.subtract_frame_transforms(
                target_pos_w.unsqueeze(0), target_quat_w.unsqueeze(0),
                sphere_pos_w.unsqueeze(0), sphere_quat_w.unsqueeze(0)
            )
            
            local_pos_list = local_pos[0].cpu().numpy().tolist()
            local_rot_list = local_rot[0].cpu().numpy().tolist()

            # --- 物理安定化 (速度リセット) ---
            self._target_object.write_root_velocity_to_sim(
                torch.zeros((1, 6), device=self._asset.device), 
                env_ids=torch.tensor([env_idx], device=self._asset.device)
            )

            # --- ジョイント定義 ---
            joint = UsdPhysics.FixedJoint.Define(self.stage, self.joint_paths[env_idx])
            
            joint.CreateBody0Rel().SetTargets([self.sphere_prims[env_idx].GetPath()])
            joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0, 0, 0))
            joint.CreateLocalRot0Attr().Set(Gf.Quatf(1, 0, 0, 0))
            
            joint.CreateBody1Rel().SetTargets([target_prim.GetPath()])
            joint.CreateLocalPos1Attr().Set(Gf.Vec3f(*local_pos_list))
            joint.CreateLocalRot1Attr().Set(Gf.Quatf(local_rot_list[0], local_rot_list[1], local_rot_list[2], local_rot_list[3]))
            
            joint.CreateExcludeFromArticulationAttr().Set(True)

            # ★★★ 追加: 安全装置 (Break Force) ★★★
            # 無理な力がかかったらジョイントが壊れるようにする (爆発防止の切り札)
            # 値は十分大きく、かつ無限ではない値 (例: 1000.0ニュートン)
            # User requested to keep this. Setting to 10000.0 to allow throwing but avoid startup glitches.
            joint.CreateBreakForceAttr().Set(10000.0) 
            joint.CreateBreakTorqueAttr().Set(10000.0)
            
        except Exception as e:
            print(f"[Critical Error] Failed to create joint: {e}")

    def _remove_joint(self, env_idx):
        joint_path = self.joint_paths[env_idx]
        if self.stage.GetPrimAtPath(joint_path).IsValid():
            self.stage.RemovePrim(joint_path)

    def reset(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = torch.arange(self._env.num_envs, device=self._asset.device)
        
        # リセットされた環境の吸着を解除
        ids_cpu = env_ids.cpu().numpy()
        for i in ids_cpu:
            if self.is_attached[i]:
                self._remove_joint(i)
                self.is_attached[i] = False

        if env_ids is None:
            # First time reset or full reset
            if self.cfg.start_attached:
                # Force attach for all envs (simplification, ideally only for reset envs if logic allows)
                # But actions.reset is called with env_ids.
                pass
        
        # Handle start_attached for reset environments
        if self.cfg.start_attached:
             # We need to make sure the object is at the gripper BEFORE creating the joint
             # This assumes the object has been teleported to the gripper by an EventTerm already.
             # We just create the joint here.
             ids_cpu = env_ids.cpu().numpy()
             for i in ids_cpu:
                 # Only attach if not already attached (though reset usually implies clear state)
                 if not self.is_attached[i]:
                     # We assume object is close enough because of reset_object_to_gripper event
                     # But we should find the target prim.
                     stage_idx = i
                     target_prim_path = f"/World/envs/env_{stage_idx}/{self._target_prim_name}"
                     target_prim = self.stage.GetPrimAtPath(target_prim_path)
                     if target_prim.IsValid():
                         self._create_joint(stage_idx, target_prim)
                         self.is_attached[stage_idx] = True

    @property
    def action_dim(self) -> int: return 1
    @property
    def raw_actions(self) -> torch.Tensor: return self._raw_actions
    @property
    def processed_actions(self) -> torch.Tensor: return self._processed_actions
    
    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        self._processed_actions[:] = actions

@configclass
class MagicianSuctionActionCfg(ActionTermCfg):
    asset_name: str = "robot"
    target_object_name: str = "object"
    threshold: float = 0.01 # テスト用（本番は0.035程度に戻す）
    offset: tuple = (0.06, 0.0, -0.059)
    radius: float = 0.005
    start_attached: bool = False
    force_hold: bool = False # New: Force hold for velocity training
    class_type: type = MagicianSuctionAction