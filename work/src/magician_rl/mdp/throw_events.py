from __future__ import annotations
import torch
import numpy as np
import isaaclab.utils.math as math_utils
from isaaclab.managers import SceneEntityCfg
from isaaclab.envs import ManagerBasedEnv
from isaaclab.assets import Articulation

def reset_joints_with_mimic(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    position_range: tuple[float, float],
    velocity_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """
    Resets the robot joints with random values, ENFORCING the mimic joint constraints.
    Magician: q_mimic_1 = -q_2, q_mimic_2 = -q_3
    This ensures the end-effector is horizontal at reset.
    """
    asset: Articulation = env.scene[asset_cfg.name]

    # 1. Randomize all joints first (to get base randomness)
    # We can use the default method logic or just sample from range.
    # asset.data.default_joint_pos is the reference.
    
    # Get joint limits or use provided range relative to default?
    # Usually reset_joints_by_scale scales the default position.
    # explicit_range means we sample uniformly in [min, max].
    # Let's assume we want to sample around default pose with some noise, 
    # OR simpler: just sample valid ranges for active joints.
    
    # Let's use the logic:
    # q = default * scale (where scale is random)
    
    # Sample scales
    range_len = position_range[1] - position_range[0]
    scales = torch.rand((len(env_ids), asset.num_joints), device=env.device) * range_len + position_range[0]
    
    # Current default positions
    default_pos = asset.data.default_joint_pos[env_ids]
    
    # Apply scale
    new_pos = default_pos * scales
    
    # Apply velocity
    vel_range_len = velocity_range[1] - velocity_range[0]
    new_vel = torch.rand((len(env_ids), asset.num_joints), device=env.device) * vel_range_len + velocity_range[0]

    # --- ENFORCE MIMIC CONSTRAINTS ---
    # We need to know indices.
    # Joint names: magician_joint_1, magician_joint_2, magician_joint_3, magician_joint_mimic_1, magician_joint_mimic_2
    # We find them dynamically to be safe.
    
    joint_names = asset.joint_names
    try:
        idx_2 = joint_names.index("magician_joint_2")
        idx_3 = joint_names.index("magician_joint_3")
        idx_mimic_1 = joint_names.index("magician_joint_mimic_1")
        idx_mimic_2 = joint_names.index("magician_joint_mimic_2")
        
        # Enforce: mimic = -source
        new_pos[:, idx_mimic_1] = -new_pos[:, idx_2]
        new_pos[:, idx_mimic_2] = -new_pos[:, idx_3]
        
        # Zero velocity for simplicity or mimic it too?
        new_vel[:, idx_mimic_1] = -new_vel[:, idx_2]
        new_vel[:, idx_mimic_2] = -new_vel[:, idx_3]
        
    except ValueError as e:
        print(f"[Warning] Joint name not found in reset_joints_with_mimic: {e}")

    # Set into simulation
    asset.write_joint_state_to_sim(new_pos, new_vel, env_ids=env_ids)


def reset_object_to_gripper(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    offset: tuple = (0.06, 0.0, -0.059), # Magician gripper offset
):
    """
    Resets the object position to be exactly at the gripper's suction point.
    This ensures that when the simulation starts, the object is in position for the FixedJoint to be created.
    """
    # Resolve assets
    object_asset = env.scene[object_cfg.name]
    robot_asset = env.scene[robot_cfg.name]

    # Find the end effector link index
    # We assume the link name is known or we search for it.
    # In this specific case, we know it's "magician_link_4" from actions.py context
    link_name = "magician_link_4"
    link_ids, _ = robot_asset.find_bodies(link_name)
    if len(link_ids) == 0:
        print(f"[Warning] reset_object_to_gripper: '{link_name}' not found.")
        return
    ee_link_idx = link_ids[0]

    # Get Robot EE State
    # shape: (num_envs, 3), (num_envs, 4)
    ee_pos_w = robot_asset.data.body_pos_w[env_ids, ee_link_idx]
    ee_quat_w = robot_asset.data.body_quat_w[env_ids, ee_link_idx]

    # Calculate Object Position
    # object_pos = ee_pos + quat_apply(ee_quat, offset)
    offset_tensor = torch.tensor(offset, device=env.device).unsqueeze(0).repeat(len(env_ids), 1)
    object_pos_w = ee_pos_w + math_utils.quat_apply(ee_quat_w, offset_tensor)
    
    # Object Orientation: Same as gripper (or aligned as needed)
    # Let's align it with gripper rotation for now.
    object_quat_w = ee_quat_w.clone()

    # Set State (Velocity Zero)
    object_asset.write_root_pose_to_sim(
        torch.cat([object_pos_w, object_quat_w], dim=-1),
        env_ids=env_ids
    )
    object_asset.write_root_velocity_to_sim(
        torch.zeros((len(env_ids), 6), device=env.device),
        env_ids=env_ids
    )
