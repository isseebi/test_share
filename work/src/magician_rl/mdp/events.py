from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def reset_robot_and_object_coupled(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    robot_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    joint_range: dict[str, tuple[float, float]],
    base_offset: tuple[float, float, float] = (0.0, 0.0, 0.0), # Robot base offset
    suction_offset: tuple[float, float, float] = (0.06, 0.0, -0.059),
    cylinder_height: float = 0.05,
    l2: float = 0.135,
    l3: float = 0.147,
):
    """
    Resets the robot joints to a random position and places the object 
    immediately below the suction cup (End Effector).
    
    This function performs simplified Forward Kinematics for the Dobot Magician 
    to determine the EE position derived from the sampled joint angles.
    """
    
    # 1. Sample random joint positions
    # We assume 3 DOF for the arm: joint_1, joint_2, joint_3
    # joint_range should keys "j1", "j2", "j3" or similar, or we just rely on order.
    # Ideally, use the joint names from the robot.
    
    robot = env.scene[robot_cfg.name]
    object = env.scene[object_cfg.name]
    
    # Resolve joint indices
    # We assume the robot has ["magician_joint_1", "magician_joint_2", "magician_joint_3"]
    # We can get them from the asset
    
    # Create random joints tensor
    # We need to know the number of joints. Magician has 5 (incl mimics).
    # But we only randomize the active ones.
    
    # Using explicit names for clarity and safety
    joint_names = ["magician_joint_1", "magician_joint_2", "magician_joint_3"]
    joint_ids, _ = robot.find_joints(joint_names)
    
    # Generate random values
    # joint_range should be provided as implicit args or explicit
    # Default fallback if not in dict
    range_j1 = joint_range.get("j1", (-0.4, 0.4))
    range_j2 = joint_range.get("j2", (0.0, 0.5))
    range_j3 = joint_range.get("j3", (-0.2, 0.2)) # Relative?
    
    num_envs = len(env_ids)
    
    dl_j1 = range_j1[1] - range_j1[0]
    dl_j2 = range_j2[1] - range_j2[0]
    dl_j3 = range_j3[1] - range_j3[0]
    
    q1 = torch.rand(num_envs, device=env.device) * dl_j1 + range_j1[0]
    q2 = torch.rand(num_envs, device=env.device) * dl_j2 + range_j2[0]
    q3 = torch.rand(num_envs, device=env.device) * dl_j3 + range_j3[0]
    
    # 2. Calculate EE Position via Forward Kinematics
    # Magician kinematics (Approximation based on dobot_inv_kin.py structure)
    # r = l2 * cos(q2) + l3 * cos(q2 + q3)  (assuming q3 is relative to link2)
    # z = l2 * sin(q2) + l3 * sin(q2 + q3)
    
    # Note: joint 0 is Base Rotation (q1)
    #       joint 1 is Rear Arm (q2)
    #       joint 2 is Forearm (q3) - check if it is relative or absolute in Sim.
    #       In most URDFs, joints are relative to parent.
    #       So angle of link3 is q2 + q3.
    
    # Check if simulation needs mimic handling manually here? 
    # The simulation's physics handles mimic if we set targets? 
    # But for initialization we set state directly.
    # Mimic joints: magician_joint_mimic_1 = -q2, mimic_2 = -q3 (usually).
    # Let's verify behavior later. For now assume standard chain.
    
    cos_q2 = torch.cos(q2)
    sin_q2 = torch.sin(q2)
    cos_q2q3 = torch.cos(q2 + q3)
    sin_q2q3 = torch.sin(q2 + q3)
    
    r = l2 * cos_q2 + l3 * cos_q2q3
    z = l2 * sin_q2 + l3 * sin_q2q3
    
    x = r * torch.cos(q1)
    y = r * torch.sin(q1)
    
    # Add Base Offset (Robot Base height from ground)
    # Need to verify if z=0 is ground or robot base. 
    # In config, robot pos is (0,0,0). The base mesh has some height. 
    # J2 is usually ~138mm above base.
    base_height = base_offset[2] # e.g. 0.138
    
    ee_pos_x = x + base_offset[0]
    ee_pos_y = y + base_offset[1]
    ee_pos_z = z + base_height
    
    # Apply Suction Offset (Local to EE frame)
    # EE frame orientation:
    # If Magician keeps EE level, rotation is just q1 around Z.
    # Rot = RotationZ(q1)
    
    # Transform suction offset by q1
    # suction_offset is (0.06, 0.0, -0.059)
    sx, sy, sz = suction_offset
    
    # Rotation Z(q1) * (sx, sy, sz)
    # new_sx = sx * cos(q1) - sy * sin(q1)
    # new_sy = sx * sin(q1) + sy * cos(q1)
    # new_sz = sz
    
    final_x = ee_pos_x + (sx * torch.cos(q1) - sy * torch.sin(q1))
    final_y = ee_pos_y + (sx * torch.sin(q1) + sy * torch.cos(q1))
    final_z = ee_pos_z + sz
    
    target_x = 0.2
    target_y = 0.0
    target_z = cylinder_height / 2.0  # 地面に接するように配置

    final_x = torch.full((num_envs,), target_x, device=env.device)
    final_y = torch.full((num_envs,), target_y, device=env.device)
    final_z = torch.full((num_envs,), target_z, device=env.device)
    # Adjust for object origin (Cylinder center is at h/2)
    # We want top of cylinder to touch suction (or close)
    # Cylinder pos = final_z - (h/2)
    # object_z = final_z - (cylinder_height / 2.0)
    object_z = torch.full((num_envs,), cylinder_height / 2.0, device=env.device)    # 3. Apply to Simulation
    
    # Set Robot Joints
    # Construct full joint state (including mimics if needed, or rely on sim to snap them? Sim won't snap on reset unless we set them).
    # Mimic logic: mimic_1 = -q2, mimic_2 = -q3 (Based on typical Magician URDF)
    
    joint_pos_tensor = robot.data.default_joint_pos[env_ids].clone()
    
    # Fill in active joints
    # Map indices: 
    # We need to look up indices in the full joint list
    full_joint_names = robot.data.joint_names
    # This is slightly slow if done every reset, but safe. 
    # Better: pre-compute indices, but in functional style we do this:
    
    idx_1 = full_joint_names.index("magician_joint_1")
    idx_2 = full_joint_names.index("magician_joint_2")
    idx_3 = full_joint_names.index("magician_joint_3")
    
    try:
        idx_m1 = full_joint_names.index("magician_joint_mimic_1")
        idx_m2 = full_joint_names.index("magician_joint_mimic_2")
        has_mimic = True
    except ValueError:
        has_mimic = False
    
    joint_pos_tensor[:, idx_1] = q1
    joint_pos_tensor[:, idx_2] = q2
    joint_pos_tensor[:, idx_3] = q3
    
    if has_mimic:
        joint_pos_tensor[:, idx_m1] = -q2
        joint_pos_tensor[:, idx_m2] = -q3
        
    robot.write_joint_state_to_sim(joint_pos_tensor, torch.zeros_like(joint_pos_tensor), env_ids=env_ids)
    
    # Set Object Position
    # Shape: (num_envs, 3)
    # The calculated final_x/y/z are in the Robot/Environment Local Frame.
    # We need to add the Environment Origin to get World Frame positions.
    
    local_pos = torch.stack([final_x, final_y, object_z], dim=1)
    
    # Get Environment Origins
    env_origins = env.scene.env_origins[env_ids]
    
    # World Position = Env Origin + Local Position
    object_pos_w = env_origins + local_pos
    
    # Orientation: Upright (1, 0, 0, 0)
    object_rot = torch.zeros((num_envs, 4), device=env.device)
    object_rot[:, 0] = 1.0
    
    object.write_root_state_to_sim(
        torch.cat([object_pos_w, object_rot, torch.zeros((num_envs, 6), device=env.device)], dim=1),
        env_ids=env_ids
    )

    # 最初の環境 (env_id = 0) の情報を代表して表示
    if 0 in env_ids:
        # env_ids の中での 0 番目のインデックスを取得
        idx_zero = (env_ids == 0).nonzero(as_tuple=True)[0]
        
        # 吸盤の座標 (final_x, final_y, final_z)
        # 物体の座標 (final_x, final_y, object_z)
        print("-" * 30)
        print(f"[Env 0] Robot & Object Coupled Reset")
        print(f"  - Joint Angles: q1={q1[idx_zero].item():.3f}, q2={q2[idx_zero].item():.3f}, q3={q3[idx_zero].item():.3f}")
        print(f"  - Suction Tip (Local): x={final_x[idx_zero].item():.3f}, y={final_y[idx_zero].item():.3f}, z={final_z[idx_zero].item():.3f}")
        print(f"  - Object Center (Local): x={final_x[idx_zero].item():.3f}, y={final_y[idx_zero].item():.3f}, z={object_z[idx_zero].item():.3f}")
        print("-" * 30)
    # ---------------------