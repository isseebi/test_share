# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import combine_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_distance_reward(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    gripper_action_name: str = "gripper_action",
    attached_weight: float = 0.0,
    detached_weight: float = 1.0,
) -> torch.Tensor:
    """
    Reward distance from robot with varying weights depending on attachment status.
    - If attached: reward = attached_weight * distance
    - If detached: reward = detached_weight * distance
    This encourages the robot to release the object to gain higher rewards.
    """
    object: RigidObject = env.scene[object_cfg.name]
    robot: RigidObject = env.scene[robot_cfg.name]
    
    # Calculate distance from robot base (or origin, but robot base is better if robot moves)
    # Robot is fixed base, so let's use origin (0,0,0) or robot root.
    # Object root pos w: (num_envs, 3)
    # Robot root pos w: (num_envs, 3)
    # XY distance
    dist = torch.norm(object.data.root_pos_w[:, :2] - robot.data.root_pos_w[:, :2], dim=-1)
    
    # Check attached status
    try:
        # Use _terms to access action term (private but necessary here)
        if hasattr(env.action_manager, "_terms"):
             action_term = env.action_manager._terms[gripper_action_name]
        else:
             action_term = env.action_manager._terms[gripper_action_name]
        is_attached = action_term.is_attached
    except KeyError:
        # If action term not found or doesn't have attribute, assume attached? Or detached?
        # Actually safer to assume detached or default to detached_weight if we can't tell.
        # But for this task, we know the action term name.
        is_attached = torch.zeros_like(dist, dtype=torch.bool)

    # Compute weighted reward
    rewards = torch.where(is_attached, dist * attached_weight, dist * detached_weight)
    
    return rewards

def object_throwing_velocity_reward(
    env: ManagerBasedRLEnv, 
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    gripper_action_name: str = "gripper_action",
) -> torch.Tensor:
    """Reward the agent for throwing the object (high velocity) ONLY when detached."""
    object: RigidObject = env.scene[object_cfg.name]
    
    # Check attached status
    try:
        if hasattr(env.action_manager, "_terms"):
             action_term = env.action_manager._terms[gripper_action_name]
        else:
             action_term = env.action_manager._terms[gripper_action_name]
        is_attached = action_term.is_attached
    except KeyError:
        is_attached = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        
    # Velocity magnitude or specific direction?
    # Let's reward magnitude in XY plane to be direction agnostic (as long as it flies away)
    # Or enforce +X direction?
    # Since "throw as far as possible" usually implies any direction or a specific one.
    # User didn't specify direction, but typically forward (+X) is standard.
    # Let's reward velocity magnitude away from robot? Or simple speed.
    # Let's use simple speed for now, or projection onto a vector?
    
    # Let's just reward speed.
    speed = torch.norm(object.data.root_lin_vel_w[:, :2], dim=-1)
    
    # Only reward if NOT attached. If attached, speed usually comes from robot moving arm.
    # We want "release velocity".
    rewards = torch.where(is_attached, torch.zeros_like(speed), speed)
    
    return rewards

def action_rate_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize large changes in actions."""
    return torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1)

def joint_vel_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize large joint velocities."""
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_vel), dim=1)

def joint_pos_rel(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Relative joint positions."""
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.joint_pos - asset.data.default_joint_pos

def joint_vel_rel(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Relative joint velocities."""
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.joint_vel - asset.data.default_joint_vel

def object_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Position of the object relative to the robot root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    
    object_pos_w = object.data.root_pos_w
    robot_pos_w = robot.data.root_pos_w
    robot_quat_w = robot.data.root_quat_w
    
    # relative position
    rel_pos_w = object_pos_w - robot_pos_w
    
    # rotate to robot frame (inverse rotation)
    # conjugate of quat is inverse for unit quat
    import isaaclab.utils.math as math_utils
    # Use quat_apply instead of deprecated quat_rotate
    rel_pos_b = math_utils.quat_apply(math_utils.quat_inv(robot_quat_w), rel_pos_w)
    
    return rel_pos_b

def object_height_reward(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    target_height: float = 0.5, # Optional target or just maximize
) -> torch.Tensor:
    """Reward the height of the object."""
    object: RigidObject = env.scene[object_cfg.name]
    # Reward absolute height
    return object.data.root_pos_w[:, 2]

def throwing_velocity_shaped_reward(
    env: ManagerBasedRLEnv, 
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """
    Reward forward/upward velocity, penalize backward/downward and static states.
    """
    object: RigidObject = env.scene[object_cfg.name]
    
    vel_w = object.data.root_lin_vel_w
    vel_x = vel_w[:, 0]
    vel_z = vel_w[:, 2]

    # Positive rewards (Forward + Upward)
    # Clamp max to 10.0 to prevent reward explosion from physics glitches
    r_pos_x = torch.clamp(vel_x, min=0.0, max=10.0) * 1.0 
    r_pos_z = torch.clamp(vel_z, min=0.0, max=10.0) * 1.0 
    
    # Negative penalties (Backward + Downward)
    # Increased penalty to 10.0 to strictly forbid backward/downward motion
    p_neg_x = torch.clamp(vel_x, max=0.0) * 1.0 
    p_neg_z = torch.clamp(vel_z, max=0.0) * 1.0 

    # Static penalty
    # If speed is very low, give large negative reward (-2.0)
    speed = torch.norm(vel_w[:, :3], dim=-1)
    p_static = torch.where(speed < 0.01, 1.0, 0.0)
    # p_static must be tensor of shape (num_envs,) which where returns.
    # Make sure types match
    p_static = p_static.to(dtype=torch.float32)

    return r_pos_x * r_pos_z + p_neg_x + p_neg_z - p_static
    # return r_pos_x + r_pos_z + p_neg_x + p_neg_z
    # return r_pos_x + r_pos_z

def last_action(env: ManagerBasedRLEnv) -> torch.Tensor:
    """The last action applied to the environment."""
    return env.action_manager.prev_action

def time_out(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Terminate the episode when the time limit is reached."""
    return env.episode_length_buf >= env.max_episode_length
