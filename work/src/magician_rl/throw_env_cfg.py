# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
import math # for pi
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, DeformableObjectCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, ManagerBasedRLEnv
from isaaclab.assets import RigidObject, Articulation
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg, OffsetCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.sim.spawners.shapes import CuboidCfg, CylinderCfg
from isaaclab.sim.spawners.materials import PreviewSurfaceCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from . import mdp 
from .mdp import throw_events # Import custom events
from .config.magician import MAGICIAN_ROBOT_CFG

##
# Scene definition
##

@configclass
class MagicianThrowSceneCfg(InteractiveSceneCfg):
    """Configuration for the throw scene with a robot and a object."""

    # robots
    robot: ArticulationCfg = MAGICIAN_ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    
    # end-effector sensor (for reference, though actions handles logic)
    ee_frame: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/magician_link_4",
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/magician_link_4",
                name="end_effector",
                offset=OffsetCfg(pos=(0.06, 0.0, -0.059)),
            ),
        ],
    )
    
    # target object: Cylinder for throwing
    object: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.2, 0, 0.05], rot=[1, 0, 0, 0]),
        spawn=CylinderCfg(
            radius=0.015, # Small cylinder
            height=0.04,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=1.0,
                disable_gravity=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05), # Light enough to throw
            visual_material=PreviewSurfaceCfg(diffuse_color=(0.1, 0.8, 0.1)), # Green
        ),
    )

    # plane
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, -0.0]),
        spawn=GroundPlaneCfg(),
    )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


##
# MDP settings
##

@configclass
class CommandsCfg:
    """Command terms for the MDP."""
    # We might not need commands if the task is just "throw as far as possible"
    # But usually we need at least a dummy or null command to keep things consistent.
    # Let's keep it empty for now, or add a null command if needed.
    pass


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""
    arm_action: mdp.MagicianMimicActionCfg = mdp.MagicianMimicActionCfg(
        asset_name="robot",
        joint_names=["magician_joint_1", "magician_joint_2", "magician_joint_3"],
    )
    # Suction action with start_attached=True
    gripper_action: mdp.MagicianSuctionActionCfg = mdp.MagicianSuctionActionCfg(
        asset_name="robot",
        target_object_name="object",
        start_attached=True, # Start with object attached
        threshold=0.05, # Slightly larger to be safe during reset attachment
        force_hold=False, # User request: Disable force hold to allow release
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        # Position of object relative to robot root is crucial for aiming
        object_position = ObsTerm(func=mdp.object_position_in_robot_root_frame)
        # Actions history might help smoothness
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""
    # 1. Reset Robot Joints (Randomized)
    # 1. Reset Robot Joints (Randomized with Mimic Constraint)
    reset_robot_joints = EventTerm(
        func=throw_events.reset_joints_with_mimic,
        mode="reset",
        params={
            "position_range": (0.5, 1.5), # Scale for default pose
            "velocity_range": (0.0, 0.0),
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # 2. Reset Object to Gripper (Attached state)
    # This must happen AFTER robot joints are set, so the gripper position is known.
    reset_object_to_gripper = EventTerm(
        func=throw_events.reset_object_to_gripper,
        mode="reset",
        params={
            "object_cfg": SceneEntityCfg("object"),
            "robot_cfg": SceneEntityCfg("robot"),
        },
    )

@configclass
class RewardsCfg:
    """Reward terms for the MDP."""
    
    # 1. Throw Distance Reward
    # User requested to focus ONLY on velocity for now.
    # We want to maximize the distance of the object from the origin (or robot base).
    # Since we want to THROW, we should care about the distance at the end of the episode or continuous distance.
    # Continuous distance encourages moving it away.
    # distance_reward = RewTerm(
    #      func=mdp.object_distance_reward,
    #      params={
    #          "attached_weight": 1.0, 
    #          "detached_weight": 1.0, # Equal weight to prevent "drop to gain reward" hack
    #          "gripper_action_name": "gripper_action"
    #      },
    #      weight=1.0
    # )
    
    # New: Reward for lifting the object high
    # User requested to remove lift reward and focus on XZ velocity.
    # lift_reward = RewTerm(
    #     func=mdp.object_height_reward,
    #     params={"target_height": 0.5},
    #     weight=2.0 # Strong incentive to lift
    # )

    # この報酬はうまく行く
    throwing_velocity = RewTerm(
         func=mdp.throwing_velocity_shaped_reward,
         params={},
         weight=1.0
    )

    # 2. Release/Throw Velocity Reward
    # Reward velocity only when object is detached
    throwing_velocity = RewTerm(
         func=mdp.object_throwing_velocity_reward,
         params={"gripper_action_name": "gripper_action"},
         weight=0.1
    )

    # 3. Penalties
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4) # Keep small penalty to avoid jitter
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # Reset if physics explode
    high_velocity_reset = DoneTerm(
        func=mdp.root_lin_vel_magnitude_upper_bound,
        params={"threshold": 100.0}, # Relaxed to 100.0 to prevent false resets during throwing
        time_out=False
    )

    object_dropped = DoneTerm(
        func=mdp.object_dropped,
        params={"threshold": 0.02}, # Slightly higher than radius (0.015)
    )
    
@configclass
class ThrowEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the throwing environment."""
    # Scene settings
    scene: MagicianThrowSceneCfg = MagicianThrowSceneCfg(num_envs=4096, env_spacing=1.0) # Larger spacing for throwing
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        """Post initialization."""
        self.decimation = 2
        self.episode_length_s = 3.0 # Short episodes for throwing? Or longer to wait for landing?
        # Let's say 3 seconds: 1s to swing, 2s for flight.
        self.sim.dt = 0.01  # 100Hz
        self.sim.render_interval = self.decimation

        self.sim.physx.bounce_threshold_velocity = 0.2
