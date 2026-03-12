# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.assets import RigidObjectCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
import isaaclab.sim as sim_utils
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import SceneEntityCfg

from isaaclab_tasks.manager_based.manipulation.magisian import mdp
from isaaclab_tasks.manager_based.manipulation.magisian.lift_env_cfg import LiftEnvCfg, RewardsCfg, TerminationsCfg, termination_by_joint_pos_limit, termination_by_bad_state
from isaaclab_tasks.manager_based.manipulation.magisian.config.magician import MAGICIAN_ROBOT_CFG

##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip

@configclass
class ThrowingRewardsCfg(RewardsCfg):
    """Reward terms for the throwing task."""
    # Override lifting and reaching
    lifting_object = RewTerm(func=mdp.object_is_lifted, params={"minimal_height": 0.1}, weight=0.0) # Disable
    reaching_object = RewTerm(func=mdp.object_ee_distance, params={"std": 0.1}, weight=0.5) # Reduced weight
    
    # Add Throwing Reward
    throwing_velocity = RewTerm(
        func=mdp.object_throwing_velocity,
        weight=1.0,
        params={"object_cfg": SceneEntityCfg("object")}
    )

@configclass
class ThrowingTerminationsCfg(TerminationsCfg):
    """Termination terms for the throwing task."""
    # Disable object dropping by setting to None (or just not including it if we didn't inherit, but we inherit)
    # To remove an inherited field in @configclass, we can override it with None? 
    # IsaacLab managers skip None terms.
    object_dropping = None 

    # Add Joint Limit Termination (Safety)
    joint_pos_limit = DoneTerm(func=termination_by_joint_pos_limit, time_out=False)

    # Add Bad State Termination (NaN/Inf check)
    bad_state_reset = DoneTerm(func=termination_by_bad_state, time_out=False)


    # inherited: high_velocity_reset (limit=3.0)



@configclass
class FrankaCubeLiftEnvCfg(LiftEnvCfg):
    # Override the config sections
    rewards: ThrowingRewardsCfg = ThrowingRewardsCfg()
    terminations: ThrowingTerminationsCfg = ThrowingTerminationsCfg()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # --- ロボットとアクションの設定 ---
        self.scene.robot = MAGICIAN_ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        self.actions.arm_action = mdp.MagicianMimicActionCfg(
            asset_name="robot", joint_names=["magician_joint_[1-3]"], scale=0.5
        )

        self.actions.gripper_action = mdp.MagicianSuctionActionCfg(
            asset_name="robot",
            target_object_name="object",
            offset=(0.06, 0.0, -0.059), # X, Y, Z オフセット
            radius=0.005,
            threshold=0.055,
        )
        self.commands.object_pose.body_name = "Object"

        # --- オブジェクトの設定 ---
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.2, 0.0, 0.025], rot=[1, 0, 0, 0]),
            spawn=sim_utils.CylinderCfg(
                radius=0.0125,
                height=0.05,
                collision_props=sim_utils.CollisionPropertiesCfg(),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False, # ここをTrueにすると浮き続けますが、物理的に正しく吸着させたいならFalseのままが良いです
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.01),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
            ),
        )

        # --- Frame Transformerの設定 ---
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/magician_link_4",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/magician_link_4",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=[0.06, 0.0, -0.059],
                    ),
                ),
            ],
        )
        
        # ---------------------------------------------------------------------
        # Event Logic
        # ---------------------------------------------------------------------
        
        # Use Coupled Reset
        # We need to replace the default reset logic or add this as the main reset event
        
        # Note: 'reset_all' in parent uses 'reset_scene_to_default'.
        # We want to override or add to it.
        # Since we want randomization, we should use our new term.
        
        self.events.reset_robot_and_object = EventTerm(
            func=mdp.reset_robot_and_object_coupled,
            mode="reset",
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "object_cfg": SceneEntityCfg("object"),
                "joint_range": {
                    "j1": (-1.0, 1.0), # More random: +/- 1.0 rad (~57 deg)
                    "j2": (0.0, 0.6),  # Shoulder lift
                    "j3": (-0.2, 0.4),  # Elbow
                },
                "base_offset": (0.0, 0.0, 0.25), # Corrected base height for FK
                "suction_offset": (0.06, 0.0, -0.059),
                "cylinder_height": 0.05,
            },
        )
        
        # Remove conflicting events if they exist
        # We use hasattr because configclass instances are not iterable like dicts
        if hasattr(self.events, "reset_object_position"):
            del self.events.reset_object_position
            
        # Also remove reset_robot_joints if it exists (for safety, though not in base LiftEnvCfg)
        if hasattr(self.events, "reset_robot_joints"):
             del self.events.reset_robot_joints

        # --- Camera Settings for Video Recording ---
        # Isometric view looking at the robot and throwing area
        self.viewer.eye = (1.2, 1.2, 1.0)
        self.viewer.lookat = (0.0, 0.0, 0.2)


@configclass
class FrankaCubeLiftEnvCfg_PLAY(FrankaCubeLiftEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
