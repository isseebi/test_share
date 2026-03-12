# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.sim.spawners.from_files import UsdFileCfg

##
# Configuration
##

MAGICIAN_ROBOT_CFG = ArticulationCfg(
    spawn=UsdFileCfg(
        usd_path="/home/isseebi/Desktop/user/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/original/config/magcian/magician_ros2/dobot_description/model/magician_suction.usd",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        joint_pos={
            "magician_joint_1": 1.5,
            "magician_joint_2": 0.8,
            "magician_joint_3": 0.5,
            "magician_joint_mimic_1": -1.8,
            "magician_joint_mimic_2":-0.5,
        },
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["magician_joint_[1-3]", "magician_joint_suction"], # Active joints + suction(prismatic)
            effort_limit=87.0,
            velocity_limit=2.175,
            stiffness=400.0,
            damping=40.0,
        ),
        "mimic": ImplicitActuatorCfg(
            # Mimic joints need high stiffness to maintain constraints rigidly
            joint_names_expr=["magician_joint_mimic_.*"],
            effort_limit=87.0,
            velocity_limit=2.175,
            stiffness=10000.0, # Very stiff to prevent drooping
            damping=1000.0,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
