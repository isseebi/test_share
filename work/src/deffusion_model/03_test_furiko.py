from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({"headless": False})

import numpy as np
from omni.isaac.core import World
from omni.isaac.core.objects import DynamicCuboid, DynamicCapsule, DynamicSphere
from pxr import UsdPhysics, Gf, PhysxSchema

world = World(stage_units_in_meters=1.0)
world.scene.add_default_ground_plane()

# --- 座標と回転の計算 ---
# ボール(Bob)の半径が 0.08 なので、全体の中心高さを z=0.08 に統一してぴったり接地させます
# 全体をX軸に沿って配置します（Base: x=0.0, Arm: x=0.5, Bob: x=1.0）

base_pos = np.array([0.0, 0.0, 0.08])
arm_pos  = np.array([0.5, 0.0, 0.08])
bob_pos  = np.array([1.0, 0.0, 0.08])

# 棒をY軸周りに90度回転させて水平に寝かせるためのクォータニオン [w, x, y, z]
arm_orientation = np.array([0.7071068, 0.0, 0.7071068, 0.0])

# 1. 支柱 (Base)
base = world.scene.add(
    DynamicCuboid(
        prim_path="/World/Pendulum/Base", 
        position=base_pos, 
        # 中心高さ 0.08 に合わせて、サイズを 0.16 にするとぴったり接地します
        scale=np.array([0.15, 0.15, 0.15]), 
        color=np.array([0.5, 0.2, 0.2])
    )
)

# 2. 棒 (Arm)
arm = world.scene.add(
    DynamicCapsule(
        prim_path="/World/Pendulum/Arm",
        name="pendulum_arm",
        position=arm_pos,
        orientation=arm_orientation, # 90度回転を適用
        radius=0.005,
        height=1.0,
        color=np.array([0.8, 0.8, 0.8])
    )
)

# 3. 重りの球 (Bob)
bob = world.scene.add(
    DynamicSphere(
        prim_path="/World/Pendulum/Bob",
        name="pendulum_bob",
        position=bob_pos,
        radius=0.08,
        color=np.array([0.2, 0.2, 0.2])
    )
)

# --- 物理プロパティの設定 ---
UsdPhysics.MassAPI.Apply(base.prim).CreateDensityAttr(1000.0) 
UsdPhysics.MassAPI.Apply(arm.prim).CreateDensityAttr(10.0) 
UsdPhysics.MassAPI.Apply(bob.prim).CreateDensityAttr(11340.0)

LINEAR_DAMPING = 0.5 
for prim in [base.prim, arm.prim, bob.prim]:
    physx_rb = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
    physx_rb.CreateLinearDampingAttr(LINEAR_DAMPING)

# --- ジョイントの設定 ---
stage = world.stage

# A. Spherical Joint (Base <-> Arm)
spherical_joint = UsdPhysics.SphericalJoint.Define(stage, "/World/Pendulum/SphericalJoint")
spherical_joint.CreateBody0Rel().SetTargets([base.prim_path])
spherical_joint.CreateBody1Rel().SetTargets([arm.prim_path])
spherical_joint.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, 0.0))
# 棒が横に寝たため、Base側（x=0.0方向）はローカルのZ軸マイナス方向に変わります
spherical_joint.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, -0.5))

# B. Fixed Joint (Arm <-> Bob)
fixed_joint = UsdPhysics.FixedJoint.Define(stage, "/World/Pendulum/FixedJoint")
fixed_joint.CreateBody0Rel().SetTargets([arm.prim_path])
fixed_joint.CreateBody1Rel().SetTargets([bob.prim_path])
# Bob側（x=1.0方向）はローカルのZ軸プラス方向になります
fixed_joint.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, 0.5))
fixed_joint.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))

# 4. シミュレーションの初期化
world.reset()

print("Simulation started. The pendulum is completely laid flat on the ground.")

while simulation_app.is_running():
    world.step(render=True)

simulation_app.close()