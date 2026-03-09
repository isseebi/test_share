from omni.isaac.kit import SimulationApp

# 1. シミュレーションアプリの起動
simulation_app = SimulationApp({"headless": False})

import numpy as np
from omni.isaac.core import World
from omni.isaac.core.objects import DynamicCuboid
from pxr import UsdPhysics, Gf

# 2. ワールドのセットアップ
world = World(stage_units_in_meters=1.0)
world.scene.add_default_ground_plane()

# 3. 直方体の作成
floating_height = 1.5  # 空中の仮想地面の高さ
cuboid_path = "/World/MyCuboid"
cuboid = world.scene.add(
    DynamicCuboid(
        prim_path=cuboid_path,
        name="my_cuboid",
        position=np.array([0.0, 0.0, floating_height]),
        scale=np.array([0.5, 0.5, 0.5]),
        color=np.array([0.2, 0.8, 0.2]),
        mass=1.0
    )
)

# 4. D6 Joint による「空中スライド」の設定
stage = world.stage
joint_path = "/World/FloatingJoint"
d6_joint = UsdPhysics.Joint.Define(stage, joint_path)
d6_joint.CreateBody1Rel().SetTargets([cuboid_path])

# ジョイントの基準位置設定
d6_joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
d6_joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))

# --- PD Driveの設定 ---
target_pos = np.array([2.0, 2.0]) # X, Yの目標位置
kp = 20.0
kd = 10.0

# X, Y軸: 移動用のPD Drive
for i, axis in enumerate(["transX", "transY"]):
    drive = UsdPhysics.DriveAPI.Apply(d6_joint.GetPrim(), axis)
    drive.CreateTypeAttr("force")
    drive.CreateStiffnessAttr(kp)
    drive.CreateDampingAttr(kd)
    drive.CreateTargetPositionAttr(float(target_pos[i]))

# Z軸: 高さを固定するためのPD Drive (仮想的な床の役割)
z_drive = UsdPhysics.DriveAPI.Apply(d6_joint.GetPrim(), "transZ")
z_drive.CreateTypeAttr("force")
z_drive.CreateStiffnessAttr(100.0)
z_drive.CreateDampingAttr(20.0)
z_drive.CreateTargetPositionAttr(floating_height)

# 修正箇所: 回転のロック (水平を保つ)
for axis in ["rotX", "rotY", "rotZ"]:
    limit = UsdPhysics.LimitAPI.Apply(d6_joint.GetPrim(), axis)
    limit.CreateLowAttr(0.0)
    limit.CreateHighAttr(0.0)

world.reset()

# 5. シミュレーションループ
while simulation_app.is_running():
    world.step(render=True)
    
    if world.is_playing():
        current_pos, _ = cuboid.get_world_pose()
        
        # コンソールに高さを表示して、浮いていることを確認 (60ステップ毎)
        if world.current_time_step_index % 60 == 0:
            print(f"Current Position: {current_pos[0]:.2f}, {current_pos[1]:.2f}, Height: {current_pos[2]:.2f}")

simulation_app.close()