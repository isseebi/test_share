from omni.isaac.kit import SimulationApp

# 1. シミュレーションアプリの起動
simulation_app = SimulationApp({"headless": False})

import numpy as np
from omni.isaac.core import World
from omni.isaac.core.objects import DynamicCuboid, DynamicCylinder, DynamicSphere
from pxr import UsdPhysics, Gf, PhysxSchema

# 2. ワールドのセットアップ
world = World(stage_units_in_meters=1.0)
world.scene.add_default_ground_plane()

stage = world.stage

# ==========================================
# 3. オブジェクトの作成 (台車・棒・重り)
# ==========================================
floating_height = 1.5
cuboid_size = 0.5
arm_length = 0.5
arm_radius = 0.01
bob_radius = 0.05

# 配置座標の計算 (上から下へ吊り下げる)
cuboid_pos = np.array([0.0, 0.0, floating_height])
# 円柱の中心は、直方体の底面から棒の長さの半分だけ下
arm_pos = cuboid_pos - np.array([0.0, 0.0, (cuboid_size/2) + (arm_length/2)])
# 球の中心は、円柱の底面から球の半径分だけ下
bob_pos = arm_pos - np.array([0.0, 0.0, (arm_length/2) + bob_radius])

# ① スライドする直方体 (Base)
cuboid_path = "/World/MyCuboid"
cuboid = world.scene.add(
    DynamicCuboid(
        prim_path=cuboid_path, name="my_cuboid",
        position=cuboid_pos, scale=np.array([cuboid_size, cuboid_size, cuboid_size]),
        color=np.array([0.2, 0.8, 0.2]), mass=5.0
    )
)

# ② 細い円柱 (Arm)
arm_path = "/World/Arm"
arm = world.scene.add(
    DynamicCylinder(
        prim_path=arm_path, name="arm",
        position=arm_pos, radius=arm_radius, height=arm_length,
        color=np.array([0.8, 0.8, 0.8])
    )
)

# ③ 重りの球 (Bob)
bob_path = "/World/Bob"
bob = world.scene.add(
    DynamicSphere(
        prim_path=bob_path, name="bob",
        position=bob_pos, radius=bob_radius,
        color=np.array([0.2, 0.2, 0.2])
    )
)

# --- 物理プロパティ(密度と減衰)の設定 ---
UsdPhysics.MassAPI.Apply(arm.prim).CreateDensityAttr(10.0)    # 棒は軽く
UsdPhysics.MassAPI.Apply(bob.prim).CreateDensityAttr(11340.0) # 重りは非常に重く(鉛の密度)

LINEAR_DAMPING = 0.1
for prim in [arm.prim, bob.prim]:
    physx_rb = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
    physx_rb.CreateLinearDampingAttr(LINEAR_DAMPING)

# ==========================================
# 4. ジョイントの作成 (振り子の接続)
# ==========================================
# ① 直方体と棒を繋ぐ球関節 (SphericalJoint: 3Dにスイングできる)
spherical_joint = UsdPhysics.SphericalJoint.Define(stage, "/World/SphericalJoint")
spherical_joint.CreateBody0Rel().SetTargets([cuboid_path])
spherical_joint.CreateBody1Rel().SetTargets([arm_path])
# 直方体の底面を支点にする
spherical_joint.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, -cuboid_size / 2.0))
# 棒の上面を支点にする
spherical_joint.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, arm_length / 2.0))

# ② 棒と重りを繋ぐ固定関節 (FixedJoint: 棒の先端に球を固定)
fixed_joint = UsdPhysics.FixedJoint.Define(stage, "/World/FixedJoint")
fixed_joint.CreateBody0Rel().SetTargets([arm_path])
fixed_joint.CreateBody1Rel().SetTargets([bob_path])
# 棒の底面
fixed_joint.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, -arm_length / 2.0))
# 球の上面
fixed_joint.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, bob_radius))

# ==========================================
# 5. 台車の空中スライド移動制御 (D6 Joint)
# ==========================================
slide_joint_path = "/World/FloatingJoint"
d6_joint = UsdPhysics.Joint.Define(stage, slide_joint_path)
d6_joint.CreateBody1Rel().SetTargets([cuboid_path])
d6_joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
d6_joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))

target_pos = np.array([2.0, 2.0]) 
kp = 20.0
kd = 10.0

# X, Y軸: PD制御で移動
for i, axis in enumerate(["transX", "transY"]):
    drive = UsdPhysics.DriveAPI.Apply(d6_joint.GetPrim(), axis)
    drive.CreateTypeAttr("force")
    drive.CreateStiffnessAttr(kp)
    drive.CreateDampingAttr(kd)
    drive.CreateTargetPositionAttr(float(target_pos[i]))

# Z軸: 空中高さを維持
z_drive = UsdPhysics.DriveAPI.Apply(d6_joint.GetPrim(), "transZ")
z_drive.CreateTypeAttr("force")
z_drive.CreateStiffnessAttr(500.0) # 重い振り子がぶら下がるので固めに設定
z_drive.CreateDampingAttr(50.0)
z_drive.CreateTargetPositionAttr(floating_height)

# 台車(直方体)の回転をロック
for axis in ["rotX", "rotY", "rotZ"]:
    limit = UsdPhysics.LimitAPI.Apply(d6_joint.GetPrim(), axis)
    limit.CreateLowAttr(0.0)
    limit.CreateHighAttr(0.0)

world.reset()

# ==========================================
# 6. シミュレーションループ
# ==========================================
# 動的に目標位置を変えるテスト用の変数
frame_count = 0

while simulation_app.is_running():
    world.step(render=True)
    
    if world.is_playing():
        frame_count += 1
        
        # 300フレームごとに目標位置をランダムに変更して、振り子を揺らす
        if frame_count % 300 == 0:
            new_x = np.random.uniform(-2.0, 2.0)
            new_y = np.random.uniform(-2.0, 2.0)
            
            # X, Yのドライブのターゲット位置を更新
            drive_x = UsdPhysics.DriveAPI.Get(d6_joint.GetPrim(), "transX")
            drive_y = UsdPhysics.DriveAPI.Get(d6_joint.GetPrim(), "transY")
            drive_x.GetTargetPositionAttr().Set(new_x)
            drive_y.GetTargetPositionAttr().Set(new_y)
            
            print(f"New Target -> X: {new_x:.2f}, Y: {new_y:.2f}")

simulation_app.close()