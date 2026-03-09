# 振り子の振動を抑制する制御
# 振動抑制はフィードバック制御

from omni.isaac.kit import SimulationApp

# 1. シミュレーションアプリの起動
simulation_app = SimulationApp({"headless": False})

import numpy as np
from omni.isaac.core import World
from omni.isaac.core.objects import DynamicCuboid, DynamicCylinder, DynamicSphere
from pxr import UsdPhysics, Gf, PhysxSchema
import matplotlib.pyplot as plt  # 【追加】グラフ描画用ライブラリ
import os

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
arm_pos = cuboid_pos - np.array([0.0, 0.0, (cuboid_size/2) + (arm_length/2)])
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
UsdPhysics.MassAPI.Apply(arm.prim).CreateDensityAttr(10.0)    
UsdPhysics.MassAPI.Apply(bob.prim).CreateDensityAttr(11340.0) 

LINEAR_DAMPING = 0.1
for prim in [arm.prim, bob.prim]:
    physx_rb = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
    physx_rb.CreateLinearDampingAttr(LINEAR_DAMPING)

# ==========================================
# 4. ジョイントの作成 (振り子の接続)
# ==========================================
spherical_joint = UsdPhysics.SphericalJoint.Define(stage, "/World/SphericalJoint")
spherical_joint.CreateBody0Rel().SetTargets([cuboid_path])
spherical_joint.CreateBody1Rel().SetTargets([arm_path])
spherical_joint.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, -cuboid_size / 2.0))
spherical_joint.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, arm_length / 2.0))

fixed_joint = UsdPhysics.FixedJoint.Define(stage, "/World/FixedJoint")
fixed_joint.CreateBody0Rel().SetTargets([arm_path])
fixed_joint.CreateBody1Rel().SetTargets([bob_path])
fixed_joint.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, -arm_length / 2.0))
fixed_joint.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, bob_radius))

# ==========================================
# 5. 台車の空中スライド移動制御 (D6 Joint)
# ==========================================
slide_joint_path = "/World/FloatingJoint"
d6_joint = UsdPhysics.Joint.Define(stage, slide_joint_path)
d6_joint.CreateBody1Rel().SetTargets([cuboid_path])
d6_joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
d6_joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))

# 初期目標位置
target_pos_x = 2.0
target_pos_y = 0.0
kp = 20.0
kd = 10.0

# X, Y軸のドライブ設定
drive_x = UsdPhysics.DriveAPI.Apply(d6_joint.GetPrim(), "transX")
drive_x.CreateTypeAttr("force")
drive_x.CreateStiffnessAttr(kp)
drive_x.CreateDampingAttr(kd)
drive_x.CreateTargetPositionAttr(target_pos_x)

drive_y = UsdPhysics.DriveAPI.Apply(d6_joint.GetPrim(), "transY")
drive_y.CreateTypeAttr("force")
drive_y.CreateStiffnessAttr(kp)
drive_y.CreateDampingAttr(kd)
drive_y.CreateTargetPositionAttr(target_pos_y)

# Z軸のドライブ設定 (高さ維持)
z_drive = UsdPhysics.DriveAPI.Apply(d6_joint.GetPrim(), "transZ")
z_drive.CreateTypeAttr("force")
z_drive.CreateStiffnessAttr(5000.0) 
z_drive.CreateDampingAttr(500.0)
z_drive.CreateTargetPositionAttr(floating_height)

# 回転ロック
for axis in ["rotX", "rotY", "rotZ"]:
    limit = UsdPhysics.LimitAPI.Apply(d6_joint.GetPrim(), axis)
    limit.CreateLowAttr(0.0)
    limit.CreateHighAttr(0.0)

world.reset()

# ==========================================
# 6. シミュレーションループ (振動補償の追加)
# ==========================================
frame_count = 0

# 振動補償用のゲイン (環境に合わせて調整してください)
k_p_sway = 1.5  # 位置のズレに対する補償ゲイン
k_d_sway = 0.5  # 速度のズレに対する補償ゲイン

# データ記録用リスト
base_history = []
bob_history = []
steps = 0
max_steps = 900

while simulation_app.is_running():
    world.step(render=True)
    
    if world.is_playing():
        frame_count += 1
        
        # 300フレームごとに目標位置をランダムに変更
        if frame_count % 300 == 0:
            # target_pos_x = np.random.uniform(-2.0, 2.0)
            # target_pos_y = np.random.uniform(0.0, 0.0)
            target_pos_x = [2.0, -2.0][frame_count // 300 % 2]  # 交互に左右に動かす
            target_pos_y = 0.0
            print(f"New Target -> X: {target_pos_x:.2f}, Y: {target_pos_y:.2f}")

        # --- 振動抑制（Anti-Sway）制御 ---
        # 1. 台車と重りの現在の位置と速度を取得
        base_pos, _ = cuboid.get_world_pose()
        bob_pos, _ = bob.get_world_pose()
        
        base_vel = cuboid.get_linear_velocity()
        bob_vel = bob.get_linear_velocity()
        
        # 2. 偏差（ズレ）の計算
        delta_pos = bob_pos - base_pos
        delta_vel = bob_vel - base_vel
        
        # 3. フィードバックを適用した動的なコマンド位置の計算
        cmd_x = target_pos_x + (k_p_sway * delta_pos[0]) + (k_d_sway * delta_vel[0])
        cmd_y = target_pos_y + (k_p_sway * delta_pos[1]) + (k_d_sway * delta_vel[1])
        
        # 4. ジョイントドライブに更新したターゲット位置を毎フレーム設定
        drive_x.GetTargetPositionAttr().Set(cmd_x)
        drive_y.GetTargetPositionAttr().Set(cmd_y)

        # 記録
        base_history.append(base_pos.copy())
        bob_history.append(bob_pos.copy())
        
    steps += 1
    if steps >= max_steps:
        break

# ========== グラフのプロットと保存 ==========
base_history = np.array(base_history)
bob_history = np.array(bob_history)
time_axis = np.arange(max_steps)

# グラフの作成
# 注：システム設定により plt.figure() は省略していますが、ローカル環境でサイズ調整が必要な場合は plt.figure(figsize=(10, 6)) を追加してください
plt.plot(time_axis, base_history[:, 0], label="X (Base)")
plt.plot(time_axis, bob_history[:, 0], label="X (Bob)", alpha=0.8) # 重なりを見やすくするため少し透過

# ターゲットライン（1つにまとめて表示）
plt.axhline(y=2.0, color='r', linestyle='--', label="Target X")
plt.axhline(y=-2.0, color='r', linestyle='--') # 凡例が重複するためラベルは片方のみ

plt.title("Position History (Base and Bob)")
plt.xlabel("Steps")
plt.ylabel("Position ($m$)")
plt.legend()
plt.grid(True)

plt.tight_layout()

# 保存先のパスを指定
save_dir = "/home/isseebi/Desktop/user/Reinforcement_learning/study/OpenManipulation/src"

# ファイル名を設定して保存
save_path = os.path.join(save_dir, "no_10.png")
plt.savefig(save_path)

# メモリ解放のためにフィギュアを閉じる
plt.close()

simulation_app.close()