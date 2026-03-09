from omni.isaac.kit import SimulationApp

# 1. シミュレーションアプリの起動
simulation_app = SimulationApp({"headless": False})

import numpy as np
from omni.isaac.core import World
from omni.isaac.core.objects import DynamicCuboid, DynamicCylinder, DynamicSphere
from pxr import UsdPhysics, Gf, PhysxSchema
import matplotlib.pyplot as plt
import os

# 2. ワールドのセットアップ
world = World(stage_units_in_meters=1.0)
world.scene.add_default_ground_plane()

stage = world.stage

# ==========================================
# 3. オブジェクトの作成
# ==========================================
floating_height = 1.5
cuboid_size = 0.5
arm_length = 0.5
arm_radius = 0.01
bob_radius = 0.05

cuboid_pos = np.array([0.0, 0.0, floating_height])
arm_pos = cuboid_pos - np.array([0.0, 0.0, (cuboid_size/2) + (arm_length/2)])
bob_pos = arm_pos - np.array([0.0, 0.0, (arm_length/2) + bob_radius])

cuboid_path = "/World/MyCuboid"
cuboid = world.scene.add(
    DynamicCuboid(
        prim_path=cuboid_path, name="my_cuboid",
        position=cuboid_pos, scale=np.array([cuboid_size, cuboid_size, cuboid_size]),
        color=np.array([0.2, 0.8, 0.2]), mass=5.0
    )
)

arm_path = "/World/Arm"
arm = world.scene.add(
    DynamicCylinder(
        prim_path=arm_path, name="arm",
        position=arm_pos, radius=arm_radius, height=arm_length,
        color=np.array([0.8, 0.8, 0.8])
    )
)

bob_path = "/World/Bob"
bob = world.scene.add(
    DynamicSphere(
        prim_path=bob_path, name="bob",
        position=bob_pos, radius=bob_radius,
        color=np.array([0.2, 0.2, 0.2])
    )
)

UsdPhysics.MassAPI.Apply(arm.prim).CreateDensityAttr(10.0)    
UsdPhysics.MassAPI.Apply(bob.prim).CreateDensityAttr(11340.0) 

# ==========================================
# 4. ジョイントの作成
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
# 5. 台車の制御用ジョイント
# ==========================================
slide_joint_path = "/World/FloatingJoint"
d6_joint = UsdPhysics.Joint.Define(stage, slide_joint_path)
d6_joint.CreateBody1Rel().SetTargets([cuboid_path])
d6_joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
d6_joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))

current_target_x = 0.0
current_target_y = 0.0
kp = 500.0
kd = 50.0

drive_x = UsdPhysics.DriveAPI.Apply(d6_joint.GetPrim(), "transX")
drive_x.CreateTypeAttr("force")
drive_x.CreateStiffnessAttr(kp)
drive_x.CreateDampingAttr(kd)
drive_x.CreateTargetPositionAttr(current_target_x)

drive_y = UsdPhysics.DriveAPI.Apply(d6_joint.GetPrim(), "transY")
drive_y.CreateTypeAttr("force")
drive_y.CreateStiffnessAttr(kp)
drive_y.CreateDampingAttr(kd)
drive_y.CreateTargetPositionAttr(current_target_y)

z_drive = UsdPhysics.DriveAPI.Apply(d6_joint.GetPrim(), "transZ")
z_drive.CreateTypeAttr("force")
z_drive.CreateStiffnessAttr(50000.0) 
z_drive.CreateDampingAttr(500.0)
z_drive.CreateTargetPositionAttr(floating_height)

for axis in ["rotX", "rotY", "rotZ"]:
    limit = UsdPhysics.LimitAPI.Apply(d6_joint.GetPrim(), axis)
    limit.CreateLowAttr(0.0)
    limit.CreateHighAttr(0.0)

world.reset()

# ==========================================
# 6. 振動補償ゲインを用いたフィードフォワード制御ループ
# ==========================================
frame_count = 0

g = 9.81
L = arm_length + bob_radius
K_ff = L / g
# K_ff = (L / g)

omega_n = 5.0
zeta = 1.0

x_r, v_r = 0.0, 0.0
y_r, v_r_y = 0.0, 0.0
target_x, target_y = 0.0, 0.0 

last_time = 0.0

# --- 【追加】軌道タイプの設定 ---
# "step": 従来の直線ステップ軌道
# "circle": 円軌道
# "figure8": 8の字軌道
trajectory_type = "figure8" 

base_history = []
bob_history = []
target_history = [] # 【追加】ターゲットの軌跡も記録
ref_history = [] # 【追加】平滑化されたリファレンスの軌跡
steps = 0
max_steps = 700 # 軌道が分かりやすいように長めに設定

while simulation_app.is_running():
    world.step(render=True)
    
    if world.is_playing():
        current_time = world.current_time
        dt = current_time - last_time
        if dt <= 0:
            dt = 1.0 / 60.0
        last_time = current_time
        frame_count += 1
        
        # ==========================================
        # 目標軌道の生成
        # ==========================================
        if trajectory_type == "step":
            if frame_count % 300 == 0:
                target_x = [2.0, -2.0][frame_count // 300 % 2]
                target_y = 0.0
                
        elif trajectory_type == "circle":
            cycle_time = 10.0 # 1周にかかる時間(秒)
            omega = 2.0 * np.pi / cycle_time
            target_x = 2.0 * np.cos(omega * current_time)
            target_y = 2.0 * np.sin(omega * current_time)
            
        elif trajectory_type == "figure8":
            cycle_time = 10.0 # 1周にかかる時間(秒)
            omega = 2.0 * np.pi / cycle_time
            target_x = 2.0 * np.sin(omega * current_time)
            target_y = 2.0 * np.sin(2.0 * omega * current_time)

        # 理想的な軌道の計算 (2次遅れ系)
        a_r_x = (omega_n ** 2) * (target_x - x_r) - 2 * zeta * omega_n * v_r
        v_r += a_r_x * dt
        x_r += v_r * dt

        a_r_y = (omega_n ** 2) * (target_y - y_r) - 2 * zeta * omega_n * v_r_y
        v_r_y += a_r_y * dt
        y_r += v_r_y * dt

        # 指令値の計算
        # cmd_x = x_r + K_ff * a_r_x
        # cmd_y = y_r + K_ff * a_r_y
        # cmd_x = target_x
        # cmd_y = target_y
        cmd_x = x_r
        cmd_y = y_r

        drive_x.GetTargetPositionAttr().Set(cmd_x)
        drive_y.GetTargetPositionAttr().Set(cmd_y)

        base_pos, _ = cuboid.get_world_pose()
        bob_pos, _ = bob.get_world_pose()
        
        # 履歴の記録
        base_history.append(base_pos.copy())
        bob_history.append(bob_pos.copy())
        target_history.append([target_x, target_y]) 
        ref_history.append([x_r, y_r])

    steps += 1
    if steps >= max_steps:
        break

# ========== グラフのプロットと保存 ==========
base_history = np.array(base_history)
bob_history = np.array(bob_history)
target_history = np.array(target_history)
ref_history = np.array(ref_history)

# XY平面での軌跡を描画するよう変更
plt.figure(figsize=(8, 8))
plt.plot(target_history[:, 0], target_history[:, 1], 'r--', label="Target")
# plt.plot(ref_history[:, 0], ref_history[:, 1], 'k-', linewidth=2, label="Reference (2nd-order smoothed)")
plt.plot(base_history[:, 0], base_history[:, 1], 'b-', label="Base")
plt.plot(bob_history[:, 0], bob_history[:, 1], 'g-', alpha=0.8, label="Bob")

plt.title(f"XY Trajectory ({trajectory_type})")
plt.xlabel("X Position ($m$)")
plt.ylabel("Y Position ($m$)")
plt.legend()
plt.grid(True)
plt.axis('equal') # アスペクト比を1:1にする
plt.tight_layout()

save_dir = "/home/isseebi/Desktop/user/Reinforcement_learning/study/OpenManipulation/src"
os.makedirs(save_dir, exist_ok=True) # ディレクトリが存在しない場合のエラーを防止
save_path = os.path.join(save_dir, "no_12_4.png")
plt.savefig(save_path)
plt.close()

simulation_app.close()