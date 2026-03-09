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
# 3. オブジェクトの作成 (変更なし)
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

# LINEAR_DAMPING = 0.
# for prim in [arm.prim, bob.prim]:
#     physx_rb = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
#     physx_rb.CreateLinearDampingAttr(LINEAR_DAMPING)

# ==========================================
# 4. ジョイントの作成 (変更なし)
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
# 5. 台車の制御用ジョイント (変更なし)
# ==========================================
slide_joint_path = "/World/FloatingJoint"
d6_joint = UsdPhysics.Joint.Define(stage, slide_joint_path)
d6_joint.CreateBody1Rel().SetTargets([cuboid_path])
d6_joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
d6_joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))

current_target_x = 0.0
current_target_y = 0.0
kp = 20.0
kd = 10.0

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

# --- 物理モデルに基づくパラメータ ---
g = 9.81  # 重力加速度
L = arm_length + bob_radius  # 振り子の長さ (0.55m)
K_ff = L / g  # 振動補償ゲイン (L/g)

# --- 軌道生成用のフィルタパラメータ (2次遅れ系) ---
omega_n = 2.0  # 応答の速さ (大きいほど目標到達が早い)
zeta = 1.0     # 減衰比 (1.0で揺れ落ちなくピタッと止まる臨界減衰)

# --- 内部状態の初期化 ---
x_r, v_r = 0.0, 0.0    # X方向の理想的な位置と速度
y_r, v_r_y = 0.0, 0.0  # Y方向の理想的な位置と速度
target_x, target_y = 2.0, 0.0 # 最終目標地点

last_time = 0.0

# データ記録用リスト
base_history = []
bob_history = []
steps = 0
max_steps = 900

while simulation_app.is_running():
    world.step(render=True)
    
    if world.is_playing():
        current_time = world.current_time
        # 前フレームからの経過時間(dt)を計算
        dt = current_time - last_time
        if dt <= 0:
            dt = 1.0 / 60.0  # 初回や時間が進まない場合の安全策
        last_time = current_time
        
        frame_count += 1
        
        # 約5秒(300フレーム)ごとに新しい目標位置を生成
        if frame_count % 300 == 0:
            # target_x = np.random.uniform(-2.0, 2.0)
            # target_y = np.random.uniform(0.0, 0.0) # 今回はY方向も少し動かしてみます
            target_x = [2.0, -2.0][frame_count // 300 % 2]  # 交互に左右に動かす
            target_y = 0.0
            print(f"\n[{current_time:.2f}s] New Target -> X: {target_x:.2f}, Y: {target_y:.2f}")

        # ==========================================
        # Step 1: 理想的な軌道 (リファレンスモデル) の計算
        # 2次遅れ系モデルを用いて、ステップ入力を滑らかな軌道に変換
        # ==========================================
        # X方向の理想的な加速度・速度・位置を更新
        a_r_x = (omega_n ** 2) * (target_x - x_r) - 2 * zeta * omega_n * v_r
        v_r += a_r_x * dt
        x_r += v_r * dt

        # Y方向の理想的な加速度・速度・位置を更新
        a_r_y = (omega_n ** 2) * (target_y - y_r) - 2 * zeta * omega_n * v_r_y
        v_r_y += a_r_y * dt
        y_r += v_r_y * dt

        # ==========================================
        # Step 2: 振動補償ゲインを用いた指令値の計算
        # ==========================================
        # 台車への指令位置 = 理想の荷物位置 + (補償ゲイン * 理想の加速度)
        cmd_x = x_r + K_ff * a_r_x
        cmd_y = y_r + K_ff * a_r_y

        # 台車の駆動ジョイントに計算した指令値をセット
        drive_x.GetTargetPositionAttr().Set(cmd_x)
        drive_y.GetTargetPositionAttr().Set(cmd_y)

        base_pos, _ = cuboid.get_world_pose()
        bob_pos, _ = bob.get_world_pose()
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
save_path = os.path.join(save_dir, "no_11.png")
plt.savefig(save_path)

# メモリ解放のためにフィギュアを閉じる
plt.close()

simulation_app.close()