from traject_gen import RandomPathGenerator
from omni.isaac.kit import SimulationApp

# ==========================================
# 1. シミュレーションアプリの起動 (ヘッドレスモード)
# ==========================================
# headlessをTrueに設定し、UIの立ち上げをスキップします
simulation_app = SimulationApp({"headless": False})

import numpy as np
from omni.isaac.core import World
from omni.isaac.core.objects import DynamicCuboid, DynamicCylinder, DynamicSphere
from pxr import UsdPhysics, Gf
import os

# ==========================================
# 2. ワールドのセットアップと並列環境の構築
# ==========================================
world = World(stage_units_in_meters=1.0)
world.scene.add_default_ground_plane()
stage = world.stage

# 並列シミュレーションの設定
num_envs = 2          # 同時に動かす台車の数
grid_size = int(np.ceil(np.sqrt(num_envs)))
spacing = 5.0          # 各環境の間隔（m）

floating_height = 1.5
cuboid_size = 0.5
arm_length = 0.5
arm_radius = 0.01
bob_radius = 0.05

cuboids = []
drives_x = []
drives_y = []

offsets_x = np.zeros(num_envs)
offsets_y = np.zeros(num_envs)

for i in range(num_envs):
    env_path = f"/World/Env_{i}"
    
    offset_x = (i % grid_size) * spacing
    offset_y = (i // grid_size) * spacing
    offset = np.array([offset_x, offset_y, 0.0])
    
    offsets_x[i] = offset_x
    offsets_y[i] = offset_y

    cuboid_pos = np.array([0.0, 0.0, floating_height]) + offset
    arm_pos = cuboid_pos - np.array([0.0, 0.0, (cuboid_size/2) + (arm_length/2)])
    bob_pos = arm_pos - np.array([0.0, 0.0, (arm_length/2) + bob_radius])

    cuboid = world.scene.add(DynamicCuboid(prim_path=f"{env_path}/MyCuboid", name=f"cuboid_{i}", position=cuboid_pos, scale=np.array([cuboid_size]*3), color=np.array([0.2, 0.8, 0.2]), mass=5.0))
    arm = world.scene.add(DynamicCylinder(prim_path=f"{env_path}/Arm", name=f"arm_{i}", position=arm_pos, radius=arm_radius, height=arm_length, color=np.array([0.8, 0.8, 0.8])))
    bob = world.scene.add(DynamicSphere(prim_path=f"{env_path}/Bob", name=f"bob_{i}", position=bob_pos, radius=bob_radius, color=np.array([0.2, 0.2, 0.2])))
    
    # === 追加: 接続された部品同士の衝突判定を無効化 ===
    # 台車とアームの衝突を無効化
    filter_cuboid = UsdPhysics.FilteredPairsAPI.Apply(cuboid.prim)
    filter_cuboid.CreateFilteredPairsRel().AddTarget(arm.prim.GetPath())

    # アームとおもりの衝突を無効化
    filter_arm = UsdPhysics.FilteredPairsAPI.Apply(arm.prim)
    filter_arm.CreateFilteredPairsRel().AddTarget(bob.prim.GetPath())

    UsdPhysics.MassAPI.Apply(arm.prim).CreateDensityAttr(10.0)    
    UsdPhysics.MassAPI.Apply(bob.prim).CreateDensityAttr(11340.0) 

    spherical_joint = UsdPhysics.SphericalJoint.Define(stage, f"{env_path}/SphericalJoint")
    spherical_joint.CreateBody0Rel().SetTargets([f"{env_path}/MyCuboid"])
    spherical_joint.CreateBody1Rel().SetTargets([f"{env_path}/Arm"])
    spherical_joint.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, -cuboid_size / 2.0))
    spherical_joint.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, arm_length / 2.0))

    fixed_joint = UsdPhysics.FixedJoint.Define(stage, f"{env_path}/FixedJoint")
    fixed_joint.CreateBody0Rel().SetTargets([f"{env_path}/Arm"])
    fixed_joint.CreateBody1Rel().SetTargets([f"{env_path}/Bob"])
    fixed_joint.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, -arm_length / 2.0))
    fixed_joint.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, bob_radius))

    d6_joint = UsdPhysics.Joint.Define(stage, f"{env_path}/FloatingJoint")
    d6_joint.CreateBody1Rel().SetTargets([f"{env_path}/MyCuboid"])

    d6_joint.CreateLocalPos0Attr(Gf.Vec3f(float(offset_x), float(offset_y), 0.0))
    d6_joint.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))

    kp, kd = 800.0, 100.0
    drive_x = UsdPhysics.DriveAPI.Apply(d6_joint.GetPrim(), "transX")
    drive_x.CreateTypeAttr("force"); drive_x.CreateStiffnessAttr(kp); drive_x.CreateDampingAttr(kd)
    drive_y = UsdPhysics.DriveAPI.Apply(d6_joint.GetPrim(), "transY")
    drive_y.CreateTypeAttr("force"); drive_y.CreateStiffnessAttr(kp); drive_y.CreateDampingAttr(kd)
    # z_drive = UsdPhysics.DriveAPI.Apply(d6_joint.GetPrim(), "transZ")
    # z_drive.CreateStiffnessAttr(50000.0); z_drive.CreateTargetPositionAttr(floating_height)

    limit_z = UsdPhysics.LimitAPI.Apply(d6_joint.GetPrim(), "transZ")
    limit_z.CreateLowAttr(floating_height)
    limit_z.CreateHighAttr(floating_height)

    for axis in ["rotX", "rotY", "rotZ"]:
        limit = UsdPhysics.LimitAPI.Apply(d6_joint.GetPrim(), axis)
        limit.CreateLowAttr(0.0); limit.CreateHighAttr(0.0)

    cuboids.append(cuboid)
    drives_x.append(drive_x)
    drives_y.append(drive_y)

# 保存先ディレクトリの設定
save_dir = "/home/isseebi/Desktop/user/Reinforcement_learning/study/OpenManipulation/src/dataset"
os.makedirs(save_dir, exist_ok=True)

# ==========================================
# 3. エピソードループ
# ==========================================
num_episodes = 100 

g, L = 9.81, arm_length + bob_radius
K_ff = L / g
omega_n, zeta = 3.0, 1.0 

print("データ収集を開始します（ヘッドレスモード）...")

for episode in range(num_episodes):
    
    target_paths_x = []
    target_paths_y = []
    path_lengths = []

    for i in range(num_envs):
        path_gen = RandomPathGenerator(step_size=0.02)
        tx, ty = path_gen.generate()
        target_paths_x.append(tx)
        target_paths_y.append(ty)
        path_lengths.append(len(tx))

    max_path_steps = max(path_lengths)
    
    world.reset()

    # ==========================================
    # 修正1: ドライブの目標値を強制的に初期位置(0.0)に戻す
    # ==========================================
    for i in range(num_envs):
        drives_x[i].GetTargetPositionAttr().Set(0.0)
        drives_y[i].GetTargetPositionAttr().Set(0.0)

    # ==========================================
    # 修正2: 物理エンジンを安定させるためのウォームアップ
    # ==========================================
    for _ in range(10):
        world.step(render=False)

    # ==========================================
    # 修正3: ウォームアップ "後" に時間を取得し、状態変数を初期化
    # ==========================================
    last_time = world.current_time
    x_r = np.zeros(num_envs)
    y_r = np.zeros(num_envs)
    v_r_x = np.zeros(num_envs)
    v_r_y = np.zeros(num_envs)

    current_step = 0
    base_history = [] 

    # ==========================================
    # 1つに統合されたメインループ
    # ==========================================
    while simulation_app.is_running():
        world.step(render=True)
        
        if world.is_playing():
            current_time = world.current_time
            dt = current_time - last_time
            
            # 修正4: 初回ステップや処理落ちによる dt のスパイクを防ぐ
            if dt <= 0 or dt > 0.05: 
                dt = 1.0 / 60.0
            last_time = current_time
            
            if current_step > max_path_steps + 50: 
                break

            target_x = np.zeros(num_envs)
            target_y = np.zeros(num_envs)
            
            for i in range(num_envs):
                idx = min(current_step, path_lengths[i] - 1)
                target_x[i] = target_paths_x[i][idx]
                target_y[i] = target_paths_y[i][idx]

            a_r_x = (omega_n**2) * (target_x - x_r) - 2 * zeta * omega_n * v_r_x
            v_r_x += a_r_x * dt
            x_r += v_r_x * dt
            
            a_r_y = (omega_n**2) * (target_y - y_r) - 2 * zeta * omega_n * v_r_y
            v_r_y += a_r_y * dt
            y_r += v_r_y * dt

            cmd_x = x_r + K_ff * a_r_x
            cmd_y = y_r + K_ff * a_r_y

            step_positions = np.zeros((num_envs, 2))
            
            for i in range(num_envs):
                drives_x[i].GetTargetPositionAttr().Set(float(cmd_x[i]))
                drives_y[i].GetTargetPositionAttr().Set(float(cmd_y[i]))

                b_pos, _ = cuboids[i].get_world_pose()
                step_positions[i, 0] = b_pos[0] - offsets_x[i]
                step_positions[i, 1] = b_pos[1] - offsets_y[i]

            base_history.append(step_positions)
            current_step += 1

    # データ保存処理
    if len(base_history) > 0:
        trajectory_data = np.array(base_history)
        dataset_xy = np.transpose(trajectory_data, (1, 0, 2))

        dataset_path = os.path.join(save_dir, f"dataset_ep{episode:03d}.npy")
        np.save(dataset_path, dataset_xy)
        print(f"Episode {episode + 1}/{num_episodes} 完了 -> 保存: {dataset_path} | Shape: {dataset_xy.shape}")
        
print("全エピソードのシミュレーションとデータ収集が完了しました。")
simulation_app.close()