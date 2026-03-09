from traject_gen import RandomPathGenerator
from omni.isaac.kit import SimulationApp
import os

# ==========================================
# 1. シミュレーションアプリの起動 (ヘッドレスモード)
# ==========================================
simulation_app = SimulationApp({"headless": True})

import torch
import numpy as np
from omni.isaac.core import World
from omni.isaac.core.prims import RigidPrimView
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.objects import DynamicCuboid, DynamicCylinder, DynamicSphere
from pxr import UsdPhysics, Gf, PhysxSchema

# ==========================================
# 2. ワールドのセットアップと並列環境の構築
# ==========================================
device = "cuda:0"
world = World(stage_units_in_meters=1.0, backend="torch", device=device)
world.scene.add_default_ground_plane()
stage = world.stage

# --- GPU物理メモリ設定の拡張 (PhysXエラー対策) ---
physics_scene_prim = stage.GetPrimAtPath("/physicsScene")
if not physics_scene_prim.IsValid():
    from pxr import UsdPhysics
    UsdPhysics.Scene.Define(stage, "/physicsScene")
    physics_scene_prim = stage.GetPrimAtPath("/physicsScene")

physx_scene_api = PhysxSchema.PhysxSceneAPI.Apply(physics_scene_prim)
# エラーが出ている各キャパシティを、必要量以上に引き上げます
physx_scene_api.GetGpuFoundLostAggregatePairsCapacityAttr().Set(262144) # 十分な大きさ
physx_scene_api.GetGpuTotalAggregatePairsCapacityAttr().Set(262144)
physx_scene_api.GetGpuMaxRigidContactCountAttr().Set(1048576)
physx_scene_api.GetGpuMaxRigidPatchCountAttr().Set(163840)
# -----------------------------------------------age

num_envs = 100
grid_size = int(np.ceil(np.sqrt(num_envs)))
spacing = 5.0

floating_height = 1.5
cuboid_size = 0.5
arm_length = 0.5
arm_radius = 0.01
bob_radius = 0.05

offsets_tensor = torch.zeros((num_envs, 2), device=device)

print("環境を構築中（剛体BaseLink + FixedJointによる完全GPU Articulation化）...")
for i in range(num_envs):
    env_path = f"/World/Env_{i}"
    
    offset_x = (i % grid_size) * spacing
    offset_y = (i // grid_size) * spacing
    offsets_tensor[i, 0] = offset_x
    offsets_tensor[i, 1] = offset_y

    base_pos = np.array([offset_x, offset_y, floating_height])

    # --- 1. 土台 (BaseLink) ---
    # 【修正】質量を持たせた剛体として作成し、ここをArticulation Rootにします
    base_link = world.scene.add(DynamicCuboid(
        prim_path=f"{env_path}/BaseLink", name=f"base_link_{i}", 
        position=base_pos, scale=np.array([0.01, 0.01, 0.01]), mass=1.0
    ))
    base_link.prim.GetAttribute("visibility").Set("invisible")

    # BaseLink を World(空間) に FixedJoint で完全に固定
    root_joint = UsdPhysics.FixedJoint.Define(stage, f"{env_path}/RootFixedJoint")
    root_joint.CreateBody1Rel().SetTargets([f"{env_path}/BaseLink"])

    # 【重要】BaseLink を Articulation Root に設定
    UsdPhysics.ArticulationRootAPI.Apply(base_link.prim)
    PhysxSchema.PhysxArticulationAPI.Apply(base_link.prim)

    # --- 2. X軸移動用のダミーリンク (SliderX) ---
    slider_x = world.scene.add(DynamicCuboid(
        prim_path=f"{env_path}/SliderX", name=f"slider_x_{i}", 
        position=base_pos, scale=np.array([0.01, 0.01, 0.01]), mass=0.1
    ))
    slider_x.prim.GetAttribute("visibility").Set("invisible")

    # --- 3. 実際の台車 (MyCuboid) ---
    cuboid_pos = base_pos
    cuboid = world.scene.add(DynamicCuboid(
        prim_path=f"{env_path}/MyCuboid", name=f"cuboid_{i}", 
        position=cuboid_pos, scale=np.array([cuboid_size]*3), color=np.array([0.2, 0.8, 0.2]), mass=5.0
    ))

    # --- 4. 振り子 (Arm & Bob) ---
    arm_pos = cuboid_pos - np.array([0.0, 0.0, (cuboid_size/2) + (arm_length/2)])
    bob_pos = arm_pos - np.array([0.0, 0.0, (arm_length/2) + bob_radius])

    arm = world.scene.add(DynamicCylinder(prim_path=f"{env_path}/Arm", name=f"arm_{i}", position=arm_pos, radius=arm_radius, height=arm_length, color=np.array([0.8, 0.8, 0.8])))
    bob = world.scene.add(DynamicSphere(prim_path=f"{env_path}/Bob", name=f"bob_{i}", position=bob_pos, radius=bob_radius, color=np.array([0.2, 0.2, 0.2])))
    UsdPhysics.MassAPI.Apply(arm.prim).CreateDensityAttr(10.0)    
    UsdPhysics.MassAPI.Apply(bob.prim).CreateDensityAttr(11340.0) 

    # ================= 関節の構築 =================
    # X軸直動関節: BaseLink -> SliderX
    px_joint = UsdPhysics.PrismaticJoint.Define(stage, f"{env_path}/PrismaticX")
    px_joint.CreateAxisAttr("X")
    px_joint.CreateBody0Rel().SetTargets([f"{env_path}/BaseLink"])
    px_joint.CreateBody1Rel().SetTargets([f"{env_path}/SliderX"])
    px_joint.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    px_joint.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    drive_x = UsdPhysics.DriveAPI.Apply(px_joint.GetPrim(), "linear")
    drive_x.CreateTypeAttr("force"); drive_x.CreateStiffnessAttr(800.0); drive_x.CreateDampingAttr(100.0)

    # Y軸直動関節: SliderX -> MyCuboid
    py_joint = UsdPhysics.PrismaticJoint.Define(stage, f"{env_path}/PrismaticY")
    py_joint.CreateAxisAttr("Y")
    py_joint.CreateBody0Rel().SetTargets([f"{env_path}/SliderX"])
    py_joint.CreateBody1Rel().SetTargets([f"{env_path}/MyCuboid"])
    py_joint.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    py_joint.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    drive_y = UsdPhysics.DriveAPI.Apply(py_joint.GetPrim(), "linear")
    drive_y.CreateTypeAttr("force"); drive_y.CreateStiffnessAttr(800.0); drive_y.CreateDampingAttr(100.0)

    # 振り子関節 (Cuboid -> Arm -> Bob)
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

# --- GPU Viewの作成 ---
# 【修正】ArticulationRoot がついている BaseLink を指定します
envs_view = ArticulationView(prim_paths_expr="/World/Env_.*/BaseLink", name="envs_view")
world.scene.add(envs_view)

# 台車の状態取得用
cuboids_view = RigidPrimView(prim_paths_expr="/World/Env_.*/MyCuboid", name="cuboids_view")
world.scene.add(cuboids_view)

save_dir = "/home/isseebi/Desktop/user/Reinforcement_learning/study/OpenManipulation/src/dataset"
os.makedirs(save_dir, exist_ok=True)

# ==========================================
# 3. エピソードループ
# ==========================================
num_episodes = 100 
g, L = 9.81, arm_length + bob_radius
K_ff = L / g
omega_n, zeta = 3.0, 1.0 

print("データ収集を開始します（完全GPU処理）...")

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
    
    # --- 軌道パディング ---
    max_sim_steps = max_path_steps + 60
    target_traj_x = torch.zeros((max_sim_steps, num_envs), device=device)
    target_traj_y = torch.zeros((max_sim_steps, num_envs), device=device)
    
    for i in range(num_envs):
        l = path_lengths[i]
        target_traj_x[:l, i] = torch.tensor(target_paths_x[i], device=device, dtype=torch.float32)
        target_traj_x[l:, i] = target_paths_x[i][-1]
        target_traj_y[:l, i] = torch.tensor(target_paths_y[i], device=device, dtype=torch.float32)
        target_traj_y[l:, i] = target_paths_y[i][-1]
    
    world.reset()

    # --- 直動関節のインデックスを動的に取得 ---
    dof_names = envs_view.dof_names
    if episode == 0:
        print(f"✅ 認識された関節（DOF）一覧: {dof_names}")
    
    try:
        x_idx = [i for i, name in enumerate(dof_names) if "PrismaticX" in name][0]
        y_idx = [i for i, name in enumerate(dof_names) if "PrismaticY" in name][0]
    except IndexError:
        print("❌ エラー: 'PrismaticX' または 'PrismaticY' が dof_names に見つかりません。")
        simulation_app.close()
        exit()

    x_r = torch.zeros(num_envs, device=device)
    y_r = torch.zeros(num_envs, device=device)
    v_r_x = torch.zeros(num_envs, device=device)
    v_r_y = torch.zeros(num_envs, device=device)

    last_time = world.current_time
    current_step = 0
    base_history = [] 

    while simulation_app.is_running():
        world.step(render=False)
        
        if world.is_playing():
            current_time = world.current_time
            dt = current_time - last_time
            if dt <= 0: dt = 1.0 / 60.0
            last_time = current_time
            
            if current_step > max_path_steps + 50: 
                break

            target_x = target_traj_x[current_step]
            target_y = target_traj_y[current_step]

            a_r_x = (omega_n**2) * (target_x - x_r) - 2 * zeta * omega_n * v_r_x
            v_r_x += a_r_x * dt
            x_r += v_r_x * dt
            
            a_r_y = (omega_n**2) * (target_y - y_r) - 2 * zeta * omega_n * v_r_y
            v_r_y += a_r_y * dt
            y_r += v_r_y * dt

            cmd_x = x_r + K_ff * a_r_x
            cmd_y = y_r + K_ff * a_r_y

            # テンソルによるGPU一括書き込み
            cmds = torch.stack([cmd_x, cmd_y], dim=-1)
            envs_view.set_joint_position_targets(cmds, joint_indices=[x_idx, y_idx])

            # GPUで全台車の座標を一括取得
            positions, _ = cuboids_view.get_world_poses()
            step_positions = positions[:, :2] - offsets_tensor
            
            base_history.append(step_positions.clone())
            current_step += 1

    if len(base_history) > 0:
        trajectory_data = torch.stack(base_history).cpu().numpy()
        dataset_xy = np.transpose(trajectory_data, (1, 0, 2))

        dataset_path = os.path.join(save_dir, f"dataset_ep{episode:03d}.npy")
        np.save(dataset_path, dataset_xy)
        print(f"Episode {episode + 1}/{num_episodes} 完了 -> 保存: {dataset_path} | Shape: {dataset_xy.shape}")

print("全エピソードのシミュレーションとデータ収集が完了しました。")
simulation_app.close()