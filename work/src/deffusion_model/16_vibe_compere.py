from traject_gen import CustomPathGenerator
from omni.isaac.kit import SimulationApp
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

# 1. アプリ起動
simulation_app = SimulationApp({"headless": True})

from omni.isaac.core import World
from omni.isaac.core.prims import RigidPrimView
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.objects import DynamicCuboid, DynamicCylinder, DynamicSphere
from pxr import UsdPhysics, Gf, PhysxSchema

# 2. ワールド構築
device = "cuda:0"
world = World(stage_units_in_meters=1.0, backend="torch", device=device)
world.scene.add_default_ground_plane()
stage = world.stage

physics_scene_prim = stage.GetPrimAtPath("/physicsScene")
if not physics_scene_prim.IsValid():
    UsdPhysics.Scene.Define(stage, "/physicsScene")
    physics_scene_prim = stage.GetPrimAtPath("/physicsScene")
physx_scene_api = PhysxSchema.PhysxSceneAPI.Apply(physics_scene_prim)
physx_scene_api.GetGpuMaxRigidContactCountAttr().Set(1048576)

num_envs = 1
grid_size = int(np.ceil(np.sqrt(num_envs)))
spacing = 5.0
floating_height = 1.5
cuboid_size = 0.5
arm_length = 0.5
bob_radius = 0.05

offsets_tensor = torch.zeros((num_envs, 2), device=device)

print("環境を構築中...")
for i in range(num_envs):
    env_path = f"/World/Env_{i}"
    offset_x, offset_y = (i % grid_size) * spacing, (i // grid_size) * spacing
    offsets_tensor[i, 0], offsets_tensor[i, 1] = offset_x, offset_y
    base_pos = np.array([offset_x, offset_y, floating_height])

    # ================= オブジェクト生成（質量を安定化） =================
    # 極端に軽い(0.1kg)と重い(5kg)が混在すると計算が爆発するため、全体的に質量を底上げ
    base_link = world.scene.add(DynamicCuboid(prim_path=f"{env_path}/BaseLink", name=f"base_link_{i}", position=base_pos, scale=np.array([0.01]*3), mass=2.0))
    slider_x = world.scene.add(DynamicCuboid(prim_path=f"{env_path}/SliderX", name=f"slider_x_{i}", position=base_pos, scale=np.array([0.01]*3), mass=2.0))
    cuboid = world.scene.add(DynamicCuboid(prim_path=f"{env_path}/MyCuboid", name=f"cuboid_{i}", position=base_pos, scale=np.array([cuboid_size]*3), color=np.array([0.2, 0.8, 0.2]), mass=5.0))
    
    arm_pos = base_pos - np.array([0.0, 0.0, (cuboid_size/2) + (arm_length/2)])
    arm = world.scene.add(DynamicCylinder(prim_path=f"{env_path}/Arm", name=f"arm_{i}", position=arm_pos, radius=0.01, height=arm_length, mass=1.0))
    bob_pos = arm_pos - np.array([0.0, 0.0, (arm_length/2) + bob_radius])
    bob = world.scene.add(DynamicSphere(prim_path=f"{env_path}/Bob", name=f"bob_{i}", position=bob_pos, radius=bob_radius, mass=2.0))

    base_link.prim.GetAttribute("visibility").Set("invisible")
    slider_x.prim.GetAttribute("visibility").Set("invisible")

    # ================= 衝突無効化 =================
    UsdPhysics.CollisionAPI(base_link.prim).CreateCollisionEnabledAttr().Set(False)
    UsdPhysics.CollisionAPI(slider_x.prim).CreateCollisionEnabledAttr().Set(False)

    filter_cuboid = UsdPhysics.FilteredPairsAPI.Apply(cuboid.prim)
    filter_cuboid.CreateFilteredPairsRel().AddTarget(arm.prim.GetPath())

    filter_arm = UsdPhysics.FilteredPairsAPI.Apply(arm.prim)
    filter_arm.CreateFilteredPairsRel().AddTarget(bob.prim.GetPath())

    # ================= 関節設定 =================
    UsdPhysics.ArticulationRootAPI.Apply(base_link.prim)
    PhysxSchema.PhysxArticulationAPI.Apply(base_link.prim)

    # 【修正点1】RootJointを空中の正しい位置に固定
    root_joint = UsdPhysics.FixedJoint.Define(stage, f"{env_path}/RootJoint")
    root_joint.CreateBody1Rel().SetTargets([base_link.prim.GetPath()])
    root_joint.CreateLocalPos0Attr().Set(Gf.Vec3f(float(base_pos[0]), float(base_pos[1]), float(base_pos[2])))
    root_joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0, 0, 0))

    px_joint = UsdPhysics.PrismaticJoint.Define(stage, f"{env_path}/PrismaticX")
    px_joint.CreateAxisAttr("X")
    px_joint.CreateBody0Rel().SetTargets([base_link.prim.GetPath()])
    px_joint.CreateBody1Rel().SetTargets([slider_x.prim.GetPath()])
    drive_x = UsdPhysics.DriveAPI.Apply(px_joint.GetPrim(), "linear")
    drive_x.CreateStiffnessAttr(8000.0); drive_x.CreateDampingAttr(100.0)

    py_joint = UsdPhysics.PrismaticJoint.Define(stage, f"{env_path}/PrismaticY")
    py_joint.CreateAxisAttr("Y")
    py_joint.CreateBody0Rel().SetTargets([slider_x.prim.GetPath()])
    py_joint.CreateBody1Rel().SetTargets([cuboid.prim.GetPath()])
    drive_y = UsdPhysics.DriveAPI.Apply(py_joint.GetPrim(), "linear")
    drive_y.CreateStiffnessAttr(8000.0); drive_y.CreateDampingAttr(100.0)

    # 【修正点2】SphericalJointのアンカーをペアで設定
    sph_joint = UsdPhysics.SphericalJoint.Define(stage, f"{env_path}/SphericalJoint")
    sph_joint.CreateBody0Rel().SetTargets([cuboid.prim.GetPath()])
    sph_joint.CreateBody1Rel().SetTargets([arm.prim.GetPath()])
    # Cuboidのスケール前の底面位置(-0.5)と、Armの上面位置(arm_length/2)を指定
    sph_joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0, 0, -0.5))
    sph_joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0, 0, arm_length/2))

    fix_joint = UsdPhysics.FixedJoint.Define(stage, f"{env_path}/FixedJoint")
    fix_joint.CreateBody0Rel().SetTargets([arm.prim.GetPath()])
    fix_joint.CreateBody1Rel().SetTargets([bob.prim.GetPath()])
    fix_joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0, 0, -arm_length/2))
    fix_joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0, 0, bob_radius / 1.0)) # scale=1.0想定

# 3. View設定
envs_view = ArticulationView(prim_paths_expr="/World/Env_.*/BaseLink", name="envs_view")
world.scene.add(envs_view)
cuboids_view = RigidPrimView(prim_paths_expr="/World/Env_.*/MyCuboid", name="cuboids_view")
world.scene.add(cuboids_view)

save_dir = "./dataset"
os.makedirs(save_dir, exist_ok=True)

# 4. シミュレーションループ
bobs_view = RigidPrimView(prim_paths_expr="/World/Env_.*/Bob", name="bobs_view")
world.scene.add(bobs_view)

save_dir = "./dataset"
os.makedirs(save_dir, exist_ok=True)

# 4. シミュレーションループ
num_episodes = 1
K_ff = (arm_length + bob_radius) / 9.81
omega_n, zeta = 3.0, 1.0

my_waypoints = [
        [0.0, 0.0],#始点
        [1.0, 2.0],#途中の経由点
        # [1.150, 2.473],
        [1.0, 0.0]#ゴール
    ]

for episode in range(num_episodes):
    target_paths_x, target_paths_y, path_lengths = [], [], []
# ================= 軌道データの読み込み =================
    # optimized_trajectory.txt からデータを読み込む
    # X_Position, Y_Position のヘッダーを飛ばして読み込み
    # traj_data = np.loadtxt("optimized_trajectory.txt", delimiter=",", skiprows=1)
    # tx_from_file = traj_data[:, 0].tolist()
    # ty_from_file = traj_data[:, 1].tolist()
    # print(f"ファイルから {len(tx_from_file)} ステップの軌道を読み込みました。")
# ================= 軌道データの読み込み =================

    for i in range(num_envs):

        path_gen = CustomPathGenerator(step_size=0.02)#ここでなめらかな軌道が生成される
        tx, ty = path_gen.generate(my_waypoints)
        target_paths_x.append(tx); target_paths_y.append(ty); path_lengths.append(len(tx))

        # ファイルから読み込んだデータを使用
        # tx, ty = tx_from_file, ty_from_file
        # target_paths_x.append(tx)
        # target_paths_y.append(ty)
        # path_lengths.append(len(tx))

    max_steps = max(path_lengths)+100
    
    world.reset()
    envs_view.set_joint_position_targets(torch.zeros((num_envs, envs_view.num_dof), device=device))
    
    for _ in range(30):
        world.step(render=False)

    last_time = world.current_time
    x_r, y_r = torch.zeros(num_envs, device=device), torch.zeros(num_envs, device=device)
    v_r_x, v_r_y = torch.zeros(num_envs, device=device), torch.zeros(num_envs, device=device)
    current_step = 0
    
    # --- 修正：履歴保存用のリストを初期化 ---
    base_history = []
    bob_history = []
    target_history = []
    ref_history = []

    dof_names = envs_view.dof_names
    x_idx = [i for i, n in enumerate(dof_names) if "PrismaticX" in n][0]
    y_idx = [i for i, n in enumerate(dof_names) if "PrismaticY" in n][0]

    while simulation_app.is_running():
        world.step(render=True)
        if world.is_playing():
            dt = world.current_time - last_time
            if dt <= 0 or dt > 0.05: dt = 1.0/60.0
            last_time = world.current_time

            if current_step >= max_steps: break

            # ターゲット位置
            t_x = torch.tensor([target_paths_x[i][min(current_step, path_lengths[i]-1)] for i in range(num_envs)], device=device)
            t_y = torch.tensor([target_paths_y[i][min(current_step, path_lengths[i]-1)] for i in range(num_envs)], device=device)
            print(t_y)

            # 2次遅れ系フィルタ（Reference）
            a_r_x = (omega_n**2) * (t_x - x_r) - 2*zeta*omega_n*v_r_x
            v_r_x += a_r_x * dt
            x_r += v_r_x * dt
            cmd_x = x_r + K_ff * a_r_x
            # cmd_x = t_x

            a_r_y = (omega_n**2) * (t_y - y_r) - 2*zeta*omega_n*v_r_y
            v_r_y += a_r_y * dt
            y_r += v_r_y * dt
            cmd_y = y_r + K_ff * a_r_y
            # cmd_y = t_y

            cmds = torch.zeros((num_envs, envs_view.num_dof), device=device)
            cmds[:, x_idx] = cmd_x
            cmds[:, y_idx] = cmd_y
            envs_view.set_joint_position_targets(cmds)

            # --- 修正：各データの現在値をリストに追加 ---
            pos_base, _ = cuboids_view.get_world_poses()
            pos_bob, _ = bobs_view.get_world_poses()
            
            # env[0] のデータのみをプロット対象として保存
            base_history.append((pos_base[0, :2] - offsets_tensor[0]).cpu().numpy())
            bob_history.append((pos_bob[0, :2] - offsets_tensor[0]).cpu().numpy())
            target_history.append(np.array([t_x[0].item(), t_y[0].item()]))
            ref_history.append(np.array([x_r[0].item(), y_r[0].item()]))

            current_step += 1

    # データ保存（既存の処理）
    if len(base_history) > 0:
        dataset_xy = np.array(base_history) # 簡易化
        np.save(os.path.join(save_dir, f"dataset_ep{episode:03d}.npy"), dataset_xy)
        print(f"Episode {episode} saved.")

# ========== グラフのプロットと保存 ==========
# リストをnumpy配列に変換
base_history = np.array(base_history)
bob_history = np.array(bob_history)
target_history = np.array(target_history)

plt.figure(figsize=(10, 10))

# 1. 全体の軌跡を細い線で描画
# plt.plot(target_history[:, 0], target_history[:, 1], 'r--', alpha=0.5, label="Target Path")
# plt.plot(base_history[:, 0], base_history[:, 1], 'b-', alpha=0.3, label="Base Trace")

# 2. 一定の間隔（例：50ステップごと）で通過点にプロットを打つ
plot_interval = 5
plt.scatter(
    base_history[::plot_interval, 0], 
    base_history[::plot_interval, 1], 
    color='blue', 
    marker='o', 
    s=30, 
)

# 3. スタートとゴールを強調
plt.plot(base_history[0, 0], base_history[0, 1], 'go', markersize=10, label="START")
plt.plot(base_history[-1, 0], base_history[-1, 1], 'rx', markersize=10, label="END")
plt.plot(target_history[:, 0], target_history[:, 1], 'r--', label="Target Path")
# plt.plot(ref_history[:, 0], ref_history[:, 1], 'k:', label="Reference (Filtered)")
plt.plot(base_history[:, 0], base_history[:, 1], 'b-', alpha=0.5, linewidth=3.0, label="Base (Cuboid)")
plt.plot(bob_history[:, 0], bob_history[:, 1], 'g-', alpha=0.8, label="Bob (Load)")
# plt.plot(my_waypoints[1][0], my_waypoints[1][1], 'co', markersize=8, label="Waypoint")
plt.plot(1.0, 2.0, 'm*', markersize=15, label="Target Point")

plt.title("XY Trajectory")
plt.xlabel("X Position (m)")
plt.ylabel("Y Position (m)")
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.tight_layout()

save_path = "trajectory_with_markers.png"
plt.savefig(save_path)
print(f"Graph saved at {save_path}")

simulation_app.close()