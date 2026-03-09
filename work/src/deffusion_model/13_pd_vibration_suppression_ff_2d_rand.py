from traject_gen import RandomPathGenerator

from omni.isaac.kit import SimulationApp

# 1. シミュレーションアプリの起動
simulation_app = SimulationApp({"headless": False})

import numpy as np
from omni.isaac.core import World
from omni.isaac.core.objects import DynamicCuboid, DynamicCylinder, DynamicSphere
from pxr import UsdPhysics, Gf
import matplotlib.pyplot as plt
import os
from scipy.interpolate import make_interp_spline, interp1d

# ==========================================
# 追加: 目標値生成クラス
# ==========================================
class RandomPathGenerator:
    def __init__(self, step_size=0.01, num_points_range=(4, 7)):
        self.step_size = step_size
        self.num_points = np.random.randint(num_points_range[0], num_points_range[1])
        self.raw_points = None
        self.path_fine = {}
        self.path_approx = {}

    def generate(self):
        self._generate_random_points()
        self._create_smooth_base()
        self._resample_by_step_size()
        return self.path_approx['x'], self.path_approx['y']

    def _generate_random_points(self):
        # 0.0からスタートし、±2.0の範囲でランダムな点を生成
        random_coords = np.random.uniform(-2.0, 2.0, size=(self.num_points, 2))
        self.raw_points = {
            'x': np.insert(random_coords[:, 0], 0, 0.0),
            'y': np.insert(random_coords[:, 1], 0, 0.0)
        }

    def _create_smooth_base(self, resolution=2000):
        x, y = self.raw_points['x'], self.raw_points['y']
        t = np.linspace(0, 1, len(x))
        t_fine = np.linspace(0, 1, resolution)
        k = min(3, len(x) - 1)
        spl_x = make_interp_spline(t, x, k=k)
        spl_y = make_interp_spline(t, y, k=k)
        self.path_fine['x'] = spl_x(t_fine)
        self.path_fine['y'] = spl_y(t_fine)

    def _resample_by_step_size(self):
        x_f, y_f = self.path_fine['x'], self.path_fine['y']
        dx = np.diff(x_f); dy = np.diff(y_f)
        cumulative_dist = np.insert(np.cumsum(np.sqrt(dx**2 + dy**2)), 0, 0)
        total_length = cumulative_dist[-1]
        sampling_distances = np.arange(0, total_length, self.step_size)
        
        interp_x = interp1d(cumulative_dist, x_f, kind='linear')
        interp_y = interp1d(cumulative_dist, y_f, kind='linear')
        self.path_approx['x'] = interp_x(sampling_distances)
        self.path_approx['y'] = interp_y(sampling_distances)

# ==========================================
# 2. ワールドのセットアップ
# ==========================================
world = World(stage_units_in_meters=1.0)
world.scene.add_default_ground_plane()
stage = world.stage

floating_height = 1.5
cuboid_size = 0.5
arm_length = 0.5
arm_radius = 0.01
bob_radius = 0.05

# 初期位置の計算
cuboid_pos = np.array([0.0, 0.0, floating_height])
arm_pos = cuboid_pos - np.array([0.0, 0.0, (cuboid_size/2) + (arm_length/2)])
bob_pos = arm_pos - np.array([0.0, 0.0, (arm_length/2) + bob_radius])

cuboid = world.scene.add(DynamicCuboid(prim_path="/World/MyCuboid", name="my_cuboid", position=cuboid_pos, scale=np.array([cuboid_size]*3), color=np.array([0.2, 0.8, 0.2]), mass=5.0))
arm = world.scene.add(DynamicCylinder(prim_path="/World/Arm", name="arm", position=arm_pos, radius=arm_radius, height=arm_length, color=np.array([0.8, 0.8, 0.8])))
bob = world.scene.add(DynamicSphere(prim_path="/World/Bob", name="bob", position=bob_pos, radius=bob_radius, color=np.array([0.2, 0.2, 0.2])))

UsdPhysics.MassAPI.Apply(arm.prim).CreateDensityAttr(10.0)    
UsdPhysics.MassAPI.Apply(bob.prim).CreateDensityAttr(11340.0) 

# ジョイント作成
spherical_joint = UsdPhysics.SphericalJoint.Define(stage, "/World/SphericalJoint")
spherical_joint.CreateBody0Rel().SetTargets(["/World/MyCuboid"])
spherical_joint.CreateBody1Rel().SetTargets(["/World/Arm"])
spherical_joint.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, -cuboid_size / 2.0))
spherical_joint.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, arm_length / 2.0))

fixed_joint = UsdPhysics.FixedJoint.Define(stage, "/World/FixedJoint")
fixed_joint.CreateBody0Rel().SetTargets(["/World/Arm"])
fixed_joint.CreateBody1Rel().SetTargets(["/World/Bob"])
fixed_joint.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, -arm_length / 2.0))
fixed_joint.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, bob_radius))

# 台車制御用D6ジョイント
d6_joint = UsdPhysics.Joint.Define(stage, "/World/FloatingJoint")
d6_joint.CreateBody1Rel().SetTargets(["/World/MyCuboid"])

kp, kd = 800.0, 100.0
drive_x = UsdPhysics.DriveAPI.Apply(d6_joint.GetPrim(), "transX")
drive_x.CreateTypeAttr("force"); drive_x.CreateStiffnessAttr(kp); drive_x.CreateDampingAttr(kd)
drive_y = UsdPhysics.DriveAPI.Apply(d6_joint.GetPrim(), "transY")
drive_y.CreateTypeAttr("force"); drive_y.CreateStiffnessAttr(kp); drive_y.CreateDampingAttr(kd)
z_drive = UsdPhysics.DriveAPI.Apply(d6_joint.GetPrim(), "transZ")
z_drive.CreateStiffnessAttr(50000.0); z_drive.CreateTargetPositionAttr(floating_height)

for axis in ["rotX", "rotY", "rotZ"]:
    limit = UsdPhysics.LimitAPI.Apply(d6_joint.GetPrim(), axis)
    limit.CreateLowAttr(0.0); limit.CreateHighAttr(0.0)

# ==========================================
# 3. 経路の生成
# ==========================================
path_gen = RandomPathGenerator(step_size=0.02) # シミュレーションの進みに合わせて細かく
target_path_x, target_path_y = path_gen.generate()
max_path_steps = len(target_path_x)

world.reset()

# 制御パラメータ
g, L = 9.81, arm_length + bob_radius
K_ff = L / g
omega_n, zeta = 3.0, 1.0 # 追従性を上げるため少し高めに設定

x_r, v_r = 0.0, 0.0
y_r, v_r_y = 0.0, 0.0
last_time = 0.0
current_path_idx = 0

base_history, bob_history, target_history, ref_history = [], [], [], []

# ==========================================
# 4. メインループ
# ==========================================
while simulation_app.is_running():
    world.step(render=True)
    
    if world.is_playing():
        current_time = world.current_time
        dt = current_time - last_time
        if dt <= 0: dt = 1.0 / 60.0
        last_time = current_time
        
        # 経路から現在の目標値を取得
        if current_path_idx < max_path_steps:
            target_x = target_path_x[current_path_idx]
            target_y = target_path_y[current_path_idx]
            current_path_idx += 1
        else:
            # 経路が終わったら少し待って終了
            if current_path_idx > max_path_steps + 100: break
            current_path_idx += 1

        # 2次遅れ系フィルタ & フィードフォワード
        a_r_x = (omega_n**2) * (target_x - x_r) - 2 * zeta * omega_n * v_r
        v_r += a_r_x * dt; x_r += v_r * dt
        a_r_y = (omega_n**2) * (target_y - y_r) - 2 * zeta * omega_n * v_r_y
        v_r_y += a_r_y * dt; y_r += v_r_y * dt

        cmd_x = x_r + K_ff * a_r_x
        cmd_y = y_r + K_ff * a_r_y

        drive_x.GetTargetPositionAttr().Set(cmd_x)
        drive_y.GetTargetPositionAttr().Set(cmd_y)

        # 記録
        b_pos, _ = cuboid.get_world_pose()
        p_pos, _ = bob.get_world_pose()
        base_history.append(b_pos.copy())
        bob_history.append(p_pos.copy())
        target_history.append([target_x, target_y])
        ref_history.append([x_r, y_r])

# グラフ描画
base_h = np.array(base_history); bob_h = np.array(bob_history)
t_h = np.array(target_history); r_h = np.array(ref_history)

plt.figure(figsize=(8, 8))
plt.plot(t_h[:, 0], t_h[:, 1], 'r--', label="Random Target")
plt.plot(r_h[:, 0], r_h[:, 1], 'k-', label="Smooth Ref")
plt.plot(base_h[:, 0], base_h[:, 1], 'b-', label="Base (Cart)")
plt.plot(bob_h[:, 0], bob_h[:, 1], 'g-', alpha=0.6, label="Bob (Load)")
plt.legend(); plt.grid(True); plt.axis('equal'); plt.title("Random Path Tracking with Swing Suppression")
save_dir = "/home/isseebi/Desktop/user/Reinforcement_learning/study/OpenManipulation/src"
os.makedirs(save_dir, exist_ok=True) # ディレクトリが存在しない場合のエラーを防止
save_path = os.path.join(save_dir, "no_13.png")
plt.savefig(save_path)
plt.close()
simulation_app.close()

# from traject_gen import RandomPathGenerator

# if __name__ == "__main__":
#     generator = RandomPathGenerator(step_size=0.15)
#     x_approx, y_approx = generator.generate()

#     print(x_approx)
#     print(f"生成されたポイント数: {len(x_approx)}")
#     print(f"経路の全長: {generator.total_length:.2f}")
    

#     # generator.plot()