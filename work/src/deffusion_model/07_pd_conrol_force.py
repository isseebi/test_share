from omni.isaac.kit import SimulationApp
import numpy as np
import matplotlib.pyplot as plt

# 1. シミュレーションの起動
simulation_app = SimulationApp({"headless": False})

from omni.isaac.core import World
from omni.isaac.core.objects import DynamicCuboid, DynamicCapsule, DynamicSphere
from omni.isaac.core.prims import RigidPrimView
from pxr import UsdPhysics, Gf

class ForcePDController:
    def __init__(self, kp, kd):
        self.kp = kp
        self.kd = kd

    def compute(self, current_pos, target_pos, current_vel):
        error = target_pos - current_pos
        force = (self.kp * error) - (self.kd * current_vel)
        
        # 平面制御: Z方向の力は加えない（0にする）
        force[2] = 0.0
        return force

class FlatPendulum:
    def __init__(self, world, prim_path="/World/Pendulum", position=np.array([0.0, 0.0, 0.5]), arm_length=0.2):
        self.world = world
        self.prim_path = prim_path
        self.base_init_pos = np.array(position)
        self.arm_length = arm_length
        self.name_prefix = prim_path.split("/")[-1]
        
        self._build_pendulum()
        
        # BaseとBobをトラッキングするためのView
        self.base_view = RigidPrimView(prim_paths_expr=f"{self.prim_path}/Base", name="base_view")
        self.bob_view = RigidPrimView(prim_paths_expr=f"{self.prim_path}/Bob", name="bob_view")
        
        self.world.scene.add(self.base_view)
        self.world.scene.add(self.bob_view)

    def _build_pendulum(self):
        stage = self.world.stage
        base_pos = self.base_init_pos
        
        # Base
        self.world.scene.add(DynamicCuboid(prim_path=f"{self.prim_path}/Base", name=f"{self.name_prefix}_base",
                                          position=base_pos, scale=np.array([0.05, 0.05, 0.05]), color=np.array([0.5, 0.2, 0.2])))
        
        # Arm
        arm_pos = base_pos + np.array([0.0, 0.0, -self.arm_length / 2.0])
        self.world.scene.add(DynamicCapsule(prim_path=f"{self.prim_path}/Arm", name=f"{self.name_prefix}_arm",
                                           position=arm_pos, radius=0.005, height=self.arm_length, color=np.array([0.8, 0.8, 0.8])))

        # Bob
        bob_pos = base_pos + np.array([0.0, 0.0, -self.arm_length])
        self.world.scene.add(DynamicSphere(prim_path=f"{self.prim_path}/Bob", name=f"{self.name_prefix}_bob",
                                          position=bob_pos, radius=0.02, color=np.array([0.2, 0.2, 0.2])))

        # 物理プロパティ設定
        UsdPhysics.MassAPI.Apply(stage.GetPrimAtPath(f"{self.prim_path}/Base")).CreateDensityAttr(1000.0)
        UsdPhysics.MassAPI.Apply(stage.GetPrimAtPath(f"{self.prim_path}/Bob")).CreateDensityAttr(11340.0)
        
        # --- ジョイント設定 ---
        # 1. Baseをワールド空間に対して拘束（Z軸移動と全回転のロック）
        d6_joint = UsdPhysics.Joint.Define(stage, f"{self.prim_path}/BaseConstraint")
        d6_joint.CreateBody0Rel().SetTargets([]) # 空（None）にするとWorldに対する拘束になる
        d6_joint.CreateBody1Rel().SetTargets([f"{self.prim_path}/Base"])
        
        # ★修正: ジョイントのアンカー位置を初期位置の高さ(Z=0.5)に設定して空中の仮想平面を作る
        d6_joint.CreateLocalPos0Attr(Gf.Vec3f(float(base_pos[0]), float(base_pos[1]), float(base_pos[2])))
        d6_joint.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
        
        # Z軸の移動(transZ)をロック
        z_limit = UsdPhysics.LimitAPI.Apply(d6_joint.GetPrim(), "transZ")
        z_limit.CreateLowAttr(0.0)
        z_limit.CreateHighAttr(0.0)
        
        # ベースが回転しないように回転(rotX, rotY, rotZ)もロック
        for axis in ["rotX", "rotY", "rotZ"]:
            rot_limit = UsdPhysics.LimitAPI.Apply(d6_joint.GetPrim(), axis)
            rot_limit.CreateLowAttr(0.0)
            rot_limit.CreateHighAttr(0.0)

        # 2. Base と Arm のジョイント
        sj = UsdPhysics.SphericalJoint.Define(stage, f"{self.prim_path}/SphericalJoint")
        sj.CreateBody0Rel().SetTargets([f"{self.prim_path}/Base"])
        sj.CreateBody1Rel().SetTargets([f"{self.prim_path}/Arm"])
        sj.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, 0.0))
        sj.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, self.arm_length / 2.0))

        # 3. Arm と Bob のジョイント
        fj = UsdPhysics.FixedJoint.Define(stage, f"{self.prim_path}/FixedJoint")
        fj.CreateBody0Rel().SetTargets([f"{self.prim_path}/Arm"])
        fj.CreateBody1Rel().SetTargets([f"{self.prim_path}/Bob"])
        fj.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, -self.arm_length / 2.0))
        fj.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))

# --- メイン処理 ---
world = World(stage_units_in_meters=1.0)
world.scene.add_default_ground_plane()
pendulum = FlatPendulum(world=world)

# ゲイン調整
controller = ForcePDController(kp=10.0, kd=2.0)

world.reset()

target_position = np.array([0.5, 0.0, 0.5])

# データ記録用リスト
base_history = []
bob_history = []
steps = 0
max_steps = 300

print("シミュレーション開始中... 300ステップ後にグラフを保存します。")

while simulation_app.is_running():
    # 状態取得
    b_pos, _ = pendulum.base_view.get_world_poses()
    b_vel = pendulum.base_view.get_linear_velocities()
    bob_pos, _ = pendulum.bob_view.get_world_poses()
    
    current_pos = b_pos[0]
    current_vel = b_vel[0]
    current_bob_pos = bob_pos[0]
    
    # 記録
    base_history.append(current_pos.copy())
    bob_history.append(current_bob_pos.copy())
    
    # Z方向がロックされたため、重力補償なしのクリーンなPD力のみを加える
    pd_force = controller.compute(current_pos, target_position, current_vel)
    
    pendulum.base_view.apply_forces(pd_force.reshape(1, 3))
    world.step(render=True)
    
    steps += 1
    if steps >= max_steps:
        break

# --- グラフの描画と保存 ---
base_history = np.array(base_history)
bob_history = np.array(bob_history)
time_axis = np.arange(max_steps)

plt.figure(figsize=(12, 5))

# 1. ベースの座標グラフ
plt.subplot(1, 2, 1)
plt.plot(time_axis, base_history[:, 0], label="X (Base)")
plt.plot(time_axis, base_history[:, 1], label="Y (Base)")
plt.plot(time_axis, base_history[:, 2], label="Z (Base)")
plt.axhline(y=0.5, color='r', linestyle='--', label="Target X/Z")
plt.title("Base Position History")
plt.xlabel("Steps")
plt.ylabel("Position (m)")
plt.legend()
plt.grid(True)

# 2. 振り子（重り）の座標グラフ
plt.subplot(1, 2, 2)
plt.plot(time_axis, bob_history[:, 0], label="X (Bob)")
plt.plot(time_axis, bob_history[:, 1], label="Y (Bob)")
plt.plot(time_axis, bob_history[:, 2], label="Z (Bob)")
plt.title("Bob Position History")
plt.xlabel("Steps")
plt.ylabel("Position (m)")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("pendulum_control_plot.png")
print("グラフを 'pendulum_control_plot.png' として保存しました。")

simulation_app.close()