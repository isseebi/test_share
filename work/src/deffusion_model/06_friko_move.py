from omni.isaac.kit import SimulationApp
import os
simulation_app = SimulationApp({"headless": False})

from omni.isaac.core import World
from omni.isaac.core.objects import DynamicCuboid, DynamicCapsule, DynamicSphere
import numpy as np
from pxr import UsdPhysics, Gf, PhysxSchema

class FlatPendulum:
    def __init__(self, world, prim_path="/World/Pendulum", position=np.array([0.0, 0.0, 0.0]), arm_length=0.1):
        self.world = world
        self.prim_path = prim_path
        self.base_position = np.array(position)
        self.arm_length = arm_length
        self.name_prefix = prim_path.split("/")[-1]
        self._build_pendulum()

    def _build_pendulum(self):
        stage = self.world.stage
        base_pos = self.base_position
        
        # --- 修正点1: 座標を-Z方向（真下）に計算する ---
        # Armの中心はBaseから長さの半分だけ下
        arm_pos  = base_pos + np.array([0.0, 0.0, -self.arm_length / 2.0])
        # Bobの中心はBaseから長さ分だけ下
        bob_pos  = base_pos + np.array([0.0, 0.0, -self.arm_length])
        
        # --- 修正点2: 向きをデフォルト（垂直）にする ---
        # [w, x, y, z] = [1, 0, 0, 0] は回転なしの状態
        arm_orientation = np.array([1.0, 0.0, 0.0, 0.0])

        # 1. 支柱 (Base)
        self.base = self.world.scene.add(
            DynamicCuboid(
                prim_path=f"{self.prim_path}/Base", 
                name=f"{self.name_prefix}_base",
                position=base_pos, 
                scale=np.array([0.03, 0.03, 0.03]), 
                color=np.array([0.5, 0.2, 0.2])
            )
        )

        # 2. 棒 (Arm)
        self.arm = self.world.scene.add(
            DynamicCapsule(
                prim_path=f"{self.prim_path}/Arm",
                name=f"{self.name_prefix}_arm",
                position=arm_pos,
                orientation=arm_orientation,
                radius=0.002,
                height=self.arm_length,
                color=np.array([0.8, 0.8, 0.8])
            )
        )

        # 3. 重りの球 (Bob)
        self.bob = self.world.scene.add(
            DynamicSphere(
                prim_path=f"{self.prim_path}/Bob",
                name=f"{self.name_prefix}_bob",
                position=bob_pos,
                radius=0.01,
                color=np.array([0.2, 0.2, 0.2])
            )
        )

        # --- 物理プロパティの設定 ---
        UsdPhysics.MassAPI.Apply(self.base.prim).CreateDensityAttr(1000.0) 
        UsdPhysics.MassAPI.Apply(self.arm.prim).CreateDensityAttr(10.0) 
        UsdPhysics.MassAPI.Apply(self.bob.prim).CreateDensityAttr(11340.0)
        
        LINEAR_DAMPING = 0.5 
        for prim in [self.base.prim, self.arm.prim, self.bob.prim]:
            physx_rb = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
            physx_rb.CreateLinearDampingAttr().Set(LINEAR_DAMPING)

        # --- ジョイントの設定 ---
        self.root_joint = UsdPhysics.FixedJoint.Define(stage, f"{self.prim_path}/RootFixedJoint")
        self.root_joint.CreateBody0Rel().SetTargets([self.base.prim_path])
        self.root_joint.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, 0.0))
        self.root_joint.CreateLocalPos1Attr(Gf.Vec3f(*(base_pos.tolist())))

        # Base と Arm をつなぐ回転ジョイント
        spherical_joint = UsdPhysics.SphericalJoint.Define(stage, f"{self.prim_path}/SphericalJoint")
        spherical_joint.CreateBody0Rel().SetTargets([self.base.prim_path])
        spherical_joint.CreateBody1Rel().SetTargets([self.arm.prim_path])
        spherical_joint.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, 0.0))
        # --- 修正点3: Armの中心から見て上方向(+Z)にジョイントがあるように設定 ---
        spherical_joint.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, self.arm_length / 2.0)) 

        # Arm と Bob をつなぐ固定ジョイント
        fixed_joint = UsdPhysics.FixedJoint.Define(stage, f"{self.prim_path}/FixedJoint")
        fixed_joint.CreateBody0Rel().SetTargets([self.arm.prim_path])
        fixed_joint.CreateBody1Rel().SetTargets([self.bob.prim_path])
        # Armの中心から見て下方向(-Z)にBobをつなぐ
        fixed_joint.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, -self.arm_length / 2.0)) 
        fixed_joint.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))

# --- メイン処理 ---
world = World(stage_units_in_meters=1.0)
world.scene.add_default_ground_plane()

# 垂直に配置されるため、シミュレーション開始時に揺れなくなります
pendulum1 = FlatPendulum(
    world=world, 
    prim_path="/World/Pendulum_1", 
    position=np.array([0.0, 0.0, 0.5]), 
    arm_length=0.2
)

world.reset()

# 移動先のターゲット
target_position = np.array([0.5, 0.0, 0.5]) 
move_speed = 0.2 

while simulation_app.is_running():
    current_pos_gf = pendulum1.root_joint.GetLocalPos1Attr().Get()
    current_pos = np.array([current_pos_gf[0], current_pos_gf[1], current_pos_gf[2]])
    
    direction = target_position - current_pos
    distance = np.linalg.norm(direction)
    
    if distance > 0.001:
        step_size = move_speed * (1.0 / 60.0) 
        if distance < step_size:
            new_pos = target_position
        else:
            new_pos = current_pos + (direction / distance) * step_size
        
        pendulum1.root_joint.GetLocalPos1Attr().Set(Gf.Vec3f(*(new_pos.tolist())))

    world.step(render=True)

simulation_app.close()