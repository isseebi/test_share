import numpy as np
from omni.isaac.core.objects import DynamicCuboid, DynamicCapsule, DynamicSphere
from pxr import UsdPhysics, Gf, PhysxSchema

class FlatPendulum:
    # 【追加】引数に arm_length を追加（デフォルトを0.5に設定）
    def __init__(self, world, prim_path="/World/Pendulum", position=np.array([0.0, 0.0, 0.0]), arm_length=0.1):
        self.world = world
        self.prim_path = prim_path
        self.base_position = np.array(position)
        self.arm_length = arm_length  # 棒の長さを保持
        
        self.name_prefix = prim_path.split("/")[-1]
        self._build_pendulum()

    def _build_pendulum(self):
        stage = self.world.stage

        # --- 座標と回転の計算 ---
        z_offset = 0.08
        pos_offset = self.base_position + np.array([0.0, 0.0, z_offset])
        
        # 【修正】arm_length を基準に配置座標を自動計算
        base_pos = pos_offset
        arm_pos  = pos_offset + np.array([self.arm_length / 2.0, 0.0, 0.0]) # 棒の中心は長さの半分
        bob_pos  = pos_offset + np.array([self.arm_length, 0.0, 0.0])       # 重りは長さの先端

        arm_orientation = np.array([0.7071068, 0.0, 0.7071068, 0.0])

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
                radius=0.001,
                height=self.arm_length,  # 【修正】変数を使用
                color=np.array([0.8, 0.8, 0.8])
            )
        )

        # 3. 重りの球 (Bob)
        self.bob = self.world.scene.add(
            DynamicSphere(
                prim_path=f"{self.prim_path}/Bob",
                name=f"{self.name_prefix}_bob",
                position=bob_pos,
                radius=0.005,
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
            physx_rb.CreateLinearDampingAttr(LINEAR_DAMPING)

        # --- ジョイントの設定 ---
        spherical_joint = UsdPhysics.SphericalJoint.Define(stage, f"{self.prim_path}/SphericalJoint")
        spherical_joint.CreateBody0Rel().SetTargets([self.base.prim_path])
        spherical_joint.CreateBody1Rel().SetTargets([self.arm.prim_path])
        spherical_joint.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, 0.0))
        # 【修正】棒の中心から根元へのオフセットを自動計算（マイナス方向）
        spherical_joint.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, -self.arm_length / 2.0)) 

        fixed_joint = UsdPhysics.FixedJoint.Define(stage, f"{self.prim_path}/FixedJoint")
        fixed_joint.CreateBody0Rel().SetTargets([self.arm.prim_path])
        fixed_joint.CreateBody1Rel().SetTargets([self.bob.prim_path])
        # 【修正】棒の中心から先端へのオフセットを自動計算（プラス方向）
        fixed_joint.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, self.arm_length / 2.0)) 
        fixed_joint.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))