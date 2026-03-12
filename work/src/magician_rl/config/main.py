from isaacsim import SimulationApp

# 1. アプリ起動 (他のomniモジュールをインポートする前に必須)
simulation_app = SimulationApp({"headless": False})

import numpy as np
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage, get_current_stage
from omni.isaac.core.utils.viewports import set_camera_view
from omni.isaac.core.robots import Robot
from omni.isaac.core.objects import DynamicCylinder
from pxr import Usd, UsdGeom, UsdPhysics, Gf, Sdf

class MagicianController:
    def __init__(self, world: World, usd_path: str, robot_prim_path: str, obstacle_config: dict):
        self.world = world
        self.stage = world.stage
        self.robot_prim_path = robot_prim_path
        
        # --- ロボットのロード ---
        add_reference_to_stage(usd_path=usd_path, prim_path=robot_prim_path)
        self.robot = Robot(prim_path=robot_prim_path, name="magician")
        self.world.scene.add(self.robot)

        # --- 障害物のロード ---
        self.obstacle_path = obstacle_config["path"]
        self.obstacle = world.scene.add(
            DynamicCylinder(
                prim_path=self.obstacle_path,
                name="obstacle",
                position=obstacle_config["position"],
                radius=obstacle_config["radius"],
                height=obstacle_config["height"],
                color=obstacle_config["color"],
                mass=obstacle_config["mass"]
            )
        )

        # --- 手先(吸着パッド)の構築 ---
        self._setup_end_effector()

        # --- 初期化 (Joint Indexの取得などはReset後に行う必要があるため) ---
        self.world.reset()
        self._setup_joint_indices()

        # 吸着状態管理
        self.is_attached = False
        self.joint_path = f"{self.sphere_path}/AdhocFixedJoint"
        
        # ターゲット（吸着対象）のPrimキャッシュ
        self.target_prim = self.stage.GetPrimAtPath(self.obstacle_path)

    def _setup_end_effector(self):
        """手先に赤い球体(物理判定付き)を追加する"""
        target_link_path = f"{self.robot_prim_path}/magician_link_4"
        if not self.stage.GetPrimAtPath(target_link_path):
            target_link_path = self.robot_prim_path 

        self.sphere_path = f"{target_link_path}/TipContactSphere"
        sphere_geom = UsdGeom.Sphere.Define(self.stage, self.sphere_path)
        sphere_geom.GetRadiusAttr().Set(0.005) 
        sphere_geom.GetDisplayColorAttr().Set([(1.0, 0.0, 0.0)]) 
        sphere_geom.AddTranslateOp().Set(Gf.Vec3d(0.06, 0.0, -0.059)) 

        # 物理判定追加
        self.sphere_prim = sphere_geom.GetPrim()
        UsdPhysics.CollisionAPI.Apply(self.sphere_prim)

    def _setup_joint_indices(self):
        """関節名のインデックスを特定する"""
        dof_names = self.robot.dof_names
        dof_dict = {name: i for i, name in enumerate(dof_names)}
        self.idx_j1 = dof_dict.get('magician_joint_1')
        self.idx_j2 = dof_dict.get('magician_joint_2')
        self.idx_j3 = dof_dict.get('magician_joint_3')
        self.idx_m1 = dof_dict.get('magician_joint_mimic_1')
        self.idx_m2 = dof_dict.get('magician_joint_mimic_2')

    def apply_actions(self, joint_angles: list, suction_on: bool):
        """
        ロボットを制御するメイン関数
        
        Args:
            joint_angles (list): [j1, j2, j3] の目標角度 (ラジアン)
            suction_on (bool): Trueなら吸着ON、Falseなら吸着OFF
        """
        # 1. 関節角度の適用 (Mimic関節の計算を含む)
        current_positions = self.robot.get_joint_positions()
        
        if len(joint_angles) >= 3:
            t_j1, t_j2, t_j3 = joint_angles[0], joint_angles[1], joint_angles[2]

            if self.idx_j1 is not None: current_positions[self.idx_j1] = t_j1
            if self.idx_j2 is not None: current_positions[self.idx_j2] = t_j2
            if self.idx_j3 is not None: current_positions[self.idx_j3] = t_j3
            
            # 平行リンク(Mimic)の計算: 一般的に j_mimic = -j_drive
            if self.idx_m1 is not None: current_positions[self.idx_m1] = -1.0 * t_j2
            if self.idx_m2 is not None: current_positions[self.idx_m2] = -1.0 * t_j3

        self.robot.set_joint_positions(current_positions)

        # 2. グリッパ(吸着)の制御
        if suction_on:
            self._attach_logic()
        else:
            self._detach_logic()

    def _attach_logic(self, threshold=0.035):
        """吸着を実行する (距離が近ければ固定ジョイント作成)"""
        if self.is_attached:
            return # 既にくっついている

        if not self.target_prim or not self.sphere_prim:
            return

        # 距離計算
        xform_cache = UsdGeom.XformCache()
        mat_sphere = xform_cache.GetLocalToWorldTransform(self.sphere_prim)
        mat_target = xform_cache.GetLocalToWorldTransform(self.target_prim)
        
        pos_sphere = mat_sphere.ExtractTranslation()
        pos_target = mat_target.ExtractTranslation()
        dist = np.linalg.norm(np.array(pos_sphere) - np.array(pos_target))

        if dist < threshold:
            print(f"Hit! Distance: {dist:.4f} -> Attaching...")
            
            # FixedJointを作成
            joint = UsdPhysics.FixedJoint.Define(self.stage, self.joint_path)
            joint.CreateBody0Rel().SetTargets([self.sphere_prim.GetPath()])
            joint.CreateBody1Rel().SetTargets([self.target_prim.GetPath()])
            joint.CreateExcludeFromArticulationAttr().Set(True)
            
            self.is_attached = True

    def _detach_logic(self):
        """吸着を解除する (固定ジョイント削除)"""
        if not self.is_attached:
            return # くっついていない

        print("Detaching...")
        # ジョイントのPrimを削除して結合を解く
        self.stage.RemovePrim(self.joint_path)
        self.is_attached = False

# =========================================================
#  実行スクリプト (呼び出し側の例)
# =========================================================
if __name__ == "__main__":
    # 1. ワールド作成
    world = World(stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()

    # パス設定 (環境に合わせて変更してください)
    USD_PATH = "/home/isseebi/Desktop/user/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/original/config/magcian/magician_ros2/dobot_description/model/magician_suction.usd"
    
    obstacle_config = {
        "path": "/World/ObstacleCylinder",
        "position": np.array([0.22, 0.0, 0.025]),
        "radius": 0.02,
        "height": 0.05,
        "color": np.array([0.0, 0.0, 1.0]),
        "mass": 0.1 
    }

    # 2. クラスのインスタンス化
    controller = MagicianController(
        world=world, 
        usd_path=USD_PATH, 
        robot_prim_path="/World/Magician",
        obstacle_config=obstacle_config
    )

    # カメラ位置
    set_camera_view(eye=[0.6, 0.6, 0.4], target=[0.0, 0.0, 0.0])

    print("Isaac Sim 実行中... (クラス呼び出しデモ)")
    
    i = 0
    # グリッパのON/OFFを切り替えるためのフラグ変数
    suction_command = True 

    while simulation_app.is_running():
        # --- 呼び出し側での制御指令 ---
        
        # 例: 正弦波で動かす
        target_j1 = np.sin(i * 0.02) * 0.5
        target_j2 = 0.45 + np.sin(i * 0.02) * 0.2
        target_j3 = 0.3 + np.cos(i * 0.02) * 0.2
        
        joint_targets = [target_j1, target_j2, target_j3]

        # 例: 一定周期で吸着ON/OFFを切り替え
        if i % 300 == 0:
            suction_command = not suction_command
            print(f"Suction Command: {suction_command}")

        # ★ クラスのメソッドを使用してロボットを操作 ★
        controller.apply_actions(joint_angles=joint_targets, suction_on=suction_command)

        # ステップ進行
        world.step(render=True)
        i += 1

    simulation_app.close()