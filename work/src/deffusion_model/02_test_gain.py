from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({"headless": False})

from omni.isaac.core import World
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.types import ArticulationAction
import numpy as np
from pxr import Usd, UsdPhysics

# ワールドとパスの設定
world = World(stage_units_in_meters=1.0)
world.scene.add_default_ground_plane()

usd_path = "/home/isseebi/Desktop/user/Reinforcement_learning/study/OpenManipulation/open_manipulator/open_manipulator_description/urdf/open_manipulator_x/open_manipulator_x/open_manipulator_x.usd"
prim_path = "/World/open_manipulator_x"

add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)
my_robot = world.scene.add(Articulation(prim_path=prim_path, name="open_manipulator"))

world.reset()

# ========== 制御パラメータ（ゲイン）の設定 ==========
num_dof = my_robot.num_dof

# 以下の2つの数値を変更して、挙動の違いを確認してください
STIFFNESS = 100.0  # 比例ゲイン(Kp): 目標角度に向かう力の強さ（バネの硬さ）
DAMPING = 5.0      # 微分ゲイン(Kd): 動きに対する抵抗力（空気抵抗やブレーキのようなもの）

# 全関節に同じゲインを適用する配列を作成
kps = np.ones(num_dof) * STIFFNESS
kds = np.ones(num_dof) * DAMPING

# コントローラーにゲインを設定
my_robot.get_articulation_controller().set_gains(kps=kps, kds=kds)
# ====================================================

step_count = 0

print("シミュレーション開始: ゲインのテストを行います...")

while simulation_app.is_running():
    world.step(render=True)
    
    if world.is_playing():
        if step_count == 0:
                    my_robot.post_reset()
                    
                    print("--- 物理パラメータの確認 ---")
                    stage = world.scene.stage
                    robot_prim = stage.GetPrimAtPath(prim_path) # ロボットのルートパス
                    
                    # ロボットを構成する全パーツ（Prim）を順番にチェック
                    for prim in Usd.PrimRange(robot_prim):
                        # 質量（MassAPI）が設定されているパーツか判定
                        if prim.HasAPI(UsdPhysics.MassAPI):
                            mass_api = UsdPhysics.MassAPI(prim)
                            current_mass = mass_api.GetMassAttr().Get()
                            print(f"リンク名: {prim.GetName()}, 現在の質量: {current_mass} kg")
                            
                            # 【解決策1】もし質量がNoneや0.001などの極小値なら、ここで強制上書き（例: 0.2kg）
                            # mass_api.GetMassAttr().Set(0.2)
                            
                    print("--------------------------------")

        # ゲインの挙動を確認しやすくするため、約2秒(120ステップ)ごとに目標角度を急に切り替える
        if (step_count // 120) % 2 == 0:
            target_angle = -1.0  # 約57度
        else:
            target_angle = 0.0 # 約-57度
            
        target_positions = np.ones(num_dof) * target_angle
        action = ArticulationAction(joint_positions=target_positions)
        
        my_robot.get_articulation_controller().apply_action(action)
        
        step_count += 1

simulation_app.close()