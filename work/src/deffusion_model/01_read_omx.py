from omni.isaac.kit import SimulationApp

# シミュレーションアプリの起動
simulation_app = SimulationApp({"headless": False})

from omni.isaac.core import World
from omni.isaac.core.robots import Robot
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.utils.stage import add_reference_to_stage
import numpy as np
from omni.isaac.core.utils.types import ArticulationAction

world = World(stage_units_in_meters=1.0)
world.scene.add_default_ground_plane()

usd_path = "/home/isseebi/Desktop/user/Reinforcement_learning/study/OpenManipulation/open_manipulator/open_manipulator_description/urdf/open_manipulator_x/open_manipulator_x/open_manipulator_x.usd"
prim_path = "/World/open_manipulator_x"

add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)

my_robot = world.scene.add(Articulation(prim_path=prim_path, name="open_manipulator"))

world.reset()

print("シミュレーション開始...")
step_count = 0

# while simulation_app.is_running():
#     world.step(render=True)
    
#     if world.is_playing():
#         if step_count == 0:
#             my_robot.post_reset()
        
#         num_dof = my_robot.num_dof
        
#         # 目標となる角度（NumPy配列）を計算
#         target_positions = np.sin(step_count * 0.02) * np.ones(num_dof) * 0.3
        
#         action = ArticulationAction(joint_positions=target_positions)
        
#         # 指令の適用
#         my_robot.get_articulation_controller().apply_action(action)
        
#         step_count += 1

while simulation_app.is_running():
    world.step(render=True)
    
    if world.is_playing():
        # --- 目標値の与え方 A: 全関節に配列で渡す ---
        # 順序は my_robot.dof_names の並び順に従います
        target_pos = np.array([0.5, -0.2, 0.3, 0.1, 0.0, 0.0]) # 例: 6自由度分
        
        # --- 目標値の与え方 B: 特定の関節だけ名前で指定する (推奨) ---
        # joint_indices を使うことで、順番を気にせず制御できます
        action = ArticulationAction(
            joint_positions=np.array([1.0]), 
            joint_indices=[my_robot.get_dof_index("joint1")] # joint2だけを1.0radへ
        )

        # 指令を送る
        my_robot.get_articulation_controller().apply_action(action)
        
        step_count += 1

simulation_app.close()