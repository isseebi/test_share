from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({"headless": False})

from omni.isaac.core import World
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.types import ArticulationAction
import numpy as np
from pxr import Usd, UsdPhysics
from friko_class import FlatPendulum

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
STIFFNESS = 50.0  # 比例ゲイン(Kp): 目標角度に向かう力の強さ（バネの硬さ）
DAMPING = 5.0      # 微分ゲイン(Kd): 動きに対する抵抗力（空気抵抗やブレーキのようなもの）

# 全関節に同じゲインを適用する配列を作成
kps = np.ones(num_dof) * STIFFNESS
kds = np.ones(num_dof) * DAMPING

# 【改善ポイント1】グリッパ主関節（5個目：インデックス4）のゲインを大幅に上げる
# kps[4] = 1000.0  # 物体をガッチリ掴むための強い力
# kds[4] = 100.0   # 振動を抑えるためのダンピング

# 【改善ポイント2】ミミック関節（6個目：インデックス5）のPD制御を無効化する
# これにより、6個目の関節はスクリプトからの指令を無視し、ミミック制約（物理ギア）のみに従うようになります
# kps[5] = 1000.0  # ← 0.0ではなく、5個目と同じ強いゲインを設定
# kds[5] = 100.0   # ← 同様にダンピングも設定

# コントローラーにゲインを設定
my_robot.get_articulation_controller().set_gains(kps=kps, kds=kds)

# max_efforts = np.ones(num_dof) * 1000.0
# my_robot.get_articulation_controller().set_max_efforts(max_efforts)
# ====================================================

step_count = 0

# 1つ目の振り子を原点に配置
pendulum1 = FlatPendulum(
    world=world, 
    prim_path="/World/Pendulum_1", 
    position=[0.32, 0.0, 0.0]
)

print("シミュレーション開始: ゲインのテストを行います...")

# 1. 目標地点のリストを作成
target_list = [
    # np.array([0.0, 0.0, 0.0, 0.0, 0.5, 0.5]),
    np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    np.array([0.0, 0.6, 0.2, -0.5, 0.1, 0.1]),
    np.array([0.0, 0.6, 0.2, -0.5, -0.1, -0.1]),
    np.array([0.0, 0.0, 0.0, 0.0, -0.1, -0.1]),
    np.array([0.3, 0.0, 0.0, 0.0, -0.1, -0.1]),
    np.array([-0.3, 0.0, 0.0, 0.0, -0.1, -0.1])

]

current_target_idx = 0
threshold = 0.05
# 待機用の変数
is_waiting = False
wait_counter = 0
max_wait_steps = 60  # 何ステップ停止させるか (例: 60fpsなら約1秒)

while simulation_app.is_running():
    world.step(render=True)
    
    if world.is_playing():
        if current_target_idx < len(target_list):
            target_pos = target_list[current_target_idx]
            
            # 現在の関節角度を取得して誤差を計算
            current_joint_pos = my_robot.get_joint_positions()
            error = np.mean(np.abs(current_joint_pos - target_pos))

            if not is_waiting:
                # --- 通常の移動モード ---
                if error < threshold:
                    print(f"Target {current_target_idx} reached! Waiting...")
                    is_waiting = True  # 待機モードへ移行
                    wait_counter = 0   # カウンタをリセット
            else:
                # --- 待機モード ---
                wait_counter += 1
                if wait_counter >= max_wait_steps:
                    print(f"Wait finished. Moving to next target.")
                    is_waiting = False
                    current_target_idx += 1 # 次の目標へ
            
            # 待機中も同じ目標値(target_pos)を送り続けることで、その場を維持させる
            action = ArticulationAction(joint_positions=target_pos)
            my_robot.get_articulation_controller().apply_action(action)
            
        else:
            print("All targets and pauses completed.")

simulation_app.close()