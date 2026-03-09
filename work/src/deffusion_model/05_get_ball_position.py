from omni.isaac.kit import SimulationApp
import os
simulation_app = SimulationApp({"headless": False})

from omni.isaac.core import World
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.types import ArticulationAction
import numpy as np
import matplotlib.pyplot as plt  # 【追加】グラフ描画用ライブラリ
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
STIFFNESS = 50.0  
DAMPING = 5.0      
kps = np.ones(num_dof) * STIFFNESS
kds = np.ones(num_dof) * DAMPING

my_robot.get_articulation_controller().set_gains(kps=kps, kds=kds)
# ====================================================

# 1つ目の振り子を配置
pendulum1 = FlatPendulum(
    world=world, 
    prim_path="/World/Pendulum_1", 
    position=[0.32, 0.0, 0.0]
)

print("シミュレーション開始: ゲインのテストを行います...")

target_list = [
    np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    np.array([0.0, 0.6, 0.2, -0.5, 0.1, 0.1]),
    np.array([0.0, 0.6, 0.2, -0.5, -0.1, -0.1]),
    np.array([0.0, 0.0, 0.0, 0.0, -0.1, -0.1]),
    np.array([0.3, 0.0, 0.0, 0.0, -0.1, -0.1]),
    np.array([-0.3, 0.0, 0.0, 0.0, -0.1, -0.1])
]

current_target_idx = 0
threshold = 0.05
is_waiting = False
wait_counter = 0
max_wait_steps = 60 

# 【追加】グラフ描画用のデータ保存リスト
time_steps = []
bob_x = []
bob_y = []
bob_z = []

current_time = 0.0
dt = world.get_physics_dt() # 1ステップあたりの時間

while simulation_app.is_running():
    world.step(render=True)
    
    if world.is_playing():
        # 【追加】球(Bob)の現在位置を取得して記録
        bob_pos, bob_ori = pendulum1.bob.get_world_pose()
        time_steps.append(current_time)
        bob_x.append(bob_pos[0])
        bob_y.append(bob_pos[1])
        bob_z.append(bob_pos[2])
        current_time += dt

        if current_target_idx < len(target_list):
            target_pos = target_list[current_target_idx]
            
            # 現在の関節角度を取得して誤差を計算
            current_joint_pos = my_robot.get_joint_positions()
            error = np.mean(np.abs(current_joint_pos - target_pos))

            if not is_waiting:
                if error < threshold:
                    print(f"Target {current_target_idx} reached! Waiting...")
                    is_waiting = True  
                    wait_counter = 0   
            else:
                wait_counter += 1
                if wait_counter >= max_wait_steps:
                    print(f"Wait finished. Moving to next target.")
                    is_waiting = False
                    current_target_idx += 1 
            
            action = ArticulationAction(joint_positions=target_pos)
            my_robot.get_articulation_controller().apply_action(action)
            
        else:
            print("All targets and pauses completed.")
            break # 【変更】すべての目標に到達したらループを抜けてグラフ描画へ移行


# ========== グラフのプロットと保存 ==========
print("グラフを画像として保存します...")
plt.figure(figsize=(10, 6))
plt.plot(time_steps, bob_x, label='X Position')
plt.plot(time_steps, bob_y, label='Y Position')
plt.plot(time_steps, bob_z, label='Z Position')

plt.title('Pendulum Bob Position over Time')
plt.xlabel('Time [s]')
plt.ylabel('Position [m]')
plt.legend()
plt.grid(True)

# 保存先のパスを指定
save_dir = "/home/isseebi/Desktop/user/Reinforcement_learning/study/OpenManipulation/src"
os.makedirs(save_dir, exist_ok=True)  # ディレクトリがない場合は作成

# ファイル名を設定して保存
save_path = os.path.join(save_dir, "pendulum_position.png")
plt.savefig(save_path)

# メモリ解放のためにフィギュアを閉じる
plt.close()

print(f"保存完了: {save_path}")

simulation_app.close()