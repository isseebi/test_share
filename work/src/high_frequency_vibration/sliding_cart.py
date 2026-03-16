import numpy as np
import sys

# Isaac SimのSimulationAppを初期化
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

from omni.isaac.core import World
from omni.isaac.core.objects import DynamicCuboid

def main():
    world = World()
    world.scene.add_default_ground_plane()

    # 台車（小さな板）を作成
    # 位置A (x=0) に配置
    cart = world.scene.add(
        DynamicCuboid(
            prim_path="/World/Cart",
            name="cart",
            position=np.array([0.0, 0.0, 0.055]), # 地面より少し上 (スケールZが0.1なので、0.05だとぴったり。0.055で少し浮かせる)
            scale=np.array([0.5, 0.3, 0.1]),     # 板状
            color=np.array([0.8, 0.2, 0.2]),
            mass=1.0,
        )
    )

    world.reset()

    # パラメータ設定
    point_A_x = 0.0
    point_B_x = 2.0
    force_magnitude = 5.0 # N
    cart_mass = 1.0
    stop_triggered = False

    print("--- シミュレーション開始 ---", flush=True)

    step_count = 0
    post_stop_steps = 0
    current_v_x = 0.0  # 外部で速度を管理して摩擦によるリセットを防ぐ

    while simulation_app.is_running():
        world.step(render=True) # Headless mode doesn't need rendering
        step_count += 1

        if world.current_time_step_index == 0:
            world.reset()

        pos, rot = cart.get_world_pose()
        vel = cart.get_linear_velocity()
        current_x = pos[0]

        if step_count % 50 == 0 and not stop_triggered:
            print(f"移動中... 現在のx座標: {current_x:.2f}m", flush=True)

        if current_x < point_B_x and not stop_triggered:
            # x軸方向に力を加える (シミュレーション上で速度を更新)
            # F = ma -> a = F/m
            # v = v0 + a * dt
            dt = world.get_physics_dt()
            a_x = force_magnitude / cart_mass
            current_v_x += a_x * dt
            cart.set_linear_velocity(np.array([current_v_x, 0.0, 0.0]))
        elif current_x >= point_B_x and not stop_triggered:
            # B地点で急停車
            print(f"B地点 ({current_x:.2f}m) に到達しました。急停車します！", flush=True)
            current_v_x = 0.0
            cart.set_linear_velocity(np.array([0.0, 0.0, 0.0]))
            stop_triggered = True
            
        if stop_triggered:
            post_stop_steps += 1
            # 停止状態を維持
            cart.set_linear_velocity(np.array([0.0, 0.0, 0.0]))
            if post_stop_steps > 30:
                print("シミュレーションを終了します。", flush=True)
                break
                
        # 異常終了用（タイムアウト）
        if step_count > 1000:
            print("タイムアウト。終了します。", flush=True)
            break

    simulation_app.close()

if __name__ == '__main__':
    main()
