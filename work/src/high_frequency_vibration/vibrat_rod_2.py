import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})

from omni.isaac.core import World
from omni.isaac.core.objects import DynamicCuboid, DynamicCylinder
from omni.isaac.core.articulations import Articulation
# PhysxSchema をインポートに追加
from pxr import UsdPhysics, Gf, Usd, PhysxSchema

def main():
    # 高周波振動(1000〜4000Hz)を取得するため、20000Hz(0.00005秒)でシミュレーション
    physics_dt = 1.0 / 20000.0
    world = World(physics_dt=physics_dt, rendering_dt=1.0/60.0)
    world.scene.add_default_ground_plane()

    # 台車を作成
    cart_mass = 10.0
    cart = world.scene.add(
        DynamicCuboid(
            prim_path="/World/Cart",
            name="cart",
            position=np.array([0.0, 0.0, 0.055]),
            scale=np.array([0.5, 0.3, 0.1]),
            color=np.array([0.8, 0.2, 0.2]),
            mass=cart_mass,
        )
    )

    # 棒を作成
    rod_length = 0.5
    rod = world.scene.add(
        DynamicCylinder(
            prim_path="/World/Rod",
            name="rod",
            position=np.array([0.0, 0.0, 0.055 + 0.05 + rod_length/2.0]), 
            radius=0.01,
            height=rod_length,
            color=np.array([0.2, 0.8, 0.2]),
            mass=1.0, 
        )
    )

    # アーティキュレーションルートを設定
    UsdPhysics.ArticulationRootAPI.Apply(cart.prim)

    # ジョイントを作成
    stage = world.stage
    joint_path = "/World/RevoluteJoint"
    joint = UsdPhysics.RevoluteJoint.Define(stage, joint_path)
    joint.CreateBody0Rel().SetTargets(["/World/Cart"])
    joint.CreateBody1Rel().SetTargets(["/World/Rod"])
    joint.CreateAxisAttr("Y") 
    joint.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, 0.05)) 
    joint.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, -rod_length/2.0)) 

    # --- 摩擦をゼロにする設定 ---
    # PhysX固有のジョイント属性を適用し、関節摩擦を 0.0 に設定
    physx_joint = PhysxSchema.PhysxJointAPI.Apply(joint.GetPrim())
    physx_joint.CreateJointFrictionAttr(0.0)

    # ばね特性(Stiffness)を設定
    drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular")
    drive.CreateTypeAttr("force")
    drive.CreateTargetPositionAttr(0.0) 
    
    # 剛性設定
    drive.CreateStiffnessAttr(1000.0) 
    # 【変更】減衰（Damping）を 0.0 に設定し、粘性摩擦を排除
    drive.CreateDampingAttr(0.0)    

    world.reset()

    # Articulationラッパーを作成
    robot = Articulation("/World/Cart")
    world.scene.add(robot)
    world.reset() 

    point_A_x = 0.0
    point_B_x = 1.0 
    force_magnitude = 100.0 
    stop_triggered = False

    print(f"--- 物理ステップ: {physics_dt}秒 (20000Hz) 摩擦ゼロ設定で開始 ---", flush=True)

    step_count = 0
    post_stop_steps = 0
    current_v_x = 0.0
    
    times = []
    displacements = []
    cart_xs = []
    tip_xs = []
    record_data = False
    
    while simulation_app.is_running():
        world.step(render=False)
        step_count += 1
        current_time = step_count * physics_dt

        if world.current_time_step_index == 0:
            world.reset()

        pos, rot = cart.get_world_pose()
        rod_pos, rod_rot = rod.get_world_pose()
        current_x = pos[0]

        if current_x < point_B_x and not stop_triggered:
            a_x = force_magnitude / cart_mass
            current_v_x += a_x * physics_dt
            robot.set_linear_velocity(np.array([current_v_x, 0.0, 0.0]))
        elif current_x >= point_B_x and not stop_triggered:
            print(f"B地点 ({current_x:.2f}m) に到達。急停車します。", flush=True)
            v_0 = current_v_x
            current_v_x = 0.0
            robot.set_linear_velocity(np.array([0.0, 0.0, 0.0]))
            
            # 角速度の付与
            L_eff = 2.0 / 3.0 * rod_length
            joint_omega = v_0 / L_eff
            robot.set_joint_velocities(np.array([joint_omega]))
            
            stop_triggered = True
            record_data = True
            
        if record_data:
            robot.set_linear_velocity(np.array([0.0, 0.0, 0.0]))
            
            rot_rod_np = np.array([rod_rot[1], rod_rot[2], rod_rot[3], rod_rot[0]])
            r = R.from_quat(rot_rod_np)
            
            tip_local = np.array([0.0, 0.0, rod_length/2.0])
            tip_world = rod_pos + r.apply(tip_local)
            expected_tip_x = pos[0]
            displacement_x = tip_world[0] - expected_tip_x
            
            # 摩擦ゼロの効果を確認するため、1.0秒間（長めに）記録
            if post_stop_steps * physics_dt <= 1.0: 
                times.append(current_time)
                displacements.append(displacement_x)
                cart_xs.append(pos[0])
                tip_xs.append(tip_world[0])
            
            post_stop_steps += 1
            if post_stop_steps * physics_dt > 1.0: 
                print("シミュレーションを終了します。", flush=True)
                break
                
        if current_time > 10.0:
            break

    # グラフの作成
    if len(times) > 0:
        try:
            times_arr = np.array(times)
            times_arr -= times_arr[0]
            
            plt.figure(figsize=(20, 5))
            plt.plot(times_arr * 1000, np.array(displacements) * 1000, label="Tip Displacement (X)")
            plt.xlabel("Time after sudden stop (ms)")
            plt.ylabel("Displacement (mm)")
            plt.title("Frictionless Rod Vibration (Perpetual Motion)")
            plt.grid(True)
            plt.legend()
            graph_path = "./vibration_graph_frictionless.png"
            plt.savefig(graph_path)
            print(f"グラフを保存しました: {graph_path}", flush=True)
        except Exception as e:
            print(f"グラフ保存エラー: {e}", flush=True)

    simulation_app.close()

if __name__ == '__main__':
    main()