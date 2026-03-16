import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})

from omni.isaac.core import World
from omni.isaac.core.objects import DynamicCuboid, DynamicCylinder
from omni.isaac.core.articulations import Articulation
from pxr import UsdPhysics, Gf, Usd

def main():
    # 高周波振動(1000〜4000Hz)を取得するため、ナイキスト定理を考慮して20000Hz(0.00005秒)で物理シミュレーションを回します
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

    # 上にそそり立つ棒を作成
    rod_length = 0.5
    rod = world.scene.add(
        DynamicCylinder(
            prim_path="/World/Rod",
            name="rod",
            position=np.array([0.0, 0.0, 0.055 + 0.05 + rod_length/2.0]), 
            radius=0.01,
            height=rod_length,
            color=np.array([0.2, 0.8, 0.2]),
            mass=1.0, # Solverの安定化のため質量を1.0kgに増やす（軽すぎると減衰される）
        )
    )

    # アーティキュレーションルートを設定して剛性の高いばねと正確な関節DOF管理を行う
    UsdPhysics.ArticulationRootAPI.Apply(cart.prim)

    # ジョイントを作成して台車と棒を接続
    stage = world.stage
    joint_path = "/World/RevoluteJoint"
    joint = UsdPhysics.RevoluteJoint.Define(stage, joint_path)
    joint.CreateBody0Rel().SetTargets(["/World/Cart"])
    joint.CreateBody1Rel().SetTargets(["/World/Rod"])
    joint.CreateAxisAttr("Y") # X軸に沿って動くので、Y軸を中心に前後に倒れる/振動する
    joint.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, 0.05)) # 台車の上部中心
    joint.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, -rod_length/2.0)) # 棒の底部

    # ばね特性(Stiffness)を設定して高周波振動させる
    drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular")
    drive.CreateTypeAttr("force")
    drive.CreateTargetPositionAttr(0.0) # 常に直立状態(0度)へ戻ろうとする
    
    # 棒の慣性モーメント I = m * L^2 / 3 = 0.01 * 0.25 / 3 = 0.000833
    # Articulation時の高剛性スプリングを安定して動作させる
    drive.CreateStiffnessAttr(1000.0) # ばね定数 (Articulationではスケールが異なるため適切な範囲に)
    drive.CreateDampingAttr(0.05)    # 適切な減衰

    world.reset()

    # Articulationラッパーを作成して関節速度などへアクセス可能にする
    robot = Articulation("/World/Cart")
    world.scene.add(robot)
    world.reset() # robot追加後に再度resetして初期化

    point_A_x = 0.0
    point_B_x = 1.0 # 走行距離
    force_magnitude = 100.0 # B地点まで加速するための力(質量10倍に合わせて100N)
    stop_triggered = False

    print(f"--- 物理ステップ: {physics_dt}秒 (20000Hz) でシミュレーション開始 ---", flush=True)

    step_count = 0
    post_stop_steps = 0
    current_v_x = 0.0
    
    # 振動データを保存するための配列
    times = []
    displacements = []
    cart_xs = []
    tip_xs = []
    record_data = False
    
    while simulation_app.is_running():
        # 画面描画（render=True）を20000Hzで毎ステップ行うと非常に重くなるため、
        # 約60Hz（333物理ステップに1回）の頻度でレンダリングを有効にします
        # should_render = (step_count % 333 == 0)
        # world.step(render=should_render) 
        world.step(render=False)
        step_count += 1
        current_time = step_count * physics_dt

        if world.current_time_step_index == 0:
            world.reset()

        pos, rot = cart.get_world_pose()
        rod_pos, rod_rot = rod.get_world_pose()
        current_x = pos[0]

        # プログラムによる台車の強制移動
        if current_x < point_B_x and not stop_triggered:
            # 徐々に加速させる
            a_x = force_magnitude / cart_mass
            current_v_x += a_x * physics_dt
            robot.set_linear_velocity(np.array([current_v_x, 0.0, 0.0]))
        elif current_x >= point_B_x and not stop_triggered:
            print(f"B地点 ({current_x:.2f}m) に到達しました。急停車します！", flush=True)
            
            # 物理的な急停止の衝撃（インパルス）を計算
            v_0 = current_v_x
            current_v_x = 0.0
            
            # 急停車（ルートの速度を0にして以後固定）
            robot.set_linear_velocity(np.array([0.0, 0.0, 0.0]))
            
            # 棒の重心の速度はv_0を維持し、関節の拘束により回転運動に変換される。
            # 重心距離 L_eff = 2/3 * L に相当する物理的角速度をArticulationのJointの速度として与える
            L_eff = 2.0 / 3.0 * rod_length
            joint_omega = v_0 / L_eff
            
            # 1自由度のRevoluteJointに速度を直接セット！
            # Y軸周りに+または-（設定により位相反転するので絶対値だけ確認できればOK）
            robot.set_joint_velocities(np.array([joint_omega]))
            
            stop_triggered = True
            record_data = True
            
        if record_data:
            robot.set_linear_velocity(np.array([0.0, 0.0, 0.0])) # 台車へのブレーキを保つ
            
            # 棒の回転クォータニオン (Isaac Sim: [w, x, y, z])
            # Scipyのクォータニオンは [x, y, z, w] なので並べ替える
            rot_rod_np = np.array([rod_rot[1], rod_rot[2], rod_rot[3], rod_rot[0]])
            r = R.from_quat(rot_rod_np)
            
            # 棒のローカルの先端位置
            tip_local = np.array([0.0, 0.0, rod_length/2.0])
            # ワールド座標での先端位置
            tip_world = rod_pos + r.apply(tip_local)
            
            # 台車にくっついて棒が垂直に立っていた場合の本来のX座標
            expected_tip_x = pos[0]
            
            # X方向の変位量(振動)を計算
            displacement_x = tip_world[0] - expected_tip_x
            
            # グラフが見やすくなるように、停止直後の最初の10ms分(200ステップ)のみ記録する
            if post_stop_steps * physics_dt <= 1.0: # driftを見るために少し長く(50ms)記録
                times.append(current_time)
                displacements.append(displacement_x)
                cart_xs.append(pos[0])
                tip_xs.append(tip_world[0])
            
            post_stop_steps += 1
            
            # 急停止後、余韻を画面で確認できるように0.5秒間シミュレーションを継続してから終了
            if post_stop_steps * physics_dt > 1.0: 
                print("シミュレーションを終了します。", flush=True)
                break
                
        # タイムアウトのフェイルセーフ (実質稼働時間5秒のシミュレーション時間)
        if current_time > 5.0:
            print("タイムアウト。終了します。", flush=True)
            break

    # グラフの作成と保存
    print(f"記録したデータ点数: {len(times)}", flush=True)
    if len(times) > 0:
        try:
            times_arr = np.array(times)
            times_arr -= times_arr[0]
            
            plt.figure(figsize=(20, 5))
            plt.plot(times_arr * 1000, np.array(displacements) * 1000, label="Tip Displacement (X)")
            plt.xlabel("Time after sudden stop (ms)")
            plt.ylabel("Displacement (mm)")
            plt.title("Rod Vibration Displacement (First 10ms upon Sudden Stop)")
            plt.grid(True)
            plt.legend()
            graph_path = "/home/isseebi/Desktop/user/deffusion_model/antigravity/vibration_graph.png"
            plt.savefig(graph_path)
            print(f"振動データをグラフに保存しました: {graph_path}", flush=True)
        except Exception as e:
            print(f"グラフ保存中にエラーが発生しました: {e}", flush=True)
            
        # 常にCSVに保存して詳細を分析できるようにする
        csv_path = "/home/isseebi/Desktop/user/deffusion_model/antigravity/vibration_data.csv"
        np.savetxt(csv_path, np.column_stack((times_arr, displacements, cart_xs, tip_xs)), delimiter=",", header="time(s),displacement(m),cart_x(m),tip_x(m)", comments="")
        print(f"CSVに保存しました: {csv_path}", flush=True)

    simulation_app.close()

if __name__ == '__main__':
    main()
