import time
import math
import rclpy
from rclpy.node import Node
from dobot_driver.dobot_handle import bot

class TestCPMotion(Node):
    def fwd_kin(self, j1, j2, j3):
        # Dobot Magician Dimensions (mm)
        L1 = 135.0
        L2 = 147.0
        L3 = 60.0 

        q1 = math.radians(j1)
        q2 = math.radians(j2)
        q3 = math.radians(j3)

        # 順運動学計算 (すべての関節が0度のとき、アームは最大リーチで水平になります)
        r = L1 * math.cos(q2) + L2 * math.cos(q3) + L3
        x = r * math.cos(q1)
        y = r * math.sin(q1)
        z = L1 * math.sin(q2) + L2 * math.sin(q3)
        
        return x, y, z

    def __init__(self):
        super().__init__('test_cp_motion')
        
        self.get_logger().info("Moving to Zero Position (0, 0, 0)...")
        # 全関節を0度にセット
        start_j1, start_j2, start_j3 = 0.0, 0.0, 0.0
        bot.set_point_to_point_command(4, start_j1, start_j2, start_j3, 0.0, queue=True) 

        time.sleep(4) 
        
        self.get_logger().info("Starting CP motion around 0 degrees...")
        bot.clear_queue()
        bot.start_queue()
        
        # 20ステップで極小範囲を移動
        for i in range(20):
            # J1: 0度から 2度までゆっくり旋回
            target_j1 = start_j1 + i * 0.1 
            
            # J2: 0度を中心に ±1.5度 の微細な上下（前後）運動
            target_j2 = start_j2 + math.sin(i * 0.5) * 1.5
            
            # J3: 0度で固定
            target_j3 = start_j3
            
            # 座標計算
            x, y, z = self.fwd_kin(target_j1, target_j2, target_j3)
            
            # 非常にゆっくり動かす (15.0 mm/s)
            vel = 15.0 
            
            # 連続軌道コマンド
            bot.set_continous_trajectory_command(1, x, y, z, vel, queue=True)
            time.sleep(0.05)
            
        time.sleep(2)
        self.get_logger().info("Finished.")

def main(args=None):
    rclpy.init(args=args)
    node = TestCPMotion()
    try:
        # 実行完了を待つロジック（実際の実装に合わせて調整してください）
        time.sleep(5)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()