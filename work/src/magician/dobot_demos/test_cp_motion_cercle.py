import time
import rclpy
import math
from rclpy.node import Node
from dobot_driver.dobot_handle import bot

class TestSmallCircle(Node):
    def __init__(self):
        super().__init__('test_small_circle')
        
        # --- 1. パラメータ設定 ---
        cx, cy = 150.0, 0.0    # 円の中心 (ベースに近い安全圏)
        r = 10.0               # 半径 (10mm)
        z = 50.0               # 高さ (50mm)
        num_points = 36        # <--- ここを追加しました（36分割 = 10度ずつ）
        speed = 10.0           # 速度 (mm/s)
        
        # 2. 開始点（角度0度）の計算
        start_x = cx + r * math.cos(0)
        start_y = cy + r * math.sin(0)
        
        self.get_logger().info(f"Moving to Circle Start: ({start_x}, {start_y})")
        # PTPで開始点へ移動
        bot.set_point_to_point_command(1, start_x, start_y, z, 0.0, queue=True)
        time.sleep(3) 
        
        # 3. CP（連続軌跡）設定
        self.get_logger().info("Setting CP Params...")
        bot.set_continous_trajectory_params(50, 50, 50, queue=True)
        
        # キューをクリアして開始
        bot.clear_queue()
        bot.start_queue()
        
        self.get_logger().info("Drawing a small circle...")
        
        # 4. 円周上の点を計算して送信
        for i in range(num_points + 1):  # 0から36まで
            # 角度を計算
            angle = (2 * math.pi) * (i / num_points)
            x = cx + r * math.cos(angle)
            y = cy + r * math.sin(angle)
            
            # CPコマンド送信
            bot.set_continous_trajectory_command(1, x, y, z, speed, queue=True)
            
            # 通信を安定させるためのわずかな待機
            time.sleep(0.02)
            
        time.sleep(5)
        self.get_logger().info("Circle movement done.")

def main(args=None):
    rclpy.init(args=args)
    node = TestSmallCircle()
    # 実行が完了したらノードを破棄
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()