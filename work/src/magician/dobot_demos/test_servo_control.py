import time
import math
import rclpy
from rclpy.node import Node
from dobot_driver.dobot_handle import bot

class TestServoControl(Node):
    def __init__(self):
        super().__init__('test_servo_control')
        self.timer = self.create_timer(0.05, self.timer_callback) # 20Hz
        self.t = 0.0
        self.radius = 20.0
        self.center_j1 = 0.0
        
        # Initial pose
        self.get_logger().info("Moving to start position...")
        bot.set_point_to_point_command(4, 0.0, 0.0, 0.0, 0.0, queue=True)
        time.sleep(5) # Wait for move
        self.get_logger().info("Starting servo loop...")

    def timer_callback(self):
        self.t += 0.05
        
        # Sine wave on Joint 1
        j1 = self.center_j1 + self.radius * math.sin(self.t)
        j2 = 0.0
        j3 = 0.0
        j4 = 0.0
        
        # Send without queue to simulate servo
        # Mode 4: MOVJ_ANGLE
        bot.set_point_to_point_command(4, j1, j2, j3, j4, queue=False)
        self.get_logger().info(f"Sent J1={j1:.2f}")

def main(args=None):
    rclpy.init(args=args)
    node = TestServoControl()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
