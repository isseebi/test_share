import time
import rclpy
from rclpy.node import Node
from dobot_driver.dobot_handle import bot

class TestPTPSmoothness(Node):
    def __init__(self):
        super().__init__('test_ptp_smoothness')
        self.get_logger().info("Setting params to 100%...")
        bot.set_point_to_point_common_params(100, 100, queue=True)
        bot.set_point_to_point_joint_params([100, 100, 100, 100], [100, 100, 100, 100], queue=True)
        
        self.get_logger().info("Moving to Start (0)...")
        bot.set_point_to_point_command(4, 0.0, 0.0, 0.0, 0.0, queue=True)
        time.sleep(4) 
        
        self.get_logger().info("Queueing 10 close points (0->10 deg)...")
        
        # Queue 10 points, 1 degree apart
        bot.clear_queue()
        bot.start_queue()
        for i in range(1, 11):
            bot.set_point_to_point_command(4, float(i), 0.0, 0.0, 0.0, queue=True)
        
        self.get_logger().info("Executing...")
        # Queue is auto-executing? Or need specific start?
        # Standard driver starts executing queue immediately if not stopped?
        # Let's ensure it's not stopped?
        # bot.start_queue() is usually called by driver init?
        
        time.sleep(5)
        self.get_logger().info("Done.")

def main(args=None):
    rclpy.init(args=args)
    node = TestPTPSmoothness()
    try:
        pass
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
