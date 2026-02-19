import time
import rclpy
from rclpy.node import Node
from dobot_driver.dobot_handle import bot

class TestNonBlocking(Node):
    def __init__(self):
        super().__init__('test_non_blocking')
        self.get_logger().info("Moving to Start (0)...")
        bot.set_point_to_point_command(4, 0.0, 0.0, 0.0, 0.0, queue=True)
        time.sleep(5) 
        
        self.get_logger().info("Sending J1=40 (Queue=False)...")
        # Mode 4: MOVJ_ANGLE
        bot.set_point_to_point_command(4, 40.0, 0.0, 0.0, 0.0, queue=False)
        
        time.sleep(0.5) # Wait a bit, robot should be moving towards 40
        
        self.get_logger().info("Sending J1=-40 (Queue=False) - Should Interrupt!")
        bot.set_point_to_point_command(4, -40.0, 0.0, 0.0, 0.0, queue=False)
        
        time.sleep(5)
        self.get_logger().info("Done.")

def main(args=None):
    rclpy.init(args=args)
    node = TestNonBlocking()
    try:
        pass
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
