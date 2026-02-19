import time
import rclpy
from rclpy.node import Node
from dobot_driver.dobot_handle import bot

class TestForceStop(Node):
    def __init__(self):
        super().__init__('test_force_stop')
        self.get_logger().info("Moving to Start (0)...")
        bot.set_point_to_point_command(4, 0.0, 0.0, 0.0, 0.0, queue=True)
        time.sleep(5) 
        
        self.get_logger().info("Sending J1=40 (Queue=False)...")
        bot.set_point_to_point_command(4, 40.0, 0.0, 0.0, 0.0, queue=False)
        
        time.sleep(0.5) 
        
        self.get_logger().info("Attempting FORCE STOP and new command J1=-40...")
        
        # Force stop the current command
        bot.stop_queue(force=True)
        bot.clear_queue()
        bot.start_queue()
        
        # Send new command
        bot.set_point_to_point_command(4, -40.0, 0.0, 0.0, 0.0, queue=False)
        
        time.sleep(5)
        self.get_logger().info("Done.")

def main(args=None):
    rclpy.init(args=args)
    node = TestForceStop()
    try:
        pass
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
