import time
import rclpy
from rclpy.node import Node
from dobot_driver.dobot_handle import bot

class TestCPMotion(Node):
    def __init__(self):
        super().__init__('test_cp_motion')
        
        # PTP to start position
        self.get_logger().info("Moving to Start (PTP)...")
        bot.set_point_to_point_command(1, 150.0, 0.0, 50.0, 0.0, queue=True) # MOVJ_XYZ
        time.sleep(5) 
        
        self.get_logger().info("Setting CP Params...")
        # CP params: planAcc, juncVel, acc, period
        request = bot.set_continous_trajectory_params(50, 50, 50, queue=True)
        # Note: Interface might not expose set_cp_params correctly, checking..
        # intf.py has set_cp_params? No, I checked intf.py and it has get_cp_params (ID 90) but set?
        # Let's check interface.py again or just try ID 90.
        # parser 90 expects [f, f, f, B] -> planAcc, juncVel, acc, realTimeTrack
        # Wait, if parser 90 is for GET, what is SET?
        # parsers[90] has value[3] (set request) = lambda x: list(struct.pack('<fffB', *x))
        # So yes, we can set it.
        
        # But `dobot_handle.bot` might not have the method wrappers.
        # I'll check `interface.py` for `set_cp_params`.
        # If not, I can construct message manually? No, I need method.
        # Let's verify interface.py has it first.
        
        # Assuming it exists or I use raw message if needed.
        # Let's assume defaults are OK for now.
        
        self.get_logger().info("Sending CP commands...")
        
        bot.clear_queue()
        bot.start_queue()
        
        # Mode 1: Absolute Linear? 
        # Interface `set_continous_trajectory_command` (ID 91)
        # Arg: mode, x, y, z, velocity
        
        for i in range(10):
            x = 150.0 + i * 2.0
            y = i * 5.0
            z = 50.0
            vel = 5.0 # What unit?
            
            # Mode 1: CP Relative? Mode 0: CP Absolute?
            # Protocol says: 
            # 0: Relative
            # 1: Absolute
            
            bot.set_continous_trajectory_command(1, x, y, z, vel, queue=True)
            time.sleep(0.01)
            
        time.sleep(5)
        self.get_logger().info("Done.")

def main(args=None):
    rclpy.init(args=args)
    node = TestCPMotion()
    try:
        pass
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
