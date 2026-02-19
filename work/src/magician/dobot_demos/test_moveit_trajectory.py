#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from builtin_interfaces.msg import Duration

class TrajectoryTestClient(Node):

    def __init__(self):
        super().__init__('trajectory_test_client')
        self._action_client = ActionClient(self, FollowJointTrajectory, 'dobot_arm_controller/follow_joint_trajectory')

    def send_goal(self):
        self.get_logger().info('Waiting for action server...')
        self._action_client.wait_for_server()

        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.trajectory.joint_names = ['magician_joint_1', 'magician_joint_2', 'magician_joint_3', 'magician_joint_4']
        
        # Point 1: Home
        point1 = JointTrajectoryPoint()
        point1.positions = [0.0, 0.0, 0.0, 0.0]
        point1.time_from_start = Duration(sec=2, nanosec=0)
        
        # Point 2: Move J1
        point2 = JointTrajectoryPoint()
        point2.positions = [0.5, 0.0, 0.0, 0.0] # ~28 degrees
        point2.time_from_start = Duration(sec=4, nanosec=0)
        
        # Point 3: Move J2
        point3 = JointTrajectoryPoint()
        point3.positions = [0.5, 0.2, 0.0, 0.0] 
        point3.time_from_start = Duration(sec=6, nanosec=0)
        
        # Point 4: Back to Home
        point4 = JointTrajectoryPoint()
        point4.positions = [0.0, 0.0, 0.0, 0.0] 
        point4.time_from_start = Duration(sec=8, nanosec=0)

        goal_msg.trajectory.points = [point1, point2, point3, point4]

        self.get_logger().info('Sending goal...')
        self._send_goal_future = self._action_client.send_goal_async(goal_msg)
        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected :(')
            return

        self.get_logger().info('Goal accepted :)')

        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        result = future.result().result
        self.get_logger().info('Result received')
        rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    action_client = TrajectoryTestClient()
    action_client.send_goal()
    rclpy.spin(action_client)

if __name__ == '__main__':
    main()
