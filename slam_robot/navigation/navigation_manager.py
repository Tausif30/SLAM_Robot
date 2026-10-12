import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile
from std_msgs.msg import Empty, String


class NavigationManager(Node):
    """The only node that talks to Nav2.

    Topic  in : /target_goal         (geometry_msgs/PoseStamped)  "go here" (map frame)
    Topic  in : /cancel_navigation   (std_msgs/Empty)             "stop"
    Action out: navigate_to_pose     (nav2_msgs/NavigateToPose)   goal -> Nav2
    Topic  out: /navigation_status   (std_msgs/String)  IDLE | NAVIGATING | SUCCEEDED | FAILED | CANCELED
    """

    def __init__(self):
        super().__init__('navigation_manager')

        self._client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.create_subscription(PoseStamped, 'target_goal', self.on_goal, 10)
        self.create_subscription(Empty, 'cancel_navigation', self.on_cancel, 10)

        # "Latched" status: a node that starts later still receives the last value.
        latched = QoSProfile(depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        self._status_pub = self.create_publisher(String, 'navigation_status', latched)

        self._serial = 0          # id of the newest goal; results of older goals are ignored
        self._goal_handle = None  # handle of the goal Nav2 is working on
        self._set_status('IDLE')
        self.get_logger().info('navigation_manager ready: publish a PoseStamped to /target_goal')

    # Status Handling: /navigation_status
    def _set_status(self, state, detail=''):
        self._status_pub.publish(String(data=state))
        self.get_logger().info(f'status: {state}' + (f' ({detail})' if detail else ''))

    # Goal Handling: /target_goal -> Nav2 action server
    def on_goal(self, pose):
        if not pose.header.frame_id:
            pose.header.frame_id = 'map'
        if not self._client.server_is_ready():
            self._set_status('FAILED', 'Nav2 action server is not available')
            return

        self._serial += 1
        serial = self._serial
        goal = NavigateToPose.Goal()
        goal.pose = pose
        p = pose.pose.position
        self._set_status('NAVIGATING', f'goal #{serial}: x={p.x:.2f} y={p.y:.2f} in {pose.header.frame_id}')

        # A new goal while one is active simply replaces it (Nav2 supports preemption).
        send = self._client.send_goal_async(
            goal, feedback_callback=lambda fb: self._on_feedback(serial, fb))
        send.add_done_callback(lambda f: self._on_goal_response(serial, f))

    def on_cancel(self, _msg):
        if self._goal_handle is not None:
            self.get_logger().info('cancel requested')
            self._goal_handle.cancel_goal_async()

    # Nav2 Reply Handling
    def _on_goal_response(self, serial, future):
        handle = future.result()
        if not handle.accepted:
            if serial == self._serial:
                self._set_status('FAILED', 'Nav2 rejected the goal')
            return
        if serial != self._serial:   # a newer goal already took over
            return
        self._goal_handle = handle
        handle.get_result_async().add_done_callback(lambda f: self._on_result(serial, f))

    def _on_feedback(self, serial, feedback_msg):
        if serial != self._serial:
            return
        fb = feedback_msg.feedback
        self.get_logger().info(
            f'distance remaining: {fb.distance_remaining:.2f} m',
            throttle_duration_sec=2.0)

    def _on_result(self, serial, future):
        if serial != self._serial:   # result of a goal we already replaced
            return
        status = future.result().status
        self._goal_handle = None
        if status == GoalStatus.STATUS_SUCCEEDED:
            self._set_status('SUCCEEDED')
        elif status == GoalStatus.STATUS_CANCELED:
            self._set_status('CANCELED')
        else:
            self._set_status('FAILED', f'Nav2 status code {status}')


def main(args=None):
    rclpy.init(args=args)
    node = NavigationManager()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
