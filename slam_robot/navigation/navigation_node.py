import math
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist

# Subscribe to /scan, publish /cmd_vel: drive forward, turn when blocked.
class NavigationNode(Node):

    def __init__(self):
        super().__init__('navigation_node')

        # Parameters
        self.declare_parameter('forward_speed', 0.15)   # m/s
        self.declare_parameter('turn_speed', 0.6)       # rad/s
        self.declare_parameter('safe_distance', 0.5)    # m

        # Sensor QoS = Best Effort, matching what the simulator publishes
        self.sub = self.create_subscription(
            LaserScan, 'scan', self.on_scan, qos_profile_sensor_data)
        self.pub = self.create_publisher(Twist, 'cmd_vel', 10)

        self.get_logger().info('Navigation_Node started')

    # Smallest valid distance in the angle window [lo, hi] degrees.
    @staticmethod
    def sector_min(scan, lo_deg, hi_deg):
        lo, hi = math.radians(lo_deg), math.radians(hi_deg)
        best = float('inf')
        for i, r in enumerate(scan.ranges):
            if not math.isfinite(r) or r < scan.range_min:
                continue
            a = scan.angle_min + i * scan.angle_increment
            a = math.atan2(math.sin(a), math.cos(a))  # wrap to [-pi, pi]
            if lo <= a <= hi:
                best = min(best, r)
        return best

    def on_scan(self, scan):
        safe = self.get_parameter('safe_distance').value
        fwd = self.get_parameter('forward_speed').value
        turn = self.get_parameter('turn_speed').value

        front = self.sector_min(scan, -30, 30)
        left = self.sector_min(scan, 30, 90)
        right = self.sector_min(scan, -90, -30)

        cmd = Twist()
        if front < safe:
            # Blocked: stop and rotate toward the side with more room
            cmd.angular.z = turn if left > right else -turn
        else:
            cmd.linear.x = fwd
        self.pub.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = NavigationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.pub.publish(Twist())  # stop the robot on exit
        node.destroy_node()
        rclpy.shutdown()

