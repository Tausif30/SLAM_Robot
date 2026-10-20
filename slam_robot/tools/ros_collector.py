"""ros_collector.py - the ROS node that talks to Gazebo: finds the robot, teleports it and grabs a fresh
RGB + depth + odom set. Used by collect.py and teleport.py.

ROS is imported inside ros_collector_class(), not at the top, so the rest of the pipeline
(label / build / train / evaluate) works on a machine without ROS.
"""
import math
import sys
import time
import numpy as np

from camera_geometry import angle_diff, quat_to_yaw
from pipeline_config import POSE_TOL_DEG, POSE_TOL_M, ROBOT_HINTS, SETTLE_SIM_S, WORLD_MODELS
from teleport import teleport_entity


def pick_robot(names):
    others = [n for n in names if n not in WORLD_MODELS]
    hinted = [n for n in others if any(h in n.lower() for h in ROBOT_HINTS)]
    candidates = hinted if hinted else others
    return candidates[0] if len(candidates) == 1 else None


def stamp_s(header):
    return header.stamp.sec + header.stamp.nanosec * 1e-9


def ready(rgb, depth, odom, t_min):
    """True when RGB, depth and odom all arrived and were recorded at least at simulated time t_min."""
    return all(m is not None and m[0] >= t_min for m in (rgb, depth, odom))


def ros_collector_class():
    """Build the Collector node class (ROS is imported here, see the top of this file)."""
    import rclpy
    from gazebo_msgs.msg import ModelStates
    from gazebo_msgs.srv import SetEntityState
    from nav_msgs.msg import Odometry
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import Image

    class Collector(Node):
        def __init__(self):
            super().__init__("collect_dataset")
            self.rgb = self.depth = self.odom = None
            self.names = None
            self.robot = None
            self.create_subscription(Image, "/camera/image_raw", self.on_rgb, qos_profile_sensor_data)
            self.create_subscription(Image, "/camera/depth/image_raw", self.on_depth, qos_profile_sensor_data)
            self.create_subscription(Odometry, "/odom", self.on_odom, qos_profile_sensor_data)
            self.create_subscription(ModelStates, "/gazebo/model_states", self.on_states, 10)
            self.client = self.create_client(SetEntityState, "/gazebo/set_entity_state")

        def on_rgb(self, msg):
            if msg.encoding != "rgb8":
                self.get_logger().error(f"unexpected RGB encoding {msg.encoding}")
                return
            self.rgb = (stamp_s(msg.header), np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, 3).copy())

        def on_depth(self, msg):
            if msg.encoding != "32FC1":
                self.get_logger().error(f"unexpected depth encoding {msg.encoding}")
                return
            self.depth = (stamp_s(msg.header), np.frombuffer(msg.data, np.float32).reshape(msg.height, msg.width).copy())

        def on_odom(self, msg):
            p, q = msg.pose.pose.position, msg.pose.pose.orientation
            self.odom = (stamp_s(msg.header), p.x, p.y, quat_to_yaw(q.x, q.y, q.z, q.w))

        def on_states(self, msg):
            self.names = list(msg.name)

        def spin_until(self, cond, seconds):
            end = time.time() + seconds
            while rclpy.ok() and time.time() < end and not cond():
                rclpy.spin_once(self, timeout_sec=0.05)
            return cond()

        def start(self):
            if not self.spin_until(lambda: self.names is not None, 10):
                sys.exit("No /gazebo/model_states: is the simulation running with the gazebo_ros_state plugin?")
            self.robot = pick_robot(self.names)
            if self.robot is None:
                sys.exit(f"Could not tell which model is the robot: {self.names}")
            if not self.client.wait_for_service(timeout_sec=5.0):
                sys.exit("Service /gazebo/set_entity_state not available.")
            if not self.spin_until(lambda: self.rgb and self.depth and self.odom, 10):
                sys.exit("No RGB / depth / odom messages yet.")
            print(f"robot entity: '{self.robot}'")

        def teleport(self, x, y, yaw):
            return teleport_entity(self, self.client, self.robot, x, y, yaw)

        def capture_at(self, x, y, yaw):
            t_ref = self.odom[0] if self.odom else 0.0
            if not self.teleport(x, y, yaw):
                return None, "teleport failed"
            t_min = t_ref + SETTLE_SIM_S
            if not self.spin_until(lambda: ready(self.rgb, self.depth, self.odom, t_min), 20):
                return None, "timeout waiting for fresh images"
            _, ox, oy, oyaw = self.odom
            err_m = math.hypot(ox - x, oy - y)
            err_deg = math.degrees(abs(angle_diff(oyaw, yaw)))
            if err_m > POSE_TOL_M or err_deg > POSE_TOL_DEG:
                return None, f"robot was pushed ({err_m * 100:.0f} cm, {err_deg:.1f} deg) at ({x:.2f},{y:.2f})"
            return dict(rgb=self.rgb[1], depth=self.depth[1], odom=(ox, oy, oyaw), stamp=self.rgb[0]), None

    return Collector
