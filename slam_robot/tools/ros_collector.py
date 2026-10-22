# Collects RGB, depth and odometry data from the simulation.
import json
import math
import os
import sys
import time
import numpy as np
from camera_geometry import angle_diff, finalize_rotation, merge_calibration, quat_to_rot, quat_to_yaw, to_optical
from pipeline_config import BASE_FRAME, CALIB, POSE_TOL_DEG, POSE_TOL_M, ROBOT_NAME, SETTLE_SIM_S
from teleport import teleport_entity

# Gazebo Model
def pick_robot(names):
    return ROBOT_NAME if ROBOT_NAME in names else None

def stamp_s(header):
    return header.stamp.sec + header.stamp.nanosec * 1e-9

# Check if the RGB / depth / odom messages are all fresh (after a teleport).
def ready(rgb, depth, odom, t_min):
    return all(m is not None and m[0] >= t_min for m in (rgb, depth, odom))


# Build the Collector node class.
def ros_collector_class():
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
                sys.exit(f"No model called '{ROBOT_NAME}' in /gazebo/model_states: {self.names}")
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

# Calibration: read the camera_info and TF, write slam_robot/tools/calib.json.
def calibrate_arguments(p):
    """No options: everything comes from /camera/camera_info and TF."""


def run_calibrate(a):
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from rclpy.time import Time
    from sensor_msgs.msg import CameraInfo
    from tf2_ros import Buffer, TransformListener

    class Calib(Node):
        def __init__(self):
            super().__init__("camera_calib")
            self.info = self.depth_info = None
            self.create_subscription(CameraInfo, "/camera/camera_info", self.on_info, qos_profile_sensor_data)
            self.create_subscription(CameraInfo, "/camera/depth/camera_info", self.on_depth_info, qos_profile_sensor_data)
            self.tf_buffer = Buffer()
            self.tf_listener = TransformListener(self.tf_buffer, self)

        def on_info(self, msg):
            self.info = msg

        def on_depth_info(self, msg):
            self.depth_info = msg

        def spin_until(self, cond, seconds):
            end = time.time() + seconds
            while rclpy.ok() and time.time() < end and not cond():
                rclpy.spin_once(self, timeout_sec=0.1)
            return cond()

    rclpy.init()
    node = Calib()
    if not node.spin_until(lambda: node.info is not None, 10):
        sys.exit("No message on /camera/camera_info. Is the simulation running?")
    node.spin_until(lambda: node.depth_info is not None, 3)
    info = node.info
    K = list(info.k)
    print(f"Camera_info: frame_id='{info.header.frame_id}'  size={info.width}x{info.height}")
    print(f"  fx={K[0]:.3f} fy={K[4]:.3f} cx={K[2]:.3f} cy={K[5]:.3f}")
    if node.depth_info is None:
        print("  (No /camera/depth/camera_info received; assuming the depth camera has the same optics)")
    else:
        same = list(node.depth_info.k) == K and node.depth_info.header.frame_id == info.header.frame_id
        print(f"  Depth camera_info: frame_id='{node.depth_info.header.frame_id}', identical to RGB: {same}")

    frame = info.header.frame_id
    if not node.spin_until(lambda: node.tf_buffer.can_transform(BASE_FRAME, frame, Time()), 8):
        print(f"TF lookup {BASE_FRAME} <- {frame} failed. Frames TF knows about:")
        sys.exit(node.tf_buffer.all_frames_as_string())
    tf = node.tf_buffer.lookup_transform(BASE_FRAME, frame, Time())
    t, q = tf.transform.translation, tf.transform.rotation
    R_opt, converted = to_optical(quat_to_rot(q.x, q.y, q.z, q.w))
    R_final, snapped = finalize_rotation(R_opt)
    print(f"TF {BASE_FRAME} -> {frame}: translation = ({t.x:.4f}, {t.y:.4f}, {t.z:.4f}) m")
    print(f"  camera looks along {np.round(R_opt[:, 2], 3).tolist()} in the robot frame (1,0,0 = straight ahead)"
          + ("; snapped to the exact optical rotation" if snapped else ""))

    calib = {"width": int(info.width), "height": int(info.height), "fx": K[0], "fy": K[4], "cx": K[2], "cy": K[5],
             "camera_info_frame": frame, "base_frame": BASE_FRAME, "t_base_cam": [t.x, t.y, t.z],
             "R_base_cam_optical": R_final.tolist(), "R_base_cam_optical_from_tf": R_opt.tolist(),
             "converted_from_body_convention": converted}
    old = json.load(open(CALIB)) if os.path.exists(CALIB) else {}
    calib = merge_calibration(calib, old)
    if "refine" not in calib:
        print("WARNING: no 'refine' block (the measured camera pitch / yaw / height correction). Without it label "
              "boxes can be off by a few pixels; check the overlays that 'label' writes.")
    with open(CALIB, "w") as f:
        json.dump(calib, f, indent=2)
    print(f"saved {CALIB}")
    node.destroy_node()
    rclpy.shutdown()