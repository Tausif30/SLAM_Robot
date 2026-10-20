#!/usr/bin/env python3
"""calibrate.py - read the camera model from the running simulation -> config/camera_calibration.json
                                                                                      [ROS + simulation]
  python calibrate.py

Same as 'python ml_pipeline.py calibrate'. Intrinsics come from /camera/camera_info, the camera mounting from TF.
A hand-fitted 'refine' block already in the file is kept.
"""
import argparse
import json
import os
import sys
import time
import numpy as np

from camera_geometry import quat_to_rot
from pipeline_config import CALIB

BASE_FRAME = "base_footprint"
# Optical frame: x right, y down, z forward.  Body frame: x forward, y left, z up.
# v_body = M_BODY_FROM_OPTICAL @ v_optical
M_BODY_FROM_OPTICAL = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])


def to_optical(R):
    """R: orientation of the camera frame in the robot frame. Return the optical-frame
    orientation and whether a conversion was needed. Raises if the camera does not face forward."""
    forward = np.array([1.0, 0.0, 0.0])
    if R[:, 2] @ forward > 0.8:
        return R, False                      # frame is already optical (z forward)
    if R[:, 0] @ forward > 0.8:
        return R @ M_BODY_FROM_OPTICAL, True  # body convention (x forward): convert
    raise ValueError(f"camera does not face forward; rotation matrix:\n{R}")


def finalize_rotation(R_opt, tol=0.01):
    """The simulator renders with exact angles, but the TF tree can carry rounding (1.57 instead of pi/2).
    If the TF rotation is within tol of the exact optical rotation, use the exact one."""
    if np.abs(np.asarray(R_opt) - M_BODY_FROM_OPTICAL).max() < tol:
        return M_BODY_FROM_OPTICAL.copy(), True
    return np.asarray(R_opt), False


def merge_calibration(new, old):
    """Keep the hand-fitted 'refine' block (camera pitch / yaw / height correction) of an existing file."""
    merged = dict(new)
    if old and old.get("refine"):
        merged["refine"] = old["refine"]
    return merged


# ---------------------------------------------------------------- command line
def add_arguments(p):
    """No options: everything comes from /camera/camera_info and TF."""


def run(a):
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
    print(f"camera_info: frame_id='{info.header.frame_id}'  size={info.width}x{info.height}")
    print(f"  fx={K[0]:.3f} fy={K[4]:.3f} cx={K[2]:.3f} cy={K[5]:.3f}")
    if node.depth_info is None:
        print("  (no /camera/depth/camera_info received; assuming the depth camera has the same optics)")
    else:
        same = list(node.depth_info.k) == K and node.depth_info.header.frame_id == info.header.frame_id
        print(f"  depth camera_info: frame_id='{node.depth_info.header.frame_id}', identical to RGB: {same}")

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


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_arguments(ap)
    run(ap.parse_args(argv))


if __name__ == "__main__":
    main()
