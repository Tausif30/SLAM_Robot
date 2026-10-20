#!/usr/bin/env python3
"""teleport.py - put the robot at x y yaw in the running Gazebo simulation.          [ROS + simulation]

  python teleport.py 3.7 -2.0 -26.6        x (m), y (m), yaw (degrees)

Same as 'python ml_pipeline.py teleport 3.7 -2.0 -26.6'. Afterwards it prints where /odom says the robot is.
teleport_entity() is the one place that calls /gazebo/set_entity_state; the frame collector (collect.py) uses it too.
"""
import argparse
import math
import sys

from camera_geometry import angle_diff, yaw_to_quat


def teleport_entity(node, client, name, x, y, yaw):
    """Place Gazebo model `name` at (x, y) on the floor, facing yaw (radians). `client` is a SetEntityState
    service client created on `node`. Returns True if Gazebo accepted the request."""
    import rclpy
    from gazebo_msgs.srv import SetEntityState
    req = SetEntityState.Request()
    req.state.name = name
    req.state.reference_frame = "world"
    req.state.pose.position.x, req.state.pose.position.y, req.state.pose.position.z = x, y, 0.01
    (req.state.pose.orientation.x, req.state.pose.orientation.y,
     req.state.pose.orientation.z, req.state.pose.orientation.w) = yaw_to_quat(yaw)
    future = client.call_async(req)
    rclpy.spin_until_future_complete(node, future, timeout_sec=5.0)
    return future.done() and future.result() is not None and future.result().success


# ---------------------------------------------------------------- command line
def add_arguments(p):
    p.add_argument("x", type=float, help="metres")
    p.add_argument("y", type=float, help="metres")
    p.add_argument("yaw_deg", type=float, help="degrees")


def run(a):
    import rclpy
    from ros_collector import ros_collector_class       # imported here because ros_collector imports this file
    rclpy.init()
    col = ros_collector_class()()
    col.start()
    if not col.teleport(a.x, a.y, math.radians(a.yaw_deg)):
        sys.exit("set_entity_state failed (see the Gazebo terminal for the reason).")
    col.odom = None
    col.spin_until(lambda: False, 1.5)                       # let the robot settle and /odom update
    print(f"teleported '{col.robot}' to x={a.x:.2f} y={a.y:.2f} yaw={a.yaw_deg:.1f} deg")
    if col.odom is None:
        print("warning: no /odom message received afterwards")
    else:
        _, ox, oy, oyaw = col.odom
        print(f"/odom now says x={ox:.3f} y={oy:.3f} yaw={math.degrees(oyaw):.1f} deg "
              f"(error {math.hypot(ox - a.x, oy - a.y) * 100:.1f} cm, "
              f"{math.degrees(abs(angle_diff(oyaw, math.radians(a.yaw_deg)))):.1f} deg)")
    col.destroy_node()
    rclpy.shutdown()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_arguments(ap)
    run(ap.parse_args(argv))


if __name__ == "__main__":
    main()
