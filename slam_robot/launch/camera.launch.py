"""
Camera window: the detector plus a window showing its annotated picture (boxes around what it recognises).

The demo runs in three windows:
  1  ros2 launch slam_robot simulation.launch.py      Gazebo
  2  ros2 launch slam_robot navigation.launch.py      Nav2 + RViz (map, costmaps, set goals)
  3  ros2 launch slam_robot camera.launch.py          this: what the robot recognises

  ros2 launch slam_robot camera.launch.py target_label:=blue_box      which object gets the thick green box
  ros2 launch slam_robot camera.launch.py detector_rate_hz:=5.0       lower it if the simulation slows down
  ros2 launch slam_robot camera.launch.py viewer:=false               no window (headless, Docker)
"""
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg = get_package_share_directory('slam_robot')

    return LaunchDescription([
        DeclareLaunchArgument(
            'target_label', default_value='',
            description='object that gets the thick green box (red_box, blue_box, blue_cylinder); empty = the parameter file'),
        DeclareLaunchArgument(
            'detector_rate_hz', default_value='8.0',
            description='most images per second the detector processes (empty = the parameter file)'),
        DeclareLaunchArgument(
            'viewer', default_value='true', description='open the window with the annotated camera picture'),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(pkg, 'launch', 'detector.launch.py')),
            launch_arguments={'target_label': LaunchConfiguration('target_label'),
                              'max_rate_hz': LaunchConfiguration('detector_rate_hz')}.items()),

        # rqt_image_view takes the topic as its first argument; it also starts before the topic exists
        # (the detector only publishes the picture while someone is looking at it)
        Node(package='rqt_image_view', executable='rqt_image_view', output='screen',
             arguments=['/detector/debug_image'],
             condition=IfCondition(LaunchConfiguration('viewer'))),
    ])