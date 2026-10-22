"""
Camera window showing the robot's viewpoint with Object Detection.
  ros2 launch slam_robot camera.launch.py detector_rate_hz:=3.0    pictures analysed per second (default 8)
  ros2 launch slam_robot camera.launch.py confidence:=0.6          ignore detections below this (default 0.4)
  ros2 launch slam_robot camera.launch.py viewer:=false            no window (headless, Docker)

Both numbers can also be changed while it runs:   ros2 param set /detector_node max_rate_hz 3.0
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'detector_rate_hz', default_value='8.0',
            description='most pictures per second the detector analyses'),
        DeclareLaunchArgument(
            'confidence', default_value='0.4',
            description='ignore detections below this confidence (0 to 1)'),
        DeclareLaunchArgument(
            'viewer', default_value='true',
            description='open the window with the annotated camera picture'),

        Node(package='slam_robot', executable='detector_node', name='detector_node', output='screen',
             parameters=[{
                 'max_rate_hz': ParameterValue(LaunchConfiguration('detector_rate_hz'), value_type=float),
                 'confidence_threshold': ParameterValue(LaunchConfiguration('confidence'), value_type=float),
             }]),

        # rqt_image_view takes the topic as its first argument; it also starts before the topic exists
        # (the detector only analyses pictures while someone is looking at them)
        Node(package='rqt_image_view', executable='rqt_image_view', output='screen',
             arguments=['/detector/debug_image'],
             condition=IfCondition(LaunchConfiguration('viewer'))),
    ])
