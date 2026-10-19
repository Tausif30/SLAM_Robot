"""
Commands:
ros2 launch slam_robot detector.launch.py
ros2 launch slam_robot detector.launch.py target_label:=blue_box
"""
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def launch_setup(context):
    parameters = [LaunchConfiguration('params_file').perform(context)]
    overrides = {}
    for name in ('target_label', 'model_path'):
        value = LaunchConfiguration(name).perform(context)
        if value:
            overrides[name] = value
    if overrides:
        parameters.append(overrides)
    return [Node(package='slam_robot', executable='detector_node', name='detector_node',
                 output='screen', parameters=parameters)]


def generate_launch_description():
    pkg = get_package_share_directory('slam_robot')
    return LaunchDescription([
        DeclareLaunchArgument(
            'params_file',
            default_value=os.path.join(pkg, 'config', 'detector_params.yaml'),
            description='parameter file for the detector node'),
        DeclareLaunchArgument(
            'target_label', default_value='',
            description='override target_label (red_box, blue_box, blue_cylinder); empty = use the file'),
        DeclareLaunchArgument(
            'model_path', default_value='',
            description='override the path of the YOLO weights; empty = use the file'),
        OpaqueFunction(function=launch_setup),
    ])