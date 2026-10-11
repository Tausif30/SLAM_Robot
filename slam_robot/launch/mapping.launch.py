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
    slam_toolbox = get_package_share_directory('slam_toolbox')
    nav2_bringup = get_package_share_directory('nav2_bringup')

    use_sim_time = LaunchConfiguration('use_sim_time')
    use_rviz = LaunchConfiguration('rviz')

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument('rviz', default_value='true'),

        # SLAM: subscribes to /scan and TF (odom -> base_footprint),
        # publishes /map and the TF map -> odom.
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(slam_toolbox, 'launch', 'online_async_launch.py')),
            launch_arguments={
                'use_sim_time': use_sim_time,
                'slam_params_file': os.path.join(pkg, 'config', 'slam_params.yaml'),
            }.items()),

        # Visualization (a custom RViz config comes later in the project)
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen',
            arguments=['-d', os.path.join(
                nav2_bringup, 'rviz', 'nav2_default_view.rviz')],
            parameters=[{'use_sim_time': use_sim_time}],
            condition=IfCondition(use_rviz)),
    ])
