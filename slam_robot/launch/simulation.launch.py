import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, IncludeLaunchDescription,
                            SetEnvironmentVariable)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg = get_package_share_directory('slam_robot')
    gazebo_ros = get_package_share_directory('gazebo_ros')

    use_sim_time = LaunchConfiguration('use_sim_time')
    world = LaunchConfiguration('world')
    x_pose = LaunchConfiguration('x_pose')
    y_pose = LaunchConfiguration('y_pose')
    gui = LaunchConfiguration('gui')

    # Let Gazebo find the models via model://name, keeping whatever model paths were already set.
    model_paths = [os.path.join(pkg, 'models')]
    if os.environ.get('GAZEBO_MODEL_PATH'):
        model_paths.append(os.environ['GAZEBO_MODEL_PATH'])
    set_model_path = SetEnvironmentVariable(
        'GAZEBO_MODEL_PATH', os.pathsep.join(model_paths))

    # robot_state_publisher publishes the fixed TF frames from the URDF
    with open(os.path.join(pkg, 'urdf', 'slam_robot_waffle.urdf'), 'r') as f:
        robot_description = f.read()

    # The SDF model is what Gazebo simulates (body, LiDAR, RGB-D camera, drive)
    model_sdf = os.path.join(pkg, 'models', 'slam_robot_waffle', 'model.sdf')

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument(
            'world',
            default_value=os.path.join(pkg, 'worlds', 'object_search.world')),
        DeclareLaunchArgument('x_pose', default_value='-3.0'),
        DeclareLaunchArgument('y_pose', default_value='-2.0'),

        set_model_path,

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(gazebo_ros, 'launch', 'gzserver.launch.py')),
            launch_arguments={'world': world}.items()),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(gazebo_ros, 'launch', 'gzclient.launch.py'))),

        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            output='screen',
            parameters=[{'use_sim_time': use_sim_time,
                         'robot_description': robot_description}]),

        Node(
            package='gazebo_ros',
            executable='spawn_entity.py',
            arguments=['-entity', 'slam_robot', '-file', model_sdf,
                       '-x', x_pose, '-y', y_pose, '-z', '0.01'],
            output='screen'),
    ])