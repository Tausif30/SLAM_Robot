# SLAM Robot: finding and recognising objects in ROS 2 (Gazebo, Nav2, YOLO26)

A simulated TurtleBot3-Waffle-style robot maps a custom Gazebo world with SLAM, localises on that map, drives wherever
you click in RViz, and recognises red and blue objects with a neural network which was made by fine tuning a pre-trained YOLO26 Model. 
Everything is available in one Docker image making it easy for anyone to use.

## About the project

The world is a rectangular room, split into two rooms by a wall doorway. It holds
one **red box**, three **blue boxes** and a blue crate, one **blue cylinder**, and four grey
**look-alikes** (two cylinders, a sphere and a crate) that the detector must *not* mistake for the targets.

The robot carries a 5 Hz LiDAR, a 640 x 480 RGB-D camera (15 Hz), an IMU and a differential drive. The project has four
parts:

1. **SLAM.** SLAM Toolbox builds an occupancy-grid map from the LiDAR and the wheel odometry (232 x 136 cells of 5 cm,
   mean wall error 2.7 cm).
2. **Navigation.** AMCL localises the robot on the saved map and Nav2 plans and drives to any point you click in RViz,
   avoiding obstacles with costmaps built from the LiDAR.
3. **Perception.** A YOLO26n network recognises the red box, the blue boxes and the blue cylinder in the camera picture.
   It was fine-tuned only on pictures from the simulator that are **labelled automatically**: the world file says where
   every object is, the camera model says where it appears in the picture, and the depth image says whether it is really
   visible. An independent colour test checks every label, so no picture was labelled by hand.
4. **Packaging.** Everything runs from one Docker image, in three windows.

**At a glance** (details and charts are in the report):

* On the test pictures (taken at floor positions never seen in training) the detector found **214 of 215** objects with
  **no false alarm** at any confidence from 0.05 to 0.7. The off-the-shelf zero-shot detector found 4 of the 90 red boxes
  and raised 48 false alarms.
* On a separate set aimed at the grey look-alikes it found 197 of 199 objects with no false alarm from confidence 0.25 up.
* Inside the container Gazebo runs at real time (factor 1.00) and the detector needs 84 to 121 ms per picture, when the
  GPU draws the windows.

## Project Video


https://github.com/user-attachments/assets/c9a3614a-505e-47d2-8e82-04ffca539c8f


## Try it yourself with Docker

**You need:** Docker with Compose v2, a PC with an Intel or AMD processor (amd64), about 7 GB of free disk space (the
download is about 1.5 GB), and a screen the container can draw on: Windows 11 with WSL2 (WSLg provides it), or a Linux
desktop (run `xhost +local:` once first). Docker must be running.

```bash
git clone https://github.com/Tausif30/SLAM_Robot.git
cd SLAM_Robot
docker compose pull         # Downloads the Package
docker compose up -d
```

Then open three terminals in the same folder, one per window, and start them in this order:

```bash
docker compose exec robot slam sim       # Gazebo: the world and the robot
```

```bash
docker compose exec robot slam nav       # Nav2 and RViz: map, costmaps, goals (wait about 30 s)
```

```bash
docker compose exec robot slam camera    # Camera picture with the recognised objects
```


Stop with `Ctrl+C` in each terminal, then `docker compose down`.

**Windows with a GPU.** By default the CPU draws the windows. That works everywhere but is slow: Gazebo ran at 0.85 of
real time and the detector needed about a second per picture. With WSL2 and a GPU, let the GPU draw them:

```bash
echo 'COMPOSE_FILE=docker-compose.yml:docker-compose.wsl-gpu.yml' > .env
docker compose down && docker compose up -d
docker compose exec robot slam bash -c 'glxinfo -B | grep "OpenGL renderer"'    # must say D3D12 (...), not llvmpipe
```

Tested with an AMD Radeon 780M. To go back to the CPU, delete `.env`.

### Using it

* **Send the robot somewhere:** in RViz choose **Nav2 Goal** in the toolbar, click on the map and drag to set the
  direction. The robot plans a path and drives around obstacles. If it looks misplaced on the map, use
  **2D Pose Estimate** and click where it really is.
* **See what it recognises:** The camera shows live images with a coloured box and a label such as
  around every object (red box: red outline, blue box: blue, blue cylinder: light blue). The detector
  only analyses pictures while this window is open.
* **Change how often it looks, while it runs** (write the number with a decimal point):

  ```bash
  docker compose exec robot slam ros2 param set /detector_node max_rate_hz 3.0
  docker compose exec robot slam ros2 param set /detector_node confidence_threshold 0.6
  ```

  Or at start: `docker compose exec robot slam camera detector_rate_hz:=3.0 confidence:=0.6`.

## Run it natively (WSL or Linux)

You need Ubuntu 22.04 with ROS 2 Humble installed.

```bash
sudo apt install ros-humble-gazebo-ros-pkgs ros-humble-turtlebot3-gazebo ros-humble-nav2-bringup \
     ros-humble-slam-toolbox ros-humble-rviz2 ros-humble-rqt-image-view python3-venv
mkdir -p ~/slam_ws && git clone https://github.com/Tausif30/SLAM_Robot.git ~/slam_ws/src
python3 -m venv --system-site-packages ~/slam_ws/venv
source ~/slam_ws/venv/bin/activate
pip install --upgrade pip && pip install -r ~/slam_ws/src/requirements.txt
cd ~/slam_ws && python -m colcon build --symlink-install
```

Build with `python -m colcon` while the virtual environment is active: the detector's start script then uses the
environment's Python, where PyTorch and Ultralytics are installed (and NumPy 2 stays out of ROS's own Python).

In **each** of the three terminals, first prepare the environment, then start one window:

```bash
source /opt/ros/humble/setup.bash && source ~/slam_ws/install/setup.bash
source /usr/share/gazebo/setup.sh                                  # Gazebo's own model folders (sun, ground_plane)
export SLAM_MODEL_PATH=~/slam_ws/src/weights/detector_v1.pt        # needed in the camera terminal

ros2 launch slam_robot simulation.launch.py      # terminal 1
ros2 launch slam_robot navigation.launch.py      # terminal 2
ros2 launch slam_robot camera.launch.py          # terminal 3
```

The launch options are the same as in the Docker version, for example `ros2 launch slam_robot camera.launch.py
detector_rate_hz:=3.0`.

## Training the detector yourself

The whole pipeline is in `slam_robot/tools/`. Run it inside the virtual environment; the first step needs the simulation
running (terminal 1 above).

```bash
cd ~/slam_ws/src/slam_robot/tools
python ml_pipeline.py collect 1500      # teleport the robot to random poses, save RGB + depth + pose
python ml_pipeline.py label             # ground-truth labels from the world file + colour quality check
python ml_pipeline.py build             # split by floor location into train / val / test
python ml_pipeline.py train --epochs 40
python ml_pipeline.py evaluate --sweep  # score on the test split, confidence sweep
python ml_pipeline.py figures           # copy every result picture into one folder
```

Add `collect 300 --only grey --out ~/slam_ws/dataset_stress` and `label --raw ~/slam_ws/dataset_stress` to make the
look-alike test set, and `evaluate --yoloe` to compare with the zero-shot detector. `python ml_pipeline.py -h` lists
everything. The data and results go to `~/slam_ws` (or `$SLAM_WS`). Training 40 epochs took about 1.7 hours on the CPU.

## Runtime architecture

<img width="9130" height="5927" alt="SLAM_Runtime_Architecture" src="https://github.com/user-attachments/assets/ec1e2a5f-4eba-472d-9778-586cd6b6c5f3" />

The system runs as three groups of processes, one per window. ROS 2 Humble connects them with its default middleware
(Fast DDS). Gazebo's server and its window (`gzclient`) talk to each other over Gazebo Transport instead (protocol
buffers over TCP, master at `localhost:11345`), which is not ROS.


### Processes

| Window | Launch file | Process | What it does |
|---|---|---|---|
| 1 | `simulation.launch.py` | `gzserver` | Gazebo physics (ODE, 1 ms steps) and the robot's plugins: LiDAR, camera, IMU, drive, joint states, model states |
| 1 | `simulation.launch.py` | `gzclient` | the Gazebo window |
| 1 | `simulation.launch.py` | `robot_state_publisher` | turns the URDF and the wheel angles into the transform tree |
| 1 | `simulation.launch.py` | `spawn_entity.py` | puts the robot into the world, then exits |
| 2 | `navigation.launch.py` | `nav2_container` | one process holding `map_server`, `amcl`, `controller_server`, `smoother_server`, `planner_server`, `behavior_server`, `bt_navigator`, `waypoint_follower`, `velocity_smoother` and two lifecycle managers |
| 2 | `navigation.launch.py` | `rviz2` | map, costmaps, paths, and the goal and pose tools |
| 3 | `camera.launch.py` | `detector_node` | YOLO26n on the camera pictures |
| 3 | `camera.launch.py` | `rqt_image_view` | shows `/detector/debug_image` |
| mapping | `mapping.launch.py` | `slam_toolbox`, `rviz2` | builds a new map (`slam mapping`) |

### The four ways ROS 2 nodes communicate

| Type | Pattern | Used here for |
|---|---|---|
| **Topic** | one-way stream, publishers to subscribers, no reply | sensors, odometry, transforms, velocity commands, maps, the detector's picture |
| **Service** | one request, one reply | clearing a costmap, spawning the robot, moving it for data collection, saving a map |
| **Action** | a long goal with feedback, a result and cancellation | "drive to this pose", "compute a path", "follow this path", recovery behaviours |
| **Parameter** | a named setting of a node, readable and changeable at run time | detector rate and confidence, Nav2 and SLAM tuning, `use_sim_time` |

### Topics (publish and subscribe)

`/scan` and `/camera/*` are published with the sensor-data QoS (best effort), so their subscribers must use it too, as
the detector does. Run `docker compose exec robot slam ros2 topic list -t` to see every topic; the ones below are the
ones the system uses (plus debug topics, `/rosout` and `/parameter_events`, which every node has).

| Topic | Message type | Published by | Subscribed by | Content |
|---|---|---|---|---|
| `/clock` | `rosgraph_msgs/Clock` | `gzserver` (`gazebo_ros_init`) | every node with `use_sim_time: true` | simulation time |
| `/scan` | `sensor_msgs/LaserScan` | `gzserver` (LiDAR, 5 Hz, frame `base_scan`) | `amcl`, both costmaps, `slam_toolbox` (mapping), RViz | 360 degree ranges |
| `/odom` | `nav_msgs/Odometry` | `gzserver` (diff drive, 30 Hz) | `bt_navigator`, `controller_server`, `velocity_smoother`, the data-collection tool | wheel odometry |
| `/tf` | `tf2_msgs/TFMessage` | diff drive (`odom` to `base_footprint`), `robot_state_publisher` (wheels), `amcl` or `slam_toolbox` (`map` to `odom`) | Nav2, RViz | moving transforms |
| `/tf_static` | `tf2_msgs/TFMessage` | `robot_state_publisher` | Nav2, RViz | fixed transforms (sensors, casters) |
| `/joint_states` | `sensor_msgs/JointState` | `gzserver` (30 Hz) | `robot_state_publisher` | wheel angles |
| `/robot_description` | `std_msgs/String` | `robot_state_publisher` | none by default | the URDF |
| `/imu` | `sensor_msgs/Imu` | `gzserver` (200 Hz) | none by default | orientation, rates, accelerations |
| `/camera/image_raw` | `sensor_msgs/Image` (`rgb8`, 640 x 480) | `gzserver` (camera, 15 Hz model, about 11 to 12 Hz measured) | `detector_node`, the data-collection tool | colour picture |
| `/camera/depth/image_raw` | `sensor_msgs/Image` (`32FC1`) | `gzserver` | the data-collection tool (labelling) | depth in metres |
| `/camera/camera_info`, `/camera/depth/camera_info` | `sensor_msgs/CameraInfo` | `gzserver` | the calibration tool | camera intrinsics |
| `/camera/points` | `sensor_msgs/PointCloud2` | `gzserver` | none by default | depth point cloud |
| `/gazebo/model_states`, `/gazebo/link_states` | `gazebo_msgs/ModelStates`, `LinkStates` | `gzserver` (`gazebo_ros_state`, 2 Hz) | the data-collection tool | poses of every model and link |
| `/map` | `nav_msgs/OccupancyGrid` (latched) | `map_server` (navigation) or `slam_toolbox` (mapping) | `amcl`, global costmap, RViz | the map |
| `/initialpose` | `geometry_msgs/PoseWithCovarianceStamped` | RViz (2D Pose Estimate) | `amcl` | "the robot is here" |
| `/amcl_pose` | `geometry_msgs/PoseWithCovarianceStamped` | `amcl` | none by default | estimated robot pose |
| `/particle_cloud` | `nav2_msgs/ParticleCloud` | `amcl` | RViz | the localisation particles |
| `/goal_pose` | `geometry_msgs/PoseStamped` | any publisher (RViz "2D Goal Pose" tool, `ros2 topic pub`) | `bt_navigator` | an alternative way to give a goal |
| `/plan` | `nav_msgs/Path` | `planner_server` | RViz | global path |
| `/local_plan` | `nav_msgs/Path` | `controller_server` (DWB) | RViz | local trajectory |
| `/global_costmap/costmap`, `/local_costmap/costmap` | `nav_msgs/OccupancyGrid` | the two costmaps | RViz | obstacle maps with inflation |
| `/local_costmap/costmap_raw` | `nav2_msgs/Costmap` | local costmap | `behavior_server` | costmap for the recovery behaviours |
| `/local_costmap/published_footprint` | `geometry_msgs/PolygonStamped` | local costmap | `behavior_server`, RViz | the robot's outline |
| `/cmd_vel_nav` | `geometry_msgs/Twist` | `controller_server` | `velocity_smoother` | raw velocity command |
| `/cmd_vel` | `geometry_msgs/Twist` | `velocity_smoother` | `gzserver` (diff drive) | smoothed velocity command |
| `/bond` | `bond/Status` | every Nav2 lifecycle node and the lifecycle managers | the same | heartbeat between manager and nodes |
| `/detector/debug_image` | `sensor_msgs/Image` (`rgb8`) | `detector_node` (reliable, depth 1) | `rqt_image_view` | the picture with boxes and labels |

### Services (request and reply)

| Service | Type | Server | Client | Purpose |
|---|---|---|---|---|
| `/spawn_entity` | `gazebo_msgs/SpawnEntity` | `gzserver` | `spawn_entity.py` (once at start) | put the robot into the world |
| `/gazebo/set_entity_state` | `gazebo_msgs/SetEntityState` | `gzserver` (`gazebo_ros_state`) | the `teleport` and `collect` tools | move the robot for data collection |
| `/gazebo/get_entity_state` | `gazebo_msgs/GetEntityState` | `gzserver` | none by default | read a model's pose |
| `/global_costmap/clear_entirely_global_costmap`, `/local_costmap/clear_entirely_local_costmap` | `nav2_msgs/ClearEntireCostmap` | the two costmaps | `bt_navigator` (recovery steps of its behavior tree) | forget obstacles when stuck |
| `/lifecycle_manager_navigation/manage_nodes`, `/lifecycle_manager_localization/manage_nodes` | `nav2_msgs/ManageLifecycleNodes` | the lifecycle managers | RViz's Nav2 panel | start, pause or reset Nav2 |
| `/<node>/change_state`, `/<node>/get_state` | `lifecycle_msgs/ChangeState`, `GetState` | each Nav2 lifecycle node | the lifecycle managers; you with `ros2 lifecycle get /bt_navigator` | configure and activate nodes |
| `/slam_toolbox/save_map`, `/slam_toolbox/serialize_map` | `slam_toolbox/SaveMap`, `SerializePoseGraph` | `slam_toolbox` (mapping only) | you | save the map made while mapping |

### Actions (long goals with feedback)

| Action | Type | Server | Client | Purpose |
|---|---|---|---|---|
| `/navigate_to_pose` | `nav2_msgs/NavigateToPose` | `bt_navigator` | RViz's Nav2 panel (the **Nav2 Goal** tool) | "drive to this pose", with distance feedback and cancel |
| `/compute_path_to_pose` | `nav2_msgs/ComputePathToPose` | `planner_server` (NavFn, plugin `GridBased`) | `bt_navigator` | plan a global path |
| `/follow_path` | `nav2_msgs/FollowPath` | `controller_server` (DWB) | `bt_navigator` | follow the path, avoiding obstacles |
| `/spin`, `/backup`, `/wait` | `nav2_msgs/Spin`, `BackUp`, `Wait` | `behavior_server` | `bt_navigator` (recovery: spin 1.57 rad, back up 0.30 m, wait 5 s) | get unstuck |

`behavior_server` also provides `/drive_on_heading` and `/assisted_teleop`, `smoother_server` provides `/smooth_path`, and
`bt_navigator` and `waypoint_follower` provide `/navigate_through_poses` and `/follow_waypoints`. The default behavior
tree does not call any of these.

### Parameters (settings of a node)

Every node answers `/<node>/get_parameters`, `set_parameters`, `list_parameters` and `describe_parameters`, and announces
changes on `/parameter_events`. The ones that matter here:

| Node | Parameter | Value | Where it is set |
|---|---|---|---|
| `detector_node` | `max_rate_hz` | 8.0 (most pictures analysed per second; changeable while running) | `camera.launch.py` (`detector_rate_hz:=`), `ros2 param set` |
| `detector_node` | `confidence_threshold` | 0.4 (changeable while running) | `camera.launch.py` (`confidence:=`), `ros2 param set` |
| `detector_node` | `model_path`, `imgsz`, `device` | `$SLAM_MODEL_PATH`, 640, `cpu` (read once at start) | the node's defaults |
| all simulation-time nodes | `use_sim_time` | `true` | the launch files and `config/nav2_params.yaml` |
| Nav2 (`amcl`, costmaps, `controller_server`, ...) | initial pose (-3, -2, 0), robot radius 0.22 m, resolution 0.05 m, controller 20 Hz | | `config/nav2_params.yaml` |
| `slam_toolbox` | mode `mapping`, resolution 0.05 m, update every 2 s, loop closing on | | `config/slam_params.yaml` |

### Transform tree

```
map -> odom -> base_footprint -> base_link -+-> wheel_left_link, wheel_right_link
 (amcl)  (diff drive)                       +-> caster_back_left_link, caster_back_right_link, imu_link
                                            +-> base_scan                       (LiDAR)
                                            +-> camera_link -+-> camera_rgb_frame   -> camera_rgb_optical_frame
                                                             +-> camera_depth_frame -> camera_depth_optical_frame
```

## Troubleshooting

* **No window appears.** `echo $DISPLAY` should print something like `:0`. On Linux run `xhost +local:` first. On
  Windows, Microsoft's WSLg container example also mounts `/mnt/wslg`; if the windows do not open, add
  `- /mnt/wslg:/mnt/wslg` under `volumes:` in `docker-compose.yml`.
* **Everything is slow, and Nav2 warns `Behavior Tree tick rate 100.00 was exceeded`.** The CPU is drawing the windows.
  Use the GPU file (see above) on WSL2 with a GPU, or lower the detector rate.
* **`Unable to find uri[model://sun]` or `ground_plane` in the Gazebo log.** Gazebo's own folders were not on the path.
  Start Gazebo with `slam sim` (Docker) or `source /usr/share/gazebo/setup.sh` first (native).
* **`Message Filter dropping message ... earlier than all the data in the transform cache`.** Normal for a few seconds
  while everything starts. If it never stops, the machine is overloaded.
* **The robot does not move after a goal.** Wait for Nav2 (`docker compose exec robot slam ros2 lifecycle get
  /bt_navigator` must say `active [3]`). If the robot looks misplaced, use **2D Pose Estimate**. A goal inside a wall
  can be accepted by the planner, so click on free space.
* **The camera window stays empty.** Start the simulation first. The detector only works while the window is open, and
  it prints `in view: ...` when it sees something.
* **`ros2 param set` says the parameter type is wrong.** Write the number with a decimal point (`3.0`, not `3`).
* **Native: `No module named 'ultralytics'`.** The package was built with the system Python. Rebuild with
  `python -m colcon build` while the virtual environment is active.
* **Native: `canonicalize_version() got an unexpected keyword argument 'strip_trailing_zero'`.** Run
  `pip install "packaging>=24.2"` inside the virtual environment (it is in `requirements.txt`).

## Licence

GNU Affero General Public License v3.0 (see `LICENSE`). The detector uses Ultralytics YOLO, which is AGPL-3.0, and the
weights in `weights/` were trained with it.
