"""pipeline_config.py - paths and constants shared by the steps of the ML pipeline.

Paths are relative to this file (things inside the slam_robot package) and to ~/slam_ws, or $SLAM_WS if set
(the big data, which is not in git).
"""
import os

# ---------------------------------------------------------------- paths
PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))          # .../src/slam_robot
WS = os.path.expanduser(os.environ.get("SLAM_WS", "~/slam_ws"))                # where the big data lives (not in git)
WORLD = os.path.join(PKG_DIR, "worlds", "object_search.world")
MAP_YAML = os.path.join(PKG_DIR, "maps", "object_search_map.yaml")
CALIB = os.path.join(PKG_DIR, "config", "camera_calibration.json")
RAW = os.path.join(WS, "dataset_raw")                 # frames straight from the simulator + their labels
DATASET = os.path.join(WS, "dataset")                 # the train / val / test split that YOLO reads
RUNS = os.path.join(WS, "runs")                       # training runs and evaluation reports
WEIGHTS_DIR = os.path.join(WS, "weights")             # downloaded pretrained weights (yolo26n.pt, ...)
DEFAULT_WEIGHTS = os.path.join(RUNS, "detector_v1", "weights", "best.pt")

# ---------------------------------------------------------------- classes and labelling
CLASSES = ["red_box", "blue_box", "blue_cylinder"]
NAMES = CLASSES
BASE_Z = 0.0                 # height of the robot's base_footprint above the floor (m)
DEPTH_TOL_ABS = 0.03         # a pixel belongs to the object if measured depth is within
DEPTH_TOL_REL = 0.01         #   3 cm + 1 % of the expected depth
NEAR_CLIP, FAR_CLIP = 0.1, 8.0   # camera clip planes (m): nothing outside them is drawn, in RGB or depth
MIN_PIXELS = 80              # smaller than this (about 9x9 px) is too tiny to be a label
MIN_VISIBLE_FRAC = 0.35      # less than this visible -> too occluded to label

# ---------------------------------------------------------------- collecting frames
CLEARANCE = 0.35                 # m from any wall/object: robot radius 0.22 + margin
MODES = ("red", "blue", "random")
MODE_WEIGHTS = (0.35, 0.35, 0.30)
LOOK_DIST = (0.6, 5.0)           # m from the object we look at
LOOK_JITTER_DEG = 25.0           # random aim error, so objects are sometimes off-centre / cut off
SETTLE_SIM_S = 0.8               # simulated seconds to wait after a teleport
POSE_TOL_M, POSE_TOL_DEG = 0.02, 1.5

# models that belong to the world; the one model in /gazebo/model_states that is not listed here is the robot
WORLD_MODELS = {
    "ground_plane", "object_search_walls", "red_box", "blue_box_1", "blue_box_2",
    "blue_box_3", "blue_cylinder_1", "blue_crate_1", "cylinder_1", "cylinder_2",
    "sphere_1", "crate_1", "sun",
}
ROBOT_HINTS = ("waffle", "robot", "turtle", "burger")
