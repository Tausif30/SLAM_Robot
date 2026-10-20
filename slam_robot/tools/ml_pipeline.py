#!/usr/bin/env python3
"""ml_pipeline.py - build, train and test the object detector: one command line for every step.

The detector is a YOLO26n network fine-tuned on pictures from the Gazebo world. The ground-truth boxes are
not drawn by hand: the world file says where every object is, the camera model says where it appears in the
picture, and the depth image says whether it is really visible (see ground_truth.label_frame).

Steps, in order. Run from anywhere; paths are relative to this folder and to ~/slam_ws (or $SLAM_WS):

  python ml_pipeline.py calibrate            camera model -> config/camera_calibration.json     [ROS + simulation]
  python ml_pipeline.py collect 1500         random robot poses -> ~/slam_ws/dataset_raw        [ROS + simulation]
  python ml_pipeline.py label                labels + colour quality check of dataset_raw
  python ml_pipeline.py build                split by floor location -> ~/slam_ws/dataset
  python ml_pipeline.py train --epochs 40    fine-tune yolo26n -> ~/slam_ws/runs/detector_v1    [ultralytics]
  python ml_pipeline.py evaluate --sweep     score on the test split, confidence sweep          [ultralytics]

Extras:
  python ml_pipeline.py collect 300 --only grey --out ~/slam_ws/dataset_stress     look-alike stress set
  python ml_pipeline.py label --raw ~/slam_ws/dataset_stress
  python ml_pipeline.py evaluate --raw ~/slam_ws/dataset_stress --conf 0.05
  python ml_pipeline.py teleport 3.7 -2.0 -26.6                                    put the robot somewhere

Every command also runs on its own file with the same options, e.g. 'python teleport.py 3.7 -2.0 -26.6'.
ROS and Ultralytics are imported only by the commands that need them, so label / build work without either.

Files in this folder:
  commands (each has add_arguments(p) and run(args))
    calibrate.py        calibrate     camera intrinsics from camera_info, mounting from TF
    collect.py          collect       sampling loop that saves RGB + depth + pose
    label.py            label         writes YOLO labels, colour check, qa.json
    build_dataset.py    build         train / val / test split by floor cell
    train.py            train         fine-tunes YOLO26 with Ultralytics
    evaluate.py         evaluate      ground truth for scoring, detectors, confidence sweep
    teleport.py         teleport      teleport_entity(): the one call to /gazebo/set_entity_state
  shared code
    pipeline_config.py  paths and constants (classes, depth tolerances, sampling settings, ...)
    world_objects.py    labelled objects and look-alikes read from the world file
    camera_geometry.py  rotations, camera pose, one ray per pixel, ray / box and ray / cylinder hits
    ground_truth.py     label_frame(): which object is where in a picture, and is it visible
    pose_sampling.py    free space of the saved map, random and look-at robot poses
    ros_collector.py    the ROS node that teleports the robot and grabs RGB + depth + odom
    eval_metrics.py     matching detections to ground truth, confusion matrix, precision / recall
    eval_report.py      printed report, confusion picture, failure montage
"""
import argparse

import build_dataset
import calibrate
import collect
import evaluate
import label
import teleport
import train

# (command, file that implements it, one-line help)
COMMANDS = (
    ("calibrate", calibrate, "read the camera model from the running simulation"),
    ("teleport", teleport, "put the robot at x y yaw (degrees)"),
    ("collect", collect, "save RGB + depth + pose at random free robot poses"),
    ("label", label, "ground-truth labels + colour quality check for a raw folder"),
    ("build", build_dataset, "split the labelled frames by floor location into train / val / test"),
    ("train", train, "fine-tune a pretrained YOLO26 model on the dataset (CPU)"),
    ("evaluate", evaluate, "score a model on the test split or on any raw folder"),
)


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True, metavar="command")
    for name, module, help_text in COMMANDS:
        p = sub.add_parser(name, help=help_text)
        module.add_arguments(p)
        p.set_defaults(func=module.run)
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
