#!/usr/bin/env python3
"""ml_pipeline.py - build, train and test the object detector: one command line for every step.

The detector is a YOLO26n network fine-tuned on pictures from the Gazebo world. The ground-truth boxes are not
drawn by hand: the world file says where every object is, the camera model says where it appears in the picture,
and the depth image says whether it is really visible (see ground_truth.label_frame).

The steps, in order. Run from anywhere; paths are relative to this folder and to ~/slam_ws (or $SLAM_WS):

  python ml_pipeline.py calibrate            camera model -> config/camera_calibration.json     [ROS + simulation]
  python ml_pipeline.py collect 1500         random robot poses -> ~/slam_ws/dataset_raw        [ROS + simulation]
  python ml_pipeline.py label                labels + colour quality check of dataset_raw
  python ml_pipeline.py build                split by floor location -> ~/slam_ws/dataset
  python ml_pipeline.py train --epochs 40    fine-tune yolo26n -> ~/slam_ws/runs/detector_v1    [ultralytics]
  python ml_pipeline.py evaluate --sweep     score on the test split, confidence sweep          [ultralytics]
  python ml_pipeline.py figures              copy every result picture into ~/slam_ws/figures

Extras:
  python ml_pipeline.py collect 300 --only grey --out ~/slam_ws/dataset_stress     look-alike stress set
  python ml_pipeline.py label --raw ~/slam_ws/dataset_stress
  python ml_pipeline.py evaluate --raw ~/slam_ws/dataset_stress --conf 0.05
  python ml_pipeline.py teleport 3.7 -2.0 -26.6                                    put the robot somewhere

ROS and Ultralytics are imported only by the commands that need them, so label / build / figures work without either.

The 8 files in this folder:
  ml_pipeline.py      this file: the command line
  pipeline_config.py  paths and constants (classes, depth tolerances, sampling settings, ...)
  camera_geometry.py  rotations, camera pose, one ray per pixel, ray / box and ray / cylinder hits, calibration maths
  ground_truth.py     the world file -> objects;  label_frame(): which object is where in a picture, is it visible?
  dataset.py          pose sampling, collect, label (+ colour check), build (split by floor cell)
  training.py         train, scoring (confusion matrix ...), reports, evaluate (+ confidence sweep), figures
  ros_collector.py    everything that needs ROS + the simulation: the Collector node and the calibrate command
  teleport.py         teleport_entity(): the one call to /gazebo/set_entity_state, and the teleport command
"""
import argparse

import dataset
import ros_collector
import teleport
import training


def commands():
    """(command, function that defines its options, function that runs it, one-line help).
    Looked up when the parser is built, so each command stays defined next to its code."""
    return (
        ("calibrate", ros_collector.calibrate_arguments, ros_collector.run_calibrate,
         "read the camera model from the running simulation"),
        ("teleport", teleport.add_arguments, teleport.run, "put the robot at x y yaw (degrees)"),
        ("collect", dataset.collect_arguments, dataset.run_collect,
         "save RGB + depth + pose at random free robot poses"),
        ("label", dataset.label_arguments, dataset.run_label,
         "ground-truth labels + colour quality check for a raw folder"),
        ("build", dataset.build_arguments, dataset.run_build,
         "split the labelled frames by floor location into train / val / test"),
        ("train", training.train_arguments, training.run_train,
         "fine-tune a pretrained YOLO26 model on the dataset (CPU)"),
        ("evaluate", training.evaluate_arguments, training.run_evaluate,
         "score a model on the test split or on any raw folder"),
        ("figures", training.figures_arguments, training.run_figures,
         "copy every result picture into one folder"),
    )


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True, metavar="command")
    for name, add_arguments, run, help_text in commands():
        p = sub.add_parser(name, help=help_text)
        add_arguments(p)
        p.set_defaults(func=run)
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()