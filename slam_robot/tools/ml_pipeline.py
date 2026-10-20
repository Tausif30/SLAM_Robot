#!/usr/bin/env python3
"""The detector is a YOLO26n network fine-tuned on pictures from the Gazebo world. The ground-truth boxes are not
drawn by hand: the world file says where every object is, the camera model says where it appears in the picture,
and the depth image says whether it is really visible (see ground_truth.label_frame).

Command Orders for Training:
  python ml_pipeline.py calibrate            Camera model -> config/camera_calibration.json     [ROS + simulation]
  python ml_pipeline.py collect 1500         Random robot poses -> ~/slam_ws/dataset_raw        [ROS + simulation]
  python ml_pipeline.py label                Labels + colour quality check of dataset_raw
  python ml_pipeline.py build                Split by floor location -> ~/slam_ws/dataset
  python ml_pipeline.py train --epochs 40    Fine-tune yolo26n -> ~/slam_ws/runs/detector_v1    [ultralytics]
  python ml_pipeline.py evaluate --sweep     Score on the test split, confidence sweep          [ultralytics]
  python ml_pipeline.py figures              Copy every result picture into ~/slam_ws/figures

Extras:
  python ml_pipeline.py collect 300 --only grey --out ~/slam_ws/dataset_stress     Look-alike stress set
  python ml_pipeline.py label --raw ~/slam_ws/dataset_stress
  python ml_pipeline.py evaluate --raw ~/slam_ws/dataset_stress --conf 0.05
  python ml_pipeline.py teleport x y yaw                                           Teleport Robot
"""
import argparse
import dataset
import ros_collector
import teleport
import training


# Command, function with all the options, function that runs it, one-line help.
def commands():
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