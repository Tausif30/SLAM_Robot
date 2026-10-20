#!/usr/bin/env python3
"""train.py - fine-tune a pretrained YOLO26 model on the dataset (CPU) -> ~/slam_ws/runs/<name>   [ultralytics]

  python train.py --epochs 40

Same as 'python ml_pipeline.py train ...'. Needs the dataset from build_dataset.py.
"""
import argparse
import glob
import os
import sys
import time
import yaml

from pipeline_config import DATASET, RUNS, WEIGHTS_DIR


def dataset_summary(data_yaml):
    cfg = yaml.safe_load(open(data_yaml))
    counts = {s: len(glob.glob(os.path.join(cfg["path"], cfg[s], "*.png"))) for s in ("train", "val", "test")}
    return cfg, counts


# ---------------------------------------------------------------- command line
def add_arguments(p):
    p.add_argument("--model", default="yolo26n.pt", help="pretrained weights (yolo26n.pt = smallest)")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--name", default="detector_v1")
    p.add_argument("--patience", type=int, default=20, help="stop early if validation does not improve for this many epochs")
    p.add_argument("--plots", action="store_true", help="save training curves and sample images (a little slower)")


def run(a):
    data = os.path.join(DATASET, "data.yaml")
    if not os.path.exists(data):
        sys.exit(f"{data} not found: run the 'build' step first")
    cfg, counts = dataset_summary(data)
    print(f"dataset: {counts['train']} train / {counts['val']} val / {counts['test']} test images; "
          f"classes: {list(cfg['names'].values())}")
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    os.chdir(WEIGHTS_DIR)                       # pretrained weights are downloaded into the current folder
    from ultralytics import YOLO
    model = YOLO(a.model)
    t0 = time.time()
    model.train(data=data, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, device="cpu", cache="ram",
                project=RUNS, name=a.name, exist_ok=True, patience=a.patience, plots=a.plots, seed=0)
    print(f"\ntraining finished in {(time.time() - t0) / 60:.1f} min; results in {RUNS}/{a.name}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_arguments(ap)
    run(ap.parse_args(argv))


if __name__ == "__main__":
    main()
