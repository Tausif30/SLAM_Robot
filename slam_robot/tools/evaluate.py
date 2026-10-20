#!/usr/bin/env python3
"""evaluate.py - score a model on the test split or on any raw folder.                            [ultralytics]

  python evaluate.py --sweep                                       test split + confidence sweep
  python evaluate.py --raw ~/slam_ws/dataset_stress --conf 0.05    every kept frame of a stress set

Same as 'python ml_pipeline.py evaluate ...'. Writes report.json, confusion.png and failures_montage.png
into ~/slam_ws/runs/eval_<run>_<split or folder>/.
"""
import argparse
import json
import os
import numpy as np

from camera_geometry import camera_pose_world
from eval_metrics import class_stats, evaluate
from eval_report import draw_confusion, failure_montage, fmt, print_report
from ground_truth import label_frame
from pipeline_config import CALIB, DATASET, DEFAULT_WEIGHTS, NAMES, RAW, RUNS, WEIGHTS_DIR, WORLD
from world_objects import load_world_objects


# ---------------------------------------------------------------- ground truth of the frames to score
def centre_depth(obj, pose, calib):
    """z-depth of the object's centre in the camera frame (metres)."""
    R_wc, o = camera_pose_world(pose["x"], pose["y"], pose["yaw_rad"], calib)
    return float((R_wc.T @ (obj["center"] - o))[2])


def load_gt(fid, raw, objects, calib):
    depth = np.load(f"{raw}/depth/{fid}.npy")
    pose = json.load(open(f"{raw}/meta/{fid}.json"))
    H, W = depth.shape
    by_name = {o["name"]: o for o in objects}
    gts = []
    for r in label_frame(depth, pose, calib, objects):
        if r["status"] != "LABEL":
            continue
        x1, y1, x2, y2 = r["box"]
        gts.append(dict(name=r["name"], cls=NAMES.index(r["cls"]), box=(x1, y1, x2, y2), frac=float(r["frac"]),
                        dist=centre_depth(by_name[r["name"]], pose, calib),
                        truncated=bool(x1 <= 1 or y1 <= 1 or x2 >= W - 1 or y2 >= H - 1)))
    return gts


def split_frames(split):
    ids = sorted(i for i, s in json.load(open(f"{DATASET}/split.json"))["splits"].items() if s == split)
    objects = load_world_objects(WORLD)
    calib = json.load(open(CALIB))
    return [dict(id=i, img=f"{DATASET}/images/{split}/{i}.png", gts=load_gt(i, RAW, objects, calib)) for i in ids]


def raw_frames(raw):
    """Every kept frame of a raw folder (e.g. a stress set), with ground truth from the depth-based labeller."""
    qa = json.load(open(f"{raw}/qa.json"))
    objects = load_world_objects(WORLD)
    calib = json.load(open(CALIB))
    return [dict(id=i, img=f"{raw}/rgb/{i}.png", gts=load_gt(i, raw, objects, calib))
            for i in sorted(i for i, q in qa.items() if q["keep"])]


# ---------------------------------------------------------------- detectors
def _wrap(model, conf, imgsz):
    def predict(path):
        r = model.predict(path, conf=conf, imgsz=imgsz, device="cpu", verbose=False)[0]
        return [dict(cls=int(b.cls), conf=float(b.conf), box=tuple(float(v) for v in b.xyxy[0])) for b in r.boxes]
    return predict


def finetuned_predictor(weights, conf, imgsz=640):
    from ultralytics import YOLO
    return _wrap(YOLO(weights), conf, imgsz)


def yoloe_predictor(conf, imgsz=640):
    os.chdir(WEIGHTS_DIR)                       # the text encoder lives here
    from ultralytics import YOLOE
    model = YOLOE("yoloe-26s-seg.pt")
    model.set_classes(["red box", "blue box", "blue cylinder"])      # same order as our class ids; prompts not tuned
    return _wrap(model, conf, imgsz)


# ---------------------------------------------------------------- confidence sweep
SWEEP_TAUS = [0.001, 0.05, 0.10, 0.25, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]


def sweep(frames, weights, tag):
    """Run the model once at confidence 0.001, then re-score with every threshold in SWEEP_TAUS.
    The 0.001 row is what Ultralytics' own confusion matrix shows; the robot would use 0.25 or higher."""
    base = finetuned_predictor(weights, 0.001)
    cache = {}

    def cached(path):
        if path not in cache:
            cache[path] = base(path)
        return cache[path]

    n = len(NAMES)
    print(f"\nCONFIDENCE SWEEP on '{tag}': what each threshold costs")
    print(f"{'threshold':>10}{'correct':>9}{'wrong cls':>10}{'missed':>8}{'false pos':>10}{'precision':>10}{'recall':>8}")
    rows = []
    for tau in SWEEP_TAUS:
        res = evaluate(frames, lambda p, t=tau: [d for d in cached(p) if d["conf"] >= t])
        cm = res["cm"]
        tp = int(np.trace(cm[:n, :n]))
        wrong = int(cm[:n, :n].sum()) - tp
        missed = int(cm[:n, n].sum())
        fp = int(cm[n, :n].sum())
        prec = tp / (tp + wrong + fp) if tp + wrong + fp else float("nan")
        rec = tp / (tp + wrong + missed) if tp + wrong + missed else float("nan")
        rows.append(dict(tau=tau, tp=tp, wrong=wrong, missed=missed, fp=fp, precision=prec, recall=rec))
        print(f"{tau:>10.3f}{tp:>9}{wrong:>10}{missed:>8}{fp:>10}{fmt(prec):>10}{fmt(rec):>8}")
    clean = [r for r in rows if r["wrong"] == 0 and r["fp"] == 0]
    if clean:
        best = max(clean, key=lambda r: (r["recall"], -r["tau"]))
        print(f"-> no false alarms from threshold {min(r['tau'] for r in clean):.2f} up; "
              f"best recall among those: {fmt(best['recall'])} at {best['tau']:.2f}")
    else:
        print("-> no threshold gives zero false alarms on this set")
    return rows


# ---------------------------------------------------------------- command line
def add_arguments(p):
    p.add_argument("--weights", default=DEFAULT_WEIGHTS)
    p.add_argument("--split", default="test", choices=("val", "test"))
    p.add_argument("--raw", help="score every kept frame of this raw folder instead of a dataset split (stress sets)")
    p.add_argument("--conf", type=float, default=0.25, help="confidence threshold for the fine-tuned model")
    p.add_argument("--yoloe", action="store_true", help="also score zero-shot YOLOE (conf 0.05: generous to it)")
    p.add_argument("--no-official", action="store_true", help="skip Ultralytics' own mAP evaluation")
    p.add_argument("--sweep", action="store_true", help="also print what every confidence threshold costs")


def run(a):
    run_name = os.path.basename(os.path.dirname(os.path.dirname(a.weights)))
    tag = os.path.basename(os.path.normpath(a.raw)) if a.raw else a.split
    out = f"{RUNS}/eval_{run_name}_{tag}"
    os.makedirs(out, exist_ok=True)
    print(f"loading ground truth for '{tag}' ...")
    frames = raw_frames(a.raw) if a.raw else split_frames(a.split)
    print(f"{len(frames)} frames, {sum(len(f['gts']) for f in frames)} labelled objects")
    if a.sweep:
        sweep(frames, a.weights, tag)

    res = evaluate(frames, finetuned_predictor(a.weights, a.conf))
    bd = print_report(f"FINE-TUNED  {run_name}  on '{tag}'  (confidence >= {a.conf})", res)
    draw_confusion(f"{out}/confusion.png", res["cm"])
    shown = failure_montage(f"{out}/failures_montage.png", frames, res)
    print(f"\n{len(res['bad'])} frames with at least one problem" + (f"; montage: {out}/failures_montage.png" if shown else ""))
    report = dict(source=tag, conf=a.conf, confusion=res["cm"].tolist(), classes=class_stats(res["cm"]),
                  mean_iou=res["mean_iou"], breakdown=bd, problem_frames=sorted(res["bad"]),
                  false_positives=[dict(frame=f["frame"], cls=NAMES[f["cls"]], conf=f["conf"], box=f["box"]) for f in res["fps"]])
    if a.yoloe:
        res0 = evaluate(frames, yoloe_predictor(0.05))
        print_report(f"ZERO-SHOT YOLOE (no training, text prompts)  on '{tag}'  (confidence >= 0.05)", res0)
        report["zero_shot"] = dict(confusion=res0["cm"].tolist(), classes=class_stats(res0["cm"]))
    json.dump(report, open(f"{out}/report.json", "w"), indent=1)
    if not a.no_official and not a.raw:
        from ultralytics import YOLO
        YOLO(a.weights).val(data=f"{DATASET}/data.yaml", split=a.split, imgsz=640, device="cpu", plots=True,
                            project=RUNS, name=f"{run_name}_{a.split}_official", exist_ok=True)
    print(f"\nreport: {out}/report.json")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_arguments(ap)
    run(ap.parse_args(argv))


if __name__ == "__main__":
    main()
