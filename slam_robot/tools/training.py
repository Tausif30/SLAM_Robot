"""training.py - train the detector, score it, and gather the result pictures.

  1. train      fine-tune a pretrained YOLO26 model on the dataset (CPU)                      [ultralytics]
  2. scoring    match detections to ground truth: confusion matrix, precision / recall, recall by distance
  3. reports    the printed report, the confusion picture, a montage of the frames that went wrong
  4. evaluate   ground truth of the frames to score, the detectors, the confidence sweep       [ultralytics]
  5. figures    copy every result picture into one folder

The commands are run_train / run_evaluate / run_figures (ml_pipeline.py calls them); each has a *_arguments(p)
function that defines its options. Ultralytics is imported only inside the functions that need it.
"""
import glob
import json
import os
import shutil
import sys
import time

import cv2
import numpy as np
import yaml

from camera_geometry import camera_pose_world
from ground_truth import label_frame, load_world_objects
from pipeline_config import (CALIB, DATASET, DEFAULT_WEIGHTS, FIGURES, NAMES, RAW, RUNS, WEIGHTS_DIR, WORLD, WS)


# ======================================================================================================
# 1. train: fine-tune YOLO26 with Ultralytics  [ultralytics]
# ======================================================================================================

def dataset_summary(data_yaml):
    cfg = yaml.safe_load(open(data_yaml))
    counts = {s: len(glob.glob(os.path.join(cfg["path"], cfg[s], "*.png"))) for s in ("train", "val", "test")}
    return cfg, counts


def train_arguments(p):
    p.add_argument("--model", default="yolo26n.pt", help="pretrained weights (yolo26n.pt = smallest)")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--name", default="detector_v1")
    p.add_argument("--patience", type=int, default=20, help="stop early if validation does not improve for this many epochs")
    p.add_argument("--plots", action="store_true", help="save training curves and sample images (a little slower)")


def run_train(a):
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


# ======================================================================================================
# 2. scoring: matching detections to ground truth, confusion matrix, precision / recall
# ======================================================================================================

DIST_BINS = [(0.0, 2.0), (2.0, 4.0), (4.0, 6.0), (6.0, 8.1)]

VIS_BINS = [(0.35, 0.60), (0.60, 0.90), (0.90, 1.01)]


def iou(a, b):
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def match(gts, preds, thr=0.5):
    """One-to-one matching by descending overlap (class ignored). Returns (pairs, unmatched gts, unmatched preds)."""
    cand = sorted(((iou(g["box"], p["box"]), gi, pj) for gi, g in enumerate(gts) for pj, p in enumerate(preds)),
                  reverse=True)
    used_g, used_p, pairs = set(), set(), []
    for v, gi, pj in cand:
        if v < thr:
            break
        if gi in used_g or pj in used_p:
            continue
        used_g.add(gi)
        used_p.add(pj)
        pairs.append((gi, pj))
    return (pairs, [i for i in range(len(gts)) if i not in used_g], [j for j in range(len(preds)) if j not in used_p])


def evaluate(frames, predict, iou_thr=0.5):
    """frames: list of dict(id, img, gts). predict(img_path) -> list of dict(cls, conf, box) above the chosen conf."""
    n = len(NAMES)
    cm = np.zeros((n + 1, n + 1), int)      # rows: true class (last = background); cols: predicted (last = missed)
    records, fps, bad, ious = [], [], {}, []
    for fr in frames:
        preds = predict(fr["img"])
        pairs, ug, up = match(fr["gts"], preds, iou_thr)
        issues = []
        for gi, pj in pairs:
            g, p = fr["gts"][gi], preds[pj]
            ok = g["cls"] == p["cls"]
            v = iou(g["box"], p["box"])
            cm[g["cls"], p["cls"]] += 1
            if ok:
                ious.append(v)
            else:
                issues.append(f"{g['name']} called {NAMES[p['cls']]}")
            records.append(dict(g, ok=ok, iou=v, frame=fr["id"],
                                outcome="ok" if ok else f"wrong class ({NAMES[p['cls']]})"))
        for gi in ug:
            g = fr["gts"][gi]
            cm[g["cls"], n] += 1
            records.append(dict(g, ok=False, iou=0.0, frame=fr["id"], outcome="missed"))
            issues.append(f"missed {g['name']}")
        for pj in up:
            p = preds[pj]
            cm[n, p["cls"]] += 1
            fps.append(dict(p, frame=fr["id"]))
            issues.append(f"false {NAMES[p['cls']]} (conf {p['conf']:.2f})")
        if issues:
            bad[fr["id"]] = (issues, preds)
    return dict(cm=cm, records=records, fps=fps, bad=bad, n_frames=len(frames),
                mean_iou=float(np.mean(ious)) if ious else float("nan"))


def class_stats(cm):
    n = len(NAMES)
    rows = []
    for c in range(n):
        tp = int(cm[c, c])
        wrong = int(cm[c, :n].sum() - tp)
        missed = int(cm[c, n])
        fp = int(cm[:, c].sum() - tp)           # everything predicted as c that was not a true c
        rows.append(dict(cls=NAMES[c], gt=tp + wrong + missed, tp=tp, wrong=wrong, missed=missed, fp=fp,
                         precision=tp / (tp + fp) if tp + fp else float("nan"),
                         recall=tp / (tp + wrong + missed) if tp + wrong + missed else float("nan")))
    return rows


def recall_by(records, key, bins):
    out = []
    for lo, hi in bins:
        sel = [r for r in records if lo <= key(r) < hi]
        out.append(dict(lo=lo, hi=hi, n=len(sel), recall=(sum(r["ok"] for r in sel) / len(sel)) if sel else None))
    return out


def breakdowns(res):
    rec = res["records"]
    return dict(
        distance=recall_by(rec, lambda r: r["dist"], DIST_BINS),
        visibility=recall_by(rec, lambda r: r["frac"], VIS_BINS),
        truncated=dict(cut_off=_rate([r for r in rec if r["truncated"]]), whole=_rate([r for r in rec if not r["truncated"]])),
        per_class_distance={c: recall_by([r for r in rec if r["cls"] == i], lambda r: r["dist"], DIST_BINS)
                            for i, c in enumerate(NAMES)})


def _rate(sel):
    return dict(n=len(sel), recall=(sum(r["ok"] for r in sel) / len(sel)) if sel else None)


# ======================================================================================================
# 3. reports: printed report, confusion picture, failure montage
# ======================================================================================================

def fmt(v):
    return "  n/a" if v is None or v != v else f"{v * 100:5.1f}%"


def print_report(title, res):
    cm, n = res["cm"], len(NAMES)
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")
    print(f"{'class':<15}{'objects':>8}{'correct':>9}{'wrong cls':>10}{'missed':>8}{'false pos':>10}{'precision':>10}{'recall':>8}")
    for s in class_stats(cm):
        print(f"{s['cls']:<15}{s['gt']:>8}{s['tp']:>9}{s['wrong']:>10}{s['missed']:>8}{s['fp']:>10}{fmt(s['precision']):>10}{fmt(s['recall']):>8}")
    bg_fp = int(cm[n, :].sum())
    print(f"false detections on empty background (no object there): {bg_fp}")
    if res["fps"]:
        c = sorted(f["conf"] for f in res["fps"])
        print(f"   strength of ALL false detections ({len(c)}): highest {c[-1]:.2f};  >=0.50: {sum(v >= 0.5 for v in c)},"
              f"  0.25-0.50: {sum(0.25 <= v < 0.5 for v in c)},  0.10-0.25: {sum(0.1 <= v < 0.25 for v in c)},  <0.10: {sum(v < 0.1 for v in c)}")
    print(f"mean box overlap (IoU) of correct detections: {res['mean_iou']:.3f}")
    print("\nconfusion (rows = what the object really is, columns = what the detector said):")
    head = [c[:10] for c in NAMES] + ["missed"]
    print(f"{'':<15}" + "".join(f"{h:>12}" for h in head))
    for i, name in enumerate(NAMES + ["(background)"]):
        print(f"{name:<15}" + "".join(f"{int(cm[i, j]):>12}" for j in range(n + 1)))
    b = breakdowns(res)
    print("\nrecall by distance to the object:")
    for r in b["distance"]:
        print(f"   {r['lo']:.0f}-{min(r['hi'], 8):.0f} m   {fmt(r['recall'])}   ({r['n']} objects)")
    print("recall by how much of the object is visible:")
    for r in b["visibility"]:
        print(f"   {r['lo'] * 100:.0f}-{min(r['hi'], 1) * 100:.0f}%   {fmt(r['recall'])}   ({r['n']} objects)")
    t = b["truncated"]
    print(f"recall when the image edge cuts the object off: {fmt(t['cut_off']['recall'])} ({t['cut_off']['n']})   "
          f"whole object in view: {fmt(t['whole']['recall'])} ({t['whole']['n']})")
    bad = [r for r in res["records"] if not r["ok"]]
    if bad or res["fps"]:
        print("\nevery error in detail:")
        for r in bad:
            x1, y1, x2, y2 = r["box"]
            print(f"   {r['outcome']:<26} {r['name']:<16} frame {r['frame']}  true box {x2 - x1}x{y2 - y1} px, "
                  f"{r['dist']:.1f} m, {r['frac'] * 100:.0f}% visible, {'CUT OFF by the image edge' if r['truncated'] else 'whole'}")
        for f in res["fps"]:
            x1, y1, x2, y2 = f["box"]
            print(f"   false {NAMES[f['cls']]:<20} frame {f['frame']}  box {x2 - x1:.0f}x{y2 - y1:.0f} px at x={x1:.0f}..{x2:.0f}, conf {f['conf']:.2f}")
    return b


def draw_confusion(path, cm):
    n = len(NAMES)
    cell, left, top = 110, 190, 60
    img = np.full((top + (n + 1) * cell + 30, left + (n + 1) * cell + 20, 3), 255, np.uint8)
    rowsum = cm.sum(axis=1, keepdims=True).clip(min=1)
    norm = cm / rowsum
    cols = NAMES + ["missed"]
    for j, c in enumerate(cols):
        cv2.putText(img, c[:14], (left + j * cell + 4, top - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(img, "detector said ->", (left, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    for i, r in enumerate(NAMES + ["(background)"]):
        cv2.putText(img, r, (8, top + i * cell + cell // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        for j in range(n + 1):
            v = norm[i, j]
            ink = (255, 255, 255) if v > 0.55 else (0, 0, 0)
            cv2.rectangle(img, (left + j * cell, top + i * cell), (left + (j + 1) * cell, top + (i + 1) * cell),
                          (255, int(255 - 170 * v), int(255 - 215 * v)), -1)     # white -> blue (BGR)
            cv2.rectangle(img, (left + j * cell, top + i * cell), (left + (j + 1) * cell, top + (i + 1) * cell), (120, 120, 120), 1)
            cv2.putText(img, f"{int(cm[i, j])}", (left + j * cell + 8, top + i * cell + cell // 2 + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, ink, 1, cv2.LINE_AA)
            if i < n and cm[i].sum():
                cv2.putText(img, f"{v * 100:.0f}%", (left + j * cell + 8, top + i * cell + cell // 2 + 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, ink, 1, cv2.LINE_AA)
    cv2.putText(img, "row = truth, % of the row", (8, img.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 80), 1, cv2.LINE_AA)
    cv2.imwrite(path, img)


def failure_montage(path, frames, res, limit=12):
    by_id = {f["id"]: f for f in frames}
    tiles = []
    for fid, (issues, preds) in list(res["bad"].items())[:limit]:
        img = cv2.imread(by_id[fid]["img"])
        for g in by_id[fid]["gts"]:
            cv2.rectangle(img, g["box"][:2], g["box"][2:], (0, 220, 0), 2)
        for p in preds:
            x1, y1, x2, y2 = (int(v) for v in p["box"])
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(img, f"{NAMES[p['cls']]} {p['conf']:.2f}", (x1, max(14, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)
        tile = cv2.resize(img, (320, 240))
        cv2.rectangle(tile, (0, 0), (320, 18), (0, 0, 0), -1)
        cv2.putText(tile, f"{fid}: {'; '.join(issues)}"[:52], (3, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(tile)
    if not tiles:
        return False
    while len(tiles) % 3:
        tiles.append(np.zeros((240, 320, 3), np.uint8))
    cv2.imwrite(path, np.vstack([np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)]))
    return True


# ======================================================================================================
# 4. evaluate: ground truth of the frames to score, the detectors, the confidence sweep  [ultralytics]
# ======================================================================================================

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


def evaluate_arguments(p):
    p.add_argument("--weights", default=DEFAULT_WEIGHTS)
    p.add_argument("--split", default="test", choices=("val", "test"))
    p.add_argument("--raw", help="score every kept frame of this raw folder instead of a dataset split (stress sets)")
    p.add_argument("--conf", type=float, default=0.25, help="confidence threshold for the fine-tuned model")
    p.add_argument("--yoloe", action="store_true", help="also score zero-shot YOLOE (conf 0.05: generous to it)")
    p.add_argument("--no-official", action="store_true", help="skip Ultralytics' own mAP evaluation")
    p.add_argument("--sweep", action="store_true", help="also print what every confidence threshold costs")


def run_evaluate(a):
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
    json.dump(report, open(f"{out}/report.json", "w"), indent=1)            # written first: a failing baseline must not lose it
    if a.yoloe:
        try:
            res0 = evaluate(frames, yoloe_predictor(0.05))
        except Exception as e:                                               # no network, CLIP missing, ...
            print(f"\nzero-shot baseline SKIPPED ({type(e).__name__}: {str(e).splitlines()[0][:150] if str(e) else ''})")
        else:
            print_report(f"ZERO-SHOT YOLOE (no training, text prompts)  on '{tag}'  (confidence >= 0.05)", res0)
            report["zero_shot"] = dict(confusion=res0["cm"].tolist(), classes=class_stats(res0["cm"]))
            json.dump(report, open(f"{out}/report.json", "w"), indent=1)
    if not a.no_official and not a.raw:
        from ultralytics import YOLO
        YOLO(a.weights).val(data=f"{DATASET}/data.yaml", split=a.split, imgsz=640, device="cpu", plots=True,
                            project=RUNS, name=f"{run_name}_{a.split}_official", exist_ok=True)
    print(f"\nreport: {out}/report.json")


# ======================================================================================================
# 5. figures: every result picture in ONE folder
# ======================================================================================================


def collect_figures(out=None):
    """Copy every result picture (PNG / JPG) into one folder, named <source folder>__<file>.
    Looks only at the top level of the dataset folders and of every training / evaluation run, so the thousands
    of camera frames inside rgb/ and images/ are not copied. Returns the list of copied files."""
    out = out or FIGURES
    os.makedirs(out, exist_ok=True)
    sources = sorted(glob.glob(os.path.join(WS, "dataset*"))) + sorted(glob.glob(os.path.join(RUNS, "*")))
    copied = []
    for src in sources:
        if not os.path.isdir(src):
            continue
        for path in sorted(glob.glob(os.path.join(src, "*.png")) + glob.glob(os.path.join(src, "*.jpg"))):
            dst = os.path.join(out, f"{os.path.basename(src)}__{os.path.basename(path)}")
            shutil.copy2(path, dst)
            copied.append(dst)
    return copied


def figures_arguments(p):
    p.add_argument("--out", default=FIGURES, help="folder for the copies")


def run_figures(a):
    copied = collect_figures(a.out)
    for path in copied:
        print("  " + os.path.basename(path))
    print(f"\n{len(copied)} pictures copied to {a.out}")