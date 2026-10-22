# Running simulation to a train / val / test split.
import glob
import json
import math
import os
import random
import shutil
import time
from collections import Counter
import cv2
import numpy as np
import yaml
from camera_geometry import angle_diff
from ground_truth import draw_overlay, label_frame, load_distractors, load_world_objects, yolo_lines
from pipeline_config import (CALIB, CLASSES, CLEARANCE, DATASET, LOOK_DIST, LOOK_JITTER_DEG, MAP_YAML, MODE_WEIGHTS,
                             MODES, RAW, WORLD)
from ros_collector import ros_collector_class

# Pose sampling: where may the robot stand?
# Free cells of the saved map, with clearance, line of sight and pose sampling.
class FreeSpace:

    def __init__(self, yaml_path, clearance=CLEARANCE):
        meta = yaml.safe_load(open(yaml_path))
        img_path = os.path.join(os.path.dirname(yaml_path), meta["image"])
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(img_path)
        if int(meta.get("negate", 0)):
            img = 255 - img
        self.res = float(meta["resolution"])
        self.ox, self.oy = float(meta["origin"][0]), float(meta["origin"][1])
        self.H, self.W = img.shape
        # Trinary map: free = 254, occupied = 0, unknown = 205. Only clearly free pixels count as free:
        # unknown pixels can come out as 'free' under the usual thresholds, so we do not trust them.
        self.free = img >= 250
        # Everything outside the image counts as an obstacle (a map cropped right at a wall must not
        # look like open space beyond its edge).
        padded = np.pad(self.free, 1, constant_values=False).astype(np.uint8)
        self.clear = cv2.distanceTransform(padded, cv2.DIST_L2, 5)[1:-1, 1:-1] * self.res
        rows, cols = np.nonzero(self.clear >= clearance)
        self.cand_x = self.ox + (cols + 0.5) * self.res
        self.cand_y = self.oy + (self.H - rows - 0.5) * self.res

    def _cell(self, x, y):
        col = int(math.floor((x - self.ox) / self.res))
        row = self.H - 1 - int(math.floor((y - self.oy) / self.res))
        return row, col

    def is_free(self, x, y):
        row, col = self._cell(x, y)
        return 0 <= row < self.H and 0 <= col < self.W and bool(self.free[row, col])

    def line_free(self, x0, y0, x1, y1):
        n = max(2, int(math.hypot(x1 - x0, y1 - y0) / (self.res * 0.5)))
        return all(self.is_free(x0 + (x1 - x0) * k / n, y0 + (y1 - y0) * k / n) for k in range(n + 1))

    def _jitter(self, rng, i):
        return (self.cand_x[i] + rng.uniform(-0.5, 0.5) * self.res,
                self.cand_y[i] + rng.uniform(-0.5, 0.5) * self.res)

    def random_pose(self, rng):
        x, y = self._jitter(rng, int(rng.integers(len(self.cand_x))))
        return x, y, float(rng.uniform(-math.pi, math.pi))

    # A free pose between dmin and dmax from the target with a clear line to it, aimed at it. None if no such pose is found.
    def lookat_pose(self, rng, target_xy, dmin, dmax, jitter, stop_before, tries=300):
        tx, ty = target_xy
        d = np.hypot(self.cand_x - tx, self.cand_y - ty)
        idx = np.nonzero((d >= dmin) & (d <= dmax))[0]
        for i in rng.permutation(idx)[:tries]:
            x, y = self._jitter(rng, int(i)) # a random aim error up to +-jitter
            dist = math.hypot(tx - x, ty - y)
            if dist < dmin * 0.9:
                continue
            f = max(0.0, (dist - stop_before) / dist)
            if self.line_free(x, y, x + (tx - x) * f, y + (ty - y) * f):
                yaw = math.atan2(ty - y, tx - x) + rng.uniform(-jitter, jitter)
                return x, y, angle_diff(yaw, 0.0)
        return None

def radius_of(obj):
    if obj["kind"] == "box":
        return float(math.hypot(obj["half"][0], obj["half"][1]))
    return float(obj["radius"])

    # Pick a sampling mode and a pose. Returns (mode, target name or None, (x, y, yaw)).
    # Force: always use this mode ("red", "blue", "grey" or "random"); "grey" aims at the unlabelled look-alikes.
def choose_pose(space, objects, rng, force=None, distractors=()):
    mode = force or str(rng.choice(MODES, p=MODE_WEIGHTS))
    if mode in ("red", "blue", "grey"):
        if mode == "grey":
            pool = list(distractors)
        else:
            pool = [o for o in objects if (o["cls"] == "red_box") == (mode == "red")]
        if pool:
            target = pool[int(rng.integers(len(pool)))]
            pose = space.lookat_pose(rng, target["center"][:2], LOOK_DIST[0], LOOK_DIST[1],
                                     math.radians(LOOK_JITTER_DEG), radius_of(target) + 0.10)
            if pose is not None:
                return mode, target["name"], pose
    return "random", None, space.random_pose(rng)

# Collect: random poses -> RGB + depth + pose  [ROS + simulation]
def next_index(out):
    ids = [int(os.path.splitext(os.path.basename(p))[0]) for p in glob.glob(f"{out}/meta/*.json")]
    return max(ids) + 1 if ids else 0

def make_montage(out, names, cols=3, w=320, h=240):
    tiles = []
    for name in names:
        img = cv2.resize(cv2.imread(f"{out}/rgb/{name}.png"), (w, h))
        m = json.load(open(f"{out}/meta/{name}.json"))
        cap = f"{name} {m['mode']} ({m['x']:.1f},{m['y']:.1f}) {math.degrees(m['yaw_rad']):.0f}deg"
        cv2.rectangle(img, (0, 0), (w, 18), (0, 0, 0), -1)
        cv2.putText(img, cap, (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(img)
    while len(tiles) % cols:
        tiles.append(np.zeros((h, w, 3), np.uint8))
    rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
    cv2.imwrite(f"{out}/montage_last.png", np.vstack(rows))

# Capture_at(x, y, yaw) -> (data dict or None, reason or None)
def collect_loop(col, space, objects, n, out, seed, force=None, distractors=()):
    for sub in ("rgb", "depth", "meta"):
        os.makedirs(f"{out}/{sub}", exist_ok=True)
    start = next_index(out)
    rng = np.random.default_rng(seed)
    saved, attempts, skipped, mode_counts, names = 0, 0, {}, {}, []
    t0 = time.time()
    while saved < n and attempts < 3 * n:
        attempts += 1
        mode, target, (x, y, yaw) = choose_pose(space, objects, rng, force, distractors)
        data, reason = col.capture_at(x, y, yaw)
        if data is None:
            skipped[reason.split(" (")[0]] = skipped.get(reason.split(" (")[0], 0) + 1
            print(f"  skipped: {reason}")
            continue
        name = f"{start + saved:06d}"
        ox, oy, oyaw = data["odom"]
        cv2.imwrite(f"{out}/rgb/{name}.png", data["rgb"][:, :, ::-1])
        np.save(f"{out}/depth/{name}.npy", data["depth"])
        meta = dict(id=name, x=ox, y=oy, yaw_rad=oyaw, commanded=[x, y, yaw], mode=mode,
                    target=target, sim_stamp=data["stamp"], seed=seed)
        json.dump(meta, open(f"{out}/meta/{name}.json", "w"), indent=1)
        saved += 1
        names.append(name)
        mode_counts[mode] = mode_counts.get(mode, 0) + 1
        el = time.time() - t0
        print(f"[{saved:>4}/{n}] #{name} {mode:<6} target={target or '-':<16} "
              f"pose=({ox:6.2f},{oy:6.2f},{math.degrees(oyaw):7.1f} deg)  ETA {el / saved * (n - saved) / 60:4.1f} min")
    if names:
        make_montage(out, names[-12:])
    print(f"\nSaved {saved} frames in {(time.time() - t0) / 60:.1f} min  (attempts {attempts})")
    print(f"  by mode: {mode_counts}")
    print(f"  Skipped: {skipped if skipped else 'none'}")
    return saved

def collect_arguments(p):
    p.add_argument("n", type=int, help="number of frames to collect")
    p.add_argument("--out", default=RAW)
    p.add_argument("--seed", type=int, help="random seed (default: derived from the number of existing frames)")
    p.add_argument("--only", choices=("red", "blue", "grey", "random"),
                   help="use only this mode; grey = look at the unlabelled grey objects (look-alike stress test)")

def run_collect(a):
    import rclpy
    objects = load_world_objects(WORLD)
    space = FreeSpace(MAP_YAML)
    distractors = load_distractors(WORLD)
    print(f"{len(space.cand_x)} candidate cells (>= {CLEARANCE} m clear); {len(objects)} labelled objects; "
          f"unlabelled look-alikes: {[d['name'] for d in distractors]}")
    seed = a.seed if a.seed is not None else 1000 + next_index(a.out)
    rclpy.init()
    col = ros_collector_class()()
    col.start()
    collect_loop(col, space, objects, a.n, a.out, seed, a.only, distractors)
    col.destroy_node()
    rclpy.shutdown()

# Label: ground-truth labels + colour quality check (every object in our world has one flat colour)
CLASS_COLOUR = {"red_box": "red", "blue_box": "blue", "blue_cylinder": "blue"}
PRECISION_MIN = 0.90
UNCOVERED_MIN_PX = 25

# Measured: red cube (138,8,8) and lit side (220,10,10),
# dark blue cubes (8,15,138), shaded blue cylinder about (11,22,199); grey scenery has r ~ g ~ b.
def colour_masks(bgr):
    b, g, r = (bgr[:, :, i].astype(np.int16) for i in range(3))
    return {"red": (r > 80) & (r - g > 60) & (r - b > 60),
            "blue": (b > 80) & (b - r > 60) & (b - g > 50)}

# Independent colour check of the labels. Returns (list of problems, lowest precision).
def check_frame(bgr, results):
    cm = colour_masks(bgr)
    problems, min_prec = [], 1.0
    covered = np.zeros(bgr.shape[:2], bool)
    for r in results:
        if r["status"] == "HIDDEN":
            continue
        covered |= r["mask"]
        if r["status"] == "LABEL":
            prec = (cm[CLASS_COLOUR[r["cls"]]] & r["mask"]).sum() / max(1, r["mask"].sum())
            min_prec = min(min_prec, prec)
            if prec < PRECISION_MIN:
                problems.append(f"{r['name']}: only {prec:.0%} of its pixels have the expected colour")
    near = cv2.dilate(covered.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=2).astype(bool)
    for cname, m in cm.items():
        leftover = cv2.morphologyEx((m & ~near).astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        n, _, stats, _ = cv2.connectedComponentsWithStats(leftover, connectivity=8)   # opening drops 1-2 px slivers
        for i in range(1, n):
            x, y, w, h, area = (int(v) for v in stats[i])
            if area >= UNCOVERED_MIN_PX:
                problems.append(f"unlabelled {cname} blob: {area} px at x={x} y={y} ({w}x{h})")
    return problems, float(min_prec)

def caption(img, text, bad):
    cv2.rectangle(img, (0, 0), (img.shape[1], 18), (0, 0, 0), -1)
    cv2.putText(img, text, (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (80, 80, 255) if bad else (255, 255, 255), 1, cv2.LINE_AA)
    return img

def label_folder(raw, objects, calib, montage_n=12, verbose=True):
    ids = sorted(os.path.splitext(os.path.basename(p))[0] for p in glob.glob(f"{raw}/meta/*.json"))
    if not ids:
        raise SystemExit(f"no frames found in {raw}/meta")
    os.makedirs(f"{raw}/labels", exist_ok=True)
    sample = set(random.Random(0).sample(ids, min(montage_n, len(ids))))
    qa, bad_tiles, ok_tiles = {}, [], []
    inst, frames_with = {c: 0 for c in CLASSES}, {c: 0 for c in CLASSES}
    depth_err, precs, fracs = [], [], []
    for k, fid in enumerate(ids):
        bgr = cv2.imread(f"{raw}/rgb/{fid}.png")
        depth = np.load(f"{raw}/depth/{fid}.npy")
        pose = json.load(open(f"{raw}/meta/{fid}.json"))
        H, W = depth.shape
        results = label_frame(depth, pose, calib, objects)
        problems, min_prec = check_frame(bgr, results)
        labelled = [r for r in results if r["status"] == "LABEL"]
        ambiguous = [r["name"] for r in results if r["status"] == "AMBIGUOUS"]
        with open(f"{raw}/labels/{fid}.txt", "w") as f:
            f.write("\n".join(yolo_lines(results, W, H)) + ("\n" if labelled else ""))
        keep = not ambiguous and not problems
        qa[fid] = dict(labels=len(labelled), classes=sorted(r["cls"] for r in labelled), ambiguous=ambiguous,
                       problems=problems, min_precision=round(min_prec, 3), keep=keep, mode=pose.get("mode"))
        for r in labelled:
            inst[r["cls"]] += 1
            fracs.append(r["frac"])
            if r["depth_err_mm"] is not None:
                depth_err.append(r["depth_err_mm"])
        if labelled:
            precs.append(min_prec)
        for c in {r["cls"] for r in labelled}:
            frames_with[c] += 1
        if not keep or fid in sample:
            tile = cv2.resize(draw_overlay(bgr, results), (320, 240))
            tile = caption(tile, f"{fid} L={len(labelled)} {'KEEP' if keep else 'DROP'}", not keep)
            (ok_tiles if keep else bad_tiles).append(tile)
        if verbose and (not keep or len(ids) <= 30):
            why = "; ".join(([f"ambiguous: {', '.join(ambiguous)}"] if ambiguous else []) + problems)
            print(f"  {fid} {'KEEP' if keep else 'DROP'}  labels={len(labelled)} {sorted(r['cls'] for r in labelled)}"
                  f"{'  <- ' + why if why else ''}")
        elif verbose and (k + 1) % 100 == 0:
            print(f"  ... {k + 1}/{len(ids)} frames")
    json.dump(qa, open(f"{raw}/qa.json", "w"), indent=1)

    tiles = (bad_tiles + ok_tiles)[:montage_n]
    while len(tiles) % 3:
        tiles.append(np.zeros((240, 320, 3), np.uint8))
    cv2.imwrite(f"{raw}/label_montage.png", np.vstack([np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)]))

    kept = [f for f in ids if qa[f]["keep"]]
    n_amb = sum(1 for f in ids if qa[f]["ambiguous"])
    n_prob = sum(1 for f in ids if qa[f]["problems"])
    negatives = sum(1 for f in kept if qa[f]["labels"] == 0)
    print(f"\nframes: {len(ids)}   kept: {len(kept)}   dropped: {len(ids) - len(kept)} "
          f"(Ambiguous object in {n_amb}, Colour-check problems in {n_prob})")
    n_prec = sum(1 for f in ids if any("expected colour" in p for p in qa[f]["problems"]))
    n_blob = sum(1 for f in ids if any("unlabelled" in p for p in qa[f]["problems"]))
    print(f"Drop reasons (a frame can have several): Ambiguous object {n_amb}, Low colour precision {n_prec}, "
          f"Unlabelled colour blob {n_blob}")
    print(f"Kept frames with no object (negatives): {negatives}")
    print("Instances in all frames / frames containing the class:")
    for c in CLASSES:
        print(f"  {c:<14} {inst[c]:>5} instances   {frames_with[c]:>4} frames")
    if depth_err:
        print(f"Label health: median depth error {np.median(depth_err):.1f} mm, median visible fraction "
              f"{np.median(fracs) * 100:.1f}%, lowest colour precision {min(precs) if precs else float('nan'):.2f}")
    return qa

def label_arguments(p):
    p.add_argument("--raw", default=RAW)

def run_label(a):
    label_folder(a.raw, load_world_objects(WORLD), json.load(open(CALIB)))

# Build: split by floor location (frames from nearby poses are near-duplicates)
SPLITS = ("train", "val", "test")
SPLIT_COLOUR = {"train": (180, 220, 180), "val": (110, 215, 245), "test": (225, 160, 205)}   # BGR

def load_frames(raw, cell):
    qa = json.load(open(f"{raw}/qa.json"))
    frames = []
    for fid, q in sorted(qa.items()):
        if not q["keep"]:
            continue
        m = json.load(open(f"{raw}/meta/{fid}.json"))
        frames.append(dict(id=fid, x=m["x"], y=m["y"], cell=(math.floor(m["x"] / cell), math.floor(m["y"] / cell)),
                           classes=dict(Counter(q["classes"]))))
    return frames

def split_stats(frames, split_of_cell):
    st = {s: dict(frames=0, neg=0, inst=Counter(), frames_with=Counter()) for s in SPLITS}
    for f in frames:
        s = st[split_of_cell[f["cell"]]]
        s["frames"] += 1
        s["neg"] += not f["classes"]
        for c, n in f["classes"].items():
            s["inst"][c] += n
            s["frames_with"][c] += 1
    return st

# Scoring 
# Lower is better: frame shares and per-class instance shares should match the targets.
def score(frames, split_of_cell, targets):
    st = split_stats(frames, split_of_cell)
    total = len(frames)
    sc = sum((st[s]["frames"] / total - targets[s]) ** 2 for s in SPLITS)
    for c in CLASSES:
        n = sum(st[s]["inst"][c] for s in SPLITS)
        if n == 0:
            continue
        sc += 2.0 * sum((st[s]["inst"][c] / n - targets[s]) ** 2 for s in SPLITS)     # rare classes matter
    return sc

# Random search over assignments of whole cells to splits; returns the best one.
def assign_cells(frames, targets, seed, tries=20000):
    cells = sorted({f["cell"] for f in frames})
    rng = random.Random(seed)
    probs = [targets[s] for s in SPLITS]
    best, best_sc = None, float("inf")
    for _ in range(tries):
        cand = dict(zip(cells, rng.choices(SPLITS, weights=probs, k=len(cells))))
        if len(set(cand.values())) < 3:
            continue
        sc = score(frames, cand, targets)
        if sc < best_sc:
            best, best_sc = cand, sc
    return best, best_sc


# In the TRAIN split keep at most neg_max of the images as empty scenes. Returns the set of dropped ids.
def cap_negatives(frames, split_of_cell, neg_max, seed):
    train = [f for f in frames if split_of_cell[f["cell"]] == "train"]
    pos = [f for f in train if f["classes"]]
    neg = [f for f in train if not f["classes"]]
    allowed = int(neg_max / (1.0 - neg_max) * len(pos))
    random.Random(seed).shuffle(neg)
    return {f["id"] for f in neg[allowed:]}

def draw_map(path, frames, split_of_cell, cell, objects):
    S, M = 80, 40                                          # pixels per metre, margin
    x0, x1, y0, y1 = -7.5, 7.5, -4.6, 4.6        # whole 1.5 m grid fits, wherever it lands
    W, H = int((x1 - x0) * S) + 2 * M, int((y1 - y0) * S) + 2 * M
    img = np.full((H, W, 3), 255, np.uint8)
    px = lambda x, y: (int(M + (x - x0) * S), int(M + (y1 - y) * S))
    for (cx, cy), s in split_of_cell.items():
        a, b = px(cx * cell, (cy + 1) * cell), px((cx + 1) * cell, cy * cell)
        cv2.rectangle(img, a, b, SPLIT_COLOUR[s], -1)
        cv2.rectangle(img, a, b, (120, 120, 120), 1)
    for f in frames:
        cv2.circle(img, px(f["x"], f["y"]), 1, (60, 60, 60), -1)
    for o in objects:                                      # labelled objects, drawn in their real colours
        colour = (0, 0, 200) if o["cls"] == "red_box" else (200, 60, 0)
        cv2.circle(img, px(o["center"][0], o["center"][1]), 7, colour, -1)
    cv2.line(img, px(0, -3.5), px(0, -1.5), (0, 0, 0), 3)   # partition wall with its 3 m doorway
    cv2.line(img, px(0, 1.5), px(0, 3.5), (0, 0, 0), 3)
    cv2.rectangle(img, px(-5.8, 3.4), px(5.8, -3.4), (0, 0, 0), 2)
    for i, s in enumerate(SPLITS):
        cv2.rectangle(img, (M + i * 150, 6), (M + i * 150 + 16, 22), SPLIT_COLOUR[s], -1)
        cv2.putText(img, s, (M + i * 150 + 22, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(img, "dots = robot positions, red/blue discs = labelled objects", (M + 460, 19),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.imwrite(path, img)

def build_dataset(raw, out, cell=1.5, p_val=0.15, p_test=0.15, neg_max=0.10, seed=0, objects=()):
    frames = load_frames(raw, cell)
    targets = {"train": 1.0 - p_val - p_test, "val": p_val, "test": p_test}
    split_of_cell, sc = assign_cells(frames, targets, seed)
    dropped_neg = cap_negatives(frames, split_of_cell, neg_max, seed)
    for sub in ("images", "labels"):
        shutil.rmtree(f"{out}/{sub}", ignore_errors=True)
        for s in SPLITS:
            os.makedirs(f"{out}/{sub}/{s}")
    split_of_frame = {}
    for f in frames:
        if f["id"] in dropped_neg:
            continue
        s = split_of_cell[f["cell"]]
        split_of_frame[f["id"]] = s
        shutil.copy2(f"{raw}/rgb/{f['id']}.png", f"{out}/images/{s}/{f['id']}.png")
        shutil.copy2(f"{raw}/labels/{f['id']}.txt", f"{out}/labels/{s}/{f['id']}.txt")
    yaml.safe_dump({"path": os.path.abspath(out), "train": "images/train", "val": "images/val", "test": "images/test",
                    "names": {i: c for i, c in enumerate(CLASSES)}}, open(f"{out}/data.yaml", "w"), sort_keys=False)
    json.dump({"cell_m": cell, "seed": seed, "splits": split_of_frame}, open(f"{out}/split.json", "w"), indent=1)
    kept = [f for f in frames if f["id"] not in dropped_neg]
    draw_map(f"{out}/split_map.png", kept, split_of_cell, cell, objects)
    return kept, split_of_cell, dropped_neg

def print_split_report(kept, split_of_cell, dropped_neg, n_cells_total):
    st = split_stats(kept, split_of_cell)
    n = len(kept)
    print(f"Used {n} frames ({len(dropped_neg)} surplus empty-scene frames left out of train); "
          f"{len(set(split_of_cell.values()))} splits over {n_cells_total} floor cells\n")
    print(f"{'split':<7}{'cells':>6}{'frames':>8}{'share':>7}{'empty':>7}   " +
          "   ".join(f"{c} (inst/frames)" for c in CLASSES))
    for s in SPLITS:
        cells = sum(1 for v in split_of_cell.values() if v == s)
        row = f"{s:<7}{cells:>6}{st[s]['frames']:>8}{st[s]['frames'] / n * 100:>6.0f}%{st[s]['neg']:>7}   "
        row += "   ".join(f"{st[s]['inst'][c]:>6}/{st[s]['frames_with'][c]:<5}" + " " * (len(c) + 1) for c in CLASSES)
        print(row)

def build_arguments(p):
    p.add_argument("--raw", default=RAW)
    p.add_argument("--out", default=DATASET)
    p.add_argument("--cell", type=float, default=1.5, help="floor cell size in metres")
    p.add_argument("--val", type=float, default=0.15)
    p.add_argument("--test", type=float, default=0.15)
    p.add_argument("--neg-max", type=float, default=0.10, help="max share of empty-scene images in train")
    p.add_argument("--seed", type=int, default=0)

def run_build(a):
    objects = load_world_objects(WORLD)
    kept, soc, dropped = build_dataset(a.raw, a.out, a.cell, a.val, a.test, a.neg_max, a.seed, objects)
    print_split_report(kept, soc, dropped, len(soc))
    print(f"\nWritten to {a.out}   (data.yaml, split.json, split_map.png)")