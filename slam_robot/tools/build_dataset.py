#!/usr/bin/env python3
"""build_dataset.py - split the labelled frames by floor location into train / val / test.

  python build_dataset.py                 ~/slam_ws/dataset_raw -> ~/slam_ws/dataset

Same as 'python ml_pipeline.py build ...'. The floor is cut into square cells and every cell goes whole into one
split, so the test pictures are taken from places the detector never saw in training.
Writes images/, labels/, data.yaml, split.json and split_map.png.
"""
import argparse
import json
import math
import os
import random
import shutil
from collections import Counter
import cv2
import numpy as np
import yaml

from pipeline_config import CLASSES, DATASET, RAW, WORLD
from world_objects import load_world_objects

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


def score(frames, split_of_cell, targets):
    """Lower is better: frame shares and per-class instance shares should match the targets."""
    st = split_stats(frames, split_of_cell)
    total = len(frames)
    sc = sum((st[s]["frames"] / total - targets[s]) ** 2 for s in SPLITS)
    for c in CLASSES:
        n = sum(st[s]["inst"][c] for s in SPLITS)
        if n == 0:
            continue
        sc += 2.0 * sum((st[s]["inst"][c] / n - targets[s]) ** 2 for s in SPLITS)     # rare classes matter
    return sc


def assign_cells(frames, targets, seed, tries=20000):
    """Random search over assignments of whole cells to splits; returns the best one."""
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


def cap_negatives(frames, split_of_cell, neg_max, seed):
    """In the TRAIN split keep at most neg_max of the images as empty scenes. Returns the set of dropped ids."""
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
    print(f"used {n} frames ({len(dropped_neg)} surplus empty-scene frames left out of train); "
          f"{len(set(split_of_cell.values()))} splits over {n_cells_total} floor cells\n")
    print(f"{'split':<7}{'cells':>6}{'frames':>8}{'share':>7}{'empty':>7}   " +
          "   ".join(f"{c} (inst/frames)" for c in CLASSES))
    for s in SPLITS:
        cells = sum(1 for v in split_of_cell.values() if v == s)
        row = f"{s:<7}{cells:>6}{st[s]['frames']:>8}{st[s]['frames'] / n * 100:>6.0f}%{st[s]['neg']:>7}   "
        row += "   ".join(f"{st[s]['inst'][c]:>6}/{st[s]['frames_with'][c]:<5}" + " " * (len(c) + 1) for c in CLASSES)
        print(row)


# ---------------------------------------------------------------- command line
def add_arguments(p):
    p.add_argument("--raw", default=RAW)
    p.add_argument("--out", default=DATASET)
    p.add_argument("--cell", type=float, default=1.5, help="floor cell size in metres")
    p.add_argument("--val", type=float, default=0.15)
    p.add_argument("--test", type=float, default=0.15)
    p.add_argument("--neg-max", type=float, default=0.10, help="max share of empty-scene images in train")
    p.add_argument("--seed", type=int, default=0)


def run(a):
    objects = load_world_objects(WORLD)
    kept, soc, dropped = build_dataset(a.raw, a.out, a.cell, a.val, a.test, a.neg_max, a.seed, objects)
    print_split_report(kept, soc, dropped, len(soc))
    print(f"\nwritten to {a.out}   (data.yaml, split.json, split_map.png)")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_arguments(ap)
    run(ap.parse_args(argv))


if __name__ == "__main__":
    main()
