#!/usr/bin/env python3
"""label.py - ground-truth labels + colour quality check for a raw folder.

  python label.py                                   label ~/slam_ws/dataset_raw
  python label.py --raw ~/slam_ws/dataset_stress    label any other raw folder

Same as 'python ml_pipeline.py label ...'. Writes labels/*.txt, qa.json and label_montage.png into the raw folder.
The colour check works because every object in our world has one flat colour.
"""
import argparse
import glob
import json
import os
import random
import cv2
import numpy as np

from ground_truth import draw_overlay, label_frame, yolo_lines
from pipeline_config import CALIB, CLASSES, RAW, WORLD
from world_objects import load_world_objects

CLASS_COLOUR = {"red_box": "red", "blue_box": "blue", "blue_cylinder": "blue"}
PRECISION_MIN = 0.90
UNCOVERED_MIN_PX = 25


def colour_masks(bgr):
    """Pixels that are clearly red / clearly blue. Measured: red cube (138,8,8) and lit side (220,10,10),
    dark blue cubes (8,15,138), shaded blue cylinder about (11,22,199); grey scenery has r ~ g ~ b."""
    b, g, r = (bgr[:, :, i].astype(np.int16) for i in range(3))
    return {"red": (r > 80) & (r - g > 60) & (r - b > 60),
            "blue": (b > 80) & (b - r > 60) & (b - g > 50)}


def check_frame(bgr, results):
    """Independent colour check of the labels. Returns (list of problems, lowest precision)."""
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
          f"(ambiguous object in {n_amb}, colour-check problems in {n_prob})")
    n_prec = sum(1 for f in ids if any("expected colour" in p for p in qa[f]["problems"]))
    n_blob = sum(1 for f in ids if any("unlabelled" in p for p in qa[f]["problems"]))
    print(f"drop reasons (a frame can have several): ambiguous object {n_amb}, low colour precision {n_prec}, "
          f"unlabelled colour blob {n_blob}")
    print(f"kept frames with no object (negatives): {negatives}")
    print("instances in all frames / frames containing the class:")
    for c in CLASSES:
        print(f"  {c:<14} {inst[c]:>5} instances   {frames_with[c]:>4} frames")
    if depth_err:
        print(f"label health: median depth error {np.median(depth_err):.1f} mm, median visible fraction "
              f"{np.median(fracs) * 100:.1f}%, lowest colour precision {min(precs) if precs else float('nan'):.2f}")
    return qa


# ---------------------------------------------------------------- command line
def add_arguments(p):
    p.add_argument("--raw", default=RAW)


def run(a):
    label_folder(a.raw, load_world_objects(WORLD), json.load(open(CALIB)))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_arguments(ap)
    run(ap.parse_args(argv))


if __name__ == "__main__":
    main()
