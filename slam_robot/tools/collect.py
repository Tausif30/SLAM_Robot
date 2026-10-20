#!/usr/bin/env python3
"""collect.py - teleport the robot to random free poses and save RGB + depth + true pose.   [ROS + simulation]

  python collect.py 1500                                             -> ~/slam_ws/dataset_raw
  python collect.py 300 --only grey --out ~/slam_ws/dataset_stress   look-alike stress set

Same as 'python ml_pipeline.py collect ...'. Writes rgb/*.png, depth/*.npy, meta/*.json and montage_last.png.
"""
import argparse
import glob
import json
import math
import os
import time
import cv2
import numpy as np

from pipeline_config import CLEARANCE, MAP_YAML, RAW, WORLD
from pose_sampling import FreeSpace, choose_pose
from ros_collector import ros_collector_class
from world_objects import load_distractors, load_world_objects


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


def collect_loop(col, space, objects, n, out, seed, force=None, distractors=()):
    """Main loop. `col` needs one method: capture_at(x, y, yaw) -> (data dict or None, reason or None)."""
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
    print(f"\nsaved {saved} frames in {(time.time() - t0) / 60:.1f} min  (attempts {attempts})")
    print(f"  by mode: {mode_counts}")
    print(f"  skipped: {skipped if skipped else 'none'}")
    return saved


# ---------------------------------------------------------------- command line
def add_arguments(p):
    p.add_argument("n", type=int, help="number of frames to collect")
    p.add_argument("--out", default=RAW)
    p.add_argument("--seed", type=int, help="random seed (default: derived from the number of existing frames)")
    p.add_argument("--only", choices=("red", "blue", "grey", "random"),
                   help="use only this mode; grey = look at the unlabelled grey objects (look-alike stress test)")


def run(a):
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


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_arguments(ap)
    run(ap.parse_args(argv))


if __name__ == "__main__":
    main()
