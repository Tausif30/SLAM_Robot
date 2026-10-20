"""ground_truth.py - where is every object in a picture, and is it really visible?

World file -> boxes / cylinders; camera model -> one ray per pixel; depth image -> occlusion test.
Used by label.py (to write the YOLO labels) and evaluate.py (to score the detector).
"""
import cv2
import numpy as np

from camera_geometry import camera_pose_world, cast, pixel_rays
from pipeline_config import (CLASSES, DEPTH_TOL_ABS, DEPTH_TOL_REL, FAR_CLIP, MIN_PIXELS, MIN_VISIBLE_FRAC,
                             NEAR_CLIP)


def label_frame(depth, pose, calib, objects):
    """Return one dict per object that is at least partly in view, with status
    LABEL / AMBIGUOUS (too small or too occluded) and, for LABEL, the pixel box."""
    H, W = depth.shape
    R_wc, o = camera_pose_world(pose["x"], pose["y"], pose["yaw_rad"], calib)
    D = pixel_rays(calib, R_wc)
    meas = depth.reshape(-1).astype(np.float64)
    finite = np.isfinite(meas) & (meas > 0)
    meas = np.where(finite, meas, -1000.0)
    out = []
    for obj in objects:
        t = cast(obj, o, D)
        expected = np.isfinite(t) & (t >= NEAR_CLIP) & (t <= FAR_CLIP)   # outside the clip planes: not drawn
        n_exp = int(expected.sum())
        if n_exp == 0:
            continue
        with np.errstate(invalid="ignore"):
            err = np.abs(meas - t)
            agree = finite & (err <= DEPTH_TOL_ABS + DEPTH_TOL_REL * t)
        vis = expected & agree
        n_vis = int(vis.sum())
        frac = n_vis / n_exp
        rec = dict(name=obj["name"], cls=obj["cls"], n_exp=n_exp, n_vis=n_vis, frac=frac,
                   depth_err_mm=float(np.median(err[vis & agree]) * 1000) if (vis & agree).any() else None,
                   mask=vis.reshape(H, W), box=None)
        if n_vis == 0:
            rec["status"] = "HIDDEN"
        else:
            ys, xs = np.nonzero(rec["mask"])
            rec["box"] = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
            rec["status"] = "LABEL" if (n_vis >= MIN_PIXELS and frac >= MIN_VISIBLE_FRAC) else "AMBIGUOUS"
        out.append(rec)
    return out


def yolo_lines(results, W, H):
    lines = []
    for r in results:
        if r["status"] != "LABEL":
            continue
        x1, y1, x2, y2 = r["box"]
        lines.append(f"{CLASSES.index(r['cls'])} {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} "
                     f"{(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}")
    return lines


def draw_overlay(bgr, results):
    img = bgr.copy()
    for r in results:
        if r["status"] in ("LABEL", "AMBIGUOUS"):
            tint = (0, 255, 0) if r["status"] == "LABEL" else (0, 165, 255)
            img[r["mask"]] = (0.6 * img[r["mask"]] + 0.4 * np.array(tint)).astype(np.uint8)
    for r in results:
        if r["status"] in ("LABEL", "AMBIGUOUS"):
            x1, y1, x2, y2 = r["box"]
            col = (0, 255, 0) if r["status"] == "LABEL" else (0, 165, 255)
            cv2.rectangle(img, (x1, y1), (x2 - 1, y2 - 1), col, 2)
            text = f"{r['cls']} {r['frac'] * 100:.0f}%" + ("" if r["status"] == "LABEL" else " AMBIG")
            cv2.putText(img, text, (x1, max(14, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
    return img
