"""ground_truth.py - what is in the world, and where is it in a picture?

  1. the world file -> the labelled objects (red / blue boxes, blue cylinder) and the unlabelled look-alikes
  2. label_frame(): world objects + camera model (camera_geometry.py) + depth image -> which object is where in a
     picture, and is it really visible?  yolo_lines() and draw_overlay() turn that into labels and a check picture.
Used by dataset.py (labels) and training.py (scoring the detector).
"""

import xml.etree.ElementTree as ET
import cv2
import numpy as np
from camera_geometry import camera_pose_world, cast, pixel_rays
from pipeline_config import (CLASSES, DEPTH_TOL_ABS, DEPTH_TOL_REL, FAR_CLIP, MIN_PIXELS, MIN_VISIBLE_FRAC,
                             NEAR_CLIP)

# World file
def class_of(name):
    n = name.lower()
    if n.startswith("red_box"):
        return "red_box"
    if n.startswith("blue_box") or n.startswith("blue_crate"):
        return "blue_box"
    if n.startswith("blue_cylinder"):
        return "blue_cylinder"
    return None

def parse_floats(text, n):
    vals = [float(v) for v in (text or "").split()]
    return vals + [0.0] * (n - len(vals))

def load_world_objects(path):
    """Read the labelled objects (box / cylinder models) from the SDF world file."""
    objs = []
    for model in ET.parse(path).getroot().iter("model"):
        name = model.get("name")
        cls = class_of(name)
        if cls is None:
            continue
        x, y, z, roll, pitch, yaw = parse_floats(model.findtext("pose"), 6)
        if abs(roll) > 1e-6 or abs(pitch) > 1e-6:
            raise ValueError(f"{name}: roll/pitch not supported")
        link, visual = model.find("link"), model.find("link/visual")
        for el in (link, visual):
            if el is not None and any(abs(v) > 1e-9 for v in parse_floats(el.findtext("pose"), 6)):
                raise ValueError(f"{name}: nested link/visual pose not supported")
        geom = model.find("link/visual/geometry")
        if geom.find("box") is not None:
            sx, sy, sz = parse_floats(geom.findtext("box/size"), 3)
            objs.append(dict(name=name, cls=cls, kind="box", center=np.array([x, y, z]), yaw=yaw,
                             half=np.array([sx, sy, sz]) / 2.0))
        elif geom.find("cylinder") is not None:
            r = float(geom.findtext("cylinder/radius"))
            length = float(geom.findtext("cylinder/length"))
            objs.append(dict(name=name, cls=cls, kind="cylinder", center=np.array([x, y, z]), yaw=yaw,
                             radius=r, half_len=length / 2.0))
        else:
            raise ValueError(f"{name}: unsupported geometry (only box and cylinder)")
    return objs


def load_distractors(path):
    """Unlabelled single-shape models (the grey cylinders, sphere and crate): look-alikes for the detector."""
    out = []
    for model in ET.parse(path).getroot().iter("model"):
        name = model.get("name")
        if class_of(name) is not None or len(model.findall("link")) != 1:
            continue                                   # labelled objects, and multi-link models such as the walls
        geom = model.find("link/visual/geometry")
        if geom is None:
            continue
        x, y, z = parse_floats(model.findtext("pose"), 6)[:3]
        centre = np.array([x, y, z])
        if geom.find("box") is not None:
            sx, sy, _ = parse_floats(geom.findtext("box/size"), 3)
            out.append(dict(name=name, cls="distractor", kind="box", center=centre, half=np.array([sx, sy, 0.0]) / 2.0))
        elif geom.find("cylinder") is not None:
            out.append(dict(name=name, cls="distractor", kind="cylinder", center=centre,
                            radius=float(geom.findtext("cylinder/radius"))))
        elif geom.find("sphere") is not None:
            out.append(dict(name=name, cls="distractor", kind="sphere", center=centre,
                            radius=float(geom.findtext("sphere/radius"))))
    return out


# ======================================================================================================
# 2. Which object is where in a picture?
# ======================================================================================================

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