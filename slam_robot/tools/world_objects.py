"""world_objects.py - what is in the Gazebo world, read straight from the SDF world file: the labelled
objects (red / blue boxes and the blue cylinder) and the unlabelled look-alikes (grey cylinders, sphere, crate).
"""
import xml.etree.ElementTree as ET
import numpy as np


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
