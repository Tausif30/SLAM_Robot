# Camera_pose_world + pixel_rays give one world-frame ray per pixel; 
# Ray_box / ray_cylinder say how far along each ray an object is. 
# Ground_truth.label_frame compares that distance with the depth image.
import math
import numpy as np
from pipeline_config import BASE_Z

# Rotations and angles
def rot_z(yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

def rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])

def rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])

def quat_to_rot(x, y, z, w):
    n = x * x + y * y + z * z + w * w
    x, y, z, w = (v / np.sqrt(n) for v in (x, y, z, w))
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])

def yaw_to_quat(yaw):
    return (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))

def quat_to_yaw(x, y, z, w):
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

def angle_diff(a, b):
    return (a - b + math.pi) % (2.0 * math.pi) - math.pi


# Camera model
# Return (R_wc, origin): camera orientation and position in the world frame.
# (pitch / yaw rotation and a height offset), found by fitting to real frames.
def camera_pose_world(x, y, yaw, calib):
    R_wb = rot_z(yaw)
    t_wb = np.array([x, y, BASE_Z])
    R_bc = np.array(calib["R_base_cam_optical"], dtype=float)
    t_bc = np.array(calib["t_base_cam"], dtype=float)
    ref = calib.get("refine")
    if ref:
        R_bc = R_bc @ rot_x(ref["pitch_rad"]) @ rot_y(ref["yaw_rad"])
        t_bc = t_bc + np.array([0.0, 0.0, ref["dz_m"]])
    return R_wb @ R_bc, R_wb @ t_bc + t_wb

# One ray direction per pixel, in world coordinates. Scaled so that the camera-frame
# z-component is 1: then 'distance t along the ray' equals the depth image's z-depth.
def pixel_rays(calib, R_wc):
    W, H = calib["width"], calib["height"]
    uu, vv = np.meshgrid(np.arange(W), np.arange(H))
    d_cam = np.stack([(uu - calib["cx"]) / calib["fx"], (vv - calib["cy"]) / calib["fy"],
                      np.ones((H, W))], axis=-1).reshape(-1, 3)
    return d_cam @ R_wc.T

# Ray casting
# Depth t of the first hit on a (yaw-rotated) box for every ray; inf where there is no hit.
def ray_box(o, D, center, yaw, half):
    R = rot_z(yaw)
    ol = R.T @ (o - center)
    Dl = D @ R
    Dl = np.where(np.abs(Dl) < 1e-12, 1e-12, Dl)
    t1, t2 = (-half - ol) / Dl, (half - ol) / Dl
    t_near = np.minimum(t1, t2).max(axis=1)
    t_far = np.maximum(t1, t2).min(axis=1)
    hit = (t_far >= t_near) & (t_near > 0)
    return np.where(hit, t_near, np.inf)

def ray_cylinder(o, D, center, radius, half_len):
    ox, oy, oz = o - center
    dx, dy, dz = D[:, 0], D[:, 1], D[:, 2]
    best = np.full(len(D), np.inf)
    a = dx * dx + dy * dy
    b = 2.0 * (ox * dx + oy * dy)
    c = ox * ox + oy * oy - radius * radius
    disc = b * b - 4.0 * a * c
    ok = (a > 1e-12) & (disc >= 0)
    sq = np.sqrt(np.where(ok, disc, 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (-b - sq) / (2.0 * a)
    side = ok & (t > 0) & (np.abs(oz + t * dz) <= half_len)
    best = np.where(side, t, best)
    for zc in (-half_len, half_len):
        with np.errstate(divide="ignore", invalid="ignore"):
            tc = (zc - oz) / dz
        x, y = ox + tc * dx, oy + tc * dy
        cap = (np.abs(dz) > 1e-12) & (tc > 0) & (x * x + y * y <= radius * radius) & (tc < best)
        best = np.where(cap, tc, best)
    return best

def cast(obj, o, D):
    if obj["kind"] == "box":
        return ray_box(o, D, obj["center"], obj["yaw"], obj["half"])
    return ray_cylinder(o, D, obj["center"], obj["radius"], obj["half_len"])

# Calibration maths (used by the 'calibrate' command in ros_collector.py)
# Optical frame: x right, y down, z forward.  
# Body frame: x forward, y left, z up.
# v_body = M_BODY_FROM_OPTICAL @ v_optical

M_BODY_FROM_OPTICAL = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])

# R: orientation of the camera frame in the robot frame. 
# Return the optical-frame orientation and whether a conversion was needed. 
# Raises if the camera does not face forward.
def to_optical(R):
    forward = np.array([1.0, 0.0, 0.0])
    if R[:, 2] @ forward > 0.8:
        return R, False                      # frame is already optical (z forward)
    if R[:, 0] @ forward > 0.8:
        return R @ M_BODY_FROM_OPTICAL, True  # body convention (x forward): convert
    raise ValueError(f"camera does not face forward; rotation matrix:\n{R}")

# The simulator renders with exact angles.
# If the TF rotation is within tol of the exact optical rotation, use the exact one.
def finalize_rotation(R_opt, tol=0.01):
    if np.abs(np.asarray(R_opt) - M_BODY_FROM_OPTICAL).max() < tol:
        return M_BODY_FROM_OPTICAL.copy(), True
    return np.asarray(R_opt), False

def merge_calibration(new, old):
    merged = dict(new)
    if old and old.get("refine"):
        merged["refine"] = old["refine"]
    return merged