"""pose_sampling.py - where the robot may stand: free cells of the saved map (with clearance and line of
sight), and picking a random pose or one that looks at a chosen object.
"""
import math
import os
import cv2
import numpy as np
import yaml

from camera_geometry import angle_diff
from pipeline_config import CLEARANCE, LOOK_DIST, LOOK_JITTER_DEG, MODE_WEIGHTS, MODES


class FreeSpace:
    """Free cells of the saved map, with clearance, line of sight and pose sampling."""

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

    def lookat_pose(self, rng, target_xy, dmin, dmax, jitter, stop_before, tries=300):
        """A free pose between dmin and dmax from the target with a clear line to it, aimed at it
        (plus a random aim error up to +-jitter). None if no such pose is found."""
        tx, ty = target_xy
        d = np.hypot(self.cand_x - tx, self.cand_y - ty)
        idx = np.nonzero((d >= dmin) & (d <= dmax))[0]
        for i in rng.permutation(idx)[:tries]:
            x, y = self._jitter(rng, int(i))
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


def choose_pose(space, objects, rng, force=None, distractors=()):
    """Pick a sampling mode and a pose. Returns (mode, target name or None, (x, y, yaw)).
    force: always use this mode ("red", "blue", "grey" or "random"); "grey" aims at the unlabelled look-alikes."""
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
