"""eval_metrics.py - scoring detections against ground truth: box overlap, one-to-one matching, the confusion
matrix, per-class precision / recall, and recall broken down by distance, visibility and cut-off objects.
"""
import numpy as np

from pipeline_config import NAMES

DIST_BINS = [(0.0, 2.0), (2.0, 4.0), (4.0, 6.0), (6.0, 8.1)]
VIS_BINS = [(0.35, 0.60), (0.60, 0.90), (0.90, 1.01)]


# ---------------------------------------------------------------- matching
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


# ---------------------------------------------------------------- tallies
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
