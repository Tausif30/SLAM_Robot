"""eval_report.py - showing evaluation results: the printed report, the confusion-matrix picture and a
montage of the frames that went wrong.
"""
import cv2
import numpy as np

from eval_metrics import breakdowns, class_stats
from pipeline_config import NAMES


def fmt(v):
    return "  n/a" if v is None or v != v else f"{v * 100:5.1f}%"


def print_report(title, res):
    cm, n = res["cm"], len(NAMES)
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")
    print(f"{'class':<15}{'objects':>8}{'correct':>9}{'wrong cls':>10}{'missed':>8}{'false pos':>10}{'precision':>10}{'recall':>8}")
    for s in class_stats(cm):
        print(f"{s['cls']:<15}{s['gt']:>8}{s['tp']:>9}{s['wrong']:>10}{s['missed']:>8}{s['fp']:>10}{fmt(s['precision']):>10}{fmt(s['recall']):>8}")
    bg_fp = int(cm[n, :].sum())
    print(f"false detections on empty background (no object there): {bg_fp}")
    if res["fps"]:
        c = sorted(f["conf"] for f in res["fps"])
        print(f"   strength of ALL false detections ({len(c)}): highest {c[-1]:.2f};  >=0.50: {sum(v >= 0.5 for v in c)},"
              f"  0.25-0.50: {sum(0.25 <= v < 0.5 for v in c)},  0.10-0.25: {sum(0.1 <= v < 0.25 for v in c)},  <0.10: {sum(v < 0.1 for v in c)}")
    print(f"mean box overlap (IoU) of correct detections: {res['mean_iou']:.3f}")
    print("\nconfusion (rows = what the object really is, columns = what the detector said):")
    head = [c[:10] for c in NAMES] + ["missed"]
    print(f"{'':<15}" + "".join(f"{h:>12}" for h in head))
    for i, name in enumerate(NAMES + ["(background)"]):
        print(f"{name:<15}" + "".join(f"{int(cm[i, j]):>12}" for j in range(n + 1)))
    b = breakdowns(res)
    print("\nrecall by distance to the object:")
    for r in b["distance"]:
        print(f"   {r['lo']:.0f}-{min(r['hi'], 8):.0f} m   {fmt(r['recall'])}   ({r['n']} objects)")
    print("recall by how much of the object is visible:")
    for r in b["visibility"]:
        print(f"   {r['lo'] * 100:.0f}-{min(r['hi'], 1) * 100:.0f}%   {fmt(r['recall'])}   ({r['n']} objects)")
    t = b["truncated"]
    print(f"recall when the image edge cuts the object off: {fmt(t['cut_off']['recall'])} ({t['cut_off']['n']})   "
          f"whole object in view: {fmt(t['whole']['recall'])} ({t['whole']['n']})")
    bad = [r for r in res["records"] if not r["ok"]]
    if bad or res["fps"]:
        print("\nevery error in detail:")
        for r in bad:
            x1, y1, x2, y2 = r["box"]
            print(f"   {r['outcome']:<26} {r['name']:<16} frame {r['frame']}  true box {x2 - x1}x{y2 - y1} px, "
                  f"{r['dist']:.1f} m, {r['frac'] * 100:.0f}% visible, {'CUT OFF by the image edge' if r['truncated'] else 'whole'}")
        for f in res["fps"]:
            x1, y1, x2, y2 = f["box"]
            print(f"   false {NAMES[f['cls']]:<20} frame {f['frame']}  box {x2 - x1:.0f}x{y2 - y1:.0f} px at x={x1:.0f}..{x2:.0f}, conf {f['conf']:.2f}")
    return b


def draw_confusion(path, cm):
    n = len(NAMES)
    cell, left, top = 110, 190, 60
    img = np.full((top + (n + 1) * cell + 30, left + (n + 1) * cell + 20, 3), 255, np.uint8)
    rowsum = cm.sum(axis=1, keepdims=True).clip(min=1)
    norm = cm / rowsum
    cols = NAMES + ["missed"]
    for j, c in enumerate(cols):
        cv2.putText(img, c[:14], (left + j * cell + 4, top - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(img, "detector said ->", (left, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    for i, r in enumerate(NAMES + ["(background)"]):
        cv2.putText(img, r, (8, top + i * cell + cell // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        for j in range(n + 1):
            v = norm[i, j]
            ink = (255, 255, 255) if v > 0.55 else (0, 0, 0)
            cv2.rectangle(img, (left + j * cell, top + i * cell), (left + (j + 1) * cell, top + (i + 1) * cell),
                          (255, int(255 - 170 * v), int(255 - 215 * v)), -1)     # white -> blue (BGR)
            cv2.rectangle(img, (left + j * cell, top + i * cell), (left + (j + 1) * cell, top + (i + 1) * cell), (120, 120, 120), 1)
            cv2.putText(img, f"{int(cm[i, j])}", (left + j * cell + 8, top + i * cell + cell // 2 + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, ink, 1, cv2.LINE_AA)
            if i < n and cm[i].sum():
                cv2.putText(img, f"{v * 100:.0f}%", (left + j * cell + 8, top + i * cell + cell // 2 + 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, ink, 1, cv2.LINE_AA)
    cv2.putText(img, "row = truth, % of the row", (8, img.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 80), 1, cv2.LINE_AA)
    cv2.imwrite(path, img)


def failure_montage(path, frames, res, limit=12):
    by_id = {f["id"]: f for f in frames}
    tiles = []
    for fid, (issues, preds) in list(res["bad"].items())[:limit]:
        img = cv2.imread(by_id[fid]["img"])
        for g in by_id[fid]["gts"]:
            cv2.rectangle(img, g["box"][:2], g["box"][2:], (0, 220, 0), 2)
        for p in preds:
            x1, y1, x2, y2 = (int(v) for v in p["box"])
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(img, f"{NAMES[p['cls']]} {p['conf']:.2f}", (x1, max(14, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)
        tile = cv2.resize(img, (320, 240))
        cv2.rectangle(tile, (0, 0), (320, 18), (0, 0, 0), -1)
        cv2.putText(tile, f"{fid}: {'; '.join(issues)}"[:52], (3, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(tile)
    if not tiles:
        return False
    while len(tiles) % 3:
        tiles.append(np.zeros((240, 320, 3), np.uint8))
    cv2.imwrite(path, np.vstack([np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)]))
    return True
