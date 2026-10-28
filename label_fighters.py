"""
Detect cuts, track with ByteTrack, and label fighters A/B by trunk color.

The labels come from what the fighters look like (their trunks), not from the
tracker's ID numbers, so they survive camera cuts and ID swaps.

How it works:
  1. Calibration: you pick Fighter A and Fighter B once on a clear frame
     (press the number shown on each box). Their trunk colors become the
     reference. This is saved to a JSON file so you can rerun without picking.
  2. Cut detection: a big change in the picture marks a camera cut. The
     tracker is reset at every cut.
  3. Each frame, every detected person's trunk color is compared with the two
     references. The best match becomes "A" or "B". People who match neither
     (referee, corner staff, crowd) stay unlabeled.
  4. If a fighter's trunks are hidden for a moment (clinch, turned away), the
     ByteTrack ID can carry the label for a few frames. Rows are marked
     source=color or source=track so you can tell which is which.

Outputs (in --out):
  labeled.mp4          video with skeletons colored by fighter
  fighters_pose.csv    one row per labeled fighter per frame: box + 17 keypoints
  calibration.json     the saved reference colors

Usage (venv active):
    python label_fighters.py clip.mp4
    python label_fighters.py clip.mp4 --calib sam2_out/calibration.json   # reuse a calibration
    python label_fighters.py clip.mp4 --inherit-frames 0                  # pure color labels

Needs a desktop session for the calibration window, and regular opencv-python.
Install: python -m pip install ultralytics opencv-python numpy
"""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

COLORS = {"A": (0, 0, 255), "B": (255, 128, 0)}  # BGR: A red, B blue-ish
GREY = (160, 160, 160)
NAMES = ["A", "B"]

# COCO 17-keypoint skeleton
EDGES = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12),
         (11, 13), (13, 15), (12, 14), (14, 16), (0, 1), (0, 2), (1, 3), (2, 4)]


# ---------- color signature ----------

def trunk_signature(frame, box, kxy, kc):
    """Median Lab color of the trunk area (around the hips). None if too small."""
    H, W = frame.shape[:2]
    x1, y1, x2, y2 = box
    bh, bw = y2 - y1, x2 - x1
    if bh < 40 or bw < 15:
        return None

    if kxy is not None and kc[11] > 0.4 and kc[12] > 0.4:
        hip_l, hip_r = kxy[11], kxy[12]
        cx = (hip_l[0] + hip_r[0]) / 2
        cy = (hip_l[1] + hip_r[1]) / 2
        half_w = max(0.55 * abs(hip_l[0] - hip_r[0]), 0.07 * bh)
        ya, yb = cy - 0.04 * bh, cy + 0.20 * bh
        xa, xb = cx - half_w, cx + half_w
    else:
        cx = (x1 + x2) / 2
        xa, xb = cx - 0.2 * bw, cx + 0.2 * bw
        ya, yb = y1 + 0.45 * bh, y1 + 0.65 * bh

    xa, xb = int(max(0, xa)), int(min(W, xb))
    ya, yb = int(max(0, ya)), int(min(H, yb))
    if xb - xa < 4 or yb - ya < 4:
        return None
    crop = cv2.cvtColor(frame[ya:yb, xa:xb], cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    if len(crop) < 25:
        return None
    lab = np.median(crop, axis=0)
    return np.array([lab[0] * 100 / 255, lab[1] - 128, lab[2] - 128], dtype=np.float32)


def color_dist(a, b, l_weight):
    d = a - b
    return float(np.sqrt((l_weight * d[0]) ** 2 + d[1] ** 2 + d[2] ** 2))


def assign_by_color(people, refs, max_dist, l_weight, margin):
    """Return a list: 'A', 'B' or None for each person (best joint assignment)."""
    n = len(people)
    D = np.full((n, 2), np.inf)
    for i, p in enumerate(people):
        if p["sig"] is not None:
            for k in range(2):
                D[i, k] = color_dist(p["sig"], refs[k], l_weight)

    best, best_cost = (None, None), float("inf")
    options = [None] + list(range(n))
    for i in options:
        for j in options:
            if i is not None and i == j:
                continue
            cost = 0.0
            ok = True
            for idx, k in ((i, 0), (j, 1)):
                if idx is None:
                    cost += max_dist
                elif D[idx, k] <= max_dist:
                    cost += D[idx, k]
                else:
                    ok = False
            if ok and cost < best_cost:
                best, best_cost = (i, j), cost

    labels = [None] * n
    i, j = best
    # A lone assignment must clearly beat the other fighter's color.
    if i is not None and j is None and D[i, 0] >= margin * D[i, 1]:
        i = None
    if j is not None and i is None and D[j, 1] >= margin * D[j, 0]:
        j = None
    if i is not None:
        labels[i] = "A"
    if j is not None:
        labels[j] = "B"
    return labels


# ---------- detection helpers ----------

def people_from_result(frame, r, max_people):
    if r.boxes is None or len(r.boxes) == 0:
        return []
    xyxy = r.boxes.xyxy.cpu().numpy()
    conf = r.boxes.conf.cpu().numpy()
    ids = (r.boxes.id.cpu().numpy().astype(int) if r.boxes.id is not None
           else -np.ones(len(xyxy), dtype=int))
    if r.keypoints is not None:
        kxy_all = r.keypoints.xy.cpu().numpy()
        kc_all = (r.keypoints.conf.cpu().numpy() if r.keypoints.conf is not None
                  else np.ones(kxy_all.shape[:2], dtype=np.float32))
    else:
        kxy_all = kc_all = None

    areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
    people = []
    for i in np.argsort(-areas)[:max_people]:
        kxy = kxy_all[i] if kxy_all is not None else None
        kc = kc_all[i] if kc_all is not None else None
        p = dict(box=xyxy[i], conf=float(conf[i]), id=int(ids[i]), kxy=kxy, kc=kc)
        p["sig"] = trunk_signature(frame, xyxy[i], kxy, kc) if kxy is not None else None
        people.append(p)
    return people


def reset_tracker(model):
    pred = getattr(model, "predictor", None)
    for t in getattr(pred, "trackers", None) or []:
        try:
            t.reset()
        except Exception:
            pass


def frame_hist(frame):
    small = cv2.resize(frame, (160, 90))
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
    return cv2.normalize(h, h).flatten().astype(np.float32)


def draw_person(img, p, label, text=None):
    color = COLORS.get(label, GREY)
    thick = 3 if label else 1
    x1, y1, x2, y2 = [int(v) for v in p["box"]]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thick)
    if p["kxy"] is not None:
        for a, b in EDGES:
            if p["kc"][a] > 0.3 and p["kc"][b] > 0.3:
                cv2.line(img, tuple(int(v) for v in p["kxy"][a]), tuple(int(v) for v in p["kxy"][b]),
                         color, 2 if label else 1)
    if label:
        cv2.putText(img, text or label, (x1, max(25, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)


# ---------- calibration ----------

def calibrate(args, det_model):
    cap = cv2.VideoCapture(args.video)
    idx = -1
    while True:
        ok, frame = cap.read()
        if not ok:
            cap.release()
            raise SystemExit("Ran out of video before calibration finished. "
                             "Try a clip where both fighters are clearly visible.")
        idx += 1
        if idx % args.calib_step != 0:
            continue
        r = det_model.predict(frame, conf=args.conf, verbose=False, device=args.device)[0]
        people = [p for p in people_from_result(frame, r, args.max_people) if p["sig"] is not None]
        if len(people) < 2:
            continue

        disp = frame.copy()
        for n, p in enumerate(people, start=1):
            x1, y1, x2, y2 = [int(v) for v in p["box"]]
            cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(disp, str(n), (x1 + 6, y1 + 40), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 255, 0), 4)
        scale = min(1.0, 1280 / disp.shape[1], 720 / disp.shape[0])
        if scale < 1:
            disp = cv2.resize(disp, None, fx=scale, fy=scale)

        chosen = []
        for name in NAMES:
            msg = f"Frame {idx}: press the number of FIGHTER {name}  (N = next frame, ESC = quit)"
            show = disp.copy()
            cv2.putText(show, msg, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
            cv2.putText(show, msg, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            while True:
                cv2.imshow("Calibrate", show)
                key = cv2.waitKey(0) & 0xFF
                if key == 27:
                    cv2.destroyAllWindows()
                    raise SystemExit("Calibration cancelled.")
                if key in (ord("n"), ord("N")):
                    chosen = None
                    break
                if ord("1") <= key <= ord("9"):
                    n = key - ord("0") - 1
                    if n < len(people) and n not in chosen:
                        chosen.append(n)
                        break
            if chosen is None:
                break
        if chosen is None:
            continue

        cv2.destroyAllWindows()
        cv2.waitKey(1)
        cap.release()
        refs = [people[chosen[0]]["sig"], people[chosen[1]]["sig"]]
        return refs


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--model", default="yolo11s-pose.pt", help="Ultralytics pose model")
    ap.add_argument("--tracker", default="bytetrack.yaml")
    ap.add_argument("--device", default=None, help="cpu, cuda:0, ... (default: auto)")
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--max-people", type=int, default=6, help="consider only the largest N people")
    ap.add_argument("--calib", default=None, help="calibration.json to reuse (skips the picking step)")
    ap.add_argument("--calib-step", type=int, default=15, help="frames between calibration candidates")
    ap.add_argument("--cut-threshold", type=float, default=0.5,
                    help="histogram correlation below this counts as a camera cut")
    ap.add_argument("--max-dist", type=float, default=35.0,
                    help="max color distance to accept a match (lower = stricter)")
    ap.add_argument("--l-weight", type=float, default=0.7,
                    help="weight of brightness in color distance (lower ignores lighting more)")
    ap.add_argument("--margin", type=float, default=0.8,
                    help="a lone match must be this much closer to its fighter than to the other")
    ap.add_argument("--inherit-frames", type=int, default=10,
                    help="let a tracker ID carry a label for this many frames when color is unclear "
                         "(0 = color only)")
    ap.add_argument("--out", default="labeled_out")
    args = ap.parse_args()

    from ultralytics import YOLO

    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True)

    det_model = YOLO(args.model)    # used only for calibration
    track_model = YOLO(args.model)  # used for tracking

    if args.calib:
        data = json.loads(Path(args.calib).read_text())
        refs = [np.array(data["A"], dtype=np.float32), np.array(data["B"], dtype=np.float32)]
        print(f"Loaded calibration from {args.calib}")
    else:
        refs = calibrate(args, det_model)
    calib_path = out_dir / "calibration.json"
    calib_path.write_text(json.dumps({"A": [float(v) for v in refs[0]], "B": [float(v) for v in refs[1]]}))
    print(f"Reference trunk colors (Lab): A={np.round(refs[0], 1)}  B={np.round(refs[1], 1)}")
    if color_dist(refs[0], refs[1], args.l_weight) < 2 * args.max_dist * 0.5:
        print("Warning: the two fighters' trunks look similar. Labels may swap often; "
              "lower --max-dist or check the video carefully.")

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    writer = cv2.VideoWriter(str(out_dir / "labeled.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    csv_file = open(out_dir / "fighters_pose.csv", "w", newline="")
    wr = csv.writer(csv_file)
    kp_cols = [f"kp{k}_{c}" for k in range(17) for c in ("x", "y", "conf")]
    wr.writerow(["frame", "time_s", "shot", "fighter", "source", "track_id", "color_dist",
                 "x1", "y1", "x2", "y2"] + kp_cols)

    stats = Counter()
    cut_frames = []
    prev_hist = None
    shot = 0
    last_label = {}  # track_id -> (label, frame) within the current shot
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        hist = frame_hist(frame)
        if prev_hist is not None and cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL) < args.cut_threshold:
            cut_frames.append(frame_idx)
            shot += 1
            reset_tracker(track_model)
            last_label.clear()
        prev_hist = hist

        r = track_model.track(frame, persist=True, tracker=args.tracker, conf=args.conf,
                              verbose=False, device=args.device)[0]
        people = people_from_result(frame, r, args.max_people)

        labels = assign_by_color(people, refs, args.max_dist, args.l_weight, args.margin)
        sources = ["color" if lab else None for lab in labels]

        # Let a tracker ID carry a label briefly when the color is unclear.
        if args.inherit_frames > 0:
            used = {lab for lab in labels if lab}
            for i, p in enumerate(people):
                if labels[i] is None and p["id"] >= 0 and p["id"] in last_label:
                    lab, seen = last_label[p["id"]]
                    if frame_idx - seen <= args.inherit_frames and lab not in used:
                        labels[i], sources[i] = lab, "track"
                        used.add(lab)
            for i, p in enumerate(people):
                if sources[i] == "color" and p["id"] >= 0:
                    last_label[p["id"]] = (labels[i], frame_idx)

        vis = frame.copy()
        for p, lab, src in zip(people, labels, sources):
            d = None
            if lab and p["sig"] is not None:
                d = color_dist(p["sig"], refs[NAMES.index(lab)], args.l_weight)
            draw_person(vis, p, lab, f"{lab}{'' if src == 'color' else '*'}" if lab else None)
            if lab:
                kp = []
                for k in range(17):
                    if p["kxy"] is not None:
                        kp += [round(float(p["kxy"][k][0]), 1), round(float(p["kxy"][k][1]), 1),
                               round(float(p["kc"][k]), 3)]
                    else:
                        kp += ["", "", ""]
                x1, y1, x2, y2 = [round(float(v), 1) for v in p["box"]]
                wr.writerow([frame_idx, round(frame_idx / fps, 3), shot, lab, src, p["id"],
                             "" if d is None else round(d, 1), x1, y1, x2, y2] + kp)
                stats[lab] += 1
        present = {lab for lab in labels if lab}
        if present == {"A", "B"}:
            stats["both"] += 1
        writer.write(vis)

        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"  {frame_idx}/{total} frames")

    cap.release()
    writer.release()
    csv_file.close()

    print(f"\nFrames: {frame_idx}   Shots: {shot + 1}   Cuts at frames: {cut_frames}")
    print(f"Frames with A: {stats['A']}   with B: {stats['B']}   with both: {stats['both']}")
    print(f"Saved labeled.mp4, fighters_pose.csv and calibration.json to {out_dir.resolve()}")
    print("In the video, a * after a label means it was carried by the tracker, not matched by color.")
    print("Watch the video around cuts and clinches to check that A and B stay on the same people.")


if __name__ == "__main__":
    main()
