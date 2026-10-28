"""
Compare several BoxMOT trackers on one fight clip.

One detection pass feeds every tracker, so the comparison is fair and fast.
For each tracker it saves an annotated video and prints a summary of how
many IDs it created and how many new IDs appeared right after a camera cut.

Usage (venv active):
    python compare_trackers.py clip.mp4
    python compare_trackers.py clip.mp4 --trackers botsort strongsort deepocsort
    python compare_trackers.py clip.mp4 --top2        # keep only the 2 biggest people per frame
    python compare_trackers.py clip.mp4 --device cuda:0

Install first:
    python -m pip install boxmot ultralytics opencv-python numpy
"""
import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np

# name -> (possible class names in boxmot, uses ReID weights)
# Class names have changed between boxmot versions, so several are tried.
TRACKERS = {
    "bytetrack": (["ByteTrack", "BYTETracker"], False),
    "ocsort": (["OcSort", "OCSORT"], False),
    "botsort": (["BotSort", "BoTSORT"], True),
    "strongsort": (["StrongSort", "StrongSORT"], True),
    "deepocsort": (["DeepOcSort", "DeepOCSORT"], True),
}

PALETTE = [
    (0, 255, 0), (255, 128, 0), (0, 128, 255), (255, 0, 255),
    (255, 255, 0), (0, 255, 255), (128, 0, 255), (255, 0, 0),
]


def _find_reid_config_class():
    """Locate boxmot's ReIDConfig class (its location differs between versions)."""
    import importlib
    import pkgutil

    import boxmot

    cls = getattr(boxmot, "ReIDConfig", None)
    if cls is not None:
        return cls
    for m in pkgutil.walk_packages(boxmot.__path__, "boxmot."):
        if not any(s in m.name for s in ("reid", "config", "appearance", "trackers")):
            continue
        try:
            mod = importlib.import_module(m.name)
        except Exception:
            continue
        cls = getattr(mod, "ReIDConfig", None)
        if cls is not None:
            return cls
    return None


def _config_field_names(cls):
    import dataclasses
    import inspect

    if dataclasses.is_dataclass(cls):
        return [f.name for f in dataclasses.fields(cls)]
    if hasattr(cls, "model_fields"):  # pydantic
        return list(cls.model_fields)
    try:
        return list(inspect.signature(cls).parameters)
    except (TypeError, ValueError):
        return []


def build_tracker(name, reid_weights, device):
    import boxmot

    class_names, uses_reid = TRACKERS[name]
    cls = next((getattr(boxmot, c) for c in class_names if hasattr(boxmot, c)), None)
    if cls is None:
        raise RuntimeError(
            f"'{name}' not found in your boxmot version. "
            f"Tried: {class_names}. Check: python -m pip show boxmot"
        )

    if not uses_reid:
        return cls()

    import inspect
    import torch

    weights = Path(reid_weights)
    devices = [device, torch.device(device)]
    errors = []

    info = []

    # Attempt 0: newest API. ReID settings go in a ReIDConfig passed as reid=...
    reid_cls = _find_reid_config_class()
    if reid_cls is None:
        info.append("ReIDConfig class not found")
    else:
        names = _config_field_names(reid_cls)
        info.append(f"ReIDConfig fields: {names}")
        try:
            import boxmot as _bm
            factory0 = getattr(_bm, "create_tracker", None)
        except Exception:
            factory0 = None
        for dev in devices:
            kwargs = {}
            for n in names:
                low = n.lower()
                if "weight" in low or low in ("model", "model_path", "path"):
                    kwargs[n] = weights
                elif "device" in low:
                    kwargs[n] = dev
                elif low in ("half", "fp16"):
                    kwargs[n] = False
            if not kwargs:
                continue
            try:
                cfg = reid_cls(**kwargs)
            except Exception as e:
                errors.append(f"ReIDConfig({', '.join(kwargs)}): {e}")
                continue
            builders = []
            if factory0 is not None:
                builders.append(lambda: factory0(name, reid=cfg))
            builders.append(lambda: cls(reid=cfg))
            for build in builders:
                try:
                    return build()
                except Exception as e:
                    errors.append(f"reid=ReIDConfig({', '.join(kwargs)}): {e}")

    # Attempt 1: boxmot's own factory (older versions). Its location varies.
    for mod_name in ("boxmot", "boxmot.trackers.tracker_zoo"):
        try:
            mod = __import__(mod_name, fromlist=["create_tracker"])
            factory = getattr(mod, "create_tracker", None)
        except Exception:
            factory = None
        if factory is None:
            continue
        for dev in devices:
            try:
                return factory(name, reid_weights=weights, device=dev, half=False)
            except Exception as e:
                errors.append(f"create_tracker({mod_name}, {type(dev).__name__}): {e}")

    # Attempt 2: call the class directly, passing only arguments it accepts.
    try:
        params = set(inspect.signature(cls).parameters)
    except (TypeError, ValueError):
        params = set()
    wanted = []
    for dev in devices:
        for wkey in ("reid_weights", "model_weights"):
            for hkey in ("half", "fp16"):
                wanted.append({wkey: weights, "device": dev, hkey: False})
    for kwargs in wanted:
        if params and not set(kwargs) <= params:
            continue
        try:
            return cls(**kwargs)
        except Exception as e:
            errors.append(f"{cls.__name__}({', '.join(kwargs)}): {e}")

    # Diagnostics so the problem can be pinned down.
    try:
        sig = str(inspect.signature(cls))
    except (TypeError, ValueError):
        sig = "(unavailable)"
    detail = "\n    ".join(dict.fromkeys(errors[:3] + errors[-3:]))
    raise RuntimeError(
        f"Could not construct {name}.\n  Class: {cls.__module__}.{cls.__name__}{sig}\n"
        f"  Info: {'; '.join(info)}\n"
        f"  Errors:\n    {detail}"
    )


def detect_people(model, frame, conf, top2):
    res = model(frame, classes=[0], conf=conf, verbose=False)[0]
    if res.boxes is None or len(res.boxes) == 0:
        return np.empty((0, 6), dtype=np.float32)
    xyxy = res.boxes.xyxy.cpu().numpy()
    confs = res.boxes.conf.cpu().numpy()[:, None]
    clss = res.boxes.cls.cpu().numpy()[:, None]
    dets = np.hstack([xyxy, confs, clss]).astype(np.float32)
    if top2 and len(dets) > 2:
        areas = (dets[:, 2] - dets[:, 0]) * (dets[:, 3] - dets[:, 1])
        dets = dets[np.argsort(-areas)[:2]]
    return dets


def frame_hist(frame):
    small = cv2.resize(frame, (160, 90))
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
    return cv2.normalize(h, h).flatten()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--trackers", nargs="+", default=["bytetrack", "botsort", "strongsort", "deepocsort"],
                    choices=list(TRACKERS))
    ap.add_argument("--detector", default="yolo11s.pt", help="Ultralytics detection model")
    ap.add_argument("--reid", default="osnet_x0_25_msmt17.pt", help="ReID weights (auto-downloaded)")
    ap.add_argument("--device", default="cpu", help="cpu or cuda:0")
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--top2", action="store_true", help="keep only the two largest people per frame")
    ap.add_argument("--cut-threshold", type=float, default=0.5,
                    help="histogram correlation below this counts as a camera cut")
    ap.add_argument("--out", default="tracker_compare")
    args = ap.parse_args()

    from ultralytics import YOLO

    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Could not open {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    detector = YOLO(args.detector)

    state = {}
    for name in args.trackers:
        try:
            tracker = build_tracker(name, args.reid, args.device)
        except Exception as e:
            print(f"[skip] {name}: {e}")
            continue
        writer = cv2.VideoWriter(str(out_dir / f"{name}.mp4"),
                                 cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        state[name] = dict(
            tracker=tracker, writer=writer, first_seen={}, rows=[],
            frames_with_2=0, time=0.0,
        )
    if not state:
        raise SystemExit("No trackers could be created.")

    cut_frames = []
    prev_hist = None
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        hist = frame_hist(frame)
        if prev_hist is not None:
            corr = cv2.compareHist(prev_hist.astype(np.float32), hist.astype(np.float32),
                                   cv2.HISTCMP_CORREL)
            if corr < args.cut_threshold:
                cut_frames.append(frame_idx)
        prev_hist = hist

        dets = detect_people(detector, frame, args.conf, args.top2)

        for name, s in state.items():
            t0 = time.time()
            tracks = s["tracker"].update(dets.copy(), frame)
            s["time"] += time.time() - t0

            vis = frame.copy()
            tracks = np.asarray(tracks)
            n_tracks = 0
            if tracks.size:
                for t in tracks:
                    x1, y1, x2, y2 = [int(v) for v in t[:4]]
                    tid = int(t[4])
                    n_tracks += 1
                    s["first_seen"].setdefault(tid, frame_idx)
                    s["rows"].append((frame_idx, tid, x1, y1, x2, y2))
                    color = PALETTE[tid % len(PALETTE)]
                    cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(vis, f"ID {tid}", (x1, max(20, y1 - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            if n_tracks >= 2:
                s["frames_with_2"] += 1
            cv2.putText(vis, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            s["writer"].write(vis)

        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"  {frame_idx}/{total} frames")

    cap.release()
    for s in state.values():
        s["writer"].release()

    # Summary
    print(f"\nFrames: {frame_idx}   Camera cuts detected: {len(cut_frames)} at frames {cut_frames}\n")
    header = f"{'tracker':<12}{'unique IDs':>11}{'births after cut':>18}{'frames w/ 2+':>14}{'speed (fps)':>13}"
    print(header)
    print("-" * len(header))
    summary_rows = []
    for name, s in state.items():
        ids = s["first_seen"]
        births_after_cut = sum(
            1 for f in ids.values()
            if f > 0 and any(0 <= f - c <= 3 for c in cut_frames)
        )
        speed = frame_idx / s["time"] if s["time"] else 0
        print(f"{name:<12}{len(ids):>11}{births_after_cut:>18}{s['frames_with_2']:>14}{speed:>13.1f}")
        summary_rows.append((name, len(ids), births_after_cut, s["frames_with_2"], round(speed, 1)))

        with open(out_dir / f"{name}_tracks.csv", "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["frame", "id", "x1", "y1", "x2", "y2"])
            wr.writerows(s["rows"])

    with open(out_dir / "summary.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["tracker", "unique_ids", "births_after_cut", "frames_with_2plus", "fps"])
        wr.writerows(summary_rows)

    print(f"\nSaved videos, per-frame tracks and summary.csv to: {out_dir.resolve()}")
    print("Lower 'unique IDs' and fewer 'births after cut' are better "
          "(ideal is about 2 IDs, plus a referee if not using --top2).")
    print("Always confirm by watching the videos: these numbers can't catch IDs that swap between fighters.")


if __name__ == "__main__":
    main()
