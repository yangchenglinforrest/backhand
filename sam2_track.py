"""
Track two fighters with SAM 2, re-prompting at every camera cut.

SAM 2 follows an object smoothly inside one continuous shot, but a hard camera
cut breaks it. So this script:
  1. splits the clip into shots by detecting cuts,
  2. shows you the first frame of each shot and asks you to draw a box around
     Fighter A, then Fighter B (always in the same order, so A and B stay the
     same people across every shot),
  3. runs SAM 2 over that shot and writes masks + boxes.
Skip a shot (replay, crowd, graphics) by pressing C or Enter with no box on A.

Outputs (in --out):
  sam2_tracked.mp4    video with colored masks and A/B labels
  fighter_boxes.csv   frame, shot, fighter, x1, y1, x2, y2, area, cx, cy

Usage (venv active):
    python sam2_track.py clip.mp4
    python sam2_track.py clip.mp4 --model tiny --device cpu

Needs a desktop session (it opens windows for you to draw boxes) and the
regular opencv-python package, not the headless one.

Install first (see the notes that came with this file):
    python -m pip install sam2 opencv-python numpy
"""
import argparse
import contextlib
import csv
import shutil
import urllib.request
from pathlib import Path

import cv2
import numpy as np

MODELS = {
    "tiny": ("sam2.1_hiera_tiny.pt", "configs/sam2.1/sam2.1_hiera_t.yaml"),
    "small": ("sam2.1_hiera_small.pt", "configs/sam2.1/sam2.1_hiera_s.yaml"),
    "base_plus": ("sam2.1_hiera_base_plus.pt", "configs/sam2.1/sam2.1_hiera_b+.yaml"),
    "large": ("sam2.1_hiera_large.pt", "configs/sam2.1/sam2.1_hiera_l.yaml"),
}
CKPT_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/{}"

# BGR colors: fighter A red, fighter B blue-ish, then spares.
COLORS = [(0, 0, 255), (255, 128, 0), (0, 200, 0), (255, 0, 255)]


def ensure_checkpoint(model, ckpt_dir):
    fname, _ = MODELS[model]
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / fname
    if path.exists():
        return path
    url = CKPT_URL.format(fname)
    print(f"Downloading {fname} ...")

    def hook(blocks, block_size, total):
        if total > 0 and blocks % 200 == 0:
            print(f"  {min(100, blocks * block_size * 100 // total)}%")

    urllib.request.urlretrieve(url, path, hook)
    return path


def frame_hist(frame):
    small = cv2.resize(frame, (160, 90))
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
    return cv2.normalize(h, h).flatten().astype(np.float32)


def extract_shots(video, work, cut_threshold):
    """Write frames to JPEG folders, one folder per shot. SAM 2 reads these."""
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"Could not open {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    shots, shot, prev, idx = [], None, None, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        hist = frame_hist(frame)
        is_cut = prev is not None and cv2.compareHist(prev, hist, cv2.HISTCMP_CORREL) < cut_threshold
        if shot is None or is_cut:
            d = work / f"shot_{len(shots):03d}"
            d.mkdir(parents=True, exist_ok=True)
            shot = dict(dir=d, start=idx, n=0)
            shots.append(shot)
        cv2.imwrite(str(shot["dir"] / f"{shot['n']:05d}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        shot["n"] += 1
        idx += 1
        prev = hist
    cap.release()
    return shots, fps, (w, h), idx


def pick_boxes(frame, names, title):
    """Ask the user to draw one box per fighter. Returns a list (None = not drawn).
    Returns [] if the first fighter is skipped (meaning: skip this shot)."""
    h, w = frame.shape[:2]
    scale = min(1.0, 1280 / w, 720 / h)
    base = cv2.resize(frame, None, fx=scale, fy=scale) if scale < 1 else frame.copy()
    boxes = []
    for i, name in enumerate(names):
        disp = base.copy()
        msg = f"{title} | FIGHTER {name}: drag a box, ENTER to confirm, C to skip"
        cv2.putText(disp, msg, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
        cv2.putText(disp, msg, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        for b, c in zip(boxes, COLORS):
            if b is not None:
                cv2.rectangle(disp, (int(b[0] * scale), int(b[1] * scale)),
                              (int(b[2] * scale), int(b[3] * scale)), c, 2)
        x, y, bw, bh = cv2.selectROI("Select fighter", disp, showCrosshair=False)
        if bw == 0 or bh == 0:
            if i == 0:
                cv2.destroyAllWindows()
                cv2.waitKey(1)
                return []
            boxes.append(None)
            continue
        boxes.append(np.array([x, y, x + bw, y + bh], dtype=np.float32) / scale)
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    return boxes


def render(img, masks, names, rows, gframe, shot_idx):
    out = img.copy()
    for oid, m in masks.items():
        if not m.any():
            continue
        color = COLORS[(oid - 1) % len(COLORS)]
        out[m] = (0.55 * out[m] + 0.45 * np.array(color)).astype(np.uint8)
        ys, xs = np.where(m)
        x1, x2, y1, y2 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
        name = names[oid - 1]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(out, name, (x1, max(25, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        rows.append((gframe, shot_idx, name, x1, y1, x2, y2, int(m.sum()),
                     (x1 + x2) // 2, (y1 + y2) // 2))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--model", default="small", choices=list(MODELS),
                    help="bigger = more accurate but slower (default: small)")
    ap.add_argument("--checkpoint", default=None, help="path to a .pt file (default: auto-download)")
    ap.add_argument("--device", default=None, help="cuda or cpu (default: cuda if available)")
    ap.add_argument("--names", nargs="+", default=["A", "B"], help="fighter labels, in drawing order")
    ap.add_argument("--cut-threshold", type=float, default=0.5,
                    help="histogram correlation below this counts as a camera cut")
    ap.add_argument("--min-shot", type=int, default=15,
                    help="shots shorter than this many frames are not tracked")
    ap.add_argument("--offload", action="store_true",
                    help="keep SAM 2 state in CPU memory (slower, but saves GPU memory)")
    ap.add_argument("--keep-frames", action="store_true", help="keep the extracted JPEG frames")
    ap.add_argument("--out", default="sam2_out")
    args = ap.parse_args()

    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA is not available to PyTorch, using CPU instead (this will be slow).")
        device = "cpu"
    use_cuda = device.startswith("cuda")

    from sam2.build_sam import build_sam2_video_predictor

    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True)
    work = out_dir / "frames"
    if work.exists():
        shutil.rmtree(work)

    print("Splitting video into shots ...")
    shots, fps, (w, h), total = extract_shots(args.video, work, args.cut_threshold)
    print(f"{total} frames, {len(shots)} shots")

    ckpt = Path(args.checkpoint) if args.checkpoint else ensure_checkpoint(args.model, Path("checkpoints"))
    cfg = MODELS[args.model][1]
    print(f"Loading SAM 2 ({args.model}) on {device} ...")
    predictor = build_sam2_video_predictor(cfg, str(ckpt), device=device)

    writer = cv2.VideoWriter(str(out_dir / "sam2_tracked.mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    rows = []
    autocast = torch.autocast("cuda", dtype=torch.bfloat16) if use_cuda else contextlib.nullcontext()

    for si, shot in enumerate(shots):
        files = sorted(shot["dir"].glob("*.jpg"))
        first = cv2.imread(str(files[0]))
        boxes = []
        if shot["n"] >= args.min_shot:
            title = f"Shot {si + 1}/{len(shots)} (frames {shot['start']}-{shot['start'] + shot['n'] - 1})"
            boxes = pick_boxes(first, args.names, title)
        active = [(oid, b) for oid, b in enumerate(boxes, start=1) if b is not None]

        written = set()
        if active:
            print(f"Tracking shot {si + 1}/{len(shots)} ...")
            with torch.inference_mode(), autocast:
                state = predictor.init_state(
                    video_path=str(shot["dir"]),
                    offload_video_to_cpu=True,
                    offload_state_to_cpu=args.offload,
                )
                for oid, box in active:
                    predictor.add_new_points_or_box(
                        inference_state=state, frame_idx=0, obj_id=oid, box=box)
                for fidx, obj_ids, logits in predictor.propagate_in_video(state):
                    masks = {oid: (logits[i, 0] > 0).cpu().numpy() for i, oid in enumerate(obj_ids)}
                    img = cv2.imread(str(files[fidx]))
                    writer.write(render(img, masks, args.names, rows, shot["start"] + fidx, si))
                    written.add(fidx)
                predictor.reset_state(state)
            if use_cuda:
                torch.cuda.empty_cache()
        else:
            print(f"Skipping shot {si + 1}/{len(shots)}")

        # Any frame not tracked (skipped shot, or missed frames) goes in unmarked.
        for fidx, f in enumerate(files):
            if fidx not in written:
                writer.write(cv2.imread(str(f)))

    writer.release()
    with open(out_dir / "fighter_boxes.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["frame", "shot", "fighter", "x1", "y1", "x2", "y2", "area", "cx", "cy"])
        wr.writerows(rows)

    if not args.keep_frames:
        shutil.rmtree(work, ignore_errors=True)
    print(f"\nDone. Saved sam2_tracked.mp4 and fighter_boxes.csv to {out_dir.resolve()}")
    print("Check the video around clinches and the start of each shot. If a fighter drifts, "
          "rerun with a bigger --model.")


if __name__ == "__main__":
    main()
