"""Interactive OpenCV playback of the ``observation.tactile_raw`` column from
a LeRobot-format parquet episode.

Layout (one canvas per frame):
    ┌─────────────────────────────────────────────────┐
    │ timeline scrubber                       frame N │
    ├─────────────────────┬───────────────────────────┤
    │   front_chest 6×8   │        back 5×8           │
    ├──────┬──────┬───────┴──┬─────────┬──────────────┤
    │ L sh │ L arm│  R arm   │  R sh   │              │
    └──────┴──────┴──────────┴─────────┴──────────────┘

Keyboard:
    space    pause / resume
    →  /  ← : when paused, step one frame
    [  /  ] : when paused, jump back/forward 10 frames
    0–9      jump to N×10% of the timeline
    r        restart from frame 0
    q / ESC  quit

Usage:
    .venv_data_collection/bin/python gear_sonic/scripts/visualize_tactile.py \
        --parquet outputs/tactile/data/chunk-000/episode_000001.parquet
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "JuQiao") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "JuQiao"))
from jq_tactile_skin.mappings import REGIONS  # noqa: E402

TACTILE_COL = "observation.tactile_raw"
TACTILE_DIM = 256

# Per-region cell size (px) — bigger for chest/back, smaller for limbs.
CELL_SIZE = {
    "front_chest": 38,
    "back": 38,
    "left_shoulder": 32,
    "right_shoulder": 32,
    "left_arm": 32,
    "right_arm": 32,
}
PAD = 14
TIMELINE_H = 28
LABEL_H = 22
CANVAS_BG = (24, 24, 24)


def _load_tactile(parquet_path: Path) -> np.ndarray:
    df = pd.read_parquet(parquet_path, columns=[TACTILE_COL])
    arr = np.stack(df[TACTILE_COL].to_numpy())
    if arr.shape[1] != TACTILE_DIM:
        raise ValueError(f"Expected last dim {TACTILE_DIM}, got {arr.shape}")
    return arr.astype(np.uint8, copy=False)


def _region_grid(tactile_frame: np.ndarray, region) -> np.ndarray:
    idx = np.asarray(region.indices, dtype=np.int32) - 1  # 1-based -> 0-based
    return tactile_frame[idx].reshape(region.rows, region.cols)


def _render_region(grid: np.ndarray, cell_px: int, vmax: int) -> np.ndarray:
    """Upscale (rows, cols) uint8 grid to a colored (rows*cell, cols*cell, 3) BGR image."""
    if vmax <= 0:
        vmax = 1
    norm = np.clip(grid.astype(np.float32) / vmax, 0.0, 1.0)
    norm_u8 = (norm * 255.0).astype(np.uint8)
    big = cv2.resize(
        norm_u8,
        (grid.shape[1] * cell_px, grid.shape[0] * cell_px),
        interpolation=cv2.INTER_NEAREST,
    )
    color = cv2.applyColorMap(big, cv2.COLORMAP_INFERNO)
    # Draw cell gridlines for readability.
    for r in range(1, grid.shape[0]):
        y = r * cell_px
        cv2.line(color, (0, y), (color.shape[1] - 1, y), (60, 60, 60), 1)
    for c in range(1, grid.shape[1]):
        x = c * cell_px
        cv2.line(color, (x, 0), (x, color.shape[0] - 1), (60, 60, 60), 1)
    return color


def _put_label(canvas: np.ndarray, text: str, x: int, y: int, scale: float = 0.5):
    cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (220, 220, 220), 1, cv2.LINE_AA)


def _compose_frame(
    tactile_frame: np.ndarray,
    vmax: int,
    frame_idx: int,
    n_frames: int,
    paused: bool,
    region_max_series: np.ndarray,
) -> np.ndarray:
    """Build one full canvas image for the given tactile frame."""
    grids = {r.key: _region_grid(tactile_frame, r) for r in REGIONS}
    renders = {
        key: _render_region(grids[key], CELL_SIZE[key], vmax) for key in grids
    }

    # Row 1: front_chest + back side by side.
    fc = renders["front_chest"]
    bk = renders["back"]
    row1_h = max(fc.shape[0], bk.shape[0]) + LABEL_H
    row1_w = fc.shape[1] + PAD + bk.shape[1]

    # Row 2: left_shoulder | left_arm | right_arm | right_shoulder.
    limb_keys = ["left_shoulder", "left_arm", "right_arm", "right_shoulder"]
    limbs = [renders[k] for k in limb_keys]
    row2_h = max(im.shape[0] for im in limbs) + LABEL_H
    row2_w = sum(im.shape[1] for im in limbs) + PAD * (len(limbs) - 1)

    content_w = max(row1_w, row2_w)
    canvas_w = content_w + PAD * 2
    canvas_h = TIMELINE_H + PAD + row1_h + PAD + row2_h + PAD
    canvas = np.full((canvas_h, canvas_w, 3), CANVAS_BG, dtype=np.uint8)

    # --- Timeline strip ---
    tl_y0 = PAD // 2
    tl_y1 = tl_y0 + TIMELINE_H - 12
    tl_x0 = PAD
    tl_x1 = canvas_w - PAD
    cv2.rectangle(canvas, (tl_x0, tl_y0), (tl_x1, tl_y1), (60, 60, 60), 1)
    # Region-max bar across the timeline (8-bit normalized).
    if n_frames > 1:
        bar_w = tl_x1 - tl_x0
        bar_h = tl_y1 - tl_y0 - 2
        series_norm = np.clip(region_max_series.astype(np.float32) / max(vmax, 1), 0, 1)
        # Resample series to bar_w pixels.
        xp = np.linspace(0, n_frames - 1, bar_w)
        resampled = np.interp(xp, np.arange(n_frames), series_norm)
        for i, v in enumerate(resampled):
            h = max(1, int(v * bar_h))
            cv2.line(
                canvas,
                (tl_x0 + i, tl_y1 - 1),
                (tl_x0 + i, tl_y1 - 1 - h),
                (90, 90, 140),
                1,
            )
        # Current-frame cursor.
        cur_x = tl_x0 + int((frame_idx / max(n_frames - 1, 1)) * bar_w)
        cv2.line(canvas, (cur_x, tl_y0), (cur_x, tl_y1), (255, 255, 255), 1)

    _put_label(
        canvas,
        f"frame {frame_idx + 1:>5d}/{n_frames}   max={int(tactile_frame.max())}/{vmax}   "
        f"{'PAUSED' if paused else 'PLAYING'}",
        PAD,
        TIMELINE_H + 4,
    )

    # --- Row 1 (chest / back) ---
    y0 = TIMELINE_H + PAD + LABEL_H
    x = PAD + (content_w - row1_w) // 2
    canvas[y0 : y0 + fc.shape[0], x : x + fc.shape[1]] = fc
    _put_label(canvas, f"front_chest  {grids['front_chest'].shape[0]}x{grids['front_chest'].shape[1]}", x, y0 - 4)
    x += fc.shape[1] + PAD
    canvas[y0 : y0 + bk.shape[0], x : x + bk.shape[1]] = bk
    _put_label(canvas, f"back  {grids['back'].shape[0]}x{grids['back'].shape[1]}", x, y0 - 4)

    # --- Row 2 (limbs) ---
    y0 = TIMELINE_H + PAD + row1_h + PAD + LABEL_H
    x = PAD + (content_w - row2_w) // 2
    for key, im in zip(limb_keys, limbs):
        canvas[y0 : y0 + im.shape[0], x : x + im.shape[1]] = im
        _put_label(canvas, key, x, y0 - 4)
        x += im.shape[1] + PAD

    return canvas


def play(parquet_path: Path, fps: float, window: str) -> None:
    tactile = _load_tactile(parquet_path)
    n_frames = tactile.shape[0]
    vmax = max(int(tactile.max()), 1)
    region_max_series = tactile.max(axis=1)
    print(
        f"Loaded {n_frames} frames from {parquet_path}; "
        f"vmax={vmax}, non-zero frames={(region_max_series > 0).sum()}/{n_frames}"
    )
    print("Keys: SPACE pause | ←/→ step | [/] ±10 | 0-9 jump % | r restart | q quit")

    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    frame_idx = 0
    paused = False
    last_tick = time.monotonic()
    period = 1.0 / fps

    while True:
        canvas = _compose_frame(
            tactile[frame_idx],
            vmax,
            frame_idx,
            n_frames,
            paused,
            region_max_series,
        )
        cv2.imshow(window, canvas)

        # Pick a wait time that lets us hit the target fps when playing,
        # and stay responsive (~30ms) when paused.
        if paused:
            wait_ms = 30
        else:
            elapsed = time.monotonic() - last_tick
            remaining = period - elapsed
            wait_ms = max(1, int(remaining * 1000))

        key = cv2.waitKey(wait_ms) & 0xFFFF

        if key in (ord("q"), 27):  # q or ESC
            break
        elif key == ord(" "):
            paused = not paused
            last_tick = time.monotonic()
        elif key == ord("r"):
            frame_idx = 0
            last_tick = time.monotonic()
        elif key in (81, 2424832, ord(",")):  # left arrow (xcb / win) or comma
            if paused:
                frame_idx = max(0, frame_idx - 1)
        elif key in (83, 2555904, ord(".")):  # right arrow or period
            if paused:
                frame_idx = min(n_frames - 1, frame_idx + 1)
        elif key == ord("["):
            frame_idx = max(0, frame_idx - 10)
            last_tick = time.monotonic()
        elif key == ord("]"):
            frame_idx = min(n_frames - 1, frame_idx + 10)
            last_tick = time.monotonic()
        elif ord("0") <= key <= ord("9"):
            frac = (key - ord("0")) / 10.0
            frame_idx = int(frac * (n_frames - 1))
            last_tick = time.monotonic()
        elif key == 0xFFFF:
            pass  # no key

        if not paused:
            now = time.monotonic()
            if now - last_tick >= period:
                frame_idx = (frame_idx + 1) % n_frames
                last_tick = now

    cv2.destroyWindow(window)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--parquet", type=Path, required=True, help="Episode parquet file")
    parser.add_argument(
        "--fps",
        type=float,
        default=50.0,
        help="Playback rate (Hz). Recording is 50 Hz by default.",
    )
    parser.add_argument(
        "--window",
        default="tactile",
        help="OpenCV window title.",
    )
    args = parser.parse_args()
    if not args.parquet.is_file():
        parser.error(f"Not a file: {args.parquet}")
    play(args.parquet, args.fps, args.window)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
