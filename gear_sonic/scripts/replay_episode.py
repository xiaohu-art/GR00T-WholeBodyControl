#!/usr/bin/env python3
"""Synchronized replay of one recorded episode: MuJoCo state + tactile heatmap.

Opens the MuJoCo viewer and the OpenCV tactile window together, both driven by
a single playback loop and a single frame index, so the two modalities never
drift apart — solving the timing mismatch you get from launching
``replay_state_mujoco.py`` and ``visualize_tactile.py`` as two processes.

``observation.state`` and ``observation.tactile_raw`` come from the same rows
of the same parquet, so one frame index addresses both directly — no
resampling.

This reuses the helpers from the two standalone scripts (kept intact):
``replay_state_mujoco.py`` for the qpos mapping and ``visualize_tactile.py``
for the tactile canvas renderer.

Usage (from repo root, with .venv_sim active — it has mujoco + cv2 + pandas):
    python gear_sonic/scripts/replay_episode.py \
        outputs/2026-05-14-18-07-13/data/chunk-000/episode_000000.parquet --loop

Keyboard (focus the tactile window):
    space    pause / resume
    -> / <- : when paused, step one frame (also , / .)
    [ / ]    jump back / forward 10 frames
    0-9      jump to N x 10% of the timeline
    r        restart from the first frame
    q / ESC  quit
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import pandas as pd

try:
    import mujoco
    import mujoco.viewer
except ImportError as exc:
    raise SystemExit(
        "mujoco not installed in this venv. Use .venv_sim or pip install mujoco."
    ) from exc

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# Reuse helpers from the two standalone replay scripts (left intact on purpose).
from replay_state_mujoco import (  # noqa: E402
    DEFAULT_SCENE,
    build_qpos_index_map,
    find_dataset_info,
    load_joint_names,
)
from visualize_tactile import TACTILE_COL, TACTILE_DIM, _compose_frame  # noqa: E402


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("parquet", type=Path, help="Path to episode_*.parquet")
    ap.add_argument(
        "--scene",
        type=Path,
        default=DEFAULT_SCENE,
        help=f"MuJoCo scene XML (default: {DEFAULT_SCENE})",
    )
    ap.add_argument(
        "--rate",
        type=float,
        default=0.0,
        help="playback rate Hz; 0 = use dataset fps from meta/info.json (default 50)",
    )
    ap.add_argument("--start-frame", type=int, default=0)
    ap.add_argument("--end-frame", type=int, default=-1, help="exclusive; -1 = end")
    ap.add_argument("--loop", action="store_true", help="loop playback")
    ap.add_argument(
        "--joint-names",
        nargs="+",
        help="Explicit joint name list, overrides meta/info.json lookup",
    )
    ap.add_argument(
        "--base-height",
        type=float,
        default=0.793,
        help="z height of the floating base (m), default 0.793",
    )
    ap.add_argument(
        "--use-root-orientation",
        action="store_true",
        help="apply observation.root_orientation (wxyz) to the floating base",
    )
    ap.add_argument("--window", default="tactile", help="OpenCV tactile window title")
    ap.add_argument(
        "--no-tactile",
        action="store_true",
        help="skip the tactile window (MuJoCo-only replay)",
    )
    return ap.parse_args()


def resolve_rate(parquet_path: Path, rate_arg: float) -> float:
    """Playback rate: the --rate arg if positive, else dataset fps, else 50."""
    if rate_arg > 0:
        return rate_arg
    try:
        with find_dataset_info(parquet_path).open() as f:
            return float(json.load(f).get("fps", 50))
    except Exception:
        return 50.0


def apply_state(
    data,
    row,
    indices: list[int],
    free_adr: int | None,
    has_root_orient: bool,
) -> None:
    """Write one parquet row's joint angles (and optional root quat) into qpos."""
    state = np.asarray(row["observation.state"], dtype=np.float64)
    for jidx, qadr in enumerate(indices):
        if qadr >= 0:
            data.qpos[qadr] = state[jidx]
    if has_root_orient and free_adr is not None:
        quat_wxyz = np.asarray(row["observation.root_orientation"], dtype=np.float64)
        if quat_wxyz.shape == (4,):
            data.qpos[free_adr + 3 : free_adr + 7] = quat_wxyz


def handle_key(
    key: int, frame_idx: int, paused: bool, last_tick: float, n_frames: int
) -> tuple[int, bool, float, bool]:
    """Apply one OpenCV keypress; return (frame_idx, paused, last_tick, quit)."""
    quit_requested = False
    if key in (ord("q"), 27):  # q or ESC
        quit_requested = True
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
    return frame_idx, paused, last_tick, quit_requested


def main() -> int:
    args = parse_args()
    if not args.parquet.exists():
        print(f"[replay] parquet not found: {args.parquet}", file=sys.stderr)
        return 1
    if not args.scene.exists():
        print(f"[replay] scene xml not found: {args.scene}", file=sys.stderr)
        return 1

    df = pd.read_parquet(args.parquet)
    if "observation.state" not in df.columns:
        print("[replay] parquet missing observation.state column", file=sys.stderr)
        return 1

    # --- Frame range: slice once so frame_idx is 0-based over the selection ---
    end = len(df) if args.end_frame < 0 else min(args.end_frame, len(df))
    start = max(0, args.start_frame)
    if start >= end:
        print(f"[replay] start {start} >= end {end}", file=sys.stderr)
        return 1
    df = df.iloc[start:end].reset_index(drop=True)
    n_frames = len(df)

    # --- Joint name -> qpos mapping ---
    joint_names = load_joint_names(args.parquet, args.joint_names)
    n_state = len(joint_names)
    sample_state = np.asarray(df["observation.state"].iloc[0])
    if sample_state.shape[0] != n_state:
        print(
            f"[replay] joint_names has {n_state} entries but observation.state "
            f"has {sample_state.shape[0]}; mismatch",
            file=sys.stderr,
        )
        return 1

    # --- Tactile setup (optional / graceful) ---
    has_tactile = (not args.no_tactile) and (TACTILE_COL in df.columns)
    tactile: np.ndarray | None = None
    vmax = 1
    region_max_series: np.ndarray | None = None
    if has_tactile:
        tactile = np.stack(df[TACTILE_COL].to_numpy())
        if tactile.shape[1] != TACTILE_DIM:
            print(
                f"[replay] {TACTILE_COL} dim {tactile.shape[1]} != {TACTILE_DIM}; "
                "disabling tactile window",
                file=sys.stderr,
            )
            has_tactile = False
            tactile = None
        else:
            tactile = tactile.astype(np.uint8, copy=False)
            vmax = max(int(tactile.max()), 1)
            region_max_series = tactile.max(axis=1)
    elif not args.no_tactile:
        print(f"[replay] no {TACTILE_COL} column; MuJoCo-only replay")

    rate = resolve_rate(args.parquet, args.rate)
    period = 1.0 / rate

    # --- MuJoCo scene ---
    print(f"[replay] loading scene: {args.scene}")
    model = mujoco.MjModel.from_xml_path(str(args.scene))
    data = mujoco.MjData(model)

    indices, missing = build_qpos_index_map(model, joint_names)
    if missing:
        print(
            f"[replay] WARNING: {len(missing)} joint(s) in observation.state "
            f"not found in scene; will be skipped: {missing[:10]}"
            + (" ..." if len(missing) > 10 else "")
        )
    n_mapped = sum(1 for i in indices if i >= 0)
    print(f"[replay] mapped {n_mapped}/{n_state} joints to qpos")

    # Position the floating base.
    free_joint_qpos_adr: int | None = None
    if model.njnt > 0 and model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE:
        free_joint_qpos_adr = model.jnt_qposadr[0]
        data.qpos[free_joint_qpos_adr + 2] = args.base_height
        data.qpos[free_joint_qpos_adr + 3 : free_joint_qpos_adr + 7] = [1.0, 0.0, 0.0, 0.0]

    has_root_orient = args.use_root_orientation and "observation.root_orientation" in df.columns

    print(
        f"[replay] frames [{start},{end}) ({n_frames}) @ {rate:.1f}Hz "
        f"~ {n_frames * period:.1f}s; tactile={'on' if has_tactile else 'off'}; "
        f"loop={args.loop}"
    )

    window = args.window
    if has_tactile:
        cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
        print(
            "[replay] keys (focus tactile window): SPACE pause | <-/-> step | "
            "[/] +-10 | 0-9 jump% | r restart | q quit"
        )

    frame_idx = 0
    paused = False
    last_tick = time.monotonic()
    quit_requested = False
    ended = False

    with mujoco.viewer.launch_passive(model, data) as viewer:
        try:
            while viewer.is_running() and not quit_requested:
                # --- Drive both modalities for the current shared frame ---
                apply_state(data, df.iloc[frame_idx], indices, free_joint_qpos_adr, has_root_orient)
                mujoco.mj_forward(model, data)
                viewer.sync()

                if has_tactile:
                    assert tactile is not None and region_max_series is not None
                    canvas = _compose_frame(
                        tactile[frame_idx], vmax, frame_idx, n_frames, paused, region_max_series
                    )
                    cv2.imshow(window, canvas)
                    # cv2.waitKey is the single pacing + keyboard-input mechanism.
                    if paused:
                        wait_ms = 30
                    else:
                        remaining = period - (time.monotonic() - last_tick)
                        wait_ms = max(1, int(remaining * 1000))
                    key = cv2.waitKey(wait_ms) & 0xFFFF
                    frame_idx, paused, last_tick, quit_requested = handle_key(
                        key, frame_idx, paused, last_tick, n_frames
                    )
                    if quit_requested:
                        break
                else:
                    # No keyboard source — pace with sleep; quit via window close.
                    remaining = period - (time.monotonic() - last_tick)
                    time.sleep(0.03 if paused else max(0.001, remaining))

                # --- Advance the shared frame index ---
                if not paused:
                    now = time.monotonic()
                    if now - last_tick >= period:
                        last_tick = now
                        if frame_idx + 1 >= n_frames:
                            if args.loop:
                                frame_idx = 0
                                ended = False
                            else:
                                frame_idx = n_frames - 1
                                paused = True
                                if not ended:
                                    print(
                                        "[replay] reached end; paused "
                                        "(r restart, q quit, or close a window)"
                                    )
                                    ended = True
                        else:
                            frame_idx += 1
        except KeyboardInterrupt:
            print("\n[replay] interrupted")

    if has_tactile:
        cv2.destroyWindow(window)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
