#!/usr/bin/env python3
"""Synchronized replay of one recorded episode: MuJoCo state + tactile heatmap.

Opens the MuJoCo viewer and the OpenCV tactile window together, both driven by
a single playback loop and a single frame index, so the two modalities never
drift apart — solving the timing mismatch you get from launching
``replay_state_mujoco.py`` and ``visualize_tactile.py`` as two processes.

``observation.state`` and the tactile columns come from the same rows of the
same parquet, so one frame index addresses both directly — no resampling.  A
triple-device recording renders vest, left-arm, and right-arm windows from the
same playback clock.

This reuses the helpers from the two standalone scripts (kept intact):
``replay_state_mujoco.py`` for the qpos mapping and ``visualize_tactile.py``
for the tactile canvas renderer.

Usage (from repo root, with .venv_sim active — it has mujoco + cv2 + pandas):
    python gear_sonic/scripts/replay_episode.py \
        outputs/2026-05-14-18-07-13/data/chunk-000/episode_000000.parquet --loop

Keyboard (focus the tactile window):
    space    pause / resume
    -> / <- : when paused, step one frame (also , / .)
    up/down  previous / next episode (with --playlist-controls)
    [ / ]    jump back / forward 10 frames
    0-9      jump to N x 10% of the timeline
    r        restart from the first frame
    q / ESC  quit
"""

from __future__ import annotations

import argparse
import json
import os
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
from run_tactile_viewer import _arm_canvas  # noqa: E402
from visualize_tactile import TACTILE_COL, TACTILE_DIM, _compose_frame  # noqa: E402


TRIPLE_TACTILE_COLUMNS = {
    "vest": "observation.tactile_vest",
    "left_arm": "observation.tactile_left_arm",
    "right_arm": "observation.tactile_right_arm",
}

PREVIOUS_EPISODE_EXIT = 20
NEXT_EPISODE_EXIT = 21


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
        "--exit-at-end",
        action="store_true",
        help="exit successfully after the final frame (for dataset playlists)",
    )
    ap.add_argument(
        "--playlist-controls",
        action="store_true",
        help="return dedicated exit codes for Up/Down episode navigation",
    )
    ap.add_argument(
        "--hard-exit-at-boundary",
        action="store_true",
        help="let the OS tear down mixed GLFW/Qt GUI state at playlist boundaries",
    )
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
        "--arrange-windows",
        action="store_true",
        help="place triple tactile windows along the right side of a 1920x1080 desktop",
    )
    ap.add_argument(
        "--no-tactile",
        action="store_true",
        help="skip the tactile window (MuJoCo-only replay)",
    )
    ap.add_argument(
        "--tactile-mode",
        choices=("auto", "single", "triple"),
        default="auto",
        help="auto uses vest+left_arm+right_arm when all are present; "
        "single renders --tactile-key; triple requires all three devices",
    )
    ap.add_argument(
        "--tactile-key",
        default=None,
        help=f"single tactile parquet column (default {TACTILE_COL}); "
        "specifying this forces one-window playback in auto mode",
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
    key: int,
    frame_idx: int,
    paused: bool,
    last_tick: float,
    n_frames: int,
    playlist_controls: bool = False,
) -> tuple[int, bool, float, bool, int]:
    """Apply a keypress; return frame, pause, clock, quit, episode delta."""
    quit_requested = False
    episode_delta = 0
    if key in (ord("q"), 27):  # q or ESC
        quit_requested = True
    elif playlist_controls and key in (82, 65362, 2490368):  # up arrow
        episode_delta = -1
    elif playlist_controls and key in (84, 65364, 2621440):  # down arrow
        episode_delta = 1
    elif key == ord(" "):
        paused = not paused
        last_tick = time.monotonic()
    elif key == ord("r"):
        frame_idx = 0
        last_tick = time.monotonic()
    elif key in (81, 65361, 2424832, ord(",")):  # left arrow or comma
        if paused:
            frame_idx = max(0, frame_idx - 1)
    elif key in (83, 65363, 2555904, ord(".")):  # right arrow or period
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
    return frame_idx, paused, last_tick, quit_requested, episode_delta


def hard_exit(status: int) -> None:
    """Exit without running conflicting GLFW/Qt process destructors."""
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(status)


def main() -> int:
    args = parse_args()
    if args.loop and args.exit_at_end:
        print("[replay] --loop and --exit-at-end are mutually exclusive", file=sys.stderr)
        return 1
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
    tactile_frames: dict[str, np.ndarray] = {}
    tactile_vmax: dict[str, int] = {}
    tactile_max_series: dict[str, np.ndarray] = {}
    if not args.no_tactile:
        triple_available = all(
            column in df.columns for column in TRIPLE_TACTILE_COLUMNS.values()
        )
        if args.tactile_mode == "triple" and args.tactile_key is not None:
            print(
                "[replay] --tactile-mode triple cannot be combined with "
                "--tactile-key",
                file=sys.stderr,
            )
            return 1
        if args.tactile_mode == "triple":
            missing_tactile = [
                column
                for column in TRIPLE_TACTILE_COLUMNS.values()
                if column not in df.columns
            ]
            if missing_tactile:
                print(
                    f"[replay] triple tactile columns missing: {missing_tactile}",
                    file=sys.stderr,
                )
                return 1
            tactile_columns = TRIPLE_TACTILE_COLUMNS
        elif args.tactile_key is not None:
            device = args.tactile_key.removeprefix("observation.tactile_")
            tactile_columns = {device: args.tactile_key}
        elif args.tactile_mode == "auto" and triple_available:
            tactile_columns = TRIPLE_TACTILE_COLUMNS
        else:
            tactile_columns = {"body": TACTILE_COL}

        for device, tactile_col in tactile_columns.items():
            if tactile_col not in df.columns:
                print(
                    f"[replay] no {tactile_col} column; MuJoCo-only replay",
                    file=sys.stderr,
                )
                tactile_frames.clear()
                break
            tactile = np.stack(df[tactile_col].to_numpy())
            if tactile.shape != (n_frames, TACTILE_DIM):
                print(
                    f"[replay] {tactile_col} shape {tactile.shape} != "
                    f"({n_frames}, {TACTILE_DIM})",
                    file=sys.stderr,
                )
                return 1
            tactile = tactile.astype(np.uint8, copy=False)
            tactile_frames[device] = tactile
            tactile_vmax[device] = max(int(tactile.max()), 1)
            tactile_max_series[device] = tactile.max(axis=1)
    has_tactile = bool(tactile_frames)

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
        f"~ {n_frames * period:.1f}s; "
        f"tactile={'/'.join(tactile_frames) if has_tactile else 'off'}; "
        f"loop={args.loop}"
    )

    window = args.window
    tactile_windows: dict[str, str] = {}
    if has_tactile:
        multiple_tactile = len(tactile_frames) > 1
        for device in tactile_frames:
            title = f"{window}: {device}" if multiple_tactile else window
            tactile_windows[device] = title
            cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)
        if args.arrange_windows and multiple_tactile:
            positions = {
                "vest": (1240, 0),
                "left_arm": (1120, 530),
                "right_arm": (1520, 530),
            }
            for device, position in positions.items():
                if device in tactile_windows:
                    cv2.moveWindow(tactile_windows[device], *position)
        print(
            "[replay] keys (focus any tactile window): SPACE pause | <-/-> step | "
            + ("UP previous episode | DOWN next episode | " if args.playlist_controls else "")
            + "[/] +-10 | 0-9 jump% | r restart | q quit"
        )

    frame_idx = 0
    paused = False
    last_tick = time.monotonic()
    quit_requested = False
    ended = False
    completed = False
    episode_delta = 0

    with mujoco.viewer.launch_passive(model, data) as viewer:
        try:
            while viewer.is_running() and not quit_requested:
                # --- Drive both modalities for the current shared frame ---
                apply_state(data, df.iloc[frame_idx], indices, free_joint_qpos_adr, has_root_orient)
                mujoco.mj_forward(model, data)
                viewer.sync()

                if has_tactile:
                    for device, tactile in tactile_frames.items():
                        playback_state = "PAUSED" if paused else "PLAYING"
                        status = (
                            f"{playback_state}  frame={frame_idx + 1}/{n_frames}  "
                            f"max={int(tactile[frame_idx].max())}"
                        )
                        if device in ("left_arm", "right_arm"):
                            canvas = _arm_canvas(
                                tactile[frame_idx],
                                device,
                                tactile_vmax[device],
                                status,
                            )
                        else:
                            canvas = _compose_frame(
                                tactile[frame_idx],
                                tactile_vmax[device],
                                frame_idx,
                                n_frames,
                                paused,
                                tactile_max_series[device],
                            )
                        cv2.imshow(tactile_windows[device], canvas)
                    # waitKeyEx preserves backend-specific extended arrow key codes.
                    if paused:
                        wait_ms = 30
                    else:
                        remaining = period - (time.monotonic() - last_tick)
                        wait_ms = max(1, int(remaining * 1000))
                    key = cv2.waitKeyEx(wait_ms)
                    (
                        frame_idx,
                        paused,
                        last_tick,
                        quit_requested,
                        requested_episode_delta,
                    ) = handle_key(
                        key,
                        frame_idx,
                        paused,
                        last_tick,
                        n_frames,
                        args.playlist_controls,
                    )
                    if requested_episode_delta:
                        if args.hard_exit_at_boundary:
                            hard_exit(
                                PREVIOUS_EPISODE_EXIT
                                if requested_episode_delta < 0
                                else NEXT_EPISODE_EXIT
                            )
                        episode_delta = requested_episode_delta
                        break
                    if quit_requested:
                        if args.hard_exit_at_boundary:
                            hard_exit(130)
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
                            elif args.exit_at_end:
                                completed = True
                                print("[replay] reached end; advancing playlist")
                                if args.hard_exit_at_boundary:
                                    hard_exit(0)
                                break
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
            if args.hard_exit_at_boundary:
                hard_exit(130)
        except KeyboardInterrupt:
            quit_requested = True
            print("\n[replay] interrupted")
            if args.hard_exit_at_boundary:
                hard_exit(130)
        finally:
            # OpenCV's Qt backend and MuJoCo's GLFW backend share the X11
            # connection. Tear down Qt windows while GLFW is still alive;
            # reversing this order can abort at episode boundaries.
            if has_tactile:
                cv2.destroyAllWindows()
                cv2.waitKey(1)

    if episode_delta < 0:
        return PREVIOUS_EPISODE_EXIT
    if episode_delta > 0:
        return NEXT_EPISODE_EXIT
    if args.exit_at_end and not completed:
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
