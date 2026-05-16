#!/usr/bin/env python3
"""Replay action.motion_token from a recorded parquet episode into C++ deploy.

Reads one parquet episode written by run_data_exporter, then re-publishes its
``action.motion_token`` (plus ``teleop.left_hand_joints`` / ``teleop.right_hand_joints``)
on the same ZMQ ``pose`` topic that VLA inference uses. The C++ deploy decodes
the tokens into joint targets, so this lets you replay a recorded trajectory
through the same control stack — without invoking the policy.

Usage (from repo root):
    source .venv_inference/bin/activate
    python gear_sonic/scripts/replay_motion_tokens.py \
        outputs/2026-05-14-18-07-13/data/chunk-000/episode_000000.parquet

Prerequisites:
    - C++ deploy is running (e.g. via ``launch_inference.py`` or manually) and
      bound as a SUB on the action ZMQ topic.
    - This script binds the PUB socket; you may need to stop any other
      publisher (e.g. ``run_vla_inference.py``) before starting it.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import zmq

# Local imports (resolved by walking up from this script).
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gear_sonic.utils.teleop.zmq.zmq_planner_sender import (  # noqa: E402
    build_command_message,
    pack_pose_message,
)


def pack_latent_action_message(
    motion_token: np.ndarray,
    frame_index: np.ndarray,
    left_hand_joints: np.ndarray | None = None,
    right_hand_joints: np.ndarray | None = None,
) -> bytes:
    """Same packing as run_vla_inference.pack_latent_action_message."""
    motion_token = np.asarray(motion_token, dtype=np.float32)
    if motion_token.ndim == 1:
        motion_token = motion_token.reshape(1, -1)

    frame_index = np.asarray(frame_index, dtype=np.int64)
    if frame_index.ndim == 0:
        frame_index = np.array([frame_index], dtype=np.int64)
    elif frame_index.shape[0] != 1:
        frame_index = frame_index[:1]

    pose_data = {
        "token_state": motion_token,
        "frame_index": frame_index,
    }
    if left_hand_joints is not None:
        lh = np.asarray(left_hand_joints, dtype=np.float32).reshape(1, 7)
        pose_data["left_hand_joints"] = lh
    if right_hand_joints is not None:
        rh = np.asarray(right_hand_joints, dtype=np.float32).reshape(1, 7)
        pose_data["right_hand_joints"] = rh

    return pack_pose_message(pose_data, topic="pose", version=4)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Replay action.motion_token from a parquet episode")
    ap.add_argument("parquet", type=Path, help="Path to episode_*.parquet")
    ap.add_argument("--zmq-host", default="localhost", help="bind host (default localhost)")
    ap.add_argument("--zmq-port", type=int, default=5556, help="bind port (default 5556)")
    ap.add_argument("--rate", type=float, default=50.0, help="publish rate Hz (default 50)")
    ap.add_argument("--start-frame", type=int, default=0, help="starting frame index (inclusive)")
    ap.add_argument(
        "--end-frame",
        type=int,
        default=-1,
        help="end frame index (exclusive); -1 = end of episode",
    )
    ap.add_argument("--loop", action="store_true", help="loop the replay until Ctrl-C")
    ap.add_argument(
        "--send-start-cmd",
        action="store_true",
        help="Send a C++ deploy start command (pose mode) before replay",
    )
    ap.add_argument(
        "--no-hands",
        action="store_true",
        help="do not publish left/right hand joints (let C++ keep current hand state)",
    )
    ap.add_argument(
        "--warmup-sec",
        type=float,
        default=0.5,
        help="wait this long after binding PUB before publishing (lets SUB connect)",
    )
    return ap.parse_args()


def main() -> int:
    args = parse_args()

    if not args.parquet.exists():
        print(f"[replay] parquet not found: {args.parquet}", file=sys.stderr)
        return 1

    df = pd.read_parquet(args.parquet)
    required = {"action.motion_token", "frame_index"}
    missing = required - set(df.columns)
    if missing:
        print(f"[replay] parquet missing columns: {missing}", file=sys.stderr)
        return 1

    end = len(df) if args.end_frame < 0 else min(args.end_frame, len(df))
    start = max(0, args.start_frame)
    if start >= end:
        print(f"[replay] start {start} >= end {end}, nothing to do", file=sys.stderr)
        return 1
    n_frames = end - start
    period = 1.0 / args.rate
    duration_sec = n_frames * period
    print(
        f"[replay] {args.parquet.name}: replaying frames [{start},{end}) "
        f"= {n_frames} frames @ {args.rate}Hz ≈ {duration_sec:.1f}s"
    )

    ctx = zmq.Context()
    sock = ctx.socket(zmq.PUB)
    bind_str = f"tcp://{args.zmq_host}:{args.zmq_port}"
    sock.bind(bind_str)
    print(f"[replay] PUB bound at {bind_str}, waiting {args.warmup_sec}s for SUB to connect...")
    time.sleep(args.warmup_sec)

    if args.send_start_cmd:
        cmd = build_command_message(start=True, stop=False, planner=False)
        sock.send(cmd)
        print("[replay] sent C++ start command (pose mode)")
        time.sleep(0.05)

    has_hands = (not args.no_hands) and (
        "teleop.left_hand_joints" in df.columns
        and "teleop.right_hand_joints" in df.columns
    )

    try:
        pass_count = 0
        while True:
            pass_count += 1
            t_pass_start = time.monotonic()
            for i in range(start, end):
                row = df.iloc[i]
                motion_token = np.asarray(row["action.motion_token"], dtype=np.float32)
                frame_idx = np.array([int(row["frame_index"])], dtype=np.int64)
                lh = rh = None
                if has_hands:
                    lh = np.asarray(row["teleop.left_hand_joints"], dtype=np.float32)
                    rh = np.asarray(row["teleop.right_hand_joints"], dtype=np.float32)

                msg = pack_latent_action_message(
                    motion_token=motion_token,
                    frame_index=frame_idx,
                    left_hand_joints=lh,
                    right_hand_joints=rh,
                )
                sock.send(msg)

                t_target = t_pass_start + (i - start + 1) * period
                sleep_for = t_target - time.monotonic()
                if sleep_for > 0:
                    time.sleep(sleep_for)

                if (i - start) % 50 == 0:
                    print(
                        f"[replay] pass {pass_count}: frame {i - start + 1}/{n_frames}"
                    )

            if not args.loop:
                break
    except KeyboardInterrupt:
        print("\n[replay] interrupted")
    finally:
        try:
            sock.close()
        except Exception:
            pass
        try:
            ctx.term()
        except Exception:
            pass
        print("[replay] shutdown")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
