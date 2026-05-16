#!/usr/bin/env python3
"""Open-loop diagnostic for a fine-tuned VLA policy.

Picks one frame from a recorded episode (parquet + matching MP4), builds the
exact observation dict that ``run_vla_inference.py`` would build, queries the
running PolicyServer, and compares the predicted ``motion_token`` chunk
against the recorded ``action.motion_token`` chunk from frame ``t`` to
``t + action_horizon``.

The key diagnostic:
- If predicted matches recorded closely → model is faithful; "动作飞快" is
  a deploy-side issue (action_publish_rate / control gain), NOT the model.
- If predicted diverges from recorded → model / observation pipeline broken.

Usage (from repo root):
    source .venv_inference/bin/activate
    python gear_sonic/scripts/openloop_eval.py \
        --parquet outputs/2026-05-14-18-07-13/data/chunk-000/episode_000000.parquet \
        --video outputs/2026-05-14-18-07-13/videos/chunk-000/observation.images.ego_view/episode_000000.mp4 \
        --prompt "lift pillow" \
        --frame-indices 50 100 150 \
        --action-horizon 40
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# State dim slices, matching modality.json + features_sonic_vla.py joint order.
STATE_SLICES = {
    "left_leg":   (0, 6),
    "right_leg":  (6, 12),
    "waist":      (12, 15),
    "left_arm":   (15, 22),
    "left_hand":  (22, 29),
    "right_arm":  (29, 36),
    "right_hand": (36, 43),
}


def read_video_frame(video_path: Path, frame_idx: int) -> np.ndarray:
    """Return one frame (H, W, 3) uint8 RGB from the MP4 at index ``frame_idx``."""
    try:
        import cv2
    except ImportError as exc:
        raise SystemExit("opencv-python required: pip install opencv-python") from exc
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"could not open video: {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"could not read frame {frame_idx} from {video_path}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def build_observation(
    df: pd.DataFrame,
    frame_idx: int,
    video_path: Path,
    prompt: str,
) -> dict:
    """Build the observation dict in the format Gr00tPolicy expects."""
    row = df.iloc[frame_idx]
    state_arr = np.asarray(row["observation.state"], dtype=np.float32)
    img = read_video_frame(video_path, frame_idx)
    img_5d = img[np.newaxis, np.newaxis]  # (1, 1, H, W, 3)

    state_dict: dict = {}
    for key, (lo, hi) in STATE_SLICES.items():
        state_dict[key] = state_arr[lo:hi][np.newaxis, np.newaxis]

    pg = np.asarray(row["observation.projected_gravity"], dtype=np.float32)
    state_dict["projected_gravity"] = pg[np.newaxis, np.newaxis]

    obs = {
        "video": {"ego_view": img_5d},
        "state": state_dict,
        "language": {"annotation.human.task_description": [[prompt]]},
        "q": state_arr[np.newaxis, np.newaxis],
    }
    return obs


def query_policy(host: str, port: int, observation: dict, embodiment_tag: str | None) -> dict:
    from gr00t.policy.server_client import PolicyClient

    client = PolicyClient(host=host, port=port)
    if not client.ping():
        raise SystemExit(f"PolicyServer not reachable at {host}:{port}")
    options = {"embodiment_tag": embodiment_tag} if embodiment_tag else None
    action, _info = client.get_action(observation, options=options)
    return action


def extract_predicted_token(action: dict) -> np.ndarray:
    """Return predicted motion_token chunk as (horizon, 64)."""
    for k in ("motion_token", "action.motion_token"):
        if k in action:
            v = np.asarray(action[k], dtype=np.float32)
            if v.ndim == 3:
                v = v[0]  # drop batch
            if v.ndim == 1:
                v = v[None, :]
            return v
    raise SystemExit(f"motion_token not in policy output: keys={list(action.keys())}")


def compare(predicted: np.ndarray, recorded: np.ndarray) -> dict:
    """Return dict of comparison stats. Shapes must broadcast on (H, D)."""
    h = min(predicted.shape[0], recorded.shape[0])
    p = predicted[:h]
    r = recorded[:h]
    diff = p - r
    abs_err = np.abs(diff)
    return {
        "horizon": h,
        "per_position_mae": abs_err.mean(axis=1),       # (H,)
        "per_position_rmse": np.sqrt((diff ** 2).mean(axis=1)),  # (H,)
        "overall_mae": float(abs_err.mean()),
        "overall_rmse": float(np.sqrt((diff ** 2).mean())),
        "predicted_norm_per_pos": np.linalg.norm(p, axis=1),
        "recorded_norm_per_pos": np.linalg.norm(r, axis=1),
        "predicted_range": (float(p.min()), float(p.max())),
        "recorded_range": (float(r.min()), float(r.max())),
    }


def print_report(frame_idx: int, stats: dict) -> None:
    h = stats["horizon"]
    print(f"\n── frame {frame_idx} ─────────────────────────────")
    print(f"  predicted token range: [{stats['predicted_range'][0]:.3f}, {stats['predicted_range'][1]:.3f}]")
    print(f"  recorded  token range: [{stats['recorded_range'][0]:.3f}, {stats['recorded_range'][1]:.3f}]")
    print(f"  overall MAE: {stats['overall_mae']:.4f}   RMSE: {stats['overall_rmse']:.4f}")
    print(f"  per-position MAE (h={h}):")
    mae = stats["per_position_mae"]
    sample = [0, 1, 5, 10, 20, h - 1] if h > 20 else list(range(h))
    sample = [i for i in sample if 0 <= i < h]
    for i in sample:
        print(f"    pos {i:>2}: mae={mae[i]:.4f}  pred_norm={stats['predicted_norm_per_pos'][i]:.3f}  rec_norm={stats['recorded_norm_per_pos'][i]:.3f}")


def maybe_plot(all_stats: list[tuple[int, dict]], out_path: Path | None) -> None:
    if out_path is None:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"[plot] matplotlib not installed, skipping {out_path}")
        return

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for frame_idx, stats in all_stats:
        ax[0].plot(stats["per_position_mae"], label=f"frame {frame_idx}")
        ax[1].plot(stats["predicted_norm_per_pos"], "--", label=f"pred f{frame_idx}")
        ax[1].plot(stats["recorded_norm_per_pos"], "-", label=f"rec f{frame_idx}")
    ax[0].set_title("Per-position MAE (predicted vs recorded)")
    ax[0].set_xlabel("chunk position")
    ax[0].set_ylabel("MAE")
    ax[0].legend()
    ax[0].grid(True, alpha=0.3)
    ax[1].set_title("Token chunk norm (predicted vs recorded)")
    ax[1].set_xlabel("chunk position")
    ax[1].set_ylabel("||token||")
    ax[1].legend(fontsize=7)
    ax[1].grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    print(f"[plot] saved to {out_path}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Open-loop VLA policy diagnostic")
    ap.add_argument("--parquet", type=Path, required=True)
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--prompt", type=str, required=True)
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=5550)
    ap.add_argument(
        "--embodiment-tag",
        default=None,
        help="Optional embodiment override (e.g. 'unitree_g1_sonic')",
    )
    ap.add_argument("--action-horizon", type=int, default=40)
    ap.add_argument(
        "--frame-indices",
        type=int,
        nargs="+",
        default=[50, 100, 150],
        help="Frame indices to evaluate (default: 50 100 150)",
    )
    ap.add_argument("--plot", type=Path, default=None, help="Optional PNG output path")
    return ap.parse_args()


def main() -> int:
    args = parse_args()

    if not args.parquet.exists():
        print(f"parquet not found: {args.parquet}", file=sys.stderr)
        return 1
    if not args.video.exists():
        print(f"video not found: {args.video}", file=sys.stderr)
        return 1

    df = pd.read_parquet(args.parquet)
    n = len(df)
    print(f"loaded {n} frames from {args.parquet.name}")

    recorded_tokens_all = np.stack(df["action.motion_token"].values).astype(np.float32)
    print(f"recorded token shape: {recorded_tokens_all.shape}")

    all_stats: list[tuple[int, dict]] = []
    for frame_idx in args.frame_indices:
        if frame_idx < 0 or frame_idx >= n:
            print(f"[skip] frame {frame_idx} out of range [0, {n})")
            continue
        h_end = min(frame_idx + args.action_horizon, n)
        recorded_chunk = recorded_tokens_all[frame_idx:h_end]

        obs = build_observation(df, frame_idx, args.video, args.prompt)
        action = query_policy(args.host, args.port, obs, args.embodiment_tag)
        predicted_chunk = extract_predicted_token(action)

        stats = compare(predicted_chunk, recorded_chunk)
        print_report(frame_idx, stats)
        all_stats.append((frame_idx, stats))

    maybe_plot(all_stats, args.plot)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
