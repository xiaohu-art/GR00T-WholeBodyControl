#!/usr/bin/env python3
"""Replay observation.state from a recorded parquet episode in MuJoCo.

Loads the G1 scene, then steps through ``observation.state`` row by row and
writes each value into MuJoCo's ``qpos`` by matching joint names. This is a
visualization-only replay — no physics, no controller; we just call
``mj_forward`` and sync the viewer.

Usage (from repo root, with .venv_data_collection or .venv_sim active):
    python gear_sonic/scripts/replay_state_mujoco.py \
        outputs/2026-05-14-18-07-13/data/chunk-000/episode_000000.parquet

The dataset's ``meta/info.json`` (next to the parquet) is read to recover the
joint name ordering of ``observation.state``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import mujoco
    import mujoco.viewer
except ImportError as exc:
    raise SystemExit(
        "mujoco not installed in this venv. Install via .venv_sim or pip install mujoco."
    ) from exc


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
DEFAULT_SCENE = REPO_ROOT / "gear_sonic_deploy" / "g1" / "scene_29dof_with_hand.xml"


def find_dataset_info(parquet_path: Path) -> Path:
    """Walk up from the parquet to find meta/info.json."""
    for ancestor in parquet_path.parents:
        candidate = ancestor / "meta" / "info.json"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not locate meta/info.json for {parquet_path}. "
        "Pass --joint-names explicitly to override."
    )


def load_joint_names(parquet_path: Path, explicit: list[str] | None) -> list[str]:
    if explicit:
        return list(explicit)
    info_path = find_dataset_info(parquet_path)
    with info_path.open() as f:
        info = json.load(f)
    names = info["features"]["observation.state"]["names"]
    return list(names)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Replay observation.state in MuJoCo viewer")
    ap.add_argument("parquet", type=Path, help="Path to episode_*.parquet")
    ap.add_argument(
        "--scene",
        type=Path,
        default=DEFAULT_SCENE,
        help=f"MuJoCo scene XML (default: {DEFAULT_SCENE.relative_to(REPO_ROOT)})",
    )
    ap.add_argument(
        "--rate",
        type=float,
        default=0.0,
        help="playback rate Hz; 0 = use dataset fps from meta/info.json (default 50)",
    )
    ap.add_argument("--start-frame", type=int, default=0)
    ap.add_argument("--end-frame", type=int, default=-1, help="exclusive; -1 = end")
    ap.add_argument("--loop", action="store_true", help="loop until viewer closed")
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
    return ap.parse_args()


def build_qpos_index_map(model, joint_names: list[str]) -> tuple[list[int], list[str]]:
    """Return (qpos_indices, missing_names) parallel to joint_names."""
    indices: list[int] = []
    missing: list[str] = []
    for name in joint_names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            indices.append(-1)
            missing.append(name)
        else:
            indices.append(model.jnt_qposadr[jid])
    return indices, missing


def main() -> int:
    args = parse_args()
    if not args.parquet.exists():
        print(f"[replay-mjc] parquet not found: {args.parquet}", file=sys.stderr)
        return 1
    if not args.scene.exists():
        print(f"[replay-mjc] scene xml not found: {args.scene}", file=sys.stderr)
        return 1

    df = pd.read_parquet(args.parquet)
    if "observation.state" not in df.columns:
        print("[replay-mjc] parquet missing observation.state column", file=sys.stderr)
        return 1

    joint_names = load_joint_names(args.parquet, args.joint_names)
    n_state = len(joint_names)
    sample_state = np.asarray(df["observation.state"].iloc[0])
    if sample_state.shape[0] != n_state:
        print(
            f"[replay-mjc] joint_names has {n_state} entries but observation.state "
            f"has {sample_state.shape[0]}; mismatch",
            file=sys.stderr,
        )
        return 1

    rate = args.rate
    if rate <= 0:
        try:
            info_path = find_dataset_info(args.parquet)
            with info_path.open() as f:
                info = json.load(f)
            rate = float(info.get("fps", 50))
        except Exception:
            rate = 50.0
    period = 1.0 / rate

    end = len(df) if args.end_frame < 0 else min(args.end_frame, len(df))
    start = max(0, args.start_frame)
    if start >= end:
        print(f"[replay-mjc] start {start} >= end {end}", file=sys.stderr)
        return 1

    print(f"[replay-mjc] loading scene: {args.scene}")
    model = mujoco.MjModel.from_xml_path(str(args.scene))
    data = mujoco.MjData(model)

    indices, missing = build_qpos_index_map(model, joint_names)
    if missing:
        print(
            f"[replay-mjc] WARNING: {len(missing)} joint(s) in observation.state "
            f"not found in scene; will be skipped: {missing[:10]}"
            + (" ..." if len(missing) > 10 else "")
        )
    n_mapped = sum(1 for i in indices if i >= 0)
    print(f"[replay-mjc] mapped {n_mapped}/{n_state} joints to qpos")

    # Position the floating base
    free_joint_qpos_adr = None
    if model.njnt > 0 and model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE:
        free_joint_qpos_adr = model.jnt_qposadr[0]
        data.qpos[free_joint_qpos_adr + 2] = args.base_height
        data.qpos[free_joint_qpos_adr + 3:free_joint_qpos_adr + 7] = [1.0, 0.0, 0.0, 0.0]

    has_root_orient = (
        args.use_root_orientation and "observation.root_orientation" in df.columns
    )

    print(
        f"[replay-mjc] frames [{start},{end}) @ {rate:.1f}Hz "
        f"≈ {(end - start) * period:.1f}s; loop={args.loop}"
    )

    with mujoco.viewer.launch_passive(model, data) as viewer:
        try:
            while viewer.is_running():
                t_pass_start = time.monotonic()
                for i in range(start, end):
                    if not viewer.is_running():
                        break
                    row = df.iloc[i]
                    state = np.asarray(row["observation.state"], dtype=np.float64)
                    for jname_idx, qadr in enumerate(indices):
                        if qadr >= 0:
                            data.qpos[qadr] = state[jname_idx]
                    if has_root_orient and free_joint_qpos_adr is not None:
                        quat_wxyz = np.asarray(
                            row["observation.root_orientation"], dtype=np.float64
                        )
                        if quat_wxyz.shape == (4,):
                            data.qpos[free_joint_qpos_adr + 3 : free_joint_qpos_adr + 7] = quat_wxyz

                    mujoco.mj_forward(model, data)
                    viewer.sync()

                    t_target = t_pass_start + (i - start + 1) * period
                    sleep_for = t_target - time.monotonic()
                    if sleep_for > 0:
                        time.sleep(sleep_for)

                if not args.loop:
                    print("[replay-mjc] finished; viewer staying open until you close it")
                    while viewer.is_running():
                        viewer.sync()
                        time.sleep(0.05)
        except KeyboardInterrupt:
            print("\n[replay-mjc] interrupted")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
