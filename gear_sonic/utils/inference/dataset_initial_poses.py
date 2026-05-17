"""Load per-prompt initial motion tokens from a LeRobot-format dataset.

At inference time, the user can pass ``--dataset-path /path/to/dataset`` to
``run_vla_inference.py`` to replace the hardcoded ``LATENT_INITIAL_MOTION_TOKEN``
with one derived from real demonstrations:

  * For each episode in the dataset, read the first-frame ``action.motion_token``.
  * Group those tokens by task string (from ``meta/tasks.jsonl`` +
    ``meta/episodes.jsonl``).
  * Average the per-task tokens to produce a representative starting pose.

When the user presses ``i`` during inference, the loader is queried with the
current language prompt; if a matching task is found, that prompt's average
token is sent. Otherwise, the global fallback (mean over all episodes) is used,
and finally the hardcoded constant if even that is unavailable.

Loading is fast (~0.3 s for ~100 episodes on local SSD), so it runs at startup.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Optional

import numpy as np

_MOTION_TOKEN_COL = "action.motion_token"
_MOTION_TOKEN_DIM = 64


@dataclass
class DatasetInitialPoses:
    """Per-prompt average of first-frame motion tokens loaded from a dataset."""

    by_prompt: dict[str, np.ndarray]
    """Map from task string -> averaged first-frame motion token, shape [64]."""

    global_mean: Optional[np.ndarray]
    """Mean over every episode's first-frame token, shape [64]. None if dataset empty."""

    dataset_path: Path
    """Path the tokens were loaded from (for logging)."""

    n_episodes_loaded: int
    """Total number of episodes successfully read."""

    def lookup(self, prompt: str) -> Optional[np.ndarray]:
        """Return the initial token for ``prompt``, falling back to the global mean.

        Returns ``None`` only if the dataset is empty.
        """
        token = self.by_prompt.get(prompt)
        if token is not None:
            return token
        return self.global_mean

    def summary(self) -> str:
        lines = [
            f"Loaded {self.n_episodes_loaded} episodes from {self.dataset_path}",
            f"  Tasks ({len(self.by_prompt)}):",
        ]
        for prompt in sorted(self.by_prompt.keys()):
            lines.append(f"    - {prompt!r}")
        return "\n".join(lines)


def load_dataset_initial_poses(dataset_path: str | Path) -> DatasetInitialPoses:
    """Build per-prompt initial motion token averages from a LeRobot dataset.

    Expects the standard LeRobot layout::

        dataset_path/
          meta/tasks.jsonl     -> {"task_index": int, "task": str}
          meta/episodes.jsonl  -> {"episode_index": int, "tasks": [str, ...], ...}
          data/chunk-XXX/episode_YYYYYY.parquet

    Each parquet file is expected to contain an ``action.motion_token`` column
    where every row is a 64-d float vector.
    """
    import pyarrow.parquet as pq  # local import: only needed when feature is enabled

    root = Path(dataset_path).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset path is not a directory: {root}")

    meta_dir = root / "meta"
    episodes_jsonl = meta_dir / "episodes.jsonl"
    if not episodes_jsonl.is_file():
        raise FileNotFoundError(f"Missing {episodes_jsonl}")

    episode_tasks: dict[int, list[str]] = {}
    with episodes_jsonl.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ep = json.loads(line)
            episode_tasks[int(ep["episode_index"])] = list(ep.get("tasks", []))

    data_dir = root / "data"
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Missing {data_dir}")

    by_prompt_tokens: dict[str, list[np.ndarray]] = {}
    all_tokens: list[np.ndarray] = []

    for parquet_path in sorted(data_dir.glob("chunk-*/episode_*.parquet")):
        # Filename pattern: episode_000123.parquet
        try:
            ep_idx = int(parquet_path.stem.split("_")[-1])
        except ValueError:
            continue

        tasks = episode_tasks.get(ep_idx)
        if not tasks:
            continue

        try:
            table = pq.read_table(parquet_path, columns=[_MOTION_TOKEN_COL])
        except Exception as e:
            print(
                f"[dataset_initial_poses] Skipping {parquet_path.name}: failed to read "
                f"{_MOTION_TOKEN_COL} column ({e})"
            )
            continue

        if table.num_rows == 0:
            continue

        first = np.asarray(table[_MOTION_TOKEN_COL][0].as_py(), dtype=np.float32)
        if first.shape != (_MOTION_TOKEN_DIM,):
            print(
                f"[dataset_initial_poses] Skipping {parquet_path.name}: unexpected "
                f"motion_token shape {first.shape}"
            )
            continue

        all_tokens.append(first)
        for task in tasks:
            by_prompt_tokens.setdefault(task, []).append(first)

    by_prompt = {
        task: np.mean(np.stack(toks, axis=0), axis=0).astype(np.float32)
        for task, toks in by_prompt_tokens.items()
    }
    global_mean = (
        np.mean(np.stack(all_tokens, axis=0), axis=0).astype(np.float32) if all_tokens else None
    )

    return DatasetInitialPoses(
        by_prompt=by_prompt,
        global_mean=global_mean,
        dataset_path=root,
        n_episodes_loaded=len(all_tokens),
    )
