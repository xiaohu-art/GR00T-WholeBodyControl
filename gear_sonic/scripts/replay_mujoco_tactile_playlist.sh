#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
DATASET_ROOT="${1:-${REPO_ROOT}/outputs/desk_sweep_merged_clean}"
PYTHON="${REPO_ROOT}/.venv_sim/bin/python"

if [[ ! -x "${PYTHON}" ]]; then
  echo "[playlist] Python not found: ${PYTHON}" >&2
  exit 1
fi
if [[ ! -d "${DATASET_ROOT}/data" ]]; then
  echo "[playlist] dataset data directory not found: ${DATASET_ROOT}/data" >&2
  exit 1
fi

mapfile -t episodes < <(
  find "${DATASET_ROOT}/data" -type f -name 'episode_*.parquet' -print | sort
)
total="${#episodes[@]}"
if (( total == 0 )); then
  echo "[playlist] no episode parquet files found under ${DATASET_ROOT}/data" >&2
  exit 1
fi

index=0
while (( index < total )); do
  parquet="${episodes[index]}"
  episode_name="$(basename -- "${parquet}" .parquet)"
  printf '\n[playlist] [%02d/%02d] %s\n' "$((index + 1))" "${total}" "${episode_name}"
  echo "[playlist] Up=previous  Down=next  R=restart  Q/Esc=stop all"

  "${PYTHON}" "${SCRIPT_DIR}/replay_episode.py" "${parquet}" \
    --tactile-mode triple \
    --exit-at-end \
    --playlist-controls \
    --hard-exit-at-boundary \
    --arrange-windows \
    --window "[$(printf '%02d/%02d' "$((index + 1))" "${total}")] ${episode_name} tactile"
  status=$?

  case "${status}" in
    0|21)
      ((index += 1))
      ;;
    20)
      if (( index > 0 )); then
        ((index -= 1))
      else
        echo "[playlist] already at the first episode; replaying it"
      fi
      ;;
    130)
      echo "[playlist] stopped by user"
      exit 0
      ;;
    *)
      echo "[playlist] replay failed with status ${status}: ${parquet}" >&2
      exit "${status}"
      ;;
  esac
done

echo "[playlist] completed all ${total} episodes"
