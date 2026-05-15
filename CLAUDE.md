# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository layout

This is a monorepo for NVIDIA's GR00T Whole-Body Control projects for humanoid robots (primarily the Unitree G1). It contains four largely independent subprojects, each with its own `pyproject.toml`/`setup.py` and its own intended runtime environment:

- **`gear_sonic/`** — GEAR-SONIC: PPO training stack (Hydra + TRL + accelerate) for the humanoid behavior foundation model. Houses Isaac-Lab-backed envs (`envs/manager_env/`), Hydra configs (`config/`), data processing (`data_process/`), and runtime scripts (`scripts/`: sim loop, VR teleop manager, data collection, VLA inference). Entry points: `train_agent_trl.py`, `eval_agent_trl.py`.
- **`gear_sonic_deploy/`** — C++ inference stack (CMake, C++20) that runs the trained ONNX policy on real G1 hardware. Built via `just build`; orchestrated by `deploy.sh`. Talks to the Python side over ZMQ.
- **`decoupled_wbc/`** — Older decoupled controller used in GR00T N1.5/N1.6 (RL lower body + IK upper body). Pure Python; entry point `decoupled_wbc = decoupled_wbc.control.teleop.gui.cli:cli`. The only subproject with a real pytest suite (`decoupled_wbc/tests/`).
- **`motionbricks/`** — Real-time latent generative motion model (VQVAE + pose + root). Self-contained MuJoCo demo and training scripts. Its own conda env (`motionbricks` per its README).

`external_dependencies/` is vendored third-party code and **must be excluded** from lint/format/type-check (already configured in `pyproject.toml`).

## Environments — there are several, not one

There is no single project-wide venv. Each use case has its own isolated env, mostly created automatically by helper scripts in `install_scripts/`:

| Use case | Environment | How to install |
|---|---|---|
| Train / finetune SONIC | Isaac Lab's Python env (Python **3.11.x** required) | Install Isaac Lab separately, then `pip install -e "gear_sonic/[training]"` |
| MuJoCo sim | `.venv_sim` | `bash install_scripts/install_mujoco_sim.sh` |
| VR teleop | `.venv_teleop` | `bash install_scripts/install_pico.sh` |
| Data collection | `.venv_data_collection` | `bash install_scripts/install_data_collection.sh` |
| Camera server | `.venv_camera` | `bash install_scripts/install_camera_server.sh` |
| VLA inference | `.venv_inference` | `bash install_scripts/install_inference.sh` |
| Deploy on real robot | C++ build via `just build` inside `gear_sonic_deploy/` | See `gear_sonic_deploy/deploy.sh` + deploy docs |

The install scripts use `uv` to create venvs — do not try to consolidate them. Optional dependency groups are defined in `gear_sonic/pyproject.toml`: `[teleop]`, `[sim]`, `[data_collection]`, `[camera]`, `[inference]`, `[training]`. `decoupled_wbc` has `[full]` and `[dev]`.

`python check_environment.py` is the canonical pre-flight check (use `--training` or `--deploy` to scope it). For training mode it enforces Python 3.11 (Isaac Lab's constraint).

## Git LFS is required

Mesh, ONNX, and checkpoint files are tracked in Git LFS. Without `git lfs pull`, you'll see tiny pointer files and runtime failures with no obvious cause. Several scripts (including `check_environment.py`) sanity-check by file size — anything under ~1 KB is an LFS pointer. MotionBricks checkpoints (~2.2 GB) are **opt-in**: `git lfs pull --include="motionbricks/out/**" --exclude=""`.

## Lint / format / tests

Tooling is configured at the repo root (`pyproject.toml`, `lint.sh`, `Makefile`):

```bash
# Check (matches CI)
make run-checks                  # isort --check, black --check, ruff check
./lint.sh                        # equivalent; black + ruff (incl. ruff --select I for imports)

# Auto-fix
make format                      # isort + black
./lint.sh --fix                  # ruff --fix + ruff --select I --fix + black
```

Style: `black` line length 100, `ruff` line length 115 targeting `py310`, `ruff` selects `E,F,I`, isort uses the black profile. The configured exclusions (`external_dependencies`, `gear_sonic/dexmg`, notebooks) are load-bearing — don't widen lint scope into them.

Pytest is only meaningfully wired for `decoupled_wbc`:

```bash
pytest decoupled_wbc/tests/                                  # full suite
pytest decoupled_wbc/tests/control/robot_model/robot_model_test.py::TestX  # single test
pytest decoupled_wbc/tests/ --tensorboard-log-dir=<path>     # custom log dir (see conftest.py)
```

Pytest is configured at the root to only collect from `decoupled_wbc/tests/` and to recognize both `Test*` and `*Test` class names. Many tests require model assets/replay data — they are LFS-tracked under `decoupled_wbc/tests/replay_data/`.

## Training (gear_sonic)

Training uses Hydra configs rooted at `gear_sonic/config/` and is launched via `accelerate`:

```bash
accelerate launch --num_processes=8 gear_sonic/train_agent_trl.py \
    +exp=manager/universal_token/all_modes/sonic_release \
    +checkpoint=sonic_release/last.pt \
    num_envs=4096 headless=True \
    ++manager_env.commands.motion.motion_lib_cfg.motion_file=...
```

Note: `train_agent_trl.py` patches `sys.path` at import time to avoid `from trl import ...` resolving to the local `gear_sonic/trl/` package instead of HuggingFace's `trl`. Don't "clean up" that block. The script hard-fails if `isaaclab` isn't importable.

Hydra experiment configs live under `gear_sonic/config/exp/manager/universal_token/all_modes/`. Manager envs (`gear_sonic/envs/manager_env/`) and TRL trainer/callback overrides (`gear_sonic/trl/`) are the main extension points.

## Deployment (gear_sonic_deploy)

C++ inference stack built with CMake (C++20), driven by `just`:

```bash
cd gear_sonic_deploy
./deploy.sh sim                          # loopback, MuJoCo
./deploy.sh real                         # auto-detect 192.168.123.x interface
./deploy.sh --cp <ckpt> --obs-config <yaml> --planner <onnx> real
```

`deploy.sh` resolves the network interface, sanity-checks file presence, runs `scripts/install_deps.sh` if `just`/`cmake`/`clang` are missing, sources `scripts/setup_env.sh`, builds, then `just run g1_deploy_onnx_ref ...`. Requires `TensorRT_ROOT` env var pointing at a TensorRT install. Sim mode passes `--disable-crc-check`.

The Python side communicates with the C++ binary over ZMQ (header size is 1280 bytes as of 2026-03-24's protocol v4 — `gear_sonic/scripts/pico_manager_thread_server.py` is the canonical Python-side ZMQ manager).

## Conventions worth knowing

- The decoupled_wbc package exposes a CLI: `decoupled_wbc = decoupled_wbc.control.teleop.gui.cli:cli`.
- Version is dynamic and read from `<subproject>/version.py` (`gear_sonic.version.VERSION`, `decoupled_wbc.version.VERSION`).
- Docs are Sphinx-based under `docs/source/`; the published site is at https://nvlabs.github.io/GR00T-WholeBodyControl/.
- License is dual: Apache-2.0 for code, NVIDIA Open Model License for checkpoints.
