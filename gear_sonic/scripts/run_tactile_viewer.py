#!/usr/bin/env python3
"""Live OpenCV viewer for the JuQiao tactile skin ZMQ stream.

Subscribes to the ``tactile`` topic prefix that ``tactile_publisher.py``
publishes (and that ``run_data_exporter.py`` records) and renders each device
in its own window in real time. ``--tactile-mode`` selects the device set:

    triple (default):
        tactile.vest      -> body-region layout (reused from visualize_tactile.py)
        tactile.left_arm  -> 16x16 grid
        tactile.right_arm -> 16x16 grid
    single:
        tactile.body      -> body-region layout

This viewer is read-only: a ZMQ PUB socket fans out to every SUB independently,
so running it alongside the data exporter does not steal frames or otherwise
affect recording. It is meant for connectivity verification — confirm all three
devices stream and that pressing a body part lights up the matching window.

Usage (from repo root):
    .venv_data_collection/bin/python gear_sonic/scripts/run_tactile_viewer.py \
        --tactile-zmq-host 192.168.123.164
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import zmq

# Reuse the region rendering / canvas layout from the offline parquet viewer.
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
from visualize_tactile import (  # noqa: E402
    CANVAS_BG,
    TACTILE_DIM,
    _compose_frame,
    _put_label,
    _render_region,
)

# Device list per collection mode (must match the publisher / exporter).
_DEVICES_BY_MODE = {
    "single": ("body",),
    "triple": ("vest", "left_arm", "right_arm"),
}
# Devices rendered with the body-region (vest) layout; others use the 16x16 arm grid.
_BODY_REGION_DEVICES = ("vest", "body")

# Arm sleeve raw-channel order -> 16x16 grid (spec "手臂分区1: 从左到右"):
# channels 129..256 then 1..128, here 0-based.
ARM_ORDER = np.array(list(range(128, 256)) + list(range(0, 128)), dtype=np.int32)
ARM_CELL_PX = 22


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--tactile-mode",
        choices=("single", "triple"),
        default="triple",
        help="triple=3设备(vest/left_arm/right_arm，默认)；single=单设备 body。需与 publisher 一致。",
    )
    parser.add_argument(
        "--tactile-zmq-host",
        default="localhost",
        help="Host the tactile publisher is bound to (e.g. the G1 IP).",
    )
    parser.add_argument(
        "--tactile-zmq-port", type=int, default=5558, help="Tactile publisher ZMQ port."
    )
    parser.add_argument("--topic", default="tactile", help="ZMQ topic prefix.")
    parser.add_argument("--window", default="tactile", help="OpenCV window title prefix.")
    parser.add_argument(
        "--history",
        type=int,
        default=400,
        help="Frames of activity history shown in the vest timeline strip.",
    )
    parser.add_argument(
        "--stale-sec",
        type=float,
        default=0.5,
        help="Mark a view STALE if no frame arrives within this many seconds.",
    )
    return parser.parse_args()


def _status_canvas(text: str) -> np.ndarray:
    """A small placeholder canvas shown before the first frame arrives."""
    canvas = np.full((200, 620, 3), CANVAS_BG, dtype=np.uint8)
    cv2.putText(
        canvas, text, (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA
    )
    return canvas


def _arm_canvas(frame: np.ndarray, device: str, vmax: int, status: str) -> np.ndarray:
    """Render an arm frame as a labeled 16x16 heatmap canvas."""
    grid = frame[ARM_ORDER].reshape(16, 16)
    body = _render_region(grid, ARM_CELL_PX, vmax)
    h, w = body.shape[:2]
    canvas = np.full((h + 50, max(w, 360), 3), CANVAS_BG, dtype=np.uint8)
    _put_label(canvas, f"{device}  16x16", 6, 20)
    canvas[28 : 28 + h, 0:w] = body
    cv2.putText(
        canvas, status, (6, canvas.shape[0] - 6),
        cv2.FONT_HERSHEY_SIMPLEX, 0.44, (170, 170, 170), 1, cv2.LINE_AA,
    )
    return canvas


class DeviceView:
    def __init__(self, device: str, window: str, history: int) -> None:
        self.device = device
        self.window = window
        self.last_frame: np.ndarray | None = None
        self.last_recv = 0.0
        self.rx_count = 0
        self.n_new = 0
        self.fps_ema = 0.0
        self.prev_t: float | None = None
        self.history: deque[int] = deque(maxlen=max(history, 2))
        cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)

    def update(self, payload: np.ndarray, n_new: int) -> None:
        now = time.time()
        if self.prev_t is not None and now > self.prev_t:
            inst = n_new / (now - self.prev_t)
            self.fps_ema = inst if self.fps_ema == 0.0 else 0.9 * self.fps_ema + 0.1 * inst
        self.prev_t = now
        self.last_frame = payload
        self.last_recv = now
        self.rx_count += n_new
        self.history.append(int(payload.max()))

    def render(self, endpoint: str, stale_sec: float) -> None:
        if self.last_frame is None:
            canvas = _status_canvas(f"waiting for {self.device} on {endpoint} ...")
            cv2.imshow(self.window, canvas)
            return
        age = time.time() - self.last_recv
        series = np.asarray(self.history, dtype=np.int64)
        vmax = max(int(series.max()), 1)
        status = (
            f"LIVE {self.fps_ema:4.1f}fps  rx={self.rx_count}  "
            f"age={age * 1000:4.0f}ms  vmax={vmax}"
        )
        if age > stale_sec:
            status += "  [STALE]"
        if self.device in _BODY_REGION_DEVICES:
            canvas = _compose_frame(
                self.last_frame, vmax, len(self.history) - 1, len(self.history), False, series
            )
            cv2.putText(
                canvas, status, (14, canvas.shape[0] - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.44, (170, 170, 170), 1, cv2.LINE_AA,
            )
        else:
            canvas = _arm_canvas(self.last_frame, self.device, vmax, status)
        cv2.imshow(self.window, canvas)


def main() -> int:
    args = parse_args()
    endpoint = f"tcp://{args.tactile_zmq_host}:{args.tactile_zmq_port}"

    devices = _DEVICES_BY_MODE[args.tactile_mode]
    topic_to_device = {f"tactile.{dev}".encode("utf-8"): dev for dev in devices}

    ctx = zmq.Context()
    sock = ctx.socket(zmq.SUB)
    # Do NOT use ZMQ_CONFLATE: the publisher sends multi-part messages and
    # CONFLATE aborts libzmq on multi-part. Stay real-time by draining to the
    # latest frame per device every loop instead.
    sock.setsockopt(zmq.RCVHWM, 30)
    sock.setsockopt_string(zmq.SUBSCRIBE, args.topic)  # prefix -> all 3 topics
    sock.connect(endpoint)
    print(
        f"[tactile-viewer] SUB connected to {endpoint} "
        f"(mode={args.tactile_mode}, topic prefix={args.topic!r} -> {'/'.join(devices)})"
    )
    print("[tactile-viewer] keys: q / ESC to quit")

    views = {dev: DeviceView(dev, f"{args.window}: {dev}", args.history) for dev in devices}

    try:
        while True:
            latest: dict = {}
            counts: dict = {}
            while True:
                try:
                    parts = sock.recv_multipart(zmq.NOBLOCK)
                except zmq.Again:
                    break
                if len(parts) == 3:
                    device = topic_to_device.get(parts[0])
                    if device is not None:
                        latest[device] = parts[2]
                        counts[device] = counts.get(device, 0) + 1

            for device, raw in latest.items():
                payload = np.frombuffer(raw, dtype=np.uint8)
                if payload.shape == (TACTILE_DIM,):
                    views[device].update(payload, counts[device])

            for view in views.values():
                view.render(endpoint, args.stale_sec)

            if (cv2.waitKey(15) & 0xFFFF) in (ord("q"), 27):
                break
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        sock.close()
        ctx.term()
        total = sum(v.rx_count for v in views.values())
        print(f"[tactile-viewer] shutdown, received {total} frames total")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
