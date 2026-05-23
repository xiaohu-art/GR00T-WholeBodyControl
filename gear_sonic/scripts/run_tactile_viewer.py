#!/usr/bin/env python3
"""Live OpenCV viewer for the JuQiao tactile skin ZMQ stream.

Subscribes to the same ``tactile`` ZMQ topic that ``tactile_publisher.py``
publishes (and that ``run_data_exporter.py`` records), and renders each frame in
real time using the body-region layout from ``visualize_tactile.py``.

This viewer is read-only: a ZMQ PUB socket fans out to every SUB independently,
so running it alongside the data exporter does not steal frames or otherwise
affect recording.

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
from visualize_tactile import CANVAS_BG, TACTILE_DIM, _compose_frame  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--tactile-zmq-host",
        default="localhost",
        help="Host the tactile publisher is bound to (e.g. the G1 IP).",
    )
    parser.add_argument(
        "--tactile-zmq-port", type=int, default=5558, help="Tactile publisher ZMQ port."
    )
    parser.add_argument("--topic", default="tactile", help="ZMQ topic name.")
    parser.add_argument("--window", default="tactile (live)", help="OpenCV window title.")
    parser.add_argument(
        "--history",
        type=int,
        default=400,
        help="Frames of activity history shown in the timeline strip.",
    )
    parser.add_argument(
        "--stale-sec",
        type=float,
        default=0.5,
        help="Mark the view STALE if no frame arrives within this many seconds.",
    )
    return parser.parse_args()


def _status_canvas(text: str) -> np.ndarray:
    """A small placeholder canvas shown before the first frame arrives."""
    canvas = np.full((200, 620, 3), CANVAS_BG, dtype=np.uint8)
    cv2.putText(
        canvas, text, (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA
    )
    return canvas


def main() -> int:
    args = parse_args()
    endpoint = f"tcp://{args.tactile_zmq_host}:{args.tactile_zmq_port}"
    topic_bytes = args.topic.encode("utf-8")

    ctx = zmq.Context()
    sock = ctx.socket(zmq.SUB)
    # Do NOT use ZMQ_CONFLATE here: the publisher sends multi-part messages
    # ([topic, header, payload]) and CONFLATE does not support multi-part — it
    # aborts libzmq ("Assertion failed: !_more"). We stay real-time by draining
    # to the latest frame every loop instead. A small RCVHWM (set before
    # connect, so it takes effect) caps the backlog if rendering ever stalls.
    sock.setsockopt(zmq.RCVHWM, 10)
    sock.setsockopt_string(zmq.SUBSCRIBE, args.topic)
    sock.connect(endpoint)
    print(f"[tactile-viewer] SUB connected to {endpoint} (topic={args.topic!r})")
    print("[tactile-viewer] keys: q / ESC to quit")

    history: deque[int] = deque(maxlen=max(args.history, 2))
    last_frame: np.ndarray | None = None
    last_recv = 0.0
    rx_count = 0
    fps_ema = 0.0
    prev_t: float | None = None

    cv2.namedWindow(args.window, cv2.WINDOW_AUTOSIZE)
    try:
        while True:
            # Drain everything queued; keep the most recent frame for display,
            # but count every frame so the reported rate is the true incoming
            # stream rate, not just the viewer's (slower) redraw rate.
            latest = None
            n_new = 0
            while True:
                try:
                    parts = sock.recv_multipart(zmq.NOBLOCK)
                except zmq.Again:
                    break
                if len(parts) == 3 and parts[0] == topic_bytes:
                    latest = parts[2]
                    n_new += 1

            if latest is not None:
                payload = np.frombuffer(latest, dtype=np.uint8)
                if payload.shape == (TACTILE_DIM,):
                    now = time.time()
                    if prev_t is not None and now > prev_t:
                        inst = n_new / (now - prev_t)
                        fps_ema = inst if fps_ema == 0.0 else 0.9 * fps_ema + 0.1 * inst
                    prev_t = now
                    last_frame = payload
                    last_recv = now
                    rx_count += n_new
                    history.append(int(payload.max()))

            if last_frame is None:
                canvas = _status_canvas(f"waiting for tactile data on {endpoint} ...")
            else:
                age = time.time() - last_recv
                series = np.asarray(history, dtype=np.int64)
                vmax = max(int(series.max()), 1)
                canvas = _compose_frame(
                    last_frame, vmax, len(history) - 1, len(history), False, series
                )
                status = (
                    f"LIVE  {fps_ema:4.1f} fps   rx={rx_count}   "
                    f"age={age * 1000:4.0f}ms   vmax={vmax}"
                )
                if age > args.stale_sec:
                    status += "   [STALE]"
                cv2.putText(
                    canvas,
                    status,
                    (14, canvas.shape[0] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.44,
                    (170, 170, 170),
                    1,
                    cv2.LINE_AA,
                )

            cv2.imshow(args.window, canvas)
            if (cv2.waitKey(15) & 0xFFFF) in (ord("q"), 27):
                break
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        sock.close()
        ctx.term()
        print(f"[tactile-viewer] shutdown, received {rx_count} frames")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
