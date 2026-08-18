"""Latest-frame subscriber for the three-device SONIC tactile stream."""

from collections.abc import Callable
import time

import msgpack
import numpy as np
import zmq


TACTILE_DEVICES = ("vest", "left_arm", "right_arm")
TACTILE_FRAME_DIM = 256


class TactileSubscriber:
    """Subscribe to tactile topics and expose one fresh frame per device."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5558,
        *,
        max_age_sec: float = 0.1,
        socket=None,
        context=None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if max_age_sec <= 0:
            raise ValueError("max_age_sec must be positive")

        self.max_age_sec = float(max_age_sec)
        self._clock = clock
        self._latest: dict[str, tuple[float, np.ndarray]] = {}
        self._topic_to_device = {
            f"tactile.{device}".encode("utf-8"): device for device in TACTILE_DEVICES
        }

        self._owns_context = socket is None
        if socket is None:
            self._context = context or zmq.Context()
            self._socket = self._context.socket(zmq.SUB)
            # CONFLATE aborts libzmq for this multipart wire format. Drain the
            # queue on every read and retain only the newest frame per device.
            self._socket.setsockopt(zmq.RCVHWM, 1000)
            self._socket.setsockopt_string(zmq.SUBSCRIBE, "tactile")
            self._socket.connect(f"tcp://{host}:{port}")
        else:
            self._context = context
            self._socket = socket

    def _poll(self) -> None:
        latest_parts: dict[str, list[bytes]] = {}
        while True:
            try:
                parts = self._socket.recv_multipart(zmq.NOBLOCK)
            except zmq.Again:
                break
            if len(parts) != 3:
                continue
            device = self._topic_to_device.get(parts[0])
            if device is not None:
                latest_parts[device] = parts

        received_at = self._clock()
        for device, parts in latest_parts.items():
            try:
                header = msgpack.unpackb(parts[1], raw=False)
            except Exception:
                continue
            if not isinstance(header, dict):
                continue
            payload = np.frombuffer(parts[2], dtype=np.uint8)
            if payload.shape != (TACTILE_FRAME_DIM,):
                continue
            self._latest[device] = (received_at, payload.copy())

    def read(self) -> dict[str, np.ndarray] | None:
        """Return all current device frames, or ``None`` until all are fresh."""
        self._poll()
        now = self._clock()
        if any(device not in self._latest for device in TACTILE_DEVICES):
            return None
        if any(now - self._latest[device][0] > self.max_age_sec for device in TACTILE_DEVICES):
            return None
        return {device: self._latest[device][1].copy() for device in TACTILE_DEVICES}

    def status(self) -> str:
        """Describe missing or stale streams for operator-facing diagnostics."""
        now = self._clock()
        missing = [device for device in TACTILE_DEVICES if device not in self._latest]
        if missing:
            return f"missing tactile streams: {', '.join(missing)}"
        stale = [
            device
            for device in TACTILE_DEVICES
            if now - self._latest[device][0] > self.max_age_sec
        ]
        if stale:
            return f"stale tactile streams: {', '.join(stale)}"
        return "ready"

    def close(self) -> None:
        self._socket.close(linger=0)
        if self._owns_context and self._context is not None:
            self._context.term()
