"""Realtime subscriber for the JuQiao tactile publisher."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Callable

import msgpack
import numpy as np
import zmq

TACTILE_RAW_DIM = 256
SUPPORTED_TACTILE_DEVICES = ("body", "vest", "left_arm", "right_arm")


@dataclass(frozen=True)
class TactileFrame:
    raw: np.ndarray
    topic: str
    host_time: float
    receive_time: float


class ZMQTactileSubscriber:
    """Drain one tactile device topic and retain only its newest valid frame."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5558,
        device: str = "body",
        *,
        socket=None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if device not in SUPPORTED_TACTILE_DEVICES:
            raise ValueError(
                f"Unsupported tactile device {device!r}; choose one of "
                f"{SUPPORTED_TACTILE_DEVICES}"
            )
        self.device = device
        self.topic = f"tactile.{device}"
        self._topic_bytes = self.topic.encode("utf-8")
        self._clock = clock
        self._context = None
        self._owns_socket = socket is None
        self._latest: TactileFrame | None = None
        self.last_error: str | None = None

        if socket is None:
            self._context = zmq.Context()
            socket = self._context.socket(zmq.SUB)
            socket.setsockopt(zmq.RCVHWM, 1000)
            socket.setsockopt(zmq.SUBSCRIBE, self._topic_bytes)
            socket.connect(f"tcp://{host}:{port}")
        self._socket = socket

    def _decode(self, parts: list[bytes]) -> TactileFrame | None:
        if len(parts) != 3:
            self.last_error = f"expected 3 message parts, got {len(parts)}"
            return None
        topic, packed_header, payload = parts
        if topic != self._topic_bytes:
            self.last_error = (
                f"received topic {topic.decode('utf-8', errors='replace')!r}, "
                f"expected {self.topic!r}"
            )
            return None
        try:
            header = msgpack.unpackb(packed_header, raw=False)
        except Exception as exc:
            self.last_error = f"invalid msgpack header: {exc}"
            return None
        if not isinstance(header, dict):
            self.last_error = f"header must be a dict, got {type(header).__name__}"
            return None

        header_device = header.get("device")
        if header_device is not None and header_device != self.device:
            self.last_error = (
                f"header device {header_device!r} does not match selected "
                f"device {self.device!r}"
            )
            return None

        raw = np.frombuffer(payload, dtype=np.uint8)
        if raw.shape != (TACTILE_RAW_DIM,):
            self.last_error = (
                f"tactile payload must be uint8[{TACTILE_RAW_DIM}], got {raw.shape}"
            )
            return None

        self.last_error = None
        return TactileFrame(
            raw=raw.copy(),
            topic=self.topic,
            host_time=float(header.get("host_time", 0.0)),
            receive_time=self._clock(),
        )

    def poll_latest(self) -> TactileFrame | None:
        """Drain the multipart queue and return the latest valid selected frame."""
        newest = None
        while True:
            try:
                parts = self._socket.recv_multipart(zmq.NOBLOCK)
            except zmq.Again:
                break
            frame = self._decode(parts)
            if frame is not None:
                newest = frame
        if newest is not None:
            self._latest = newest
        return self._latest

    def read_fresh(self, max_age_sec: float) -> TactileFrame | None:
        if max_age_sec <= 0:
            raise ValueError(f"max_age_sec must be positive, got {max_age_sec}")
        frame = self.poll_latest()
        if frame is None:
            if self.last_error is None:
                self.last_error = f"waiting for {self.topic}"
            return None
        age = self._clock() - frame.receive_time
        if age > max_age_sec:
            self.last_error = (
                f"{self.topic} frame is stale ({age:.3f}s > {max_age_sec:.3f}s)"
            )
            return None
        self.last_error = None
        return frame

    def close(self) -> None:
        if self._owns_socket:
            self._socket.close(linger=0)
        if self._context is not None:
            self._context.term()
