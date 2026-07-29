from collections import deque

import msgpack
import numpy as np
import zmq

from gear_sonic.utils.inference.tactile_subscriber import ZMQTactileSubscriber


class FakeSocket:
    def __init__(self, messages=()):
        self.messages = deque(messages)

    def recv_multipart(self, _flags):
        if not self.messages:
            raise zmq.Again()
        return self.messages.popleft()


def tactile_message(value: int, *, device: str = "body", size: int = 256):
    return [
        f"tactile.{device}".encode(),
        msgpack.packb({"device": device, "host_time": 12.5}, use_bin_type=True),
        np.full(size, value, dtype=np.uint8).tobytes(),
    ]


def test_subscriber_drains_to_latest_selected_frame():
    clock = [10.0]
    subscriber = ZMQTactileSubscriber(
        device="body",
        socket=FakeSocket([tactile_message(1), tactile_message(9)]),
        clock=lambda: clock[0],
    )

    frame = subscriber.read_fresh(0.1)

    assert frame is not None
    assert frame.raw.shape == (256,)
    assert frame.raw.dtype == np.uint8
    assert np.all(frame.raw == 9)


def test_subscriber_rejects_wrong_layout_and_stale_frames():
    clock = [10.0]
    subscriber = ZMQTactileSubscriber(
        device="body",
        socket=FakeSocket([tactile_message(1, size=768)]),
        clock=lambda: clock[0],
    )
    assert subscriber.read_fresh(0.1) is None
    assert "uint8[256]" in subscriber.last_error

    subscriber._socket.messages.append(tactile_message(2))
    assert subscriber.read_fresh(0.1) is not None
    clock[0] = 10.2
    assert subscriber.read_fresh(0.1) is None
    assert "stale" in subscriber.last_error


def test_subscriber_does_not_accept_a_different_device_topic():
    subscriber = ZMQTactileSubscriber(
        device="body",
        socket=FakeSocket([tactile_message(3, device="vest")]),
        clock=lambda: 1.0,
    )

    assert subscriber.read_fresh(0.1) is None
    assert "expected 'tactile.body'" in subscriber.last_error
