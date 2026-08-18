import msgpack
import numpy as np
import zmq

from gear_sonic.utils.inference.tactile_subscriber import TACTILE_DEVICES, TactileSubscriber
from gear_sonic.scripts.run_vla_inference import prepare_observation_from_sensors


class FakeSocket:
    def __init__(self):
        self.messages = []
        self.closed = False

    def recv_multipart(self, _flags):
        if not self.messages:
            raise zmq.Again()
        return self.messages.pop(0)

    def close(self, linger=0):
        assert linger == 0
        self.closed = True


def message(device: str, value: int):
    return [
        f"tactile.{device}".encode(),
        msgpack.packb({"host_time": 1.0}),
        np.full(256, value, dtype=np.uint8).tobytes(),
    ]


def test_requires_all_three_fresh_devices():
    now = [10.0]
    socket = FakeSocket()
    subscriber = TactileSubscriber(socket=socket, max_age_sec=0.1, clock=lambda: now[0])

    socket.messages.extend(message(device, i) for i, device in enumerate(TACTILE_DEVICES))
    frames = subscriber.read()
    assert frames is not None
    for i, device in enumerate(TACTILE_DEVICES):
        np.testing.assert_array_equal(frames[device], np.full(256, i, dtype=np.uint8))

    now[0] += 0.11
    assert subscriber.read() is None
    assert subscriber.status() == "stale tactile streams: vest, left_arm, right_arm"


def test_drains_to_latest_valid_frame_and_ignores_malformed_messages():
    socket = FakeSocket()
    subscriber = TactileSubscriber(socket=socket, clock=lambda: 1.0)
    socket.messages.extend(
        [
            message("vest", 1),
            message("vest", 9),
            [b"tactile.left_arm", b"invalid", bytes(256)],
            message("left_arm", 2),
            message("right_arm", 3),
        ]
    )

    frames = subscriber.read()
    assert frames is not None
    assert np.all(frames["vest"] == 9)
    assert np.all(frames["left_arm"] == 2)
    assert np.all(frames["right_arm"] == 3)

    subscriber.close()
    assert socket.closed


class StaticSource:
    def __init__(self, value):
        self.value = value

    def read(self):
        return self.value

    def get_msg(self):
        return self.value


class FakeRobotModel:
    groups = {
        "left_leg": np.arange(0, 6),
        "right_leg": np.arange(6, 12),
        "waist": np.arange(12, 15),
        "left_arm": np.arange(15, 22),
        "right_arm": np.arange(22, 29),
        "left_hand": np.arange(29, 36),
        "right_hand": np.arange(36, 43),
    }
    num_joints = 43

    def get_configuration_from_actuated_joints(self, **_kwargs):
        return np.arange(self.num_joints, dtype=np.float32)

    def get_joint_group_indices(self, group):
        return self.groups[group]


def test_observation_builder_forwards_current_stereo_state_and_tactile():
    image = np.zeros((12, 16, 3), dtype=np.uint8)
    camera = StaticSource(
        {
            "images": {"ego_view_left": image, "ego_view_right": image.copy()},
            "timestamps": {"ego_view_left": 1.0, "ego_view_right": 1.0},
        }
    )
    state = StaticSource(
        {
            "body_q": np.zeros(29, dtype=np.float32),
            "left_hand_q": np.zeros(7, dtype=np.float32),
            "right_hand_q": np.zeros(7, dtype=np.float32),
            "base_quat": np.array([1.0, 0.0, 0.0, 0.0]),
        }
    )
    tactile = StaticSource(
        {device: np.full(256, i, dtype=np.uint8) for i, device in enumerate(TACTILE_DEVICES)}
    )

    observation = prepare_observation_from_sensors(
        camera,
        state,
        FakeRobotModel(),
        "move the bucket",
        tactile_subscriber=tactile,
    )

    assert observation is not None
    assert set(observation["video"]) == {"ego_view_left", "ego_view_right"}
    assert observation["video"]["ego_view_left"].shape == (1, 1, 12, 16, 3)
    assert tuple(observation["state"]) == (
        "left_arm",
        "right_arm",
        "waist",
        "left_leg",
        "right_leg",
        "left_hand",
        "right_hand",
        "projected_gravity",
    )
    assert all(observation["tactile"][device].shape == (1, 1, 256) for device in TACTILE_DEVICES)
