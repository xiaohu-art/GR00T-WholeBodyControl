"""Generic USB webcam driver using OpenCV.

No hardware SDK needed — works with any UVC-compatible camera visible as
``/dev/video*``.  Only requires ``opencv-python``.

Supports an optional ``stereo`` mode for side-by-side stereo cameras
(e.g. a single UVC device that exposes both eyes packed into one wide
frame). In stereo mode the captured frame is split in half along the
width and emitted as two mount keys: ``<mount_position>_left`` and
``<mount_position>_right``.
"""

import time
from typing import Any

import cv2
import numpy as np

try:
    import gymnasium as gym
except ImportError:
    gym = None  # type: ignore[assignment]

from gear_sonic.camera.sensor import Sensor
from gear_sonic.camera.sensor_server import CameraMountPosition


class USBCameraConfig:
    """Configuration for generic USB camera."""

    image_dim: tuple = (640, 480)
    fps: int = 30
    device_index: int = 0
    stereo: bool = False
    use_mjpeg: bool = True


class USBCameraSensor(Sensor):
    """Sensor for generic USB cameras using OpenCV VideoCapture."""

    def __init__(
        self,
        config: USBCameraConfig = USBCameraConfig(),
        mount_position: str = CameraMountPosition.EGO_VIEW.value,
        device_index: int | None = None,
    ):
        self.config = config
        self.mount_position = mount_position
        self.stereo = bool(getattr(config, "stereo", False))

        idx = device_index if device_index is not None else config.device_index

        self.cap = cv2.VideoCapture(idx)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open USB camera at index {idx}")

        # Force MJPG fourcc BEFORE setting resolution/fps. Without this, OpenCV
        # negotiates YUYV by default and the camera caps fps at low values for
        # anything bigger than 640x480.
        if getattr(config, "use_mjpeg", True):
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.image_dim[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.image_dim[1])
        self.cap.set(cv2.CAP_PROP_FPS, config.fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        print(f"[{mount_position}] Warming up USB camera (stereo={self.stereo})...")
        for _ in range(10):
            ret, _ = self.cap.read()
            if ret:
                break
            time.sleep(0.1)

        print(f"[{mount_position}] USB camera opened at index {idx}")
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        print(f"  Resolution: {width}x{height}")
        print(f"  FPS: {actual_fps}")

        if self.stereo:
            if width % 2 != 0:
                raise RuntimeError(
                    f"[{mount_position}] stereo mode requires even width, got {width}"
                )
            self._left_key = f"{mount_position}_left"
            self._right_key = f"{mount_position}_right"
            print(f"  Stereo split: {width // 2}x{height} per eye "
                  f"-> keys ({self._left_key}, {self._right_key})")

    def read(self) -> dict[str, Any] | None:
        ret, frame = self.cap.read()
        if not ret or frame is None:
            print(f"[{self.mount_position}] USB camera read failed: ret={ret}")
            return None

        t = time.time()

        if self.stereo:
            w = frame.shape[1]
            half = w // 2
            left_bgr = frame[:, :half]
            right_bgr = frame[:, half:]
            left_rgb = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2RGB)
            right_rgb = cv2.cvtColor(right_bgr, cv2.COLOR_BGR2RGB)
            return {
                "timestamps": {self._left_key: t, self._right_key: t},
                "images": {self._left_key: left_rgb, self._right_key: right_rgb},
            }

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return {
            "timestamps": {self.mount_position: t},
            "images": {self.mount_position: frame_rgb},
        }

    def serialize(self, data: dict[str, Any]) -> dict[str, Any]:
        from gear_sonic.camera.sensor_server import ImageMessageSchema

        serialized_msg = ImageMessageSchema(timestamps=data["timestamps"], images=data["images"])
        return serialized_msg.serialize()

    def observation_space(self):
        if gym is None:
            return None
        if self.stereo:
            half_w = self.config.image_dim[0] // 2
            eye_box = gym.spaces.Box(
                low=0,
                high=255,
                shape=(self.config.image_dim[1], half_w, 3),
                dtype=np.uint8,
            )
            return gym.spaces.Dict({"left": eye_box, "right": eye_box})
        return gym.spaces.Dict(
            {
                "color_image": gym.spaces.Box(
                    low=0,
                    high=255,
                    shape=(self.config.image_dim[1], self.config.image_dim[0], 3),
                    dtype=np.uint8,
                ),
            }
        )

    def close(self):
        if self.cap is not None:
            self.cap.release()
