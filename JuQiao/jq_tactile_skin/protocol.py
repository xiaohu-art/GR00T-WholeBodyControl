from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence


HEADER = bytes([0xAA, 0x55, 0x03, 0x99])
PACKET1_DATA_LEN = 128
PACKET2_DATA_LEN = 144
PACKET1_LEN = len(HEADER) + 2 + PACKET1_DATA_LEN
PACKET2_LEN = len(HEADER) + 2 + PACKET2_DATA_LEN

SENSOR_TYPES = {
    0x01: "LEFT_ARM",
    0x02: "RIGHT_ARM",
    0x05: "VEST",
}

# Single source of truth for the 3-device (清华 V1.0) tactile suit:
# each physical USB stream self-identifies via its sensor-type byte, so the
# publisher can auto-route a serial port to a device without relying on the
# (unstable) USB enumeration order.
DEVICE_BY_SENSOR_TYPE = {
    0x05: "vest",
    0x01: "left_arm",
    0x02: "right_arm",
}
EXPECTED_DEVICES = frozenset(DEVICE_BY_SENSOR_TYPE.values())

# ZMQ topic per device. Downstream subscribes to the ``TOPIC_PREFIX`` prefix
# (ZMQ prefix matching) to receive all three in one connection, then routes by
# the full topic.
TOPIC_PREFIX = "tactile"


def topic_for_device(device: str, prefix: str = TOPIC_PREFIX) -> str:
    return f"{prefix}.{device}"


# ---------------------------------------------------------------------------
# Collection layout: single- vs three-device suits share ONE mechanism.
#
# A "layout" is just the list of tactile devices to record. The 3-device 清华
# V1.0 suit (``triple``) has vest + left_arm + right_arm, each self-identifying
# via a sensor-type byte. The single 矩侨 V2.3 skin garment (``single``) is one
# device named ``body`` that does NOT need identity routing (``sensor_type`` is
# None -> the publisher accepts whatever the one port streams and skips the
# per-device / set validation). Every downstream tool (publisher, exporter,
# viewer) iterates this list, so there is no single/triple special-casing.
# ---------------------------------------------------------------------------

TACTILE_MODES = ("single", "triple")


@dataclass(frozen=True)
class TactileDeviceSpec:
    name: str  # device id, also the modality/feature suffix, e.g. "vest" / "body"
    topic: str  # ZMQ topic, always ``tactile.<name>``
    sensor_type: int | None  # expected sensor byte for routing; None -> accept any


def tactile_layout(mode: str) -> list[TactileDeviceSpec]:
    """Return the ordered device list for a collection ``mode``."""
    if mode == "triple":
        return [
            TactileDeviceSpec("vest", topic_for_device("vest"), 0x05),
            TactileDeviceSpec("left_arm", topic_for_device("left_arm"), 0x01),
            TactileDeviceSpec("right_arm", topic_for_device("right_arm"), 0x02),
        ]
    if mode == "single":
        return [TactileDeviceSpec("body", topic_for_device("body"), None)]
    raise ValueError(f"未知 tactile mode {mode!r}；可选 {TACTILE_MODES}")


@dataclass(frozen=True)
class Packet:
    order: int
    sensor_type: int
    payload: bytes

    @property
    def sensor_name(self) -> str:
        return SENSOR_TYPES.get(self.sensor_type, f"UNKNOWN_{self.sensor_type:02X}")


@dataclass(frozen=True)
class Sample:
    sensor_type: int
    raw: bytes
    gyro: bytes
    packet1_time: float
    packet2_time: float

    @property
    def sensor_name(self) -> str:
        return SENSOR_TYPES.get(self.sensor_type, f"UNKNOWN_{self.sensor_type:02X}")


class FrameParser:
    """Incrementally parse split serial packets from the JQ tactile skin protocol."""

    def __init__(self) -> None:
        self._buf = bytearray()

    def feed(self, chunk: bytes) -> list[Packet]:
        self._buf.extend(chunk)
        packets: list[Packet] = []

        while True:
            start = self._buf.find(HEADER)
            if start < 0:
                keep = max(0, len(HEADER) - 1)
                if keep:
                    del self._buf[:-keep]
                else:
                    del self._buf[:]
                return packets
            if start:
                del self._buf[:start]
            if len(self._buf) < len(HEADER) + 2:
                return packets

            order = self._buf[len(HEADER)]
            if order == 0x01:
                total_len = PACKET1_LEN
                payload_len = PACKET1_DATA_LEN
            elif order == 0x02:
                total_len = PACKET2_LEN
                payload_len = PACKET2_DATA_LEN
            else:
                del self._buf[0]
                continue

            if len(self._buf) < total_len:
                return packets

            sensor_type = self._buf[len(HEADER) + 1]
            payload_start = len(HEADER) + 2
            payload = bytes(self._buf[payload_start : payload_start + payload_len])
            packets.append(Packet(order=order, sensor_type=sensor_type, payload=payload))
            del self._buf[:total_len]


class SampleAssembler:
    """Join packet 1 and packet 2 into one 256-byte sensor sample plus 16-byte gyro."""

    def __init__(self) -> None:
        self._packet1: dict[int, tuple[bytes, float]] = {}

    def add_packet(self, packet: Packet, received_at: float) -> Sample | None:
        if packet.order == 0x01:
            self._packet1[packet.sensor_type] = (packet.payload, received_at)
            return None

        if packet.order != 0x02:
            return None

        first = self._packet1.pop(packet.sensor_type, None)
        if first is None:
            return None

        first_payload, first_time = first
        raw = first_payload + packet.payload[:128]
        gyro = packet.payload[128:144]
        return Sample(
            sensor_type=packet.sensor_type,
            raw=raw,
            gyro=gyro,
            packet1_time=first_time,
            packet2_time=received_at,
        )


def values_at(raw: Sequence[int], one_based_indices: Iterable[int]) -> list[int | None]:
    values: list[int | None] = []
    for index in one_based_indices:
        if 1 <= index <= len(raw):
            values.append(raw[index - 1])
        else:
            values.append(None)
    return values
