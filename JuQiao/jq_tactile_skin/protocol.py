from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence


HEADER = bytes([0xAA, 0x55, 0x03, 0x99])
PACKET1_DATA_LEN = 128
PACKET2_DATA_LEN = 144
PACKET1_LEN = len(HEADER) + 2 + PACKET1_DATA_LEN
PACKET2_LEN = len(HEADER) + 2 + PACKET2_DATA_LEN

SENSOR_TYPES = {
    0x05: "WB",
}


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
