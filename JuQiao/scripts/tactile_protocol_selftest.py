#!/usr/bin/env python3
"""Offline self-test for the 3-device tactile protocol (no hardware needed).

Builds synthetic serial byte streams for each device type, feeds them through
FrameParser + SampleAssembler, and asserts device routing / payload shapes.
Also sanity-checks the arm region mapping and the device/topic constants.

Run:  python3 scripts/tactile_protocol_selftest.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jq_tactile_skin.mappings import ARM_REGION, REGIONS_BY_DEVICE
from jq_tactile_skin.protocol import (
    DEVICE_BY_SENSOR_TYPE,
    EXPECTED_DEVICES,
    HEADER,
    FrameParser,
    SampleAssembler,
    topic_for_device,
)


def _build_frame(sensor_type: int, data1: bytes, data2: bytes, gyro: bytes) -> bytes:
    """Two-packet frame: packet1 (128B) + packet2 (128B + 16B gyro)."""
    assert len(data1) == 128 and len(data2) == 128 and len(gyro) == 16
    packet1 = HEADER + bytes([0x01, sensor_type]) + data1
    packet2 = HEADER + bytes([0x02, sensor_type]) + data2 + gyro
    return packet1 + packet2


def test_device_routing_and_payload() -> None:
    parser = FrameParser()
    assembler = SampleAssembler()
    for sensor_type, expected_device in DEVICE_BY_SENSOR_TYPE.items():
        data1 = bytes((i % 256 for i in range(128)))
        data2 = bytes(((i + 50) % 256 for i in range(128)))
        gyro = bytes(range(16))
        stream = _build_frame(sensor_type, data1, data2, gyro)

        samples = []
        for packet in parser.feed(stream):
            s = assembler.add_packet(packet, received_at=0.0)
            if s is not None:
                samples.append(s)

        assert len(samples) == 1, f"type {sensor_type:#x}: expected 1 sample, got {len(samples)}"
        s = samples[0]
        assert DEVICE_BY_SENSOR_TYPE[s.sensor_type] == expected_device
        assert len(s.raw) == 256, f"raw len {len(s.raw)}"
        assert s.raw == data1 + data2
        assert len(s.gyro) == 16 and s.gyro == gyro
    print("ok: device routing + 256B payload for all three types")


def test_split_across_chunks() -> None:
    """Parser must reassemble a frame delivered in arbitrary byte chunks."""
    stream = _build_frame(0x05, bytes(128), bytes(128), bytes(16))
    parser = FrameParser()
    assembler = SampleAssembler()
    samples = []
    for i in range(0, len(stream), 7):  # awkward chunk size
        for packet in parser.feed(stream[i : i + 7]):
            s = assembler.add_packet(packet, received_at=0.0)
            if s is not None:
                samples.append(s)
    assert len(samples) == 1 and len(samples[0].raw) == 256
    print("ok: frame reassembled across fragmented chunks")


def test_constants() -> None:
    assert EXPECTED_DEVICES == {"vest", "left_arm", "right_arm"}
    assert topic_for_device("vest") == "tactile.vest"
    assert topic_for_device("left_arm") == "tactile.left_arm"
    assert topic_for_device("right_arm") == "tactile.right_arm"
    print("ok: device / topic constants")


def test_arm_region() -> None:
    assert ARM_REGION.cols * ARM_REGION.rows == 256
    assert len(ARM_REGION.indices) == 256
    assert sorted(ARM_REGION.indices) == list(range(1, 257)), "arm must cover all 256 channels once"
    assert set(REGIONS_BY_DEVICE) == {"vest", "left_arm", "right_arm"}
    assert REGIONS_BY_DEVICE["left_arm"] == [ARM_REGION]
    # Vest keeps its original 6 body regions.
    assert {r.key for r in REGIONS_BY_DEVICE["vest"]} == {
        "front_chest",
        "back",
        "left_arm",
        "left_shoulder",
        "right_arm",
        "right_shoulder",
    }
    print("ok: arm 16x16 region + per-device regions")


def main() -> int:
    test_device_routing_and_payload()
    test_split_across_chunks()
    test_constants()
    test_arm_region()
    print("\nALL PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
