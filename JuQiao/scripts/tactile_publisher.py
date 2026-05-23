#!/usr/bin/env python3
"""ZMQ publisher for JuQiao tactile skin data.

Reads frames from the JQ Industries serial protocol, performs zero-point
calibration, and publishes each calibrated 256-byte sample on a ZMQ PUB
socket so the gear_sonic data exporter can subscribe in real time.

Wire format (multipart):
    [ b"tactile", msgpack(header), bytes(256 calibrated raw values) ]

Header fields:
    host_time   (float, time.time() at packet 2 reception)
    packet1_time (float)
    packet2_time (float)
    shape        ([256])
    dtype        ("uint8")
    calibrated   (True)
"""

from __future__ import annotations

import argparse
import signal
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jq_tactile_skin.protocol import FrameParser, SampleAssembler  # noqa: E402


# Calibration / serial-read helpers are intentionally inlined here (rather than
# imported from collect_serial.py) so this publisher is self-contained: it can
# be copied to the G1 on its own and only depends on jq_tactile_skin/.
def iter_wb_samples(ser: Any, parser: FrameParser, assembler: SampleAssembler):
    """Yield assembled 256-byte WB samples from a serial-like byte stream."""
    while True:
        chunk = ser.read(ser.in_waiting or 4096)
        if not chunk:
            continue
        received_at = time.time()
        for packet in parser.feed(chunk):
            sample = assembler.add_packet(packet, received_at)
            if sample is None:
                continue
            if sample.sensor_name != "WB":
                print(f"跳过非衣服数据帧：{sample.sensor_name}", file=sys.stderr)
                continue
            yield sample


def calibrate(ser: Any, sample_count: int) -> list[float]:
    """Average ``sample_count`` idle frames into a per-channel zero baseline."""
    if sample_count <= 0:
        raise SystemExit("--calibration-samples 必须大于 0；每次采集开始前都需要校准")

    parser = FrameParser()
    assembler = SampleAssembler()
    samples = iter_wb_samples(ser, parser, assembler)
    sums = [0.0] * 256

    print(f"校准中：请保持皮肤衣静止且不要按压，采集 {sample_count} 帧零点...", file=sys.stderr)
    for idx in range(sample_count):
        sample = next(samples)
        for raw_index, value in enumerate(sample.raw):
            sums[raw_index] += value
        if (idx + 1) % 20 == 0 or idx + 1 == sample_count:
            print(f"校准进度：{idx + 1}/{sample_count}", file=sys.stderr)

    baseline = [value / sample_count for value in sums]
    print("校准完成，开始正式采集。", file=sys.stderr)
    return baseline


def apply_calibration(raw: bytes, baseline: list[float]) -> list[int]:
    """Subtract the baseline from a raw frame, clamping negative values to zero."""
    return [max(0, int(round(value - baseline[index]))) for index, value in enumerate(raw)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ZMQ publisher for JuQiao tactile skin (WB)")
    parser.add_argument("--port", required=True, help="串口设备,例如 /dev/ttyACM0")
    parser.add_argument("--baud", type=int, default=921600, help="波特率,默认 921600")
    parser.add_argument("--zmq-host", default="0.0.0.0", help="ZMQ 绑定地址,默认 0.0.0.0")
    parser.add_argument("--zmq-port", type=int, default=5558, help="ZMQ 端口,默认 5558")
    parser.add_argument("--topic", default="tactile", help="ZMQ topic 名,默认 tactile")
    parser.add_argument(
        "--calibration-samples",
        type=int,
        default=100,
        help="启动时零点校准帧数,默认 100;期间不会发布数据",
    )
    parser.add_argument("--timeout", type=float, default=0.2, help="串口读取超时(秒)")
    parser.add_argument(
        "--warmup-sec",
        type=float,
        default=0.5,
        help="bind 后等待订阅者连接的预热时间(秒),默认 0.5",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        import serial
    except ImportError as exc:
        raise SystemExit(
            "缺少依赖 pyserial,请先运行:python3 -m pip install -r requirements.txt"
        ) from exc
    try:
        import zmq
    except ImportError as exc:
        raise SystemExit(
            "缺少依赖 pyzmq,请先运行:python3 -m pip install -r requirements.txt"
        ) from exc
    try:
        import msgpack
    except ImportError as exc:
        raise SystemExit(
            "缺少依赖 msgpack,请先运行:python3 -m pip install -r requirements.txt"
        ) from exc

    topic_bytes = args.topic.encode("utf-8")

    ctx = zmq.Context()
    sock = ctx.socket(zmq.PUB)
    sock.setsockopt(zmq.SNDHWM, 100)
    sock.setsockopt(zmq.LINGER, 0)
    bind_endpoint = f"tcp://{args.zmq_host}:{args.zmq_port}"
    sock.bind(bind_endpoint)
    print(f"[tactile-pub] ZMQ PUB bound at {bind_endpoint} (topic={args.topic!r})", file=sys.stderr)

    stop = {"flag": False}

    def _sigint(signum, frame):  # noqa: ARG001
        stop["flag"] = True

    signal.signal(signal.SIGINT, _sigint)
    signal.signal(signal.SIGTERM, _sigint)

    sample_count = 0
    try:
        with serial.Serial(args.port, args.baud, timeout=args.timeout) as ser:
            baseline = calibrate(ser, args.calibration_samples)
            print("[tactile-pub] 校准完成,开始发布...", file=sys.stderr)

            time.sleep(args.warmup_sec)

            parser = FrameParser()
            assembler = SampleAssembler()

            for sample in iter_wb_samples(ser, parser, assembler):
                if stop["flag"]:
                    break
                calibrated = apply_calibration(sample.raw, baseline)
                payload = bytes(calibrated)
                header = msgpack.packb(
                    {
                        "host_time": time.time(),
                        "packet1_time": sample.packet1_time,
                        "packet2_time": sample.packet2_time,
                        "shape": [256],
                        "dtype": "uint8",
                        "calibrated": True,
                    },
                    use_bin_type=True,
                )
                try:
                    sock.send_multipart(
                        [topic_bytes, header, payload], flags=zmq.DONTWAIT
                    )
                except zmq.Again:
                    pass

                sample_count += 1
                if sample_count % 500 == 0:
                    print(
                        f"[tactile-pub] published {sample_count} frames",
                        file=sys.stderr,
                    )
    finally:
        try:
            sock.close()
        except Exception:
            pass
        try:
            ctx.term()
        except Exception:
            pass
        print(f"[tactile-pub] shutdown, published {sample_count} frames", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
