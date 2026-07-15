#!/usr/bin/env python3
"""ZMQ publisher for the 3-device JuQiao tactile skin suit (清华 V1.0).

The suit now has three independent USB serial devices — a short-sleeve vest
plus two wrap-around arm sleeves — each with its own acquisition board and
serial link. Every device speaks the *same* JQ Industries wire protocol
(header ``AA 55 03 99``, two packets, 256 data bytes + 16 gyro bytes) and
self-identifies via its sensor-type byte:

    0x05 -> vest        0x01 -> left_arm        0x02 -> right_arm

This publisher opens all three ports (order-independent), auto-routes each
port to a device by reading its sensor-type byte, performs per-device
zero-point calibration, and publishes each calibrated 256-byte sample on a
single ZMQ PUB socket under a per-device topic so the gear_sonic data exporter
can subscribe in real time. The 16-byte gyro payload is intentionally dropped.

Wire format (multipart), one message per device per frame:
    [ b"tactile.<device>", msgpack(header), bytes(256 calibrated raw values) ]

    device      = "vest" | "left_arm" | "right_arm"

Header fields:
    host_time    (float, time.time() at packet 2 reception)
    packet1_time (float)
    packet2_time (float)
    device       (str)
    shape        ([256])
    dtype        ("uint8")
    calibrated   (True)

All three devices must be plugged in and streaming; if any port produces no
data within ``--probe-timeout`` seconds, or the detected devices are not
exactly {vest, left_arm, right_arm} (missing / duplicate), the publisher exits
with an error.

Threading model: one worker thread per serial port reads + calibrates +
enqueues (topic, header, payload) tuples; the main thread is the *only* thread
that touches the ZMQ socket (libzmq sockets are not thread-safe), draining the
queue and sending.

Mode switch (``--tactile-mode``): the above describes ``triple`` (default, the
3-device 清华 V1.0 suit). ``single`` supports the older 矩侨 V2.3 single skin
garment — exactly one port, no sensor-type routing/validation, published as one
device ``body`` on topic ``tactile.body``. Both share the same code path via
``tactile_layout()``.
"""

from __future__ import annotations

import argparse
import queue
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jq_tactile_skin.protocol import (  # noqa: E402
    DEVICE_BY_SENSOR_TYPE,
    EXPECTED_DEVICES,
    TACTILE_MODES,
    FrameParser,
    SampleAssembler,
    TactileDeviceSpec,
    tactile_layout,
    topic_for_device,
)

RAW_LEN = 256


def iter_samples(ser: Any, parser: FrameParser, assembler: SampleAssembler, stop: threading.Event):
    """Yield assembled 256-byte samples from a serial-like byte stream.

    Unlike the old single-device publisher this does *not* filter by sensor
    type: every known device type is accepted and identified by the caller.
    Returns when ``stop`` is set.
    """
    while not stop.is_set():
        chunk = ser.read(ser.in_waiting or 4096)
        if not chunk:
            continue
        received_at = time.time()
        for packet in parser.feed(chunk):
            sample = assembler.add_packet(packet, received_at)
            if sample is None:
                continue
            yield sample


class Worker:
    """State for one serial port: its thread, detected device, and status.

    ``spec`` is set in single-device mode: the device identity is fixed up
    front (no sensor-type probing), so the worker skips detection/validation
    and accepts whatever the one port streams. In triple mode ``spec`` is None
    and the device is discovered from its sensor-type byte.
    """

    def __init__(self, port: str, spec: TactileDeviceSpec | None = None) -> None:
        self.port = port
        self.spec = spec
        self.device: str | None = None
        self.probed = threading.Event()  # set once the device type is known
        self.ready = threading.Event()  # set once calibration is done
        self.recalibrate = threading.Event()  # set to request a live zero-point re-cal
        self.error: BaseException | None = None
        self.thread: threading.Thread | None = None


def _worker_main(
    w: Worker,
    args: argparse.Namespace,
    out_queue: "queue.Queue",
    stop: threading.Event,
    serial_mod: Any,
    msgpack_mod: Any,
) -> None:
    """Read one port: detect device -> calibrate -> publish (enqueue) forever."""
    try:
        with serial_mod.Serial(w.port, args.baud, timeout=args.timeout) as ser:
            parser = FrameParser()
            assembler = SampleAssembler()
            samples = iter_samples(ser, parser, assembler, stop)

            # Fixed identity (single-device mode): the device is known up front,
            # so we set it immediately and accept any sensor type. Otherwise
            # (triple mode) discover it from the first frame's sensor byte.
            fixed = w.spec is not None
            if fixed:
                device = w.spec.name
                w.device = device
                w.probed.set()
                print(f"[tactile-pub] {w.port} -> {device} (single, 不校验类型)", file=sys.stderr)

            # Accumulate the zero-point baseline over calibration frames, and in
            # triple mode identify/validate the device in the same pass.
            sums = [0.0] * RAW_LEN
            device = w.spec.name if fixed else None
            collected = 0
            for sample in samples:
                if stop.is_set():
                    return
                if not fixed:
                    dev = DEVICE_BY_SENSOR_TYPE.get(sample.sensor_type)
                    if dev is None:
                        raise RuntimeError(
                            f"{w.port}: 未知传感器类型 0x{sample.sensor_type:02X}"
                        )
                    if device is None:
                        device = dev
                        w.device = dev
                        w.probed.set()
                        print(f"[tactile-pub] {w.port} -> {device}", file=sys.stderr)
                    elif dev != device:
                        raise RuntimeError(
                            f"{w.port}: 同一串口出现多种传感器类型（{device} 与 {dev}）"
                        )
                for i, value in enumerate(sample.raw):
                    sums[i] += value
                collected += 1
                if collected >= args.calibration_samples:
                    break

            if device is None or collected == 0:
                raise RuntimeError(f"{w.port}: 未收到任何数据帧")

            baseline = [s / collected for s in sums]
            topic = w.spec.topic if fixed else topic_for_device(device)
            topic_bytes = topic.encode("utf-8")
            print(
                f"[tactile-pub] {device} 校准完成（{collected} 帧），开始发布",
                file=sys.stderr,
            )
            w.ready.set()

            # Publish loop: enqueue; the main thread owns the socket.
            for sample in samples:
                if stop.is_set():
                    break

                # Live re-calibration: when requested (e.g. operator pressed both
                # PICO grips because the suit has drifted / crept), average the
                # next `recalibration_samples` frames into a fresh baseline. The
                # suit must be at rest during this short window. Frames consumed
                # for the baseline are not published.
                if w.recalibrate.is_set():
                    sums2 = [float(v) for v in sample.raw]
                    c2 = 1
                    for s2 in samples:
                        if stop.is_set():
                            break
                        for i, value in enumerate(s2.raw):
                            sums2[i] += value
                        c2 += 1
                        if c2 >= args.recalibration_samples:
                            break
                    baseline = [s / c2 for s in sums2]
                    w.recalibrate.clear()
                    print(
                        f"[tactile-pub] {device} 重新校准完成（{c2} 帧）",
                        file=sys.stderr,
                    )
                    continue

                calibrated = bytes(
                    max(0, int(round(value - baseline[idx])))
                    for idx, value in enumerate(sample.raw)
                )
                header = msgpack_mod.packb(
                    {
                        "host_time": time.time(),
                        "packet1_time": sample.packet1_time,
                        "packet2_time": sample.packet2_time,
                        "device": device,
                        "shape": [RAW_LEN],
                        "dtype": "uint8",
                        "calibrated": True,
                    },
                    use_bin_type=True,
                )
                try:
                    out_queue.put_nowait((topic_bytes, header, calibrated))
                except queue.Full:
                    # Real-time priority: drop the oldest frame, keep the newest.
                    try:
                        out_queue.get_nowait()
                        out_queue.put_nowait((topic_bytes, header, calibrated))
                    except (queue.Empty, queue.Full):
                        pass
    except BaseException as exc:  # noqa: BLE001 - surface to main thread
        w.error = exc
        # Unblock any waiter and abort the whole publisher: an incomplete suit
        # must fail loudly rather than record partial data.
        w.probed.set()
        w.ready.set()
        stop.set()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ZMQ publisher for the 3-device JuQiao tactile skin suit"
    )
    parser.add_argument(
        "--tactile-mode",
        choices=TACTILE_MODES,
        default="triple",
        help="triple=3设备清华皮肤衣(vest/left_arm/right_arm，默认)；"
        "single=旧矩侨单皮肤衣(1个设备 body)",
    )
    parser.add_argument(
        "--ports",
        required=True,
        help="逗号分隔的串口设备(顺序无所谓)：triple 需 3 个(如 "
        "/dev/ttyACM0,/dev/ttyACM1,/dev/ttyACM2)，single 需 1 个",
    )
    parser.add_argument("--baud", type=int, default=921600, help="波特率，默认 921600")
    parser.add_argument("--zmq-host", default="0.0.0.0", help="ZMQ 绑定地址，默认 0.0.0.0")
    parser.add_argument("--zmq-port", type=int, default=5558, help="ZMQ 端口，默认 5558")
    parser.add_argument(
        "--calibration-samples",
        type=int,
        default=100,
        help="每台设备启动时的零点校准帧数，默认 100；期间不会发布数据",
    )
    parser.add_argument(
        "--recalibration-samples",
        type=int,
        default=50,
        help="收到重新校准信号后重采的零点帧数，默认 50；期间该设备不发布",
    )
    parser.add_argument(
        "--manager-host",
        default="",
        help="PICO manager 的地址；设置后订阅其 'tactile_calibrate' 信号以支持在线重新校准"
        "（双手 grip 同按触发）。留空则禁用在线重新校准。",
    )
    parser.add_argument(
        "--manager-port",
        type=int,
        default=5556,
        help="PICO manager 的 PUB 端口，默认 5556",
    )
    parser.add_argument("--timeout", type=float, default=0.2, help="串口读取超时（秒）")
    parser.add_argument(
        "--warmup-sec",
        type=float,
        default=0.5,
        help="bind 后等待订阅者连接的预热时间（秒），默认 0.5",
    )
    parser.add_argument(
        "--probe-timeout",
        type=float,
        default=5.0,
        help="每个串口首帧探测超时（秒）；超时视为设备未插/未通电，默认 5.0",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    ports = [p.strip() for p in args.ports.split(",") if p.strip()]
    layout = tactile_layout(args.tactile_mode)
    expected_n = len(layout)
    if len(ports) != expected_n:
        raise SystemExit(
            f"--tactile-mode {args.tactile_mode} 需要恰好 {expected_n} 个串口，"
            f"当前收到 {len(ports)} 个：{ports}"
        )

    try:
        import serial
    except ImportError as exc:
        raise SystemExit(
            "缺少依赖 pyserial，请先运行：python3 -m pip install -r requirements.txt"
        ) from exc
    try:
        import zmq
    except ImportError as exc:
        raise SystemExit(
            "缺少依赖 pyzmq，请先运行：python3 -m pip install -r requirements.txt"
        ) from exc
    try:
        import msgpack
    except ImportError as exc:
        raise SystemExit(
            "缺少依赖 msgpack，请先运行：python3 -m pip install -r requirements.txt"
        ) from exc

    ctx = zmq.Context()
    sock = ctx.socket(zmq.PUB)
    sock.setsockopt(zmq.SNDHWM, 100)
    sock.setsockopt(zmq.LINGER, 0)
    bind_endpoint = f"tcp://{args.zmq_host}:{args.zmq_port}"
    sock.bind(bind_endpoint)
    print(f"[tactile-pub] ZMQ PUB bound at {bind_endpoint}", file=sys.stderr)

    # Optional: subscribe to the PICO manager's "tactile_calibrate" signal so the
    # operator can re-zero all devices online (both grips pressed together). We
    # only prefix-subscribe to that bare topic, so no pose-message parsing is
    # needed and the manager_state stream is untouched.
    cal_sock = None
    if args.manager_host:
        cal_sock = ctx.socket(zmq.SUB)
        cal_sock.setsockopt(zmq.RCVHWM, 10)
        cal_sock.setsockopt_string(zmq.SUBSCRIBE, "tactile_calibrate")
        cal_endpoint = f"tcp://{args.manager_host}:{args.manager_port}"
        cal_sock.connect(cal_endpoint)
        print(
            f"[tactile-pub] 在线重新校准已启用：SUB connected to {cal_endpoint} "
            "(topic='tactile_calibrate')",
            file=sys.stderr,
        )

    stop = threading.Event()

    def _sigint(signum, frame):  # noqa: ARG001
        stop.set()

    signal.signal(signal.SIGINT, _sigint)
    signal.signal(signal.SIGTERM, _sigint)

    out_queue: "queue.Queue" = queue.Queue(maxsize=1000)
    if args.tactile_mode == "single":
        # One fixed device: no sensor-type routing, so pin the spec up front.
        workers = [Worker(ports[0], spec=layout[0])]
    else:
        workers = [Worker(p) for p in ports]
    for w in workers:
        w.thread = threading.Thread(
            target=_worker_main,
            args=(w, args, out_queue, stop, serial, msgpack),
            name=f"tactile-{Path(w.port).name}",
            daemon=True,
        )
        w.thread.start()

    sample_count = 0
    try:
        # Phase 1: wait for every port to detect its device (or fail / time out).
        deadline = time.time() + args.probe_timeout
        while not all(w.probed.is_set() for w in workers):
            if stop.is_set():
                break
            if time.time() > deadline:
                missing = [w.port for w in workers if not w.probed.is_set()]
                stop.set()
                raise SystemExit(
                    f"探测超时：以下串口 {args.probe_timeout}s 内无数据，请检查设备连接："
                    f"{missing}"
                )
            time.sleep(0.05)

        _raise_worker_errors(workers)

        # Phase 2: (triple only) validate the detected set is exactly the
        # expected devices. Single mode has one fixed device, nothing to check.
        if args.tactile_mode == "triple":
            detected = [w.device for w in workers]
            if sorted(detected) != sorted(EXPECTED_DEVICES):
                missing = sorted(EXPECTED_DEVICES - set(detected))
                dupes = sorted({d for d in detected if detected.count(d) > 1})
                raise SystemExit(
                    "设备身份校验失败："
                    f"检测到 {detected}；缺失 {missing or '无'}；重复 {dupes or '无'}。"
                    "请确认短袖(0x05)、左臂(0x01)、右臂(0x02)三台设备均已连接。"
                )

        # Phase 3: wait for all per-device calibrations to finish.
        while not all(w.ready.is_set() for w in workers):
            if stop.is_set():
                break
            time.sleep(0.05)
        _raise_worker_errors(workers)

        print(
            f"[tactile-pub] {len(workers)} 台设备({args.tactile_mode})均已就绪，开始发布...",
            file=sys.stderr,
        )
        time.sleep(args.warmup_sec)

        # Phase 4: drain the queue and publish (single-threaded socket access).
        while not stop.is_set():
            # Non-blocking check for an online re-calibration request. Any frame
            # on the subscribed topic triggers a re-zero of all three devices.
            if cal_sock is not None:
                got_signal = False
                while True:
                    try:
                        cal_sock.recv(zmq.NOBLOCK)
                        got_signal = True
                    except zmq.Again:
                        break
                    except Exception:
                        break
                if got_signal:
                    for w in workers:
                        w.recalibrate.set()
                    print(
                        f"[tactile-pub] 收到重新校准信号，{len(workers)} 台设备将重新采集零点基线",
                        file=sys.stderr,
                    )

            try:
                topic_bytes, header, payload = out_queue.get(timeout=0.2)
            except queue.Empty:
                _raise_worker_errors(workers)
                continue
            try:
                sock.send_multipart([topic_bytes, header, payload], flags=zmq.DONTWAIT)
            except zmq.Again:
                pass
            sample_count += 1
            if sample_count % 500 == 0:
                print(f"[tactile-pub] published {sample_count} frames", file=sys.stderr)
    finally:
        stop.set()
        for w in workers:
            if w.thread is not None:
                w.thread.join(timeout=1.0)
        try:
            sock.close()
        except Exception:
            pass
        if cal_sock is not None:
            try:
                cal_sock.close()
            except Exception:
                pass
        try:
            ctx.term()
        except Exception:
            pass
        print(f"[tactile-pub] shutdown, published {sample_count} frames", file=sys.stderr)

    return 0


def _raise_worker_errors(workers: list[Worker]) -> None:
    for w in workers:
        if w.error is not None:
            raise SystemExit(f"[tactile-pub] 串口 {w.port} 出错：{w.error}")


if __name__ == "__main__":
    raise SystemExit(main())
