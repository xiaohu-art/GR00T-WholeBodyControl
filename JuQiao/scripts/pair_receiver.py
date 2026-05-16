#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sys
import time
from typing import Any


DEVICE_PREFIX = "JQ-WB"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="矩侨蓝牙接收器 AT 扫描/连接辅助工具")
    parser.add_argument("--port", required=True, help="接收器串口，例如 /dev/ttyUSB0 或 COM3")
    parser.add_argument("--baud", type=int, default=921600, help="波特率，默认 921600")
    parser.add_argument("--scan", action="store_true", help="只发送 AT+SCAN=1 并打印结果")
    parser.add_argument("--connect-wb", action="store_true", help="扫描并自动连接 JQ-WB 衣服")
    parser.add_argument("--addr", help="直接连接指定蓝牙地址，例如 3C8A1F2E9A36")
    parser.add_argument("--timeout", type=float, default=8.0, help="扫描等待秒数")
    return parser.parse_args()


def send_line(ser: Any, line: str) -> None:
    ser.write((line + "\r\n").encode("ascii"))
    ser.flush()


def read_for(ser: Any, seconds: float) -> str:
    end = time.monotonic() + seconds
    chunks: list[bytes] = []
    while time.monotonic() < end:
        chunk = ser.read(4096)
        if chunk:
            chunks.append(chunk)
    return b"".join(chunks).decode("utf-8", errors="replace")


def find_wb_addr(scan_text: str) -> str | None:
    for line in scan_text.splitlines():
        if DEVICE_PREFIX not in line:
            continue
        match = re.search(r"([0-9A-Fa-f]{12})", line)
        if match:
            return match.group(1).upper()
    return None


def main() -> int:
    args = parse_args()
    try:
        import serial
    except ImportError as exc:
        raise SystemExit("缺少依赖 pyserial，请先运行：python3 -m pip install -r requirements.txt") from exc

    if not args.scan and not args.connect_wb and not args.addr:
        raise SystemExit("请指定 --scan、--connect-wb 或 --addr 之一")

    with serial.Serial(args.port, args.baud, timeout=0.2) as ser:
        if args.scan or args.connect_wb:
            send_line(ser, "AT+SCAN=1")
            scan_text = read_for(ser, args.timeout)
            print(scan_text)
            if args.scan and not args.connect_wb:
                return 0
            addr = find_wb_addr(scan_text)
            if not addr:
                raise SystemExit(f"未在扫描结果中找到 {DEVICE_PREFIX}")
        else:
            addr = args.addr.upper()

        print(f"连接 {addr}", file=sys.stderr)
        send_line(ser, f"AT+CONN={addr}")
        print(read_for(ser, 3.0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
