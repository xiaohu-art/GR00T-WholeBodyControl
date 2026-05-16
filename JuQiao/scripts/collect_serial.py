#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jq_tactile_skin.mappings import REGIONS
from jq_tactile_skin.protocol import FrameParser, SampleAssembler, values_at


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="采集矩侨皮肤衣 WB 串口数据")
    parser.add_argument("--port", required=True, help="串口设备，例如 /dev/ttyUSB0、/dev/ttyACM0 或 COM3")
    parser.add_argument("--baud", type=int, default=921600, help="波特率，默认 921600；规格书也提到 3000000")
    parser.add_argument("--out", default="data/wb_live.csv", help="CSV 输出路径")
    parser.add_argument("--jsonl", default=None, help="可选：逐帧 JSONL 输出路径，保留完整 raw/gyro")
    parser.add_argument("--duration", type=float, default=0, help="采集秒数；0 表示一直采集直到 Ctrl-C")
    parser.add_argument("--max-samples", type=int, default=0, help="最大样本数；0 表示不限制")
    parser.add_argument("--raw-only", action="store_true", help="只导出 256 个原始通道，不导出分区域映射值")
    parser.add_argument("--calibration-samples", type=int, default=100, help="采集前用于零点校准的帧数，默认 100")
    parser.add_argument("--timeout", type=float, default=0.2, help="串口读取超时秒数")
    return parser.parse_args()


def open_jsonl(path: str | None):
    if not path:
        return None
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    return output.open("w", encoding="utf-8")


def make_header(raw_only: bool) -> list[str]:
    header = ["host_time", "sensor_type", "packet1_time", "packet2_time"]
    header.extend(f"raw_{i:03d}" for i in range(1, 257))
    header.extend(f"gyro_{i:02d}" for i in range(1, 17))
    if not raw_only:
        for region in REGIONS:
            header.extend(f"{region.key}_{i:03d}" for i in range(1, len(region.indices) + 1))
    return header


def read_next_wb_sample(ser: Any, parser: FrameParser, assembler: SampleAssembler):
    while True:
        chunk = ser.read(4096)
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
            return sample


def calibrate(ser: Any, sample_count: int) -> list[float]:
    if sample_count <= 0:
        raise SystemExit("--calibration-samples 必须大于 0；每次采集开始前都需要校准")

    parser = FrameParser()
    assembler = SampleAssembler()
    sums = [0.0] * 256

    print(f"校准中：请保持皮肤衣静止且不要按压，采集 {sample_count} 帧零点...", file=sys.stderr)
    for idx in range(sample_count):
        sample = read_next_wb_sample(ser, parser, assembler)
        for raw_index, value in enumerate(sample.raw):
            sums[raw_index] += value
        if (idx + 1) % 20 == 0 or idx + 1 == sample_count:
            print(f"校准进度：{idx + 1}/{sample_count}", file=sys.stderr)

    baseline = [value / sample_count for value in sums]
    print("校准完成，开始正式采集。", file=sys.stderr)
    return baseline


def apply_calibration(raw: bytes, baseline: list[float]) -> list[int]:
    return [max(0, int(round(value - baseline[index]))) for index, value in enumerate(raw)]


def calibration_path(csv_path: Path) -> Path:
    return csv_path.with_suffix(".calibration.json")


def write_calibration(path: Path, baseline: list[float], sample_count: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at": time.time(),
        "calibration_samples": sample_count,
        "method": "per-channel mean baseline; saved values are max(0, raw - baseline)",
        "raw_baseline": baseline,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def sample_to_row(sample, calibrated_raw: list[int], raw_only: bool) -> list[object]:
    row: list[object] = [time.time(), sample.sensor_name, sample.packet1_time, sample.packet2_time]
    row.extend(calibrated_raw)
    row.extend(sample.gyro)
    if not raw_only:
        for region in REGIONS:
            row.extend(values_at(calibrated_raw, region.indices))
    return row


def sample_to_json(sample, calibrated_raw: list[int], raw_only: bool) -> dict[str, object]:
    payload: dict[str, object] = {
        "host_time": time.time(),
        "sensor_type": sample.sensor_name,
        "calibrated": True,
        "packet1_time": sample.packet1_time,
        "packet2_time": sample.packet2_time,
        "raw": calibrated_raw,
        "gyro": list(sample.gyro),
    }
    if not raw_only:
        payload["regions"] = {
            region.key: {
                "title": region.title,
                "cols": region.cols,
                "rows": region.rows,
                "values": values_at(calibrated_raw, region.indices),
            }
            for region in REGIONS
        }
    return payload


def main() -> int:
    args = parse_args()
    try:
        import serial
    except ImportError as exc:
        raise SystemExit("缺少依赖 pyserial，请先运行：python3 -m pip install -r requirements.txt") from exc

    out = Path(args.out)
    if out.exists() and out.is_dir():
        raise SystemExit(f"--out 必须是 CSV 文件路径，不能是目录：{out}")
    out.parent.mkdir(parents=True, exist_ok=True)

    parser = FrameParser()
    assembler = SampleAssembler()
    sample_count = 0

    with serial.Serial(args.port, args.baud, timeout=args.timeout) as ser:
        baseline = calibrate(ser, args.calibration_samples)
        calibration_file = calibration_path(out)
        write_calibration(calibration_file, baseline, args.calibration_samples)
        start = time.monotonic()

        with out.open("w", newline="", encoding="utf-8") as csv_file:
            jsonl_file = open_jsonl(args.jsonl)
            try:
                writer = csv.writer(csv_file)
                writer.writerow(make_header(args.raw_only))

                print(f"开始采集：port={args.port}, baud={args.baud}, csv={out}", file=sys.stderr)
                print(f"校准文件：{calibration_file}", file=sys.stderr)
                while True:
                    if args.duration and time.monotonic() - start >= args.duration:
                        break
                    if args.max_samples and sample_count >= args.max_samples:
                        break

                    sample = read_next_wb_sample(ser, parser, assembler)
                    calibrated_raw = apply_calibration(sample.raw, baseline)
                    writer.writerow(sample_to_row(sample, calibrated_raw, args.raw_only))
                    if jsonl_file:
                        jsonl_file.write(json.dumps(sample_to_json(sample, calibrated_raw, args.raw_only), ensure_ascii=False) + "\n")
                    sample_count += 1
                    if sample_count % 100 == 0:
                        csv_file.flush()
                        if jsonl_file:
                            jsonl_file.flush()
                        print(f"已采集 {sample_count} 帧，最近类型 {sample.sensor_name}", file=sys.stderr)
            finally:
                if jsonl_file:
                    jsonl_file.close()

    print(f"采集完成：{sample_count} 帧 -> {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
