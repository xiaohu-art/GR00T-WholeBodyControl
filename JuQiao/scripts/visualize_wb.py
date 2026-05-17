#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jq_tactile_skin.mappings import REGIONS


RAW_COUNT = 256
GYRO_COUNT = 16


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="按区域生成矩侨皮肤衣 WB 数据离线可视化页面")
    parser.add_argument("--csv", default="data/wb_live.csv", help="输入 CSV 数据路径")
    parser.add_argument("--out", default="reports/wb_regions.html", help="输出 HTML 路径")
    parser.add_argument("--max-frames", type=int, default=2500, help="最多嵌入多少帧，避免 HTML 过大")
    parser.add_argument("--stride", type=int, default=1, help="抽帧步长，1 表示不抽帧")
    return parser.parse_args()


def numeric(value: str) -> int:
    return 0 if value == "" else int(float(value))


def read_frames(csv_path: Path, max_frames: int, stride: int) -> dict[str, object]:
    frames: list[dict[str, object]] = []
    sensor_types: set[str] = set()
    total_rows = 0

    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        raw_cols = [f"raw_{i:03d}" for i in range(1, RAW_COUNT + 1)]
        gyro_cols = [f"gyro_{i:02d}" for i in range(1, GYRO_COUNT + 1)]
        region_cols = {
            region.key: [f"{region.key}_{i:03d}" for i in range(1, len(region.indices) + 1)]
            for region in REGIONS
        }
        required_region_cols = [col for cols in region_cols.values() for col in cols]
        missing = [c for c in raw_cols + gyro_cols + required_region_cols if c not in (reader.fieldnames or [])]
        if missing:
            raise SystemExit(f"CSV 缺少必要字段：{missing[:5]}")

        for row_index, row in enumerate(reader):
            total_rows += 1
            if row_index % stride != 0:
                continue
            if len(frames) >= max_frames:
                continue

            raw = [numeric(row[c]) for c in raw_cols]
            gyro = [numeric(row[c]) for c in gyro_cols]
            sensor_type = row.get("sensor_type", "")
            sensor_types.add(sensor_type)
            regions = {
                region.key: [numeric(row[col]) for col in region_cols[region.key]]
                for region in REGIONS
            }
            frames.append(
                {
                    "t": float(row["host_time"]),
                    "sensor": sensor_type,
                    "rawSum": sum(raw),
                    "rawMax": max(raw) if raw else 0,
                    "gyro": gyro,
                    "regions": regions,
                    "regionStats": {
                        region.key: {
                            "sum": sum(v or 0 for v in regions[region.key]),
                            "max": max((v or 0 for v in regions[region.key]), default=0),
                        }
                        for region in REGIONS
                    },
                }
            )

    if not frames:
        raise SystemExit("CSV 中没有可视化帧")

    t0 = frames[0]["t"]
    for frame in frames:
        frame["dt"] = frame["t"] - t0

    return {
        "source": str(csv_path),
        "totalRows": total_rows,
        "embeddedRows": len(frames),
        "stride": stride,
        "sensorTypes": sorted(sensor_types),
        "regions": [
            {
                "key": region.key,
                "title": region.title,
                "cols": region.cols,
                "rows": region.rows,
                "count": len(region.indices),
            }
            for region in REGIONS
        ],
        "frames": frames,
    }


HTML_TEMPLATE = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>矩侨皮肤衣分区域可视化</title>
  <style>
    :root {
      --bg: #f7f8fa;
      --panel: #ffffff;
      --border: #d8dde6;
      --text: #1f2937;
      --muted: #667085;
      --accent: #0f766e;
      --accent-dark: #115e59;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--text);
    }
    header {
      padding: 16px 20px 12px;
      border-bottom: 1px solid var(--border);
      background: var(--panel);
    }
    h1 {
      margin: 0 0 6px;
      font-size: 22px;
      letter-spacing: 0;
    }
    .meta {
      display: flex;
      flex-wrap: wrap;
      gap: 10px 18px;
      color: var(--muted);
      font-size: 13px;
    }
    main {
      padding: 16px 20px 24px;
      display: grid;
      gap: 16px;
    }
    .toolbar, .panel, .stat {
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 8px;
    }
    .toolbar {
      padding: 12px;
      display: grid;
      grid-template-columns: auto minmax(220px, 1fr) auto auto auto;
      gap: 12px;
      align-items: center;
    }
    button {
      border: 1px solid var(--accent);
      background: var(--accent);
      color: white;
      height: 34px;
      padding: 0 14px;
      border-radius: 6px;
      font-weight: 650;
      cursor: pointer;
    }
    button:hover { background: var(--accent-dark); }
    input[type="range"] { width: 100%; }
    label {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      font-size: 13px;
      color: var(--muted);
      white-space: nowrap;
    }
    select {
      height: 34px;
      border: 1px solid var(--border);
      border-radius: 6px;
      background: white;
      color: var(--text);
      padding: 0 8px;
    }
    .stats {
      display: grid;
      grid-template-columns: repeat(4, minmax(120px, 1fr));
      gap: 12px;
    }
    .stat { padding: 12px; min-width: 0; }
    .stat span {
      display: block;
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 4px;
    }
    .stat strong {
      display: block;
      font-size: 20px;
      line-height: 1.2;
      overflow-wrap: anywhere;
    }
    .regions {
      display: grid;
      grid-template-columns: repeat(3, minmax(220px, 1fr));
      gap: 16px;
      align-items: start;
    }
    .charts {
      display: grid;
      grid-template-columns: minmax(320px, 1fr) minmax(320px, 1fr);
      gap: 16px;
    }
    .panel {
      padding: 12px;
      min-width: 0;
    }
    .panel h2 {
      margin: 0 0 8px;
      font-size: 15px;
      letter-spacing: 0;
    }
    canvas {
      display: block;
      width: 100%;
      background: #f2f4f7;
      border: 1px solid var(--border);
      border-radius: 6px;
    }
    .region-canvas { aspect-ratio: 1.35 / 1; }
    #summaryCanvas, #gyroCanvas { aspect-ratio: 4 / 1; }
    .hint {
      margin-top: 8px;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.45;
    }
    .colorbar {
      height: 10px;
      margin-top: 10px;
      border-radius: 999px;
      border: 1px solid var(--border);
      background: linear-gradient(90deg, #f8fafc, #99f6e4, #14b8a6, #f59e0b, #dc2626);
    }
    @media (max-width: 1100px) {
      .regions { grid-template-columns: repeat(2, minmax(220px, 1fr)); }
    }
    @media (max-width: 760px) {
      .toolbar { grid-template-columns: 1fr; }
      .stats, .regions, .charts { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <header>
    <h1>矩侨皮肤衣分区域可视化</h1>
    <div class="meta">
      <span id="sourceMeta"></span>
      <span id="rowMeta"></span>
      <span id="sensorMeta"></span>
    </div>
  </header>
  <main>
    <section class="toolbar">
      <button id="playBtn" type="button">播放</button>
      <input id="frameSlider" type="range" min="0" max="0" value="0">
      <label>帧 <strong id="frameLabel">0 / 0</strong></label>
      <label>速度
        <select id="speedSelect">
          <option value="80">慢</option>
          <option value="35" selected>中</option>
          <option value="12">快</option>
        </select>
      </label>
      <label><input id="autoScale" type="checkbox" checked> 各区自动色阶</label>
    </section>

    <section class="stats">
      <div class="stat"><span>时间</span><strong id="timeStat">0.000s</strong></div>
      <div class="stat"><span>全 raw 总和</span><strong id="rawSumStat">0</strong></div>
      <div class="stat"><span>全 raw 最大值</span><strong id="rawMaxStat">0</strong></div>
      <div class="stat"><span>当前最强区域</span><strong id="topRegionStat">-</strong></div>
    </section>

    <section id="regions" class="regions"></section>

    <section class="charts">
      <div class="panel">
        <h2>各区域压力总和</h2>
        <canvas id="summaryCanvas"></canvas>
        <div class="hint">彩色曲线表示六个区域的压力总和，竖线表示当前帧。</div>
      </div>
      <div class="panel">
        <h2>陀螺仪 16 字节</h2>
        <canvas id="gyroCanvas"></canvas>
        <div class="hint">显示当前帧第 2 包末尾 16 个 gyro 字节。</div>
      </div>
    </section>
  </main>

  <script>
    const DATA = __DATA__;
    const frames = DATA.frames;
    const regions = DATA.regions;
    const colors = ["#0f766e", "#f59e0b", "#2563eb", "#dc2626", "#7c3aed", "#16a34a"];
    const slider = document.getElementById("frameSlider");
    const playBtn = document.getElementById("playBtn");
    const speedSelect = document.getElementById("speedSelect");
    const autoScale = document.getElementById("autoScale");
    let frameIndex = 0;
    let timer = null;

    document.getElementById("sourceMeta").textContent = `来源：${DATA.source}`;
    document.getElementById("rowMeta").textContent = `总帧数：${DATA.totalRows}，嵌入：${DATA.embeddedRows}，抽帧：${DATA.stride}`;
    document.getElementById("sensorMeta").textContent = `类型：${DATA.sensorTypes.join(", ")}`;
    slider.max = Math.max(0, frames.length - 1);

    const regionRoot = document.getElementById("regions");
    regions.forEach(region => {
      const panel = document.createElement("div");
      panel.className = "panel";
      panel.innerHTML = `
        <h2>${region.title}</h2>
        <canvas id="region-${region.key}" class="region-canvas"></canvas>
        <div class="colorbar"></div>
        <div class="hint"><span id="stat-${region.key}">sum=0 max=0</span>，${region.cols}x${region.rows}，${region.count} 点</div>
      `;
      regionRoot.appendChild(panel);
    });

    const globalRegionMax = {};
    const globalRegionSumMax = {};
    regions.forEach(region => {
      globalRegionMax[region.key] = Math.max(1, ...frames.map(f => f.regionStats[region.key].max));
      globalRegionSumMax[region.key] = Math.max(1, ...frames.map(f => f.regionStats[region.key].sum));
    });
    const rawSumMax = Math.max(1, ...frames.map(f => f.rawSum));
    const rawMaxMax = Math.max(1, ...frames.map(f => f.rawMax));

    function setupCanvas(canvas) {
      const dpr = window.devicePixelRatio || 1;
      const rect = canvas.getBoundingClientRect();
      canvas.width = Math.max(1, Math.floor(rect.width * dpr));
      canvas.height = Math.max(1, Math.floor(rect.height * dpr));
      const ctx = canvas.getContext("2d");
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      return { ctx, w: rect.width, h: rect.height };
    }

    function mix(a, b, k) {
      const r = Math.round(a[0] + (b[0] - a[0]) * k);
      const g = Math.round(a[1] + (b[1] - a[1]) * k);
      const bl = Math.round(a[2] + (b[2] - a[2]) * k);
      return `rgb(${r}, ${g}, ${bl})`;
    }

    function colorFor(value, maxValue) {
      const x = Math.max(0, Math.min(1, value / Math.max(1, maxValue)));
      if (x < 0.25) return mix([248, 250, 252], [153, 246, 228], x / 0.25);
      if (x < 0.55) return mix([153, 246, 228], [20, 184, 166], (x - 0.25) / 0.30);
      if (x < 0.78) return mix([20, 184, 166], [245, 158, 11], (x - 0.55) / 0.23);
      return mix([245, 158, 11], [220, 38, 38], (x - 0.78) / 0.22);
    }

    function drawMatrix(canvasId, values, cols, rows, maxValue) {
      const canvas = document.getElementById(canvasId);
      const { ctx, w, h } = setupCanvas(canvas);
      ctx.clearRect(0, 0, w, h);
      const gap = 2;
      const cellW = (w - gap * (cols + 1)) / cols;
      const cellH = (h - gap * (rows + 1)) / rows;
      ctx.font = `${Math.max(9, Math.min(cellW, cellH) * 0.30)}px system-ui`;
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      for (let i = 0; i < cols * rows; i++) {
        const c = i % cols;
        const r = Math.floor(i / cols);
        const x = gap + c * (cellW + gap);
        const y = gap + r * (cellH + gap);
        const hasValue = i < values.length;
        const value = hasValue ? values[i] : 0;
        ctx.fillStyle = hasValue ? colorFor(value, maxValue) : "#e5e7eb";
        ctx.fillRect(x, y, cellW, cellH);
        if (hasValue && value > 0) {
          ctx.fillStyle = value / Math.max(1, maxValue) > 0.55 ? "white" : "#111827";
          ctx.fillText(String(value), x + cellW / 2, y + cellH / 2);
        }
      }
    }

    function drawAxis(ctx, w, h) {
      ctx.strokeStyle = "#d8dde6";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(0, h - 10);
      ctx.lineTo(w, h - 10);
      ctx.stroke();
    }

    function drawLine(ctx, values, maxValue, w, h, color) {
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.beginPath();
      values.forEach((v, i) => {
        const x = i / Math.max(1, values.length - 1) * w;
        const y = h - 10 - (v / Math.max(1, maxValue)) * (h - 18);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();
    }

    function drawSummary() {
      const canvas = document.getElementById("summaryCanvas");
      const { ctx, w, h } = setupCanvas(canvas);
      ctx.clearRect(0, 0, w, h);
      drawAxis(ctx, w, h);
      regions.forEach((region, idx) => {
        drawLine(ctx, frames.map(f => f.regionStats[region.key].sum), globalRegionSumMax[region.key], w, h, colors[idx]);
      });
      const x = frameIndex / Math.max(1, frames.length - 1) * w;
      ctx.strokeStyle = "#1f2937";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, h);
      ctx.stroke();
    }

    function drawGyro(values) {
      const canvas = document.getElementById("gyroCanvas");
      const { ctx, w, h } = setupCanvas(canvas);
      ctx.clearRect(0, 0, w, h);
      drawAxis(ctx, w, h);
      const maxValue = Math.max(1, ...values);
      const barW = w / values.length;
      values.forEach((v, i) => {
        const barH = v / maxValue * (h - 18);
        ctx.fillStyle = "#0f766e";
        ctx.fillRect(i * barW + 3, h - barH - 10, Math.max(1, barW - 6), barH);
      });
    }

    function render() {
      const frame = frames[frameIndex];
      slider.value = String(frameIndex);
      document.getElementById("frameLabel").textContent = `${frameIndex + 1} / ${frames.length}`;
      document.getElementById("timeStat").textContent = `${frame.dt.toFixed(3)}s`;
      document.getElementById("rawSumStat").textContent = String(frame.rawSum);
      document.getElementById("rawMaxStat").textContent = String(frame.rawMax);
      let topRegion = regions[0];
      regions.forEach(region => {
        if (frame.regionStats[region.key].sum > frame.regionStats[topRegion.key].sum) topRegion = region;
      });
      document.getElementById("topRegionStat").textContent = `${topRegion.title} ${frame.regionStats[topRegion.key].sum}`;

      regions.forEach(region => {
        const stat = frame.regionStats[region.key];
        const maxValue = autoScale.checked ? Math.max(1, stat.max) : globalRegionMax[region.key];
        drawMatrix(`region-${region.key}`, frame.regions[region.key], region.cols, region.rows, maxValue);
        document.getElementById(`stat-${region.key}`).textContent = `sum=${stat.sum} max=${stat.max}`;
      });
      drawSummary();
      drawGyro(frame.gyro);
    }

    function togglePlay() {
      if (timer) {
        clearInterval(timer);
        timer = null;
        playBtn.textContent = "播放";
        return;
      }
      playBtn.textContent = "暂停";
      timer = setInterval(() => {
        frameIndex = (frameIndex + 1) % frames.length;
        render();
      }, Number(speedSelect.value));
    }

    slider.addEventListener("input", () => {
      frameIndex = Number(slider.value);
      render();
    });
    playBtn.addEventListener("click", togglePlay);
    speedSelect.addEventListener("change", () => {
      if (timer) {
        clearInterval(timer);
        timer = null;
        togglePlay();
      }
    });
    autoScale.addEventListener("change", render);
    window.addEventListener("resize", render);
    render();
  </script>
</body>
</html>
"""


def main() -> int:
    args = parse_args()
    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise SystemExit(f"找不到 CSV 文件：{csv_path}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    data = read_frames(csv_path, args.max_frames, max(1, args.stride))
    html = HTML_TEMPLATE.replace("__DATA__", json.dumps(data, ensure_ascii=False, separators=(",", ":")))
    out.write_text(html, encoding="utf-8")
    print(f"已生成：{out}")
    print(f"总帧数：{data['totalRows']}，嵌入帧数：{data['embeddedRows']}，抽帧步长：{data['stride']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
