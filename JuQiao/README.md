# 矩侨皮肤衣串口采集

本目录用于采集矩侨织物电子皮肤“衣服/上身”数据。项目已按《【矩侨精密】织物电子皮肤（高频版）产品规格书 V2.3》收窄为 `WB` 设备专用，不再保留手套和足底入口。

## 协议要点

- 串口波特率：默认 `921600 bps`；如果设备被配置为高速模式，可试 `3000000 bps`。
- 帧头：`AA 55 03 99`。
- 衣服/上身传感器类型：`0x05`，脚本中显示为 `WB`。
- 每个采样点分两包：
  - 第 1 包：`帧头 + 0x01 + 0x05 + 128 字节传感器数据`
  - 第 2 包：`帧头 + 0x02 + 0x05 + 128 字节传感器数据 + 16 字节陀螺仪数据`
- 脚本会自动把两包拼成一帧：`256` 个原始传感器字节 + `16` 个陀螺仪字节。

## 安装依赖

```bash
cd /home/descfly/Workplace/wzh/JuQiao_TactileSkin
conda activate juqiao
python3 -m pip install -r requirements.txt
```

如果打开 `/dev/ttyACM0` 报 `Permission denied`，长期解决方式是加入串口组后重新登录：

```bash
sudo usermod -aG dialout "$USER"
```

如果之前用 `sudo` 采集过，`data/` 目录可能会变成 root 或 nobody 所有，后续普通用户覆盖同名文件会失败。可执行：

```bash
sudo chown -R "$USER":"$USER" data reports
```


## 采集

查看串口：

```bash
ls /dev/ttyUSB* /dev/ttyACM*
sudo ls /dev/ttyUSB* /dev/ttyACM*
```

你当前识别到的是 `/dev/ttyACM0`，采集衣服数据：

```bash
python3 scripts/collect_serial.py --port /dev/ttyACM0 --baud 921600 --out data/wb_live.csv --jsonl data/wb_live.jsonl
sudo /home/descfly/miniforge3/envs/juqiao/bin/python scripts/collect_serial.py --port /dev/ttyACM0 --baud 921600 --out data/wb_live.csv --jsonl data/wb_live.jsonl
```

每次采集开始前，脚本会先自动做一次 Calibration：默认采集 `100` 帧作为零点基线，这些校准帧不会写入 CSV/JSONL。校准期间请让皮肤衣保持静止、不要按压；校准完成后脚本才会正式保存数据。

如果想增加校准帧数，例如 `200` 帧：

```bash
python3 scripts/collect_serial.py --port /dev/ttyACM0 --baud 921600 --calibration-samples 200 --out data/wb_live.csv --jsonl data/wb_live.jsonl
sudo /home/descfly/miniforge3/envs/juqiao/bin/python scripts/collect_serial.py --port /dev/ttyACM0 --baud 921600 --calibration-samples 200 --out data/wb_live.csv --jsonl data/wb_live.jsonl
```

采 30 秒：

```bash
python3 scripts/collect_serial.py --port /dev/ttyACM0 --baud 921600 --duration 30 --out data/wb_30s.csv --jsonl data/wb_30s.jsonl
sudo /home/descfly/miniforge3/envs/juqiao/bin/python scripts/collect_serial.py --port /dev/ttyACM0 --baud 921600 --duration 30 --out data/wb_30s.csv --jsonl data/wb_30s.jsonl
```

只保存原始 256 通道，不保存分区域列：

```bash
python3 scripts/collect_serial.py --port /dev/ttyACM0 --raw-only --out data/wb_raw.csv
sudo /home/descfly/miniforge3/envs/juqiao/bin/python scripts/collect_serial.py --port /dev/ttyACM0 --raw-only --out data/wb_raw.csv
```

这里的“原始 256 通道”指设备两包数据拼接后的原始数组：第 1 包 `128` 字节 + 第 2 包前 `128` 字节，导出为 `raw_001` 到 `raw_256`。这些列是设备直接发出来的通道顺序，最完整、最接近底层协议。


“分区域列”指脚本按照规格书里的区域对照关系，从 raw 数组中提取并导出的区域数据：

- `front_chest_001` 起：前胸，48 点。
- `back_001` 起：后背，40 点。
- `left_arm_001` 起：左臂，8 点。
- `left_shoulder_001` 起：左肩，4 点。
- `right_arm_001` 起：右臂，8 点。
- `right_shoulder_001` 起：右肩，4 点。

这些区域列更接近你实际要看的衣服部位，但本质上仍然来自同一份 raw 数据。


## 输出字段

CSV 字段：

- `host_time`：电脑写入该样本时的时间戳。
- `sensor_type`：正常应全部为 `WB`。
- `packet1_time`、`packet2_time`：两包数据的接收时间戳。
- `raw_001` 到 `raw_256`：拼包后的 256 字节传感器数组，已扣除本次采集开始前的校准基线。
- `gyro_01` 到 `gyro_16`：第 2 包末尾 16 字节陀螺仪数据。
- `front_chest_*`、`back_*`、`left_arm_*`、`left_shoulder_*`、`right_arm_*`、`right_shoulder_*`：按规格书区域对照关系提取后的部位数据。

JSONL 每行保存同一帧的完整结构，适合后续 Python 分析。校准信息不写入 CSV/JSONL 行内，而是保存到同名 `*.calibration.json`。

## 可视化

生成离线 HTML 可视化页面：

```bash
python3 scripts/visualize_wb.py --csv data/wb_live.csv --out reports/wb_regions.html
```

生成后直接用浏览器打开：


## 蓝牙接收器配对

如果使用蓝牙接收器且需要重新配对，先扫描：

```bash
python3 scripts/pair_receiver.py --port /dev/ttyACM0 --scan
sudo /home/descfly/miniforge3/envs/juqiao/bin/python scripts/pair_receiver.py --port /dev/ttyACM0 --scan
```

扫描并自动连接 `JQ-WB` 衣服：

```bash
python3 scripts/pair_receiver.py --port /dev/ttyACM0 --connect-wb
sudo /home/descfly/miniforge3/envs/juqiao/bin/python scripts/pair_receiver.py --port /dev/ttyACM0 --connect-wb
```

也可以直接连接指定地址：

```bash
python3 scripts/pair_receiver.py --port /dev/ttyACM0 --addr 3C8A1F2E9A36
sudo /home/descfly/miniforge3/envs/juqiao/bin/python scripts/pair_receiver.py --port /dev/ttyACM0 --addr 3C8A1F2E9A36
```
