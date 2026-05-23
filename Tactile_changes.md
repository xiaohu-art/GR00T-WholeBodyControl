# 触觉皮肤衣接入 gear_sonic 数据采集

矩侨（JuQiao）织物电子皮肤衣的 256 路 raw 通道作为 `observation.tactile_raw`
落入 LeRobot 格式数据集，供 VLA 训练使用。

## 架构

```
G1 onboard PC                       开发机
─────────────                       ──────
皮肤衣 ──USB──► tactile_publisher.py
                  │ bind 0.0.0.0:5558
                  │ topic = "tactile"
                  │ payload = msgpack(header) + 256 bytes uint8
                  ▼
              ZMQ PUB ────网线────► ZMQ SUB
                                     │
                                     ▼
                          run_data_exporter.py
                          （由 launch_data_collection.py 编排）
                                     │
                                     ▼
                          LeRobot parquet:
                          observation.tactile_raw (uint8, shape=(256,))
```

- **Publisher 在 G1**：串口直连皮肤衣，启动时跑 100 帧零点校准（约 1–2 秒，
  期间不发数据），校准后约 14 fps 发出 ZMQ PUB。
- **Subscriber 在开发机**：data exporter 的 50 Hz 主循环每轮把 tactile socket
  队列排空、只留最新一帧；超过 100 ms 老旧则填零（与 SMPL 一致的宽松对齐策略）。
  详见文末「触觉时间错位修复」。

## 启动 CLI

### 1. G1 上启动 publisher

```bash
cd <JuQiao 路径>
python3 scripts/tactile_publisher.py \
    --port /dev/ttyACM0 \
    --zmq-host 0.0.0.0 \
    --zmq-port 5558
```

关键：`--zmq-host 0.0.0.0`，不能绑 localhost，否则开发机连不上。

### 2. 开发机上启动数据采集

```bash
python gear_sonic/scripts/launch_data_collection.py \
    --camera-host 192.168.123.164 \
    --record-tactile \
    --tactile-zmq-host 192.168.123.164
```

`192.168.123.164` 是 G1 的内网 IP，camera server 和 tactile publisher 都在
G1 上各自跑（默认端口 5555 / 5558）。

启动横幅会打印：
```
Camera:          192.168.123.164:5555
Tactile suit:    Yes (subscribing tcp://192.168.123.164:5558)
```

data exporter pane 起来后应看到：
```
[Tactile] Subscribed to 192.168.123.164:5558 (topic=tactile)
```

## 参数速查

| Launcher 参数 | 默认 | 说明 |
|---|---|---|
| `--record-tactile` | False | 开启触觉订阅与录制 |
| `--tactile-zmq-host` | `localhost` | publisher 所在 host（一般是 G1 IP） |
| `--tactile-zmq-port` | `5558` | publisher ZMQ 端口 |

Publisher 自己另有 `--port` / `--baud` / `--calibration-samples` 等，详见
`JuQiao/scripts/tactile_publisher.py --help`。

## 数据字段

写入 LeRobot dataset 的字段：

```
observation.tactile_raw  dtype=uint8  shape=(256,)
```

每个元素是某一路传感通道的**校准后** 8-bit 读数。原始通道顺序就是协议
拼包后的顺序（第 1 包 128 字节 + 第 2 包前 128 字节，`raw_001`–`raw_256`）。
不记 16 字节 gyro、不做区域重排（前胸 / 后背 / 左右臂等映射可在训练时再做）。

## 网络/链路自测

publisher 跑起来后，在开发机上独立验一下链路（绕开整套 sonic）：

```bash
.venv_data_collection/bin/python -c "
import zmq, msgpack, time
ctx = zmq.Context()
s = ctx.socket(zmq.SUB)
s.connect('tcp://192.168.123.164:5558')
s.subscribe(b'tactile')
start = time.time()
for i in range(20):
    t, h, p = s.recv_multipart()
    hdr = msgpack.unpackb(h)
    print(f'[{i:02d}] host_time={hdr[\"host_time\"]:.3f}  bytes={len(p)}')
print(f'received 20 frames in {time.time()-start:.2f}s, avg_fps={20/(time.time()-start):.1f}')
"
```

期望：20 行 `bytes=256`，~1.4 s，avg_fps ≈ 14。

## 数据落盘校验

录一段后：

```python
import pandas as pd, glob, os
latest = sorted(glob.glob("outputs/*/data/chunk-000/episode_000000.parquet"),
                key=os.path.getmtime)[-1]
df = pd.read_parquet(latest)
col = "observation.tactile_raw"
print(col in df.columns, df[col].iloc[0].shape, df[col].iloc[100][:16])
```

期望：列存在、shape (256,)、有非全零行。

## 注意事项

- **串口权限**：G1 用户需在 `dialout` 组里，否则开 `/dev/ttyACM0` 报
  Permission denied。临时方案：`sudo` 跑 publisher。
- **校准期间**：publisher 起来后约 1–2 秒不发数据，data exporter 在此期间
  会把 `observation.tactile_raw` 全部填零，**数据集开头几秒不要按 record**。
- **触觉进程崩溃**：exporter 会继续填零正常采集，不会中断；事后通过
  parquet 里整段全零行可识别坏区段。
- **G1 IP 飘移**：如果 G1 网络配置变了，每次启动改一下 `--tactile-zmq-host`
  和 `--camera-host` 即可，没有其它硬编码。

## 涉及到的文件

- `JuQiao/scripts/tactile_publisher.py` — 串口读取 + 校准 + ZMQ PUB
- `JuQiao/requirements.txt` — pyserial / pyzmq / msgpack
- `gear_sonic/scripts/run_data_exporter.py` — 新增 tactile SUB / 写入逻辑
- `gear_sonic/scripts/launch_data_collection.py` — `--record-tactile` / `--tactile-zmq-host` / `--tactile-zmq-port`
- `gear_sonic/data/features_sonic_vla.py` — `observation.tactile_raw` 字段注册
- `gear_sonic/scripts/visualize_tactile.py` — 离线可视化（OpenCV，读 parquet）
- `gear_sonic/scripts/run_tactile_viewer.py` — 实时可视化（OpenCV，订阅 ZMQ 流）

## 可视化（离线）

播放单条 episode 的触觉信号（OpenCV 窗口，原速 50 Hz）：

```bash
.venv_data_collection/bin/python gear_sonic/scripts/visualize_tactile.py \
    --parquet outputs/2026-05-21-21-36-11/data/chunk-000/episode_000000.parquet
```

布局：顶部一条压力时间线 + 当前帧游标，下面六个身体区域按 JuQiao
`mappings.REGIONS` 的 1-based 索引重排成 grid（front_chest 6×8、back 5×8、
左右肩 1×4、左右臂 2×4），颜色用 `COLORMAP_INFERNO`，亮度归一到本 episode
的 max 值。

键位：

| 键 | 作用 |
|---|---|
| `space` | 暂停 / 继续 |
| `←` `→` | 暂停时单帧步进（也可用 `,` `.`） |
| `[` `]` | 后退 / 前进 10 帧 |
| `0`–`9` | 跳到时间轴 0%–90% |
| `r` | 回到第 0 帧 |
| `q` / `ESC` | 退出 |

可选参数：`--fps`（默认 50，可降速便于观察细节）、`--window` 设置窗口标题。

依赖：matplotlib 不需要，只用 `cv2` / `numpy` / `pandas`；如果是 SSH 远程
登录，需要 X-forwarding（`ssh -X`），否则 `cv2.imshow` 起不来窗口。

## 可视化（实时）

数据采集进行中，在**另开一个终端**实时查看触觉信号（不影响录制）：

```bash
.venv_data_collection/bin/python gear_sonic/scripts/run_tactile_viewer.py \
    --tactile-zmq-host 192.168.123.164
```

`--tactile-zmq-host` 填 publisher 所在 host（一般是 G1 IP，与 launcher 的
`--tactile-zmq-host` 一致）。它直接订阅 `tactile_publisher.py` 的 ZMQ 流，
**不读 parquet、不依赖录制** —— publisher 一起来就能看。

布局复用离线 viewer 的 `_compose_frame`：六个身体区域热力图与离线版一致；
顶部时间线变成**滚动活动历史**（最近 `--history` 帧的峰值）；底部一行实时
状态：

```
LIVE  14.0 fps   rx=1234   age=70ms   vmax=19
```

- `fps` 实测接收帧率、`rx` 累计帧数、`age` 最新帧距今毫秒数、`vmax` 当前
  归一化上限；
- publisher 掉线（超过 `--stale-sec`，默认 0.5 s 无新帧）时状态行追加
  `[STALE]`。

**为什么不影响录制**：ZMQ PUB 向每个 SUB 独立分发，这个 viewer 只是又一个
**只读订阅者**，与 `run_data_exporter.py` 并行、不抢帧。viewer 每轮把 socket
队列排空、只渲染最新一帧，画面永远实时、不会越看越延迟。

参数：

| 参数 | 默认 | 说明 |
|---|---|---|
| `--tactile-zmq-host` | `localhost` | publisher 所在 host（一般 G1 IP） |
| `--tactile-zmq-port` | `5558` | publisher ZMQ 端口 |
| `--history` | `400` | 时间线显示的活动历史帧数 |
| `--stale-sec` | `0.5` | 超过该秒数无新帧则标记 `[STALE]` |
| `--window` | `tactile (live)` | OpenCV 窗口标题 |

键位：`q` / `ESC` 退出。SSH 远程登录同样需要 `ssh -X`。

## 录制开关丢键修复（manager_state 独立 socket）

数据采集时录制的开始/停止由 **PICO 手柄触发**：左 grip + A = `toggle_data_collection`
（开关录制），左 grip + B = `toggle_data_abort`（丢弃当前 episode）。
`launch_data_collection.py` 不启动键盘 publisher，故 exporter 里基于 ZMQ 5580 的
键盘路径在数据采集场景下不生效。

### 问题

手柄触发偶发「按了没反应」。

### 根因

`pico_manager_thread_server.py` 把 `toggle_data_collection` 做成**上升沿脉冲** ——
一次按键只在**一条** `manager_state` 消息里为 `True`，下一帧即复位。原先该消息与
高频的 `pose` / `planner` 共用同一个 ZMQ SUB socket：当 exporter 50 Hz 主循环偶发
卡顿（主要是 `poll_image` 读相机阻塞）时，`pose` 消息在 socket 缓冲里积压，一旦
触达 `RCVHWM` 上限，那条**唯一的瞬时 toggle 消息**被静默挤掉 → 按键丢失。此外
原 `RCVHWM` 是在 `connect()` 之后 `setsockopt` 的，按 ZMQ 语义对已建立的连接不生效。

### 改动（`run_data_exporter.py`）

- 新增独立 socket `_manager_zmq_socket`，**只订阅 `manager_state`**；`RCVHWM=1000`
  且在 `connect()` **之前**设置（确保生效）。该 topic 低频且无人竞争，缓冲不会积压。
- `_sonic_zmq_socket` 去掉 `manager_state` 订阅，只保留 `pose` / `planner`；其
  `RCVHWM=20` 等原配置**不变** —— 高频数据维持「小缓冲、落后即丢旧保新」的原策略。
- 新增 `_poll_manager_state_zmq()`：**无上限完整排空** manager_state socket（低频，
  成本可忽略），在 `_poll_sonic_zmq_messages()` 开头调用。
- `save_and_cleanup()` 一并关闭新 socket。

`_handle_manager_state` / `_check_recording_commands` 等录制状态逻辑未改动。

### 效果

按键 toggle 独占一根无人竞争的低频管道，主循环再卡也不会被 `pose` 积压冲掉。
改动全部在开发机端 exporter，不涉及 G1、`pico_manager` 或 Sonic 本体。

## 触觉时间错位修复（drain-to-latest）

### 问题

采集时 `run_tactile_viewer.py`（实时 viewer）看到的触觉信号，和事后用
`visualize_tactile.py` 回放 parquet 看到的对不上 —— 录进数据集的触觉相对
sonic 原生模态有一个会变化的滞后。

### 根因

`run_data_exporter.py` 的 tactile SUB socket 上 `CONFLATE=1` 完全无效：一是
设在 `connect()` 之后（ZMQ 要求在 connect 之前设才生效），二是 publisher 发的
是 `[topic, header, payload]` 3 段 multi-part 消息，CONFLATE 根本不支持
multi-part。于是该 socket 退化成普通 FIFO 队列，而 `_poll_tactile_zmq()` 每个
主循环只 `recv` 一次、取**最旧**一帧。

平时 50 Hz 循环快过 14 fps 来帧、队列浅没事；但主循环一卡（`poll_image` 读
相机偶发阻塞，与录制丢键 bug 同源），触觉帧在 FIFO 里积压，卡完后 exporter
一帧一帧慢慢排，这期间每条录制行写进去的都是滞后的旧帧 —— sonic 原生数据走
另一条 socket 始终接近实时，于是触觉相对 sonic 越拖越后。

`receive_timestamp` 取的是「收到那一刻」，旧帧出队时时间戳被刷新成「现在」，
所以 100 ms 老化保护（`tactile_max_age_sec`）也拦不住积压的旧帧。

### 改动（`run_data_exporter.py`）

- tactile socket 去掉无效的 `CONFLATE`；`RCVHWM=1000` 在 `connect()` **之前**
  设（卡顿期间整段 backlog 都留着，排空后能恢复到真正最新帧；HWM 太小反而会
  在上限处丢掉最新帧、让排空落在旧帧上）。
- `_poll_tactile_zmq()` 改成 `while` 循环排空队列、只保留最新一帧 —— 和
  `run_tactile_viewer.py` 实时 viewer 的做法一致。卡顿积压的 backlog 一轮即可
  排空，立即回到实时。

### 效果

录进 parquet 的触觉帧始终是排空时刻的最新帧，与 sonic 原生模态对齐；
`run_tactile_viewer.py` 实时所见 == `visualize_tactile.py` 回放所见。改动全部
在开发机端 exporter，不涉及 G1 / publisher。
