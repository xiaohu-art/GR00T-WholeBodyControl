# 触觉采集:单设备 / 三设备 模式开关 —— 设计文档

日期:2026-07-14 · 分支:`tactile`

## 背景与目标

当前 `tactile` 分支的触觉采集只支持**三设备(清华 V1.0)**方案:一件短袖背心(vest)+ 左臂袖套(left_arm)+ 右臂袖套(right_arm),三个独立 USB 串口。之前开发时把旧的**单皮肤衣(矩侨精密 高频版 V2.3,单 USB)**模式去掉了,导致只有一件皮肤衣的旧硬件无法采集。

目标:在数据采集链路上加一个**开关**,可选 `single`(1 个触觉设备)或 `triple`(3 个触觉设备),默认保持 `triple`(当前行为不变)。

> 说明:不需要兼容之前采过的单皮肤衣数据集,因此**不**沿用旧的 `observation.tactile_raw` schema。改为把 single 和 triple 统一成同一套 per-device 命名,让代码保持最简洁。

## 关键约定

- 统一词表:所有涉及触觉的工具都用同一个开关 `--tactile-mode {single,triple}`,默认 `triple`。
- **源端(publisher)和汇端(exporter)必须设成同一个 mode**,这是一条书面约定(和 `--stereo-ego-view` 两端必须一致同理)。不一致时不会崩溃,但会导致设备被零填充(数据里可见),文档中明确提示。
- single 与 triple 走**完全相同**的机制:一个"触觉设备列表(layout)",只是设备数量不同。single = 单一设备 `body`;triple = `vest / left_arm / right_arm`。所有下游代码对该列表做纯遍历,**无 single/triple 特殊分支**。

## 各模式的 Schema(统一 per-device 方案)

| mode | ZMQ topic | 数据集 key | modality 条目 |
|---|---|---|---|
| `single` | `tactile.body` | `observation.tactile_body` | `tactile.body` |
| `triple` | `tactile.vest` / `tactile.left_arm` / `tactile.right_arm` | `observation.tactile_{vest,left_arm,right_arm}` | `tactile.{vest,left_arm,right_arm}` |

命名规则一致:topic 恒为 `tactile.<device>`,数据集 key 恒为 `observation.tactile_<device>`,modality 子键恒为 `<device>`。单设备名定为 `body`(矩侨整体皮肤衣;如需可改 `skin`/`torso`)。

## 设备 layout(唯一抽象)

引入按 mode 选择的设备列表,每项含:`name`(用于日志/身份)、`topic`(`tactile.<name>`)、`feature_key`(`observation.tactile_<name>`)、`sensor_type`(triple 用于串口自动路由/校验;single 为 `None` 表示不探测、不校验、接受任意 sensor 字节)。

- `single` → `[Device("body", sensor_type=None)]`
- `triple` → `[Device("vest", 0x05), Device("left_arm", 0x01), Device("right_arm", 0x02)]`

## 逐文件改动

### 1. `JuQiao/scripts/tactile_publisher.py`(G1 端发布器)
- 新增 `--tactile-mode {single,triple}`,默认 `triple`。
- `triple`:行为不变(3 个串口、按 sensor-type 字节自动路由、精确校验集合为 {vest,left_arm,right_arm}、每设备一个 topic)。
- `single`:`--ports` 恰好 1 个串口;因 layout 里 `sensor_type is None`,worker **跳过** sensor-type 探测与集合校验,固定为设备 `body`(topic `tactile.body`),接受任意 sensor 字节。零点校准、双手 grip 在线重新校准、线程/队列模型都不变(只有一个 worker)。
- 小重构:worker 身份由按 mode 选择的 layout 决定;`_worker_main` 的探测/校验逻辑按 `sensor_type is None` 分支。

### 2. `gear_sonic/scripts/run_data_exporter.py`(采集/导出)
- 新增配置 `tactile_mode: str = "triple"`(CLI `--tactile-mode`)。
- 用按 mode 选择的 layout(`(topic_bytes, feature_key)` 列表)替换硬编码的 `_tactile_devices` 与 `_tactile_topic_to_device`:
  - `single` → `[(b"tactile.body", "observation.tactile_body")]`
  - `triple` → 现有三条
- `_add_tactile_to_frame_data`:遍历 layout,逐设备按 `tactile_max_age_sec` 独立零填充(逻辑不变,只是数据驱动)。
- schema 构建处(约 1080 行)把 `config.tactile_mode` 传入 feature / modality 两个辅助函数。
- ZMQ 订阅前缀仍是 `"tactile"`(前缀匹配同时覆盖两种 mode 的 `tactile.*`,无需改)。

### 3. `gear_sonic/data/features_sonic_vla.py`
- `get_tactile_features(mode="triple")` / `get_tactile_modality_config(mode="triple")` 按 mode 返回单或三设备 schema;默认 `triple`,保证现有两个无参调用点不受影响。

### 4. `gear_sonic/scripts/launch_data_collection.py`(采集启动器)
- 新增 `tactile_mode: str = "triple"` 配置;当 `record_tactile` 时,把 `--tactile-mode {mode}` 透传给 exporter 命令。
- 更新启动横幅/日志,显示当前 mode 及对应的设备/key。
- publisher 由操作员手动启动(不由 launcher 拉起),因此操作员需把 publisher 的 `--tactile-mode` 设成一致 —— 在 tutorial 中说明。

### 5. `gear_sonic/scripts/run_tactile_viewer.py`(主机端可视化)
- 新增 `--tactile-mode {single,triple}`,默认 `triple`。
- `single` → 单个窗口订阅 topic `tactile.body`,用 vest 那套 body-region 渲染器绘制;`triple` 不变。

### 6. `gear_sonic/scripts/visualize_tactile.py` / `replay_episode.py`(离线回放,最小改动)
- `visualize_tactile.py` 现在硬编码 `TACTILE_COL = "observation.tactile_raw"`;该 key 两种新 mode 都不再产出。改为可配置 `--tactile-key`,默认 `observation.tactile_body`,使 single 模式数据能回放;triple 可显式指定某个设备 key(如 `observation.tactile_vest`)。
- `replay_episode.py` 透传该 `--tactile-key`。

### 7. `tutorial.md`(文档)
- 新增"单设备(矩侨 V2.3)"小节,给出 `--tactile-mode single` 的 publisher + launcher 命令;明确提示**两端 mode 必须一致**。
- 现有三设备小节标注为默认(`triple`)。

## 明确不做(避免范围蔓延)
- 不做同时可视化 triple 三设备的合成回放窗口(现状本就只能显示单个 key);本次只把回放的 key 变为可配置,保证 single 可回放、triple 可按设备指定。
- 不新增 gear_sonic → JuQiao 的运行时耦合:mode→layout 映射在 gear_sonic 侧本地定义(沿用 exporter 现在就硬编码 topic 字节的做法),遵循"保持 JuQiao 独立"的原则。

## 验证
gear_sonic / JuQiao 没有 pytest 接线,故:
- (a) `JuQiao/scripts/tactile_protocol_selftest.py` 仍通过;
- (b) 断言 `get_tactile_features("single")` 的 key 集合 == `{"observation.tactile_body"}` 且 modality == `{"tactile": {"body": {...}}}`;`get_tactile_features("triple")` 保持三设备;
- (c) 按 tutorial 对两种 mode 各做一次手动冒烟(publisher → viewer → 采集一小段 → replay)。
