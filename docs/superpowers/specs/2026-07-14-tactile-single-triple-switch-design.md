# 触觉采集:单设备 / 三设备 模式开关 —— 设计文档

日期:2026-07-14 · 分支:`tactile`

## 背景与目标

当前 `tactile` 分支的触觉采集只支持**三设备(清华 V1.0)**方案:一件短袖背心(vest)+ 左臂袖套(left_arm)+ 右臂袖套(right_arm),三个独立 USB 串口。之前开发时把旧的**单皮肤衣(矩侨精密 高频版 V2.3,单 USB)**模式去掉了,导致只有一件皮肤衣的旧硬件无法采集。

目标:在数据采集链路上加一个**开关**,可选 `single`(1 个触觉设备)或 `triple`(3 个触觉设备),默认保持 `triple`(当前行为不变)。`single` 模式必须**精确还原**旧单皮肤衣的数据 schema,使旧数据集、可视化、回放全部沿用。

## 关键约定

- 统一词表:所有涉及触觉的工具都用同一个开关 `--tactile-mode {single,triple}`,默认 `triple`。
- **源端(publisher)和汇端(exporter)必须设成同一个 mode**,这是一条书面约定(和 `--stereo-ego-view` 两端必须一致同理)。不一致时不会崩溃,但会导致设备被零填充(数据里可见),文档中明确提示。
- 单设备的 schema 冻结为旧版(`a91e6e4` 之前)的样子,作为兼容契约。

## 各模式的 Schema(兼容契约,冻结)

| mode | ZMQ topic | 数据集 key | modality 条目 |
|---|---|---|---|
| `single` | `tactile` | `observation.tactile_raw` | `tactile.tactile_raw` |
| `triple` | `tactile.vest` / `tactile.left_arm` / `tactile.right_arm` | `observation.tactile_{vest,left_arm,right_arm}` | `tactile.{vest,left_arm,right_arm}` |

`single` 模式与旧单皮肤衣完全一致:`visualize_tactile.py`(`TACTILE_COL = observation.tactile_raw`)和 `replay_episode.py` 已经读这个 key,因此旧单皮肤衣数据集、可视化、回放**无需改动即可继续用**。

## 逐文件改动

### 1. `JuQiao/scripts/tactile_publisher.py`(G1 端发布器)
- 新增 `--tactile-mode {single,triple}`,默认 `triple`。
- `triple`:行为不变(3 个串口、按 sensor-type 字节自动路由、精确校验集合为 {vest,left_arm,right_arm}、每设备一个 topic)。
- `single`:`--ports` 恰好 1 个串口;**跳过** sensor-type 探测与集合校验;固定为单一设备(name/topic 均为 `tactile`,接受任意 sensor 字节 —— 还原旧矩侨行为)。零点校准、双手 grip 在线重新校准、线程/队列模型都不变(只有一个 worker)。
- 小重构:把 worker 的身份抽象成按 mode 选择的 `TactileDevice(name, topic, sensor_type|None)`;`_worker_main` 的探测/校验逻辑按 `sensor_type is None` 分支(为 None 即单设备,不探测不校验)。

### 2. `gear_sonic/scripts/run_data_exporter.py`(采集/导出)
- 新增配置 `tactile_mode: str = "triple"`(CLI `--tactile-mode`)。
- 用按 mode 选择的 layout(`(topic_bytes, feature_key)` 列表)替换硬编码的 `_tactile_devices` 与 `_tactile_topic_to_device`:
  - `single` → `[(b"tactile", "observation.tactile_raw")]`
  - `triple` → 现有三条
- `_add_tactile_to_frame_data`:改为遍历 layout,逐设备按 `tactile_max_age_sec` 独立零填充(逻辑不变,只是数据驱动)。
- schema 构建处(约 1080 行)把 `config.tactile_mode` 传入 feature / modality 两个辅助函数。
- ZMQ 订阅前缀仍是 `"tactile"`(前缀匹配同时覆盖 `tactile` 和 `tactile.*`,无需改)。

### 3. `gear_sonic/data/features_sonic_vla.py`
- `get_tactile_features(mode="triple")` / `get_tactile_modality_config(mode="triple")` 按 mode 返回单或三设备 schema;默认 `triple`,保证现有两个无参调用点不受影响。

### 4. `gear_sonic/scripts/launch_data_collection.py`(采集启动器)
- 新增 `tactile_mode: str = "triple"` 配置;当 `record_tactile` 时,把 `--tactile-mode {mode}` 透传给 exporter 命令。
- 更新启动横幅/日志,显示当前 mode(single 显示 `observation.tactile_raw`;triple 显示 `vest/left_arm/right_arm`)。
- publisher 由操作员手动启动(不由 launcher 拉起),因此操作员需把 publisher 的 `--tactile-mode` 设成一致 —— 在 tutorial 中说明。

### 5. `gear_sonic/scripts/run_tactile_viewer.py`(主机端可视化)
- 新增 `--tactile-mode {single,triple}`,默认 `triple`。
- `single` → 单个窗口订阅 topic `tactile`,用 vest 那套 body-region 渲染器绘制;`triple` 不变。

### 6. `tutorial.md`(文档)
- 新增"单设备(矩侨 V2.3)"小节,给出 `--tactile-mode single` 的 publisher + launcher 命令;明确提示**两端 mode 必须一致**。
- 现有三设备小节标注为默认(`triple`)。

## 明确不做(避免范围蔓延)
- `replay_episode.py` 目前只读 `observation.tactile_raw`,所以它本来就不可视化 *triple* 模式的数据集 —— 这是既有缺口,不属于本开关的范围,本次不动它。
- 不新增 gear_sonic → JuQiao 的运行时耦合:mode→layout 映射在 gear_sonic 侧本地定义(沿用 exporter 现在就硬编码 topic 字节的做法),遵循 memory 里"保持 JuQiao 独立"的原则。

## 验证
gear_sonic / JuQiao 没有 pytest 接线,故:
- (a) `JuQiao/scripts/tactile_protocol_selftest.py` 仍通过;
- (b) 断言 `get_tactile_features("single")` 的 key 集合 == `{"observation.tactile_raw"}`,且 modality == 旧版 `tactile.tactile_raw`;`get_tactile_features("triple")` 保持三设备;
- (c) 按 tutorial 对两种 mode 各做一次手动冒烟(publisher → viewer → 采集一小段 → replay)。
