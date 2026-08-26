# 回放(Replay)数据的两种方式

针对 `run_data_exporter.py` 录下来的 LeRobot 格式 parquet,有两种回放路径:

1. **送回 C++ deploy(motion token)** —— 让真机/sim 按照录制时的动作再走一遍
2. **MuJoCo 状态可视化** —— 不接控制器,只在仿真窗口里看关节姿态轨迹

数据示例路径(下文以此为例):
```
outputs/2026-05-14-18-07-13/data/chunk-000/episode_000000.parquet
```

---

## 1. Motion-Token 回放(送 C++ deploy)

### 脚本
```
gear_sonic/scripts/replay_motion_tokens.py
```

逻辑:每帧读 `action.motion_token` (64d) + `teleop.left/right_hand_joints` (7d×2),用 `pack_latent_action_message` 打包发 ZMQ `pose` topic。**协议和 `run_vla_inference.py` 完全一致**,C++ deploy 不需要任何改动。

### 前置
- C++ deploy 必须在跑(用 `launch_inference.py` 或手动 `gear_sonic_deploy/deploy.sh ... real`)
- 默认绑定 `tcp://localhost:5556`(deploy 监听的同一端口)
- **不能与 `run_vla_inference.py` 同时跑** —— 两个 PUB 同端口会冲突

### 基本用法
```bash
source .venv_inference/bin/activate
python gear_sonic/scripts/replay_motion_tokens.py \
    outputs/2026-05-14-18-07-13/data/chunk-000/episode_000000.parquet
```

### 常用参数
| 参数 | 说明 |
|------|------|
| `--rate 25` | 半速回放(默认 50 Hz),看细节 |
| `--start-frame 100 --end-frame 200` | 只回放第 100–199 帧 |
| `--loop` | 循环放直到 Ctrl-C |
| `--send-start-cmd` | 自动给 deploy 发 start(pose 模式),省得手动按键 |
| `--no-hands` | 不发手部 joints,保留 deploy 当前手势 |
| `--zmq-port 5556` | 修改 ZMQ 端口(需和 deploy `--action-zmq-port` 一致) |
| `--warmup-sec 0.5` | bind 后等 SUB 连上的预热时间 |

### 典型流程
```bash
# 终端 1:启动 deploy(假设 launch_inference 但不跑 VLA 推理 pane)
python gear_sonic/scripts/launch_inference.py --no-data-exporter --sim
#   然后 kill 掉 VLA inference pane(或直接只起 deploy.sh)

# 终端 2:回放
source .venv_inference/bin/activate
python gear_sonic/scripts/replay_motion_tokens.py \
    outputs/2026-05-14-18-07-13/data/chunk-000/episode_000035.parquet \
    --send-start-cmd --loop
```

---

## 2. MuJoCo 状态回放(可视化)

### 脚本
```
gear_sonic/scripts/replay_state_mujoco.py
```

逻辑:从 `meta/info.json` 读 43 维 `observation.state` 的 joint 名顺序,用 `mj_name2id` 一对一映射到 MuJoCo qpos,只画姿态(`mj_forward` + viewer sync,**不跑物理**)。

### 前置
- 当前 venv 里要装了 `mujoco`(`.venv_sim` 有;不行的话 `pip install mujoco`)
- 默认场景:`gear_sonic_deploy/g1/scene_29dof_with_hand.xml`

### 基本用法
```bash
source .venv_sim/bin/activate
python gear_sonic/scripts/replay_state_mujoco.py \
    outputs/2026-05-14-18-07-13/data/chunk-000/episode_000035.parquet \
    --loop
```

### 常用参数
| 参数 | 说明 |
|------|------|
| `--scene <path.xml>` | 指定其他 G1 场景文件 |
| `--rate 25` | 半速;默认 0 = 跟 `meta/info.json` 里的 fps(50 Hz) |
| `--start-frame / --end-frame` | 截取片段 |
| `--loop` | 循环放,直到关掉 viewer 窗口 |
| `--use-root-orientation` | 把 `observation.root_orientation` 应用到浮基朝向上 |
| `--base-height 0.793` | 浮基 z 高度(米) |
| `--joint-names ...` | 显式给一组 joint 名,覆盖 `meta/info.json` 的查找 |

---

## 3. 同步回放(MuJoCo + 触觉)

### 脚本
```
gear_sonic/scripts/replay_episode.py
```

逻辑:把上面的「MuJoCo 状态回放」和下面的「触觉可视化」合到**一个进程、一个播放循环**里。运行后 MuJoCo 窗口和 OpenCV 触觉窗口**同时弹出**,由同一个帧索引驱动 —— 解决两个脚本分开启动时间对不上的问题。

`observation.state` 和触觉列来自同一个 parquet 的同样行,帧数天然一致,
一个帧索引同时寻址两者,不做重采样。三路数据会同时渲染
`observation.tactile_vest` / `observation.tactile_left_arm` /
`observation.tactile_right_arm`；老数据仍可使用单路 `observation.tactile_raw`。
关节映射复用 `replay_state_mujoco.py`，触觉画布复用现有 viewer 实现。

### 前置
- venv 里要同时有 `mujoco` 和 `cv2`(`.venv_sim` 两者都有,直接用它)
- 默认场景:`gear_sonic_deploy/g1/scene_29dof_with_hand.xml`
- parquet 若没有 `observation.tactile_raw` 列,会自动退化成纯 MuJoCo 回放

### 基本用法
```bash
source .venv_sim/bin/activate
python gear_sonic/scripts/replay_episode.py \
    outputs/tactile_test/data/chunk-000/episode_000004.parquet \
    --loop
```

### 常用参数
| 参数 | 说明 |
|------|------|
| `--scene <path.xml>` | 指定其他 G1 场景文件 |
| `--rate 25` | 半速;默认 0 = 跟 `meta/info.json` 里的 fps(50 Hz) |
| `--start-frame / --end-frame` | 截取片段 |
| `--loop` | 循环放;不加则放到末帧后暂停、窗口仍可交互 |
| `--use-root-orientation` | 把 `observation.root_orientation` 应用到浮基朝向上 |
| `--base-height 0.793` | 浮基 z 高度(米) |
| `--joint-names ...` | 显式给一组 joint 名,覆盖 `meta/info.json` 的查找 |
| `--no-tactile` | 跳过触觉窗口,只放 MuJoCo |
| `--window` | OpenCV 触觉窗口标题 |
| `--tactile-mode auto\|single\|triple` | 自动识别或强制单路/三路触觉 |
| `--tactile-key <column>` | 单路模式指定 parquet 列 |

### 键位(需点中触觉窗口使其获得焦点)
| 键 | 作用 |
|---|---|
| `space` | 暂停 / 继续 |
| `←` `→` | 暂停时单帧步进(也可用 `,` `.`) |
| `[` `]` | 后退 / 前进 10 帧 |
| `0`–`9` | 跳到时间轴 0%–90% |
| `r` | 回到第 0 帧 |
| `q` / `ESC` | 退出 |

MuJoCo 和 OpenCV 是两个独立原生窗口,无法合并成一个;键盘控制由触觉窗口的 `cv2.waitKey` 统一接管,改帧索引后两个窗口一起跳。SSH 远程登录需要 `ssh -X`。

### 连续浏览 merged-clean 的全部 episode

```bash
./gear_sonic/scripts/replay_mujoco_tactile_playlist.sh
```

默认依次读取
`outputs/desk_sweep_merged_clean/data/**/episode_*.parquet`，每条都从第 0 帧播到末帧，
并同步显示 MuJoCo 关节姿态和 vest / left_arm / right_arm 三路触觉。
也可把其他 LeRobot 数据集根目录作为第一个参数：

```bash
./gear_sonic/scripts/replay_mujoco_tactile_playlist.sh \
    outputs/dynamic_load_merged_clean
```

键盘焦点需在任一触觉窗口上：

| 键 | 作用 |
|---|---|
| `↑` | 切换到上一条 episode |
| `↓` | 切换到下一条 episode |
| `r` | 当前 episode 从头重放 |
| `space` | 暂停 / 继续 |
| `←` `→` | 暂停时单帧步进 |
| `q` / `ESC` | 退出整个 playlist |

当前 episode 自然播完后会自动进入下一条；只有全部 episode 都播完或人工退出时，
playlist 才会结束。

---

## 数据文件结构提醒

每个 episode 的 parquet 包含若干行(本例 364 行,50Hz ≈ 7.3 秒),关键列:

| 列 | shape / dtype | 用途 |
|---|---|---|
| `observation.state` | (43,) float64 | G1 全身关节角(MuJoCo 回放用这个) |
| `action.wbc` | (43,) float64 | 关节空间动作(deploy 内部 WBC 产物) |
| `action.motion_token` | (64,) float64 | 隐空间动作 token(送 deploy 用这个) |
| `teleop.left_hand_joints` | (7,) float32 | 左手关节目标 |
| `teleop.right_hand_joints` | (7,) float32 | 右手关节目标 |
| `observation.root_orientation` | (4,) float64 | 浮基姿态(wxyz) |
| `frame_index` | int64 | 帧序号 |

完整 schema 在 `outputs/<session>/meta/info.json` 的 `features` 字段。

---

## 故障排查

| 现象 | 原因 / 解决 |
|------|------------|
| Motion-token 回放后机器人不动 | 检查 deploy 是否在跑、是否在 `pose` 模式(或加 `--send-start-cmd`)、端口是否一致 |
| Motion-token 回放动作飞快 | `--rate` 降到 25/30 试,确认录制时也是 50Hz 才放 50Hz |
| MuJoCo 报 "X joints not found" | 用的 scene 关节命名不一样,换 `g1_29dof_with_hand.xml`,或者用 `--joint-names` 覆盖 |
| MuJoCo viewer 一闪即过 | 没加 `--loop`,正常情况下默认会 hold 住直到你关窗口 |
| ZMQ 端口冲突 | 别和 `run_vla_inference.py` 同时跑;一个 PUB 一个端口 |

## 可视化（离线）

播放单条 episode 的触觉信号（OpenCV 窗口，原速 50 Hz）：

```bash
.venv_data_collection/bin/python gear_sonic/scripts/visualize_tactile.py \
    --parquet outputs/2026-05-21-21-45-59/data/chunk-000/episode_000004.parquet
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
