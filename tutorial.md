# Data Collection

## 1.检查g1端硬件连接

### realsense摄像头连接

系统会自启动相机服务，无需跑代码

```bash
#g1端
sudo systemctl status composed_camera_server.service
#查看相机服务是否在线
#主极端
source .venv_data_collection/bin/activate
python gear_sonic/scripts/run_camera_viewer.py --camera-host 192.168.123.164 --camera-port 5555
```

### juqiao皮肤衣服连接

触觉支持两种硬件,用 `--tactile-mode` 切换。**publisher(g1端)、launcher/采集端、viewer 三处的 mode 必须一致**,否则设备会被零填充。

#### 模式 A:三设备(清华 V1.0,默认 `triple`)

3个独立USB设备:短袖 + 左臂 + 右臂。g1端运行代码(三个串口顺序无所谓,脚本按类型字节自动识别;三台必须全部插上,否则报错退出)

```bash
cd JuQiao
python3 scripts/tactile_publisher.py \
    --tactile-mode triple \
    --ports /dev/ttyACM0,/dev/ttyACM1,/dev/ttyACM2 \
    --zmq-host 0.0.0.0 \
    --zmq-port 5558 \
    --manager-host <主机IP> \
    --manager-port 5556
```

发布三个topic(`tactile.vest` / `tactile.left_arm` / `tactile.right_arm`)到同一端口。查看串口:`ls /dev/ttyACM*`

#### 模式 B:单设备(矩侨 V2.3,`single`)

旧的单皮肤衣,只有 1 个 USB。single 模式不做类型识别/校验,直接把这一路作为设备 `body`,发布 topic `tactile.body`,数据集记为 `observation.tactile_body`。

```bash
cd JuQiao
python3 scripts/tactile_publisher.py \
    --tactile-mode single \
    --ports /dev/ttyACM0 \
    --zmq-host 0.0.0.0 \
    --zmq-port 5558 \
    --manager-host <主机IP> \
    --manager-port 5556
```

在线重新校准(双手 grip)在两种模式下都可用。

`--manager-host <主机IP>`(运行 PICO manager 的主机在 192.168.123.x 网段的地址)启用**在线重新校准**：数据采集过程中传感器发生漂移/挤压时，**双手 grip 同时按一下**即可让触觉设备重新采零点基线(默认 50 帧，期间该设备短暂不发布)。**按下时保持触觉衣放松、不受压**，否则会把当前压力当成零点。校准在源头 publisher 完成，viewer 和采集端会同时恢复正常。不传 `--manager-host` 则禁用该功能(仅启动时校准一次)。



主机端可视化验证(mode 需与 publisher 一致;triple 弹出三个窗口,single 弹出一个 `body` 窗口)

```bash
# 三设备
.venv_data_collection/bin/python gear_sonic/scripts/run_tactile_viewer.py \
    --tactile-mode triple \
    --tactile-zmq-host 192.168.123.164

# 单设备
.venv_data_collection/bin/python gear_sonic/scripts/run_tactile_viewer.py \
    --tactile-mode single \
    --tactile-zmq-host 192.168.123.164
```

## 2.连接PICO
! 一定记得校准脚环！

## 3.启动数据采集

```bash
# 三设备(默认 triple,--tactile-mode 可省)
python gear_sonic/scripts/launch_data_collection.py \
    --camera-host 192.168.123.164 \
    --stereo-ego-view \
    --record-tactile \
    --tactile-mode triple \
    --tactile-zmq-host 192.168.123.164

# 单设备(需与 publisher 的 --tactile-mode single 一致)
python gear_sonic/scripts/launch_data_collection.py \
    --camera-host 192.168.123.164 \
    --stereo-ego-view \
    --record-tactile \
    --tactile-mode single \
    --tactile-zmq-host 192.168.123.164
```

# VLA Finetuning

# VLA Inference

```bash
# On the GPU machine (from the Isaac-GR00T repo)
uv run python gr00t/eval/run_gr00t_server.py \
    --model-path /path/to/your/finetuned_model \
    --embodiment-tag UNITREE_G1_SONIC \
    --device cuda:0 \
    --port 5550
```

```bash
python gear_sonic/scripts/launch_inference.py \
      --prompt "put hand on the orange bottlecan" \
      --camera-host 192.168.123.164 \
      --dataset-path /data/humanoid-vla/GR00T-WholeBodyControl/outputs/put_hand_on_different_objects
```

# 可视化

## 同步回放(MuJoCo + 触觉)

```bash
source .venv_sim/bin/activate
python gear_sonic/scripts/replay_episode.py \
    outputs/2026-05-22-19-51-35/data/chunk-000/episode_000001.parquet \
    --loop
```

## Motion-Token 回放(送 C++ deploy)

```bash
# 终端 1:启动 deploy(假设 launch_inference 但不跑 VLA 推理 pane)
python gear_sonic/scripts/launch_inference.py --no-data-exporter --sim
#   然后 kill 掉 VLA inference pane(或直接只起 deploy.sh)

# 终端 2:回放
source .venv_inference/bin/activate
python gear_sonic/scripts/replay_motion_tokens.py \
    outputs/tactile_test/data/chunk-000/episode_000004.parquet \
    --send-start-cmd --loop
```