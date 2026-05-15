# VLA 推理的 Action Chunk 缓存机制详解

> 针对 `gear_sonic/scripts/run_vla_inference.py` 的深入剖析:从冷启动到稳态,
> Queue 信号传递,wait 阻塞点,以及"新 chunk 接管旧 chunk"的具体过程。

---

## 1. 先理清问题:为什么需要 chunk 缓存?

部署链路里有**三个速率天差地别的组件**:

| 组件 | 速率 | 每帧时间 |
|---|---|---|
| C++ 控制循环 | **50 Hz** | 20 ms |
| Python 主循环(本文件) | **50 Hz** | 20 ms |
| Isaac-GR00T VLA 推理 | **~2.5 Hz** | ~400 ms |

C++ 每 20 ms 要消费一个 motion_token,但 VLA 每 ~400 ms 才出一次结果。
如果两者直接同步,C++ 会有 380 ms 没东西可发 → 机器人姿态保持不动 →
看起来像卡顿。

**解决方案:VLA 一次输出 40 步动作的"块"(chunk),Python 主循环逐帧弹**。

```
VLA forward pass(400ms 一次)输出形状:
    motion_token:       [40, 64]    ← 40 步、每步 64 维 token
    left_hand_joints:   [40, 7]
    right_hand_joints:  [40, 7]

主循环每 20ms 取出一步 [64] / [7] / [7] 发出去,40 步够撑 800ms,
所以 VLA 在 400ms 周期内推完下一个 chunk 就够了。
```

> **关键不变量**:`action_horizon=40` 步 × `1/action_publish_rate=20ms` = **800ms** 缓冲。
> **缓冲深度必须 ≥ 推理周期**(400ms),且建议至少 2 倍以应付偶尔的推理超时。

---

## 2. 演员表:线程和共享状态

整个系统是**两线程 + 几个共享对象**:

```
┌──────────────────────────────────┐         ┌────────────────────────────────┐
│      Main Thread(50 Hz)          │         │  Inference Worker Thread       │
│                                  │         │  (~2.5 Hz)                     │
│  - 读键盘                         │         │                                │
│  - 检查 result_queue              │         │  - 等 inference_queue 信号     │
│  - 决定要不要触发新推理            │         │  - 读 camera + state           │
│  - 从 cached chunk 弹一帧         │         │  - 调 PolicyServer(阻塞 400ms)│
│  - ZMQ PUB 发出去                 │         │  - 写 result_queue             │
│  - 睡到下一个 20ms 节拍            │         │                                │
└──────────────────────────────────┘         └────────────────────────────────┘
            │             ▲                              │           ▲
            │             │                              │           │
            ▼             │                              ▼           │
    ┌─────────────────────┴──────────────────────────────┴───────────┴───┐
    │                    线程间通信对象                                   │
    │                                                                    │
    │  inference_queue: Queue(maxsize=1)   主→worker  "开始推理"信号     │
    │  result_queue:    Queue(maxsize=1)   worker→主  "结果到了"信号+数据│
    │  inference_busy_event: Event         worker→主  "我正在跑"状态     │
    │  inference_stop_event: Event         主→worker  "退出"信号         │
    │                                                                    │
    └────────────────────────────────────────────────────────────────────┘
```

**重要**:`cached_action_chunk` / `action_chunk_index` / `last_inference_time`
**只在主线程里读写**,不是共享状态。所以这三个变量根本不需要任何锁保护。

---

## 3. Bootstrap:从 None 到第一个 action 发出去

启动时初始状态:

```python
cached_action_chunk = None
action_chunk_index = 0
last_inference_time = 0.0   # epoch
pause_loop = True           # ★ 默认暂停,等用户按 'p'
```

下面追踪每个时间点发生了什么。`T` 是从启动开始的相对时间。

### T=0 ms:主线程进入 `while True` 第一圈

```python
t_start = time.monotonic()        # 比方说是 T=0
check_keyboard_input()            # 没人按键,返回 None,跳过

# === chunk 接管点 ===
try:
    processed_action, _ = result_queue.get_nowait()   # 队列空
except queue.Empty:
    pass                          # 什么都没发生

worker_is_busy = inference_busy_event.is_set()        # False
should_start = should_trigger_new_inference(
    cached_chunk_exists=False,    # ← cached_action_chunk is None
    inference_thread_running=False,
    time_since_last_inference=T - 0.0,                # 很大
    inference_interval=0.4,
)
```

打开 `should_trigger_new_inference`(`vla_utils.py:85`):

```python
def should_trigger_new_inference(cached_chunk_exists, ...):
    if not cached_chunk_exists:   # ★ 第一次推理的特殊路径
        return True
    if inference_thread_running:
        return False
    return time_since_last_inference >= inference_interval
```

`cached_chunk_exists=False` → **直接 return True,不管别的条件**。

继续:

```python
if should_start:
    try:
        inference_queue.put_nowait(None)   # ★ 给 worker 发"开始"信号
    except queue.Full:
        pass

if pause_loop:                             # True
    print("Pausing...", ...)
    time.sleep(0.2)                        # 睡 200ms
    continue                               # 跳回循环顶
```

**所以第一圈做了两件事**:
1. 给 worker 发了一次启动信号(往 `inference_queue` 塞了个 `None`);
2. 因为暂停,主线程睡 200 ms。

### T=0 ms(同时):Worker 线程被唤醒

Worker 启动后就阻塞在这里:

```python
while not stop_event.is_set():
    try:
        try:
            inference_queue.get(timeout=0.1)   # ★ 阻塞 wait
        except queue.Empty:
            continue                            # 超时:再等一次
```

主线程往 `inference_queue` 塞东西的瞬间,这个 `get(timeout=0.1)` 立刻返回(不必等到 0.1s)。Worker 继续:

```python
busy_event.set()                               # ★ 通知主线程"我开干了"
try:
    observation = prepare_obs_fn()             # 读 camera + state subscriber
    if observation is None:
        # 比如相机或状态 ZMQ 第一帧还没到
        print("[DEBUG] Worker thread: Observation is None, skipping")
        continue                                # 跳到外层 while,重新等信号
    
    inference_start_time = time.monotonic()    # 记下"开始那一刻"
    processed_action = inference_fn(observation)   # ★ 阻塞 ~400ms 调 PolicyServer
    
    if processed_action is not None:
        try:
            result_queue.put_nowait((processed_action, inference_start_time))
        except queue.Full:
            # drain-then-put,见下文
            ...
finally:
    busy_event.clear()                          # 不管成功失败都标记"空闲"
```

注意 worker **三种早退路径**:
1. `observation is None`(传感器没就绪)→ `continue`,再次去 wait 信号;
2. `processed_action is None`(推理失败,比如 PolicyServer 没起)→ 不写结果,但
   `busy_event` 仍然 clear → 主线程下圈会重新触发;
3. **任何异常**(`except Exception` 在最外层 340 行)→ 打印 traceback,**worker 不死**,继续 while 循环。

### T=200 ms:主线程从 sleep 醒来,第二圈

```python
t_start = time.monotonic()        # T=200
check_keyboard_input()            # 还没按键

# === chunk 接管点 ===
try:
    processed_action, _ = result_queue.get_nowait()   # 还是空(worker 还没推完)
except queue.Empty:
    pass

worker_is_busy = inference_busy_event.is_set()        # ★ True!worker 在跑
should_start = should_trigger_new_inference(
    cached_chunk_exists=False,
    inference_thread_running=True,                    # ← 关键
    ...
)
```

`should_trigger_new_inference`:

```python
if not cached_chunk_exists:
    return True              # 还是想触发(没 chunk)
if inference_thread_running: # ← 但是被这条拦住
    return False
```

**注意**:即使 `cached_chunk_exists=False`,函数也会先检查 `inference_thread_running`?
**不会**。重新看代码:

```python
def should_trigger_new_inference(cached_chunk_exists, ...):
    if not cached_chunk_exists:
        return True             # ← 早返回,不看 inference_thread_running
    if inference_thread_running:
        return False
    return time_since_last_inference >= inference_interval
```

第一个 `if` 命中就返回。**所以主线程会再次往 `inference_queue.put_nowait(None)`**。

但是!`inference_queue.maxsize=1`,而且队列里**已经被主线程第一圈塞了一个 None**
(worker 那边 `get(timeout=0.1)` 取走的瞬间队列就空了,但是…)

让我们仔细看时间线:

```
T=0   ms  Main:    inference_queue.put_nowait(None)           # 队列: [None], size=1
T=0+ε ms  Worker:  inference_queue.get(timeout=0.1) 成功取出   # 队列: [], size=0
T=0+ε ms  Worker:  busy_event.set()
T=0+ε ms  Worker:  开始 prepare_obs_fn() + inference_fn(...)
T=200 ms  Main:    第二圈,inference_queue.put_nowait(None)    # 队列: [None]
T=200 ms  Worker:  仍在 inference_fn() 里(还要 200ms)
```

第二个 `put_nowait` 成功了 —— 队列空。**但这一个 None 不会立刻被处理**,
因为 worker 还在跑 `inference_fn`。它要等 worker 这次跑完、回到 while 顶部、
再次调 `inference_queue.get()` 才会消费。

那这个 `None` 算是"排队"了?对,**排队但只能排 1 个**,因为 maxsize=1。
等同于一个"上次错过的门铃信号被记住了"。

继续走主线程:

```python
if pause_loop:
    time.sleep(0.2)
    continue
```

睡 200ms,跳回顶。

### T=400 ms:第三圈,同样的故事再来一次

此时 worker 仍在跑(还差 ~10ms),`busy_event` 仍 set。
主线程依然 `should_trigger=True`(因为 chunk 还是 None),
依然 `put_nowait(None)`。**这次队列满**(里面已经有上次的 None):

```python
if should_start:
    try:
        inference_queue.put_nowait(None)
    except queue.Full:
        pass                  # ★ 静默吞掉,这是关键!
```

**第三圈的请求被丢弃了** —— 没问题,因为之前已经有一个请求在队列里等了。
**Queue 维度 1 在这里起到的作用是"信号合并":多次请求自动塌缩成一个"**。

### T=410 ms:Worker 推完,把结果放进 result_queue

```python
processed_action = inference_fn(observation)        # 完成!花了 410ms
result_queue.put_nowait((processed_action, inference_start_time))  # 队列: [chunk]
busy_event.clear()                                   # 状态:空闲
# 回到 while 顶
inference_queue.get(timeout=0.1)                     # ★ 立刻拿到主线程之前塞的 None
busy_event.set()                                     # 又忙起来
# 开始下一次推理
```

**注意时序**:worker 这一次能立刻无缝接上下一次推理,**因为队列里早就排好了请求**。
所以从这一刻起,worker 进入"几乎不空闲"的稳定状态。

### T=600 ms:主线程第四圈

```python
t_start = time.monotonic()    # T=600
check_keyboard_input()

# === chunk 接管点!★★★ ===
try:
    processed_action, inference_start_time = result_queue.get_nowait()
    # ★ 成功!processed_action 是 [40, 64+7+7=78] 的字典
    inference_delay = T - inference_start_time
    # = 600 - 0 = 600ms
    # (注意:可能 inference 实际花 410ms,但主线程在 T=600 才看到,
    #  延迟从 inference_start_time 算起,所以是 600ms)
    action_chunk_index = calculate_latency_compensated_index(
        0.6, 50, 40
    )
    # round(0.6 * 50) = 30,clip 到 [0, 39] → 30
    cached_action_chunk = processed_action
    last_inference_time = time.monotonic()  # T=600
except queue.Empty:
    pass
```

`cached_action_chunk` **第一次变成非 None**。`action_chunk_index` 从初始的 0 直接跳到 30
(因为推理从 T=0 开始,到主线程看到时已过去 600ms = 30 帧)。

```python
should_start = should_trigger_new_inference(
    cached_chunk_exists=True,          # 现在变 True 了
    inference_thread_running=True,     # worker 已经开始下一次推理(T=410)
    time_since_last_inference=600-600=0,
    inference_interval=0.4,
)
```

`cached_chunk_exists=True` → 进入第二个 if → `inference_thread_running=True` → return False。
**不触发新推理**。

```python
if pause_loop:    # 仍为 True
    time.sleep(0.2)
    continue
```

继续睡 200ms。**注意:虽然 chunk 已经在缓存里,但因为 paused,根本没发出去**。
这是有意为之的安全设计 —— 用户按 'p' 之前,机器人完全静止。

### 用户按 'p':T=2000 ms(假设)

```python
elif key == "p":
    pause_loop = not pause_loop      # False
    print("Resumed policy loop")
```

下一圈主循环:

```python
if pause_loop:    # 现在 False
    ...           # 跳过
    
with telemetry.timer("total_loop"):
    if cached_action_chunk is None:
        print("[DEBUG] No cached chunk yet, waiting...")
        _sleep_remaining(t_start, loop_period)
        continue
    
    processed_action = cached_action_chunk   # ★ 拿到 chunk
    
    # ... 从 chunk[action_chunk_index] 取出一帧 ...
    # 假设 action_chunk_index 此刻已经被增长到接近 39
    # (因为从 T=600 到 T=2000 主线程跑了 ~70 圈)
    
    current_idx = min(action_chunk_index, 39)
    motion_token = motion_token_chunk[current_idx]    # 取最后一帧反复发
    zmq_socket.send(zmq_message)                       # ★ 第一个真正发出的 action
    
    action_chunk_index = min(action_chunk_index + 1, 39)  # 已饱和
```

**等等**:从 T=600 到 T=2000 之间 `action_chunk_index` 是怎么变化的?
看一下代码,`action_chunk_index += 1` 在循环末尾(672 行),**但只有 `pause_loop=False`
的分支才会执行到那一行**。在暂停期间 `continue` 直接跳过尾段,**idx 不会增长**。

修正:T=600 时 idx=30,T=2000 时 **仍然是 30**。第一次真正发动作时 idx=30。
然后每 20ms +1,到 idx=39 后饱和。

这期间 worker 早就完成了第 2 次、第 3 次……推理,但因为 `should_trigger_new_inference`
在 `inference_thread_running=True` 时返回 False,以及 worker 完成后等下个信号,
**主线程暂停时的推理节奏会变成"完成一次就立刻接着下一次"**(因为 inference_queue
里总有上一次未消费的信号)。每次完成都 `result_queue.put_nowait`,如果队列满则
drain-then-put:

```python
result_queue.put_nowait((processed_action, inference_start_time))
# 如果失败(队列满):
try:
    result_queue.get_nowait()           # 把旧 chunk 扔了
    result_queue.put_nowait(...)        # 塞新的
except queue.Empty:
    result_queue.put_nowait(...)        # 万一刚才被消费,直接塞
```

所以暂停期间 `cached_action_chunk` 实际上**一直在被刷新**,虽然没用。

---

## 4. 稳态:Queue + Event 的精确语义

启动跑通之后,系统进入稳态。这一节系统讲清楚每个同步原语的作用。

### 4.1 `inference_queue:Queue(maxsize=1)`

**用途**:主线程 → worker 的"请开始下一次推理"信号。

**注意它不传数据**:`put_nowait(None)` 塞的是 `None`,worker 取出后**完全不看这个值**,只用它的"存在"作为信号:

```python
# Worker 那边:
inference_queue.get(timeout=0.1)        # 不接收返回值
busy_event.set()
observation = prepare_obs_fn()          # ★ worker 自己去读最新观测
```

**为什么不用 `threading.Event`**?Event 也能干这事:

```python
# 假想的 Event 版本
inference_event.wait(timeout=0.1)       # 等被 set
inference_event.clear()                 # 清掉以便下次再等
busy_event.set()
...
```

Queue 比 Event 好在:
- **`put_nowait` 自动幂等**:已经满(maxsize=1)就 raise `queue.Full`,可以
  静默捕获 —— 主线程不需要先判断"是否已经在等了";
- **天然 FIFO 语义**(虽然这里只有一个槽,但符合直觉);
- **不需要手动 `clear()` 复位**:`get()` 一调用就消费掉信号。

**总结**:这里的 Queue 被用作**带去抖动的门铃信号**,不是数据通道。

### 4.2 `result_queue:Queue(maxsize=1)`

**用途**:worker → 主线程的"结果到了"信号 **+** 实际的 chunk 数据。

**为什么 maxsize=1**?让队列**永远只持有最新结果**:

| 情况 | 队列状态 | worker 行为 |
|---|---|---|
| 主线程刚消费,队列空 | `[]` | `put_nowait(chunk_new)` 成功 → `[chunk_new]` |
| 主线程还没消费,旧 chunk 在队 | `[chunk_old]` | `put_nowait(chunk_new)` 抛 `Full` → drain-then-put |

**drain-then-put 模式**(`_inference_worker_loop` 332–337 行):

```python
try:
    result_queue.put_nowait((processed_action, inference_start_time))
except queue.Full:
    try:
        result_queue.get_nowait()                # 扔旧的
        result_queue.put_nowait((processed_action, inference_start_time))
    except queue.Empty:
        # 极端情况:在 get 和 put 之间主线程刚好消费走了
        result_queue.put_nowait((processed_action, inference_start_time))
```

**这保证了**:即使主线程暂时跟不上(被键盘卡了一下、被 GIL 拖了),
worker 完成新一轮推理后**永远是最新结果**留在队里,旧的被丢。

主线程的消费:

```python
try:
    processed_action, inference_start_time = result_queue.get_nowait()
    # 这里执行体里:替换 cache,重置 idx,更新时间戳
except queue.Empty:
    pass    # 没有新结果,继续用旧 chunk
```

**`get_nowait()` 是非阻塞的**:有就拿、没有就抛 `Empty`,主线程**绝不会卡在这等**。
这是 50Hz 节拍的基本要求。

### 4.3 `inference_busy_event`

**用途**:让主线程能查询"worker 现在到底有没有在跑推理"。

不是用 Queue 表达的,因为状态比信号更简单 —— 它是一个**布尔标志**:

```python
# Worker 那边:
busy_event.set()        # 进入 prepare_obs + inference_fn
try:
    ...
finally:
    busy_event.clear()  # 不管成功失败,出来时清零

# Main 那边:
worker_is_busy = inference_busy_event.is_set()    # 查询当前状态
```

**为什么需要这个**?如果没有 busy_event,主线程会过度触发:

```python
# 假设没有 busy_event,简化版的 should_trigger:
return time_since_last_inference >= inference_interval
```

考虑这个场景:
- T=0:推理触发,worker 开始;`last_inference_time` 不变(还是上一次的)
- T=400ms:`time_since_last_inference` 满 400ms 了 → 触发 → 塞 None 进队
- T=410ms:worker 还在跑这次推理,但队列里又多了一个待办

如果 worker 每次推理花 600ms,队里的 None 会**永远满**,主线程**永远在塞**(吞 `Full`),
看起来还行,但**逻辑上每个 chunk 完成时立刻发起下一个,实际上达不到 inference_interval 节拍**
—— 因为 worker 永远在赶。

**busy_event 让主线程主动让路**:worker 忙的时候压根不下指令,worker 一空闲就立刻
被下一个 `inference_queue.get` 拿走 —— 这才是真正的"按节拍推理"。

### 4.4 wait 阻塞点全景

整个系统里实际发生**阻塞 wait**的点有以下几个:

| 阻塞点 | 在哪 | 阻塞最长时间 | 唤醒条件 |
|---|---|---|---|
| `inference_queue.get(timeout=0.1)` | Worker 线程 | 100 ms | 主线程 `put_nowait` 或超时 |
| `inference_fn(obs)` 内部 ZMQ REQ | Worker 线程 | **无限**(直到 PolicyServer 回复) | PolicyServer 回应 |
| `result_queue.get_nowait()` | 主线程 | **不阻塞**(立刻 raise Empty) | — |
| `inference_queue.put_nowait(None)` | 主线程 | **不阻塞**(立刻 raise Full) | — |
| `time.sleep(0.2)`(paused 时) | 主线程 | 200 ms | 时间到 |
| `_sleep_remaining(t_start, 0.02)` | 主线程 | ≤ 20 ms | 时间到 |
| `inference_worker_thread.join(timeout=1.0)`(shutdown) | 主线程 | 1 s | worker 退出 |

**关键观察**:
- **主线程没有任何无限阻塞点** —— 任何一个 sleep 都有明确时长。
- **Worker 唯一可能"永久卡死"的地方是 `inference_fn`**(PolicyServer 调用)—— 如果
  PolicyServer 挂了、网络分区了,worker 会卡在那。这种情况下 `stop_event` 也唤不醒它,
  shutdown 时 `join(timeout=1.0)` 等到超时,主进程退出后 Python GC 强制清理线程。
- **Worker 的 `inference_queue.get(timeout=0.1)`** 是它和外界的唯一同步点。0.1s 这个超时
  也是 `stop_event` 的"检查粒度":即使 100ms 内没有推理请求,worker 也会循环回头检查
  一次是否要退出。

### 4.5 GIL 和原子性

这套设计**完全不需要锁**,因为:

1. **`cached_action_chunk` / `action_chunk_index` / `last_inference_time` 只在主线程读写**。
   单线程访问无竞态。
2. **`Queue.get_nowait()` / `put_nowait()`** 内部用 lock,本身原子。
3. **`Event.set()` / `clear()` / `is_set()`** 内部用 lock + condition variable,原子。
4. **共享的对象引用**(比如 `processed_action` dict)在 Python 里通过 GIL 保证赋值原子。
   worker put 的对象进 Queue 时被引用计数+1,主线程 get 出来时还是同一个对象引用。

**没有用 `threading.Lock`、`threading.RLock`、`threading.Condition`**。这是 Python 多线程
代码的好实践 —— 能用 Queue 别用 Lock,出 bug 少很多。

---

## 5. 时间线图(稳态情景)

假设 PolicyServer 每次推理稳定花 410 ms,主循环 50 Hz,`inference_interval = 400 ms`,
`action_horizon = 40`,用户已经按 'p' 取消暂停。

```
                                                  消费     消费     消费
                                                 result   result   result
                                                  ▼        ▼        ▼
Main thread:    ╞══╪══╪══╪══╪══╪══╪══╪══╪══╪══╪══╪══╪══╪══╪══╪══╪══╪══╪══...
                每 20 ms 弹一帧 chunk[idx],idx++

                ▲   ▲ ▲                ▲ ▲
                │ │                    │ │
                │ │                    │ └─ 主线程检测到 busy_event 为 False
                │ │                    └─── 主线程 put_nowait 触发新推理
                │ └─ inference_queue 收到 None,worker get 出来,busy_event.set()
                └─── 主线程第一圈 put_nowait(None)

Worker thread:  ║════════════════════║     ║════════════════════║     ║...
                 inference_fn 跑 410ms      下一轮 410ms              ...

                                      ▲                          ▲
                                      │ result_queue.put_nowait  │ ...
                                      │ busy_event.clear()        │
                                      │ (此刻 inference_queue 里  │
                                      │  早就有下一个 None 在等)   │
                                      │                           │
                                      └─ 主线程在下一个 20ms tick 时
                                         get_nowait 拿到结果,
                                         action_chunk_index 跳到
                                         round(0.41 * 50) = 20

时间(ms):  0   100   200   300   400   500   600   700   800   900   1000
```

**关键观察**:
- 主线程每 20 ms 都会**尝试**消费 result_queue,99% 的循环里队列是空(`queue.Empty`),
  但**那 1% 不空的瞬间**就是 chunk 接管;
- worker 完成 → 主线程下个 tick 消费 → 中间间隔最多 20 ms,**chunk 替换的延迟很小**;
- 在两次 chunk 替换之间,主线程**单调递增 `action_chunk_index`**,每 20 ms 弹一帧;
- chunk 接管时 idx 重置为 20 左右(因为 410 ms 已过 → 20 帧),**避开了"回放过去的动作"**。

### 异常时间线:推理变慢了

假设 PolicyServer 突然变慢,某次推理花了 1200 ms:

```
Main thread:    ╞══╪══╪══╪══╪══╪══╪══╪══╪══...═╪══╪══...═╪══...═╪══...═╪══
                       │
                       └─ chunk[0..39] 全部播完,idx 饱和在 39,机器人保持姿态
                          (chunk 末尾不会突然换成 chunk[0],
                           因为 min(idx+1, 39) 自我夹紧)

Worker thread:  ║════════════════════════════════════════════════════║
                 这次推理花 1200ms

时间(ms):  0    200   400   600   800   1000  1200
                  ↑                                    ↑
            上次结果在 ~410ms 进 cache                推理超期回来,新 chunk 替换
            idx 从 20 开始,800ms / 20 = 40 步,        idx 重置为 round(1.2*50) = 60
            到 1200ms 时已经播 40 步,饱和在 39        被 clip 到 39(action_horizon-1)
```

**机器人的体感**:从 1000 ms 到 1200 ms 这段(chunk 用完后),**反复发同一帧动作**,
看起来像"姿态保持"。新 chunk 一回来,idx 被 clip 到 39,**只发新 chunk 的最后一帧** ——
这一帧大概率和上一次 chunk 的最后一帧不连续,**会有一个动作跳变**。

> **这就是为什么 `action_horizon` 要预留充足余量**。如果你看到部署时机器人有"卡顿
> + 跳跃"的现象,大概率是 PolicyServer 在某些 prompt 下变慢,chunk 用完前推不出新的。

---

## 6. 常见疑问 FAQ

### Q1:`pause_loop=True` 期间 worker 还在跑吗?

**是**。暂停只影响主线程**是否往 ZMQ 发包**,不影响 worker。
事实上,暂停期间 worker 一直在 "完成 → 下一次" 的紧凑循环里,
`result_queue` 永远是最新结果。**这是有意的**:用户取消暂停的那一瞬间,
cache 里**已经有热乎的 chunk**,马上能用。

### Q2:第一次 `inference_queue.put_nowait(None)` 之后,如果 worker 还没起来,
主线程会怎样?

不会怎样。`put_nowait` 完成后主线程立刻继续(`pause_loop=True` 则 sleep 200ms)。
worker 起来后会拿到这个 None 信号开始干活。即使 worker 永远不起来,主线程也
不会卡 —— 只是 cache 永远 None,主循环一直走 `if cached_action_chunk is None`
那条 print("No cached chunk yet, waiting...")的支路。

### Q3:如果 PolicyServer 没启动,会怎样?

worker 调 `policy.get_action(observation)`(Isaac-GR00T 的 PolicyClient),
内部走 ZMQ REQ。**ZMQ REQ 在 server 不存在时会无限等(默认无超时)**。
所以 worker 会**永远卡在那次推理上**,`busy_event` 永远 set,主线程永远
看到 worker 忙、不会再触发新推理。**整个系统死锁**,但**主线程仍能响应键盘**
(`p` / `k` / KeyboardInterrupt)。

要修这个,得给 PolicyClient 加 ZMQ recv timeout —— 但这是 Isaac-GR00T 的库,
不是本仓的代码。**实践中通过 `ping()` 提前检测**(369–373 行),warning 让用户
确认 server 起来再继续。

### Q4:`maxsize=1` 真的够吗,会丢结果吗?

**够**,**不会丢"最新"的结果**。

- worker put 时如果队满 → drain 旧的 → put 新的 → 新结果保留;
- 主线程 get 时拿到的就是 worker 最近 put 的;
- **被丢弃的永远是"上一次未及时消费"的旧 chunk**,这正好是我们想要的:
  陈旧的预测不要,要最新的。

### Q5:`action_chunk_index` 从 `calculate_latency_compensated_index` 算出
30 之后,如果新 chunk 的 horizon 只有 20 步(模型异常)怎么办?

```python
horizon = motion_token.shape[0] if motion_token.ndim == 2 else 1
current_idx = min(action_chunk_index, horizon - 1)
```

第 646 行的 `min(action_chunk_index, horizon - 1)` 防止越界。如果 idx=30 但
horizon=20,实际取 chunk[19](最后一帧)。**不崩,但行为可能怪** —— 看到部署
日志里 horizon 不是 40 时要排查 PolicyServer 配置。

### Q6:为什么主循环里"先消费 result_queue,再决定触发"?

更新 `last_inference_time` 影响 `should_trigger_new_inference` 的判断:

```python
return time_since_last_inference >= inference_interval
```

如果**先触发再消费**:刚好这个 tick 拿到了新 chunk,`last_inference_time` 应该更新,
但触发判断用的还是旧的 `last_inference_time` —— 可能 `time_since` 早就大于 interval,
触发了一次**冗余**的推理。

**先消费再触发**:`last_inference_time` 被更新到现在,`time_since=0`,触发判断
会返回 False(已经触发过了,等下一个 400ms)。

### Q7:worker 内部 `prepare_obs_fn` 为什么放在 wait 之后,而不是主线程把 obs
传过来?

如果主线程在 put_nowait 时就准备好 obs 传过去,obs 会**陈旧** —— 如果队列里那个
None 排队等了 100ms 才被消费,推理基于的 obs 已经是 100ms 前的。

放在 worker 里读 → **观测在 inference 起点最新** → 输入新鲜 → latency
compensation 的数学(`inference_delay = now - inference_start_time`)精确对应"推理
基于这一时刻"。

---

## 7. 总结:整套机制为什么干净

1. **单一主线程拥有 chunk 状态** —— 没有锁,GIL 即足够。
2. **Queue maxsize=1 + drain-then-put** —— 队列里永远只有最新元素,自动丢旧。
3. **busy_event 作为推理节拍守门员** —— 防止主线程在 worker 忙时挤进新请求。
4. **wait 全部带 timeout** —— 唯一长阻塞是 PolicyServer ZMQ REQ,可以 ping 检测。
5. **`should_trigger_new_inference` 集中节拍逻辑** —— 三条规则覆盖冷启动、忙等、稳态。
6. **`calculate_latency_compensated_index` 保证新 chunk 接管不"回放过去"** ——
   推理慢的代价是缓冲耗尽时姿态保持,但永远不会出现时间倒流。

整套设计的核心是**"主线程是节拍器,worker 是后台计算"** 的经典 producer/consumer
模式,加上一个不起眼但关键的"信号合并"机巧(maxsize=1 + put_nowait 静默 Full)。

读懂这个文件后,看 `run_data_exporter.py` 会发现它走的是**同步**模式
(主循环直接读 ZMQ 状态、写帧,不需要 worker thread)—— 因为数据采集没有
"VLA 推理慢"这个问题需要解耦。两个文件的对比能很好理解什么时候需要异步、什么时候
不需要。
