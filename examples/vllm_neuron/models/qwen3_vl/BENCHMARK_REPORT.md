# Qwen3-VL-8B-Instruct 视频推理延时基准报告

**日期**: 2026-08-11 · **机器**: trn2.3xlarge (i-0ad9fb51a3fed095e, 1 Neuron device / 4 NeuronCores / 96GB)
**软件**: vllm-neuron 0.21.0.1.0.0 (branch `release-0.21.0.1.0.0`) · vLLM 0.21.0 · Neuron SDK 2.31 · transformers 5.14.1
**模型**: `Qwen/Qwen3-VL-8B-Instruct` (bf16, 本地路径 `/mnt/nvme/models/Qwen_Qwen3-VL-8B-Instruct`)

---

## 1. 结论速览

1. **`max_model_len` 必须贴合实际负载，否则解码性能崩塌。** 解码 NEFF 默认按 `max_model_len`
   整个窗口算注意力，而这个开销是**按序列计**的，所以 batch 越大浪费越多。
   本负载实际只需 1907 token，把 `max_model_len` 从 4096 降到 2048 后：
   **batch 8 的 TPOT 从 75.16 ms 降到 23.96 ms（3.1×），吞吐从 101 涨到 284 tok/s（2.8×）**。
   batch 1 几乎无变化（11.34 → 11.23 ms）—— 这是纯粹的批处理扩展性问题（§5）。
2. **调优后的数字**（`max_model_len=2048`）: batch 1 TTFT 316 ms / TPOT 11.23 ms / E2E 3.18 s；
   batch 8 TTFT 817 ms / TPOT 23.96 ms / E2E 6.91 s / **284.1 tok/s / 1.11 req/s**。
3. **「每秒 1 次 × 256 token」的目标可以达到**：batch 8 实测 1.11 req/s（需要 1.0），
   单请求延时 6.9 s。调优前的结论「缺口 2.5 倍」是错的，那是 `max_model_len` 配置问题。
   **更推荐 batch 8 + 128 token：1.79 req/s，延时降到 4.2 s**（§7.1 有全部实测工作点）。
4. **batch 8 是最优点，batch 16 断崖式退化**（TPOT 138 ms，吞吐掉到 110 tok/s）。
   同长度纯文本对照同样复现，所以问题在通用解码路径，不在多模态路径（§5.1）。
5. **其他调优项全部无效或更差**：`decode_context_length_buckets` 慢 1.87 倍、
   `enable_chunked_prefill=False` 无差别、`attention_dp_size>1` 单节点不可用（§5.2）。
6. 过程中发现 **4 个工具链/框架问题**：连续不同视频会让引擎崩溃（§6.1）、
   多 `num_seqs_buckets` 配置会挂死（§6.2）、**TP=1 编不出来**（neuronx-cc 段错误，§6.3）、
   **TP=2 静默输出乱码**（§6.4，最危险）。
   **TP=4 是本机唯一可用的并行配置**，这否掉了所有「小 TP + 多副本」的方案。
   顺带发现默认 8 路编译并行在 12 vCPU 上有争抢，串行化能让单图编译快 4.4 倍。
7. 基准方法上有个坑：**用同一个视频重复压测会让视觉编码器被缓存跳过**，TTFT 会虚低 42%
   （171 ms vs 294 ms）。本报告所有数字均为每请求独立视频（§4.2）。

---

## 2. 输入定义与实测形状

需求为「每秒 16 帧 448×448 的视频，输出最多 256 token」，按 1 秒 1 次推理理解为**每次推理 1 秒视频 = 16 帧**。

| 项目 | 值 | 来源 |
|---|---|---|
| 帧数 / 分辨率 | 16 帧 @ 448×448 | 输入参数 |
| 声明 fps / 时长 | 16.0 fps / 1.00 s | 视频 metadata（决定时间戳 token） |
| `video_grid_thw` | `[8, 28, 28]` | HF processor 实测 |
| 原始 vision patch | 6272 | `T*H*W`，T = 帧数/2（时间 patch=2） |
| 合并后 embed token | 1568 | 2×2 spatial merge |
| prompt 总长度 | **1651 token** | HF processor 实测 |
| vision bucket | **8192** = 4 × 2048 | 分块填充后向上取整（见下） |
| 输出 token | 256（`ignore_eos=True`，每请求恰好 256） | 采样参数 |

**vision bucket 为什么是 8192 而不是 6272**: 编码器把整个时间切片打包进 `vision_attention_block_size`
的块里，且不会拆分切片。一个 448×448 帧对 = 一个 `[1,28,28]` 切片 = 784 patch；
`block_size=2048` 时每块只装得下 2 个切片（用 1792，浪费 256），8 个切片需要 4 块 = 8192。

**视频素材**: vLLM 自带 demo asset `baby_reading`（`sample_demo_1.mp4`，原片 243 帧 / 25 fps / 9.72 s / 640×360），
在整片上均匀抽 16 帧后 `cv2.resize` 到 448×448。张量形状与「1 秒 16 帧」完全一致，
但像素内容是跨 9.72 s 抽样的，不是连续 1 秒窗口 —— 对延时无影响，对输出语义有影响。

**正确性抽查**（输出可读，说明视频通路真的在工作）:

> In this video, a young child is sitting on a bed, engrossed in reading a book. The child is wearing
> glasses and a light blue shirt with pink pants. The bed has a patterned quilt, and there are some
> clothes and a crib visible in the background, suggesting the setting is a bedroom.

---

## 3. 运行配置

TP=4（本机 4 核上限；官方 recipe 的 TP=16 需要 trn2.48xlarge）。

```python
max_model_len            = 2048        # 贴合负载 1907；4096 会让解码性能崩塌，见 §5
                                       # 脚本默认自动取 >= prompt+输出 的最小 2 的幂
max_num_batched_tokens   = 2048        # >= prompt 1651，保证 prefill 单趟完成
max_num_seqs             = <batch size>
tensor_parallel_size     = 4
enable_prefix_caching    = False       # Neuron 上 APC 强依赖 segmented prefill
additional_config = {
  "neuron_config": {
    "quantization": "bf16",
    "num_batched_tokens_buckets": [2048],
    "num_seqs_buckets": [<batch size>],          # 每个 batch 单独一个引擎，见 §6.2
    "on_device_sampling_config": {"all_greedy": True},
  },
  "vision_neuron_config": {
    "num_vision_tokens_buckets": [8192],
    "vision_attention_block_size": 2048,
    "encoder_cache_num_blocks": 48,              # 必须显式设置，见 §6.1
  },
}
```

视觉并行度按默认解析为 `tp_size=1, dp_size=4`：4 个 block 各占一个 NeuronCore。

必需环境变量（本机特有）:

```bash
NEURON_SKIP_EFA_AFFINITY=1                  # 这个规格没有 EFA，否则 worker 全部启动失败
NEURON_CC_FLAGS="--temp-dir=/tmp/neuroncc_tmp"
VLLM_CACHE_ROOT=/mnt/nvme/cache/vllm
VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1200
VLLM_NEURON_COMPILATION_TIMEOUT=3600
```

---

## 4. 主结果

### 4.1 batch size 扫描

每个 batch size 独立引擎，2 次 warmup + 10 次测量迭代，每请求一个独立视频，输出固定 256 token。
batch = 同时发起的并发请求数（一次全部提交）。单位 ms。

**推荐配置（`max_model_len=2048`，贴合本负载的 1907 token）**：

| BS | 样本数 | TTFT p50 | TPOT p50 | E2E p50 | 输出 tok/s | req/s | 每序列解码 |
|---|---|---|---|---|---|---|---|
| 1 | 10 | **316** | **11.23** | **3180** | 80.4 | 0.31 | 11.23 |
| 4 | 40 | 551 | 18.72 | 5341 | 188.4 | 0.74 | 4.68 |
| 8 | 80 | 817 | **23.96** | **6905** | **284.1** | **1.11** | **3.00** |
| 16 | 160 | 1471 | 138.20 | 36484 | 110.0 | 0.43 | 8.64 |

**batch 8 是最优点，batch 16 是断崖式退化**（TPOT 5.8×、吞吐掉到 0.39 倍），见 §5.1。

**未调优（`max_model_len=4096`）—— 保留作对照，说明这个参数的影响**：

| BS | 样本数 | TTFT p50 | TTFT p90 | TTFT max | TPOT p50 | ITL p50 | E2E p50 | 输出 tok/s | req/s |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 10 | 315 | 340 | 359 | 11.34 | 11.34 | 3208 | 79.7 | 0.31 |
| 2 | 20 | 427 | 494 | 509 | 25.33 | 24.88 | 6862 | 74.0 | 0.29 |
| 4 | 40 | 563 | 723 | 761 | 42.28 | 41.06 | 11351 | 89.4 | 0.35 |
| 8 | 80 | 835 | 1076 | 1176 | 75.16 | 72.44 | 19998 | 101.0 | 0.39 |

**TTFT 不受影响**（315→316 / 563→551 / 835→817），因为 prefill 本来就按实际长度分桶；
受影响的只有解码。所有配置下每请求都恰好输出 256 token，输出文本正常。

- **TTFT** = 视频预处理 + 视觉编码 + 文本 prefill（每请求都是缓存未命中）
- **TPOT** = (E2E − TTFT) / (输出 token − 1)
- **ITL** = 逐 token 间隔；p50 ≈ TPOT 说明解码步时间均匀，没有周期性停顿
- **tok/s / req/s** = 整批聚合吞吐（总输出 token ÷ 总墙钟时间）

TTFT 随 batch 上升主要是**排队**：`max_num_batched_tokens=2048` 装不下两个 1651 token 的 prompt，
所以 prefill 逐个串行，batch 8 里最后一个请求要等前 7 个 prefill 完成。

### 4.2 视觉编码是否计入 TTFT（方法学对照）

vLLM 的 `EncoderCacheManager` 按 `mm_hash` 缓存视觉 embedding，且请求结束后条目仍留在
freeable LRU 中。**重复送同一个视频时，第 2 个请求之后整个视觉编码器被跳过。**
batch 1、20 次迭代对照：

| 模式 | TTFT p50 | 说明 |
|---|---|---|
| `--reuse-video`（同一视频） | **171 ms** | 缓存命中，只有文本 prefill |
| 默认（每请求独立视频） | **294 ms** | 缓存未命中，含预处理 + 视觉编码 |

差值 123 ms 就是视觉通路的成本。**同一视频重复压测会把 TTFT 报低 42%。**

### 4.3 TTFT 成本拆解（batch 1）

| 阶段 | 耗时 | 运行位置 |
|---|---|---|
| 视频预处理（归一化 + patchify → `pixel_values_videos [6272,1536]`） | ~40 ms | **CPU**（vLLM frontend 进程，20 次实测 p50 39.7 / p90 40.5 / max 47.5 ms，很稳） |
| 视觉编码（27 层 ViT，4 块分到 4 个 core） | ~83 ms | **Neuron 设备** |
| 文本 prefill（1651 token） | ~171 ms | Neuron 设备 |
| **合计** | **~294 ms** | |

视觉编码器确实跑在 Neuron 上，不在 CPU（日志证据：`Vision encoder compiled separately`、
`Vision warmup [1/1]: bucket=8192, num_blocks=4`、`On-device encoder cache: ... buffer_size_mb=528.0`）。
唯一在 CPU 上的是视频解码/resize 和 HF 预处理器。

> §4.1 表里 batch 1 的 TTFT 是 315 ms，§4.2/4.3 是 294 ms —— 两次独立运行（10 次 vs 20 次迭代）的
> 正常波动，约 7%。

---

## 5. 根因：解码按 `max_model_len` 整窗算注意力

最初用 `max_model_len=4096` 时，解码步时间几乎与 batch 成正比（11.3 / 24.9 / 41.1 / 72.4 ms），
每序列成本只从 11.34 降到 9.06 ms，聚合吞吐只涨 1.27 倍 —— 批处理基本失效。

**根因**：`decode_context_length_buckets` 默认为 `null`（禁用）。文档原话是
*"Second decode bucketing dimension. Compiles smaller NEFFs sized to typical context lengths
instead of `max_model_len`."* 禁用时解码 NEFF 按 **`max_model_len` 整个窗口**算注意力，
而这个开销是**按序列计**的：B 个序列各自在 4096 窗口上做注意力，总开销 ∝ B。
本负载实际 context 只有 1907 token，4096 的窗口有一半以上是纯浪费。

**验证**：只改一个参数，`max_model_len` 4096 → 2048（1651 prompt + 256 输出 = 1907，刚好放得下）：

| BS | TPOT @4096 | TPOT @2048 | 提升 | 每序列 @4096 | 每序列 @2048 |
|---|---|---|---|---|---|
| 1 | 11.34 ms | 11.23 ms | 1.01× | 11.34 | 11.23 |
| 4 | 42.28 ms | 18.72 ms | **2.26×** | 10.57 | 4.68 |
| 8 | 75.16 ms | 23.96 ms | **3.14×** | 9.39 | **3.00** |

batch 1 几乎不变、batch 越大提升越多 —— 与「浪费 ∝ batch」的预测完全一致。
每序列解码成本从 11.23 降到 3.00 ms（**3.7×**），批处理恢复正常。

注意 batch 8 的提升（3.14×）大于窗口缩小的倍数（2×），说明 4096 窗口下可能还额外触发了
更差的 kernel 路径或分块策略，不只是线性地多算。§5.2 里 `decode_context_length_buckets=[1920]`
（非 2 的幂）反而比 2048 慢 1.87 倍，也指向「context 长度的对齐方式本身影响 kernel 效率」。

**生产建议**：把 `max_model_len` 取成**贴合实际负载的最小 2 的幂**。
不要用 `decode_context_length_buckets` —— 实测反而慢 1.87 倍（§5.2）。
脚本现在默认自动这么取，显式给超过 1.5 倍余量的值会警告。

### 5.1 batch 16 的悬崖（未定位）

每序列解码成本随 batch 的曲线：**11.23 → 4.68 → 3.00 → 8.64 ms**（BS 1/4/8/16）。
8 到 16 之间有一个明确的断崖，TPOT 从 23.96 跳到 138.20 ms（**5.8×**，而 batch 只翻 2 倍），
吞吐从 284 掉到 110 tok/s。

已排除的平凡原因：

- **不是 KV cache 不足**：`GPU KV cache size: 192,928 tokens`，
  `Maximum concurrency for 2,048 tokens per request: 94.20x`，16 个请求只占 32768，
  日志中无任何 preemption。
- **不是编码器缓存缓冲的副作用**：做了控制实验，`encoder_cache_num_blocks` 48 与 72 结果
  **完全一致**（138.20 vs 138.24 ms，吞吐都是 110.0 tok/s）。
- **不是不稳定**：10 次迭代全部完成，每请求恰好 256 token，TPOT p50 138.20 / p90 139.58。

**纯文本对照：悬崖同样出现，所以问题在通用解码路径，不在多模态路径。**
用同长度（1651 token）的纯文本 prompt、无视觉输入：

| 负载 | BS 8 TPOT | BS 16 TPOT | 倍数 |
|---|---|---|---|
| 视频 | 23.96 ms | 138.20 ms | 5.77× |
| 纯文本 | 23.01 ms | 135.80 ms | 5.90× |

排除了 mrope、vision embedding 注入、编码器缓存等所有多模态相关嫌疑。
推测是 `num_seqs=16` 在 TP=4 下触发了某个编译图或 kernel 的回退路径，**未定位根因**。

顺带一个有用的量：**解码成本几乎全在文本骨干上**，视觉通路只让 BS=8 的 TPOT 多 4%
（23.96 vs 23.01）。视觉只影响 TTFT（817 vs 577 ms，差 240 ms）。

实用结论：**这台机器上 batch 8 就是最优点，不要再往上加。**

### 5.2 其他调优项：全部无效或更差（已实测）

| 调优项 | 结果 | TPOT | 吞吐 |
|---|---|---|---|
| 基线（`max_model_len=2048`, BS=8） | — | 23.96 ms | 284.1 tok/s |
| `decode_context_length_buckets=[1920]` | **更差 1.87×** | 44.76 ms | 163.8 tok/s |
| `enable_chunked_prefill=False` | **无差别** | 23.99 ms | 284.1 tok/s |
| `attention_dp_size=2` | **不可用** | — | — |

- **`decode_context_length_buckets` 反而更差**，推翻了本报告早期版本「显式配它可以在保留大
  `max_model_len` 的同时拿到同样收益」的猜测。1920 = 15×128 满足文档约束（升序、
  < `max_model_len`、能被 128 整除），但比 2048 慢 1.87 倍。推测是**非 2 的幂**的 context
  长度让解码注意力 kernel 的分块效率变差。**建议：把 `max_model_len` 取成贴合负载的
  2 的幂，不要用这个参数。**
- **`enable_chunked_prefill=False` 完全没有影响**（23.99 vs 23.96）。原因是 prompt 1651
  本来就 ≤ `max_num_batched_tokens=2048`，从来没触发过分块。README 标注该特性不支持，
  但打开与否在这个负载上无差别。
- **`attention_dp_size=2` 被配置校验拒绝**：
  `Component DP requires disaggregated inference (--kv-transfer-config). Single-node
  serving without DI is not supported with component DP.` 单节点部署用不了这个 knob。

---

## 6. 发现的两个问题

### 6.1 连续不同视频会崩掉引擎（会影响生产）

```
RuntimeError: Encoder cache full: cannot allocate mm_hash=... (1568 tokens,
4 blocks needed, 0 free, 8 active, 0 held). Scheduler should have evicted before dispatching.
```

**第 9 个不同的视频就崩，batch=1 顺序发请求也会崩，与并发无关。**

根因是两套记账口径不一致：

- scheduler 按 **token** 记账：`encoder_cache_size = max(max_num_batched_tokens, max_tokens_per_mm_item) = 16384`
  （取的是**最坏情况**视频，不是我们这个）。我们的视频 1568 token → 它认为能缓存 `16384/1568 = 10` 个。
- 设备端 `EncoderCacheBlocks` 按 **block** 分配：`cache_block_size = 2048/4 = 512`，
  我们的视频要 `ceil(1568/512) = 4` 块 = 2048 slot 装 1568 token（**1.31× 填充浪费**）。
  自动推导 `num_blocks = ceil(16384/512)+1 = 33`，可分配 32 块 → 只装得下 `32/4 = 8` 个。
- 8 < 10，scheduler 派发第 9 个时 worker 已无空闲 block。

这正是仓库里已有的 TODO：`vllm_neuron/model/neuron_config.py:304`、
`vllm_neuron/vllm/worker/neuron_model_runner.py:895`，注释明确写了「set `encoder_cache_num_blocks`
explicitly for workloads with many small images」。之前没暴露是因为压测都用同一个视频/图片 ——
一个 `mm_hash` 永远填不满缓存。

**规避**:

```
encoder_cache_num_blocks = ceil(encoder_cache_size / 每视频embed数) * 每视频block数 + 1
                         = ceil(16384/1568) * 4 + 1 = 45
```

本报告用 **48**（留余量），20 次迭代跑干净。注意该值参与图编译，**改它要全量重编译（~370 s）**。

**副作用**：缓存快满前的那两次迭代 TTFT 会飙到 703 ms / 836 ms（对比 p50 294 ms）。
这是缓存压力而非运行时抖动 —— 加到 48 块后分布很紧（max 354 ms）。

### 6.2 一个引擎里放多个 `num_seqs_buckets` 会挂死

`num_seqs_buckets: [1,2,4,8]` + `max_num_seqs=8` 的配置**能正常编译和 warmup**
（`Model warmup completed: 1 num_batched_tokens_buckets, 4 num_seqs_buckets`，
`engine ready in 178.3s`），但**第一个真实请求就永久挂住**：
4 个 worker 各占 155% CPU 空转 9 分钟无任何日志推进，无编译子进程，最后被手动终止。

改成**每个 batch size 一个独立引擎、每个只带一个 `num_seqs_buckets`** 后 4 个配置全部正常。
本报告所有数据都是这样跑的。

未深入定位（4 个核全被挂住的进程占用，无法并行诊断；等 20 分钟超时只会给出 forward pass 的
栈，信息量有限）。**影响**：想在一个服务里同时支持多种解码 batch，目前这条路不通。

---

### 6.3 TP=1 编不出来（neuronx-cc 段错误）

> 完整现场、最小复现命令、已排除原因清单见 **[`TP1_COMPILE_FAILURE.md`](TP1_COMPILE_FAILURE.md)**。

为评估「TP=1 × 4 副本」替代「TP=4 × 1 副本」的可行性，试了 TP=1，**两次都在
neuronx-cc 内部段错误**：

```
Fatal Python error: Segmentation fault
  File ".../concurrent/futures/process.py", line 263 in _process_worker
  File ".../multiprocessing/popen_fork.py", line 71 in _launch
  File ".../bin/neuronx-cc", line 8 in <module>
RuntimeError: neuronx-cc compilation failed with 70
```

| 尝试 | 编译并行度 | 结果 |
|---|---|---|
| 1 | 默认（`VLLM_NEURON_PARALLEL_COMPILE_WORKERS=8`） | 第 34.5 分钟段错误 |
| 2 | `PARALLEL_COMPILE_WORKERS=1` + `PARALLEL_TRACE_WORKERS=1` | 第 35 分钟段错误 |

**不是资源问题**：主机 124 GB RAM、崩溃前后都空着 100 GB 左右、`dmesg` 无 OOM-killer 记录。
**也不是插件的编译并行度**：串行化后照样崩，而且崩点在 **neuronx-cc 自己内部的
`ProcessPoolExecutor` fork**，那是编译器自带的多进程，插件的 knob 管不到。
看起来是编译器在未分片的 Qwen3-VL-8B 图上的 bug。

**副产物（对所有编译都有用）**：串行化让单个 HLO 的编译从 **125.1 s 降到 28.3 s（4.4×）**。
默认 8 个 worker 在这台 12 vCPU 的机器上存在明显争抢。
`envs.py` 里的注释也标了这个 TODO：*"Determine the optimal value automatically based on
CPU/RAM headroom"*。

**另一个独立的否决理由是显存**：

```
TP=4:  Neuron HBM: 5.64 GiB used, 18.36 GiB free   → KV cache 192,928 token（94× 余量）
TP=1:  Neuron HBM: 17.08 GiB used, 6.92 GiB free
```

每核预算约 24 GiB，TP=1 要用 17.08 GiB 放完整权重，只剩 6.92 GiB 给 KV + 编码器缓存。
而 DP 方案的全部价值就在于「每副本还能跑大 batch 来摊薄权重读」—— 这个前提被显存掐掉了。

### 6.4 TP=2 静默输出乱码（最危险的一个）

> 完整证据矩阵、最小复现、未做的定位工作见 **[`TP2_WRONG_OUTPUT.md`](TP2_WRONG_OUTPUT.md)**。

**TP=2 能编译、能跑、性能数字完全合理，但模型输出是乱码。** 没有任何报错。

| 配置 | 输出样本（前 100 字符） | 正确? |
|---|---|---|
| **TP=4**, 视频 | `In this video, a young child is sitting on a bed, engrossed in reading a book...` | ✅ |
| **TP=4**, 纯文本 | `a Trainium accelerator. The quick brown fox jumps over the lazy dog while...` | ✅ |
| **TP=2**, 视频, DP=1 | `\键0 numpy bl,true快手坐肚子肚子,True游ilit真真1111l把 ...` | ❌ |
| **TP=2**, 视频, DP=2 | `\键0 numpy c自己,sende  <table \ \ \ \ \ \ \, \, ...` | ❌ |
| **TP=2**, 纯文本, DP=1 | `\ \\b\b\b\l\l\l\l\l\l\l\l\l\l\l\l\l\l ...` | ❌ |
| **TP=1** | 编不出来（§6.3） | — |

**定位到 kernel 层**：纯文本同样乱码 → 与视觉塔无关；DP=1/DP=2 都乱码 → 与请求级 DP 无关；
而 **`VLLM_NEURON_DISABLE_NKI_KERNELS=1` 后 TP=2 输出正确** → 分片逻辑、集合通讯、
权重加载全部无罪，**故障是某个 NKI kernel 在 TP=2 的形状下算错**。
首要嫌疑是 QKV kernel 的尺寸比例守卫太松（`fused_qkv_dim/H` 在 TP=2 是 0.75、TP=4 是 0.375，
而代码注释已承认该 kernel 在比例过大时算错）。详见 TP2_WRONG_OUTPUT.md §5。

**为什么危险**：性能数字看起来完全正常，甚至符合物理直觉（TP=2 的 TPOT 约 44–45 ms，
是 TP=4 的 23.96 ms 的 1.85 倍，正好对应核数减半）：

| 配置 | 总并发 | TTFT p50 | TPOT p50 | E2E p50 | tok/s | req/s |
|---|---|---|---|---|---|---|
| TP=4 × DP=1（正确） | 8 | 817 | 23.96 | 6905 | 284.1 | 1.11 |
| ~~TP=2 × DP=2~~（乱码） | 16 | 1006 | 44.08 | 12377 | ~~310.9~~ | ~~1.21~~ |
| ~~TP=2 × DP=1~~（乱码） | 8 | 1158 | 45.31 | 12711 | ~~158.4~~ | ~~0.62~~ |

如果只看吞吐表，会得出「TP=2 × DP=2 比 TP=4 快 9.4%，突破了 batch 16 悬崖」的结论 ——
**完全错误，因为算出来的东西是错的**。这些数字全部作废。

**教训：任何新的并行配置都必须先核对输出文本，再看性能数字。** 本脚本每次运行都会打印
一段生成结果（`--- sample output ---`），就是为此。

**对本机的含义：TP=4 是 Qwen3-VL-8B 在这个插件上唯一可用的并行配置。**
TP=1 编不出来、TP=2 静默算错、TP=3 不能整除 head 数。这也解释了为什么模型 recipe 只记录了
TP=16（trn2.48xlarge）—— 其他 TP 值看起来没有被验证过。

## 7. 对「每秒 1 次推理」目标的容量分析

目标：到达率 1 req/s，每请求 1 秒视频 + 最多 256 输出 token。

### 7.1 输出长度 × batch 的可选工作点（全部实测）

| BS | 输出 token | TTFT p50 | TPOT p50 | **E2E p50** | tok/s | **req/s** | 达到 1 req/s |
|---|---|---|---|---|---|---|---|
| 1 | 256 | 316 | 11.23 | 3180 | 80.4 | 0.31 | ✗ |
| 1 | 128 | 293 | 11.25 | 1722 | 74.0 | 0.58 | ✗ |
| 1 | 64 | 304 | 11.23 | **1011** | 62.5 | 0.98 | 差一点 |
| 8 | 256 | 817 | 23.96 | 6905 | 284.1 | **1.11** | ✓ 余量 11% |
| 8 | 128 | 806 | 26.66 | 4197 | 228.8 | **1.79** | ✓ 余量 79% |
| 8 | 64 | 804 | 32.11 | **2845** | 164.1 | **2.56** | ✓ 余量 156% |

两个推荐工作点：

- **吞吐优先**：BS=8 + 256 token → 1.11 req/s，单请求 6.9 s。满足需求但余量薄。
- **均衡**（建议）：BS=8 + 128 token → **1.79 req/s，单请求 4.2 s**。余量 79%，
  延时还降了 39%。
- **延时优先**：BS=1 + 64 token → 单请求 **1.01 s**，但只有 0.98 req/s，**刚好差一点**
  撑不住 1 req/s 的到达率。要低延时又要吞吐得用 BS=8 + 64 token（2.85 s / 2.56 req/s）。

注意 BS=8 下 TPOT 随输出变短而上升（23.96 → 26.66 → 32.11），BS=1 下则完全不变
（11.23 / 11.25 / 11.23）。原因是 BS=8 有 8 次串行 prefill（约 800 ms）与解码争抢，
输出越短这部分固定开销摊到的 token 越少。

### 7.2 结论

**能达到 —— 前提是 `max_model_len` 配对（§5）并且用 batch 8。**

- 需要的输出吞吐 = 1 req/s × 256 token = **256 tok/s**
- 实测（`max_model_len=2048`, batch 8）= **284.1 tok/s / 1.11 req/s** → 有约 11% 余量
- 单请求延时代价：batch 8 下 E2E **6.9 s**（batch 1 是 3.18 s）。到达率 1 req/s、
  服务时间 6.9 s，按 Little's law 稳态在途请求数 ≈ 6.9，正好在 batch 8 的容量内。
- batch 4 只有 188.4 tok/s / 0.74 req/s，**撑不住** 1 req/s。

对比：调优前（`max_model_len=4096`）batch 8 只有 101 tok/s / 0.39 req/s，
所以本报告早期版本的结论「缺口 2.5 倍、加并发也没用」是错的 —— 那是配置问题，不是硬件上限。

余量偏薄（11%），如果要更稳：

| 方案 | 效果 | 代价 |
|---|---|---|
| **砍输出到 128 token** | **已测**：1.79 req/s，E2E 降到 4.2 s | 输出变短 |
| ~~加大 batch~~ | **已测，走不通** —— batch 16 掉到 110 tok/s / 0.43 req/s（§5.1） | — |
| ~~`decode_context_length_buckets`~~ | **已测，更差 1.87 倍**（§5.2） | — |
| ~~`enable_chunked_prefill=False`~~ | **已测，无差别**（§5.2） | — |
| ~~`attention_dp_size>1`~~ | **不可用**，需要 disaggregated inference（§5.2） | — |
| 换 trn2.48xlarge（TP=16） | 4 倍核数 | 成本 |

> 全部工作点见 §7.1，均为实测，无外推。

---

## 8. 复现方式

脚本：`examples/vllm_neuron/models/qwen3_vl/benchmark_video_latency.py`（新增）。
用离线 `AsyncLLM` 流式接口测量，不经 HTTP —— 这样帧数是精确的 16 帧。
走 OpenAI server 的话服务端会按默认 fps 重采样，1 秒视频会被采成 2 帧，测的就不是目标负载。

```bash
V=/opt/aws_neuronx_venv_pytorch_inference_vllm_0_21_0_1_0_0

# 先看解析出的形状和 bucket 数学，不加载权重
PATH=$V/bin:$PATH $V/bin/python \
  examples/vllm_neuron/models/qwen3_vl/benchmark_video_latency.py --dry-run

# 单个 batch size（多个 batch size 放一个引擎会挂，见 6.2）
for bs in 1 2 4 8; do
  PATH=$V/bin:$PATH VLLM_CACHE_ROOT=/mnt/nvme/cache/vllm $V/bin/python \
    examples/vllm_neuron/models/qwen3_vl/benchmark_video_latency.py \
    --batch-sizes $bs --num-iters 10 --warmup-iters 2 \
    --encoder-cache-num-blocks 48 --max-model-len 2048 \
    --output-json /mnt/nvme/bench_results/bs$bs.json
done
```

脚本要点：

- vision bucket **不是硬编码**：先在 CPU 上跑一次 HF processor 拿到真实 `video_grid_thw`，
  再按分块填充算出 8192。改 `--fps` / `--duration-sec` / `--resolution` 会自动重算。
- 默认**每请求独立视频**（改一个角落像素的 3 个字节，形状不变），避免 §4.2 的缓存假象；
  `--reuse-video` 可测缓存命中路径。
- `ignore_eos=True` 强制每请求恰好 256 token，否则不同 batch 间 TPOT 不可比。
- 本机必需的 EFA / prefix-caching flag 已内置。
- `--video-source synthetic` 可完全离线（确定性噪声）。

编译耗时参考：冷启动全量编译 ~370–410 s；`VLLM_CACHE_ROOT` 命中后引擎启动 ~105–180 s。
每个新的 `num_seqs_buckets` 或 `encoder_cache_num_blocks` 值都会触发重编译。

---

## 9. 原始数据

原始数据留在测试机本地 `/mnt/nvme/bench_results/`，**未提交进仓库**（JSON 合计 ~2.4 MB，
含每请求逐 chunk 时间戳）：

| 文件 | 内容 |
|---|---|
| `REPORT.md` | 本报告 |
| `bs{1,2,4,8}.json` | §4.1 每个 batch 的完整结果（配置 + 汇总 + 每请求逐 chunk 时间戳） |
| `bench_bs{1,2,4,8}.log` | 对应的完整引擎日志 |
| `bench_qwen3vl_video_bs1_final.json` | batch 1，20 次迭代，独立视频 |
| `bench_qwen3vl_video_bs1_unique.json` | §4.2 对照：独立视频 |
| `bench_qwen3vl_video_bs1_reuse.json` | §4.2 对照：复用视频（缓存命中） |

## 10. 未做的事

- §5.1 batch 16 悬崖的根因（已排除 KV cache、编码器缓存缓冲、多模态路径；纯文本同样复现）
- `vision_attention_block_size` / vision TP-DP 切分的调优（TTFT 方向，未动过）
- 自定义视频文件输入（脚本目前只支持 demo asset 和合成噪声，可加 `--video-path`）
- 在线 serving（HTTP）路径的延时，需要先解决服务端重采样帧数的问题
