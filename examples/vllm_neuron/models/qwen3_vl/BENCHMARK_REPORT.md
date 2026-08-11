# Qwen3-VL-8B-Instruct 视频推理延时基准报告

**日期**: 2026-08-11 · **机器**: trn2.3xlarge (i-0ad9fb51a3fed095e, 1 Neuron device / 4 NeuronCores / 96GB)
**软件**: vllm-neuron 0.21.0.1.0.0 (branch `release-0.21.0.1.0.0`) · vLLM 0.21.0 · Neuron SDK 2.31 · transformers 5.14.1
**模型**: `Qwen/Qwen3-VL-8B-Instruct` (bf16, 本地路径 `/mnt/nvme/models/Qwen_Qwen3-VL-8B-Instruct`)

---

## 1. 结论速览

1. **batch 1 单请求**: TTFT **315 ms**、TPOT **11.34 ms**、256 token 端到端 **3.21 s**。
2. **提高 batch 几乎不提升吞吐**: batch 1→8 并发翻 8 倍，输出吞吐只从 79.7 涨到 **101.0 tok/s（1.27×）**，
   而单请求端到端从 3.2 s 恶化到 **20.0 s（6.2×）**。解码步时间随 batch 近似线性增长
   （11.3 / 24.9 / 41.1 / 72.4 ms），说明解码阶段**没有获得应有的批处理收益**（见 §5）。
3. **本机达不到「每秒 1 次 × 256 token」的目标**，缺口 2.5 倍：需要 256 tok/s，实测天花板 101 tok/s。
   可行方案见 §7。
4. 过程中发现 **2 个会影响生产的问题**：连续不同视频会让引擎崩溃（§6.1），
   多 `num_seqs_buckets` 配置会挂死（§6.2）。
5. 基准方法上有个坑：**用同一个视频重复压测会让视觉编码器被缓存跳过**，TTFT 会虚低 42%
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
max_model_len            = 4096
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

| BS | 样本数 | TTFT p50 | TTFT mean | TTFT p90 | TTFT max | TPOT p50 | ITL p50 | E2E p50 | E2E p90 | 输出 tok/s | req/s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 10 | **315** | 317 | 340 | 359 | **11.34** | 11.34 | **3208** | 3235 | 79.7 | 0.31 |
| 2 | 20 | 427 | 411 | 494 | 509 | 25.33 | 24.88 | 6862 | 6903 | 74.0 | 0.29 |
| 4 | 40 | 563 | 551 | 723 | 761 | 42.28 | 41.06 | 11351 | 11401 | 89.4 | 0.35 |
| 8 | 80 | 835 | 868 | 1076 | 1176 | 75.16 | 72.44 | 19998 | 20194 | **101.0** | **0.39** |

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

## 5. 解码阶段批处理效率异常

| BS | 解码步时间 (ITL p50) | 每序列摊薄 | 相对 batch 1 加速 |
|---|---|---|---|
| 1 | 11.34 ms | 11.34 ms | 1.00× |
| 2 | 24.88 ms | 12.44 ms | 0.91× |
| 4 | 41.06 ms | 10.27 ms | 1.10× |
| 8 | 72.44 ms | 9.06 ms | 1.25× |

解码步时间几乎与 batch 成正比。正常情况下 8B 模型的解码是**权重带宽受限**的，
batch 8 的单步时间应该接近 batch 1（每步只多算 8 个 token 的矩阵乘），
每序列成本应该接近 1/8。这里每序列只快了 1.25 倍，聚合吞吐只涨 1.27 倍。

粗算权重带宽下限：8B bf16 = 16 GB 权重 / 4 core = 4 GB/core/步，即使按 700 GB/s 算也只有 ~5.7 ms，
所以 batch 8 的 72 ms **不是**权重带宽造成的 —— 解码图里存在与 batch 成正比的计算量。

**尚未定位根因。** 排除项：ITL p50≈p90 说明不是周期性停顿；8 个请求确实在同一个解码批次里
（TTFT max 1176 ms 说明 1.2 s 内全部完成 prefill，E2E ≈ 835 + 256×72.4 ms 与单批解码吻合）。
README 里 Qwen3-VL 的「Perf Tuning」状态是 *In progress*，这可能就是已知的调优缺口。

**下一步建议**：跑一组同等长度的**纯文本** prompt 做对照。若纯文本也线性增长，说明是通用解码路径问题；
若纯文本正常，则问题在多模态解码路径（如 mrope 或 vision embedding 注入）。这个对照没跑。

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

## 7. 对「每秒 1 次推理」目标的容量分析

目标：到达率 1 req/s，每请求 1 秒视频 + 最多 256 输出 token。

**结论：本机做不到，缺口约 2.5 倍。**

- 需要的输出吞吐 = 1 req/s × 256 token = **256 tok/s**
- 实测天花板（batch 8）= **101 tok/s** → 实际只能撑 **0.39 req/s**
- 瓶颈完全在 decode：batch 1 时 decode 占端到端的 **91%**（2893 / 3208 ms），prefill 只占 9%。
  视觉编码（83 ms）和预处理（40 ms）都不是问题。
- 提高并发无法解决：§5 显示 batch 8 只带来 1.27× 吞吐，代价是端到端从 3.2 s 涨到 20.0 s。

可选方案：

| 方案 | 效果 | 代价 |
|---|---|---|
| **砍输出长度到 ~60 token** | batch 1 下 E2E ≈ 315 + 60×11.34 ≈ **1.0 s**，刚好 1 req/s | 输出变短 |
| 输出 ~80–100 token + batch 4 | 吞吐 ~89 tok/s 可撑 ~1 req/s | 端到端升到数秒 |
| **换 trn2.48xlarge（TP=16）** | 4 倍算力，官方 recipe 的目标配置 | 成本 |
| 先解决 §5 的解码批处理效率 | 若 batch 8 能达到理论值，吞吐可望 4–6 倍 | 需要定位根因 |

> 注：以上「砍 token」的推算基于实测 TPOT 线性外推，未实测。建议直接跑
> `--max-tokens 64` 验证。

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
    --encoder-cache-num-blocks 48 \
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

- §5 解码批处理效率异常的根因定位（建议先做纯文本对照）
- 更短输出长度（64 / 128 token）的实测验证
- batch 16 及以上
- 自定义视频文件输入（脚本目前只支持 demo asset 和合成噪声，可加 `--video-path`）
- 在线 serving（HTTP）路径的延时，需要先解决服务端重采样帧数的问题
