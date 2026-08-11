# neuronx-cc 段错误：Qwen3-VL-8B 在 TP=1 下编不出 prefill 图

**日期**: 2026-08-11 · **复现次数**: 2/2（100%）· **状态**: 未解决，已绕过（用 TP=4）

一句话：`tensor_parallel_size=1` 时，Qwen3-VL-8B-Instruct 的 **prefill 图**在
`neuronx-cc` 的 `Tensorizer/TensorInitialization` pass 中段错误退出（exit 70）。
同样的模型/配置在 `tensor_parallel_size=4` 下编译正常。

---

## 1. 环境

| 项 | 值 |
|---|---|
| 实例 | `trn2.3xlarge`（i-0ad9fb51a3fed095e） |
| Neuron 设备 | 1 device / 4 逻辑 NeuronCore（`logical-neuroncore-config: 2`）/ 96 GB HBM |
| 主机 | 12 vCPU / 124 GB RAM / **无 swap** |
| 内核 | Linux 6.17.0-1019-aws |
| Neuron 驱动 | 2.29.0.0 |
| venv | `/opt/aws_neuronx_venv_pytorch_inference_vllm_0_21_0_1_0_0`（DLAMI 自带） |
| **neuronx-cc** | **2.26.6360.0+6f180f47**（HWM 2.26.0.6360++6f180f47） |
| Python / NumPy | 3.12.3 / 2.3.5 |
| nki | 0.5.0+28631259367.ga768afa6 |
| torch / vllm / transformers | 2.11.0 / 0.21.0 / 5.14.1 |
| vllm-neuron plugin | `release-0.21.0.1.0.0`（本仓库，测试时 HEAD = `db9bf66`） |
| 模型 | `Qwen/Qwen3-VL-8B-Instruct`，bf16，本地 `/mnt/nvme/models/Qwen_Qwen3-VL-8B-Instruct` |

## 2. 复现步骤

```bash
V=/opt/aws_neuronx_venv_pytorch_inference_vllm_0_21_0_1_0_0
PATH=$V/bin:$PATH VLLM_CACHE_ROOT=/mnt/nvme/cache/vllm \
  NEURON_SKIP_EFA_AFFINITY=1 NEURON_CC_FLAGS="--temp-dir=/tmp/neuroncc_tmp" \
  VLLM_NEURON_COMPILATION_TIMEOUT=3600 \
  $V/bin/python examples/vllm_neuron/models/qwen3_vl/benchmark_video_latency.py \
  --batch-sizes 8 --tensor-parallel-size 1 --encoder-cache-num-blocks 48
```

引擎配置（`--tensor-parallel-size 1` 是唯一与可用配置的差异）：

```python
max_model_len          = 2048          # 自动取值：prompt 1651 + 输出 256 = 1907 → 2048
max_num_batched_tokens = 2048
max_num_seqs           = 8
tensor_parallel_size   = 1             # ← 触发条件；4 则正常
enable_prefix_caching  = False
additional_config = {
  "neuron_config": {
    "quantization": "bf16",
    "num_batched_tokens_buckets": [2048],
    "num_seqs_buckets": [8],
    "on_device_sampling_config": {"all_greedy": True},
  },
  "vision_neuron_config": {
    "num_vision_tokens_buckets": [8192],
    "vision_attention_block_size": 2048,
    "encoder_cache_num_blocks": 48,
  },
}
```

输入：16 帧 448×448 视频（`video_grid_thw=[8,28,28]`，1568 embed token，prompt 1651 token）。

## 3. 现象

两次尝试，唯一区别是插件侧的编译并行度：

| # | batch | 编译并行度 | 崩溃时刻 | 结果 |
|---|---|---|---|---|
| 1 | 1 | 默认（`VLLM_NEURON_PARALLEL_COMPILE_WORKERS=8`，`PARALLEL_TRACE_WORKERS=8`） | 启动后 **34分32秒** | segfault → exit 70 |
| 2 | 8 | `PARALLEL_COMPILE_WORKERS=1` + `PARALLEL_TRACE_WORKERS=1` | 启动后 **约 35 分** | segfault → exit 70 |

尝试 2 的时间线（`/tmp/tp1_bs8_serial.log`）：

```
10:56:32  进程启动
10:57:05  Neuron HBM: 17.08 GiB used, 6.92 GiB free      ← 权重加载完成
11:31:30  Compiled HLO 6cabdf42a53a37a1741800a9aab005c2 in 28.3s   ← 前一个图成功
11:31:31  Compiling...                                   ← 失败的图开始
11:34:15  Fatal Python error: Segmentation fault
11:34:17  EXIT=1
```

**两次撞的是同一个图**：cache 目录 `6dcc944f1be638ed5bb225055e4286f1`
（尝试 1 在 09:50:23 也引用了它）。batch 1 和 batch 8 都命中，说明该图与
`num_seqs` 无关。

## 4. 失败的是 prefill 图

`.../6dcc944f1be638ed5bb225055e4286f1/dev0_0.rank0/example_inputs.txt`：

```
Input 0:  Shape: (2048,)   Dtype: int64    ← token ids，= max_num_batched_tokens
Input 1:  Shape: (2048,)   Dtype: int32
Input 2:  Shape: ()        Dtype: int32
```

Input 0 是 2048 长度的 token id 向量 → **prefill 图**（2048 batched tokens）。
这与「batch 1 和 batch 8 都失败」一致：prefill 图不随 `num_seqs` 变化。

`graph.hlo` 大小 **9.5 MB**，整个 cache 目录 130 MB。

## 5. 编译器侧证据

编译器调用（`command.txt` 原文）：

```
neuronx-cc compile .../graph.hlo --framework XLA --target trn2 \
  --output .../graph_6dcc944f1be638ed5bb225055e4286f1.neff \
  --logfile .../log-neuron-cc.txt --auto-cast=none --verbose=35 -O1 \
  '--internal-hlo2tensorizer-options=--modular-flow-mac-threshold=10 --experimental-unsafe-fp8e4m3fn-as-fp8e4m3' \
  --internal-backend-options=--enable-verifier=false
```

`log-neuron-cc.txt` 里崩溃前的最后几个 pass：

```
11:34:06  [sg0000/Tensorizer/InferPSumTensor]:      Running InferPSumTensor_iteration_1
11:34:06  [sg0000/Tensorizer/LegalizeType]:         Running LegalizeType
11:34:06  [sg0000/Tensorizer/WeightCoalescing]:     Running WeightCoalescing
11:34:06  [sg0000/Tensorizer/LegalizeSundaAccess]:  Running LegalizeSundaAccess
11:34:08  [sg0000/Tensorizer/RelaxPredicates]:      Running RelaxPredicates
11:34:09  [sg0000/Tensorizer/TensorInitialization]: Running TensorInitialization   ← 最后一个
11:34:15  USER: A process in the process pool was terminated abruptly ...
11:34:15  ERROR: Type: <class 'concurrent.futures.process.BrokenProcessPool'>
          Subcommand returned with exitcode=70
```

**失败 pass：`sg0000/Tensorizer/TensorInitialization`**（进入后 6 秒崩）。
异常类型 `BrokenProcessPool` —— 是编译器内部 worker 进程死了，不是编译器自己抛的诊断。

编译器自报的内存占用：

```
Current memory usage for neuronxcc is 1578528 Kilobytes.     ← 约 1.5 GB
```

宿主 Python 侧的 fatal error（`neuronx-cc` 顶层）：

```
Fatal Python error: Segmentation fault

Current thread (most recent call first):
  File ".../concurrent/futures/process.py", line 263 in _process_worker
  File ".../multiprocessing/process.py",   line 108 in run
  File ".../multiprocessing/popen_fork.py", line 71 in _launch
  ...
  File ".../concurrent/futures/process.py", line 807 in _spawn_process
  File ".../concurrent/futures/process.py", line 797 in _launch_processes
  File ".../concurrent/futures/process.py", line 770 in _start_executor_manager_thread
  File ".../concurrent/futures/process.py", line 831 in submit
  ...
  File "/opt/aws_neuronx_venv_pytorch_inference_vllm_0_21_0_1_0_0/bin/neuronx-cc", line 8 in <module>
```

段错误发生在 **`neuronx-cc` 自己的 `ProcessPoolExecutor` fork 新 worker 的时候**。

## 6. 已排除的原因

| 假设 | 排除依据 |
|---|---|
| **主机内存不足** | 124 GB RAM，崩溃前后 `free` 显示空闲 ~100 GB；`dmesg` 无 OOM-killer 记录；无 swap 但也没到需要 swap 的程度。**编译器自报仅用 1.5 GB。** |
| **插件的编译并行度过高** | `VLLM_NEURON_PARALLEL_COMPILE_WORKERS=1` + `VLLM_NEURON_PARALLEL_TRACE_WORKERS=1` 串行化后**照样崩**，且崩点在 neuronx-cc 内部的 fork —— 该 knob 只控制同时起几个 neuronx-cc *调用*，管不到编译器自身的多进程。 |
| **设备 HBM 不足** | 编译是纯主机侧行为。且权重已成功加载：`17.08 GiB used, 6.92 GiB free`（每核约 24 GiB 预算）。 |
| **编译超时** | `VLLM_NEURON_COMPILATION_TIMEOUT=3600`，崩溃发生在第 35 分钟，远未超时；退出码是 70 而非超时。 |
| **模型/配置本身有问题** | 完全相同的模型、输入、bucket、encoder cache 配置，仅把 `tensor_parallel_size` 改成 4 就能编译成功（约 370 s）并正常推理。 |

结论：看起来是 **neuronx-cc 2.26.6360.0 在未分片（TP=1）的 Qwen3-VL-8B prefill 图上的
缺陷**，触发点在 `Tensorizer/TensorInitialization`。

## 7. 对照：TP=4 正常

| | TP=1 | TP=4 |
|---|---|---|
| prefill 图编译 | **segfault，exit 70** | 成功 |
| 引擎启动 | 失败 | 约 370 s（冷）/ 105–180 s（缓存命中） |
| Neuron HBM | 17.08 GiB used, 6.92 GiB free | 5.64 GiB used, 18.36 GiB free |
| 可用性 | ❌ | ✅ 已完成完整 benchmark |

## 8. 影响

- **无法评估「TP=1 × N 副本」替代「TP=N × 1 副本」的方案。** 该方案的动机是去掉
  TP 的集合通讯、并绕开 batch 16 的性能悬崖（见 `BENCHMARK_REPORT.md` §5.1）。
- 该方案另有一个独立的否决理由（显存）：TP=1 时 17.08 GiB 权重占掉每核约 24 GiB 预算的
  71%，只剩 6.92 GiB 给 KV cache + 编码器缓存，而 DP 副本方案的价值前提正是
  「每副本仍能跑大 batch 来摊薄权重读」。所以即使编译问题修复，该方案也未必可行。
- 生产上不受影响：TP=4 工作正常，且是本机 4 核的最优取法。
- **中间方案 TP=2 也不可行**：它能编译，但**静默输出乱码**（文本骨干分片错误，
  纯文本同样复现）。详见 `BENCHMARK_REPORT.md` §6.4。所以 TP=1 编译修好之后，
  也只剩 TP=4 一个可用配置。

## 9. 保留的现场

| 内容 | 路径 |
|---|---|
| 失败图 + 编译器日志 + 各 pass 中间结果（130 MB） | `/mnt/nvme/cache/vllm/neuron/compile_cache/6dcc944f1be638ed5bb225055e4286f1/dev0_0.rank0/` |
| ├ 编译器调用原文 | `command.txt` |
| ├ 编译器日志（297 KB） | `log-neuron-cc.txt` |
| ├ HLO（9.5 MB） | `graph.hlo`，以及 `hlo_passes/step1..4_*.hlo` |
| ├ FX 图（5.2 MB）与 pass 中间态 | `fxgraph.txt`、`passes/00..05_*.txt` |
| └ 输入形状 | `example_inputs.txt` |
| 尝试 1 引擎日志（TP=1, bs1） | `/tmp/tp1_bs1.log` |
| 尝试 2 引擎日志（TP=1, bs8, 串行编译） | `/tmp/tp1_bs8_serial.log` |

**关键现场已另存一份（2.0 MB），不受编译缓存清理影响**：
`/mnt/nvme/bench_results/tp1_compile_failure/`

```
command.txt          661 B   编译器调用原文
log-neuron-cc.txt    297 K   编译器日志（含失败 pass 与 BrokenProcessPool）
example_inputs.txt    27 K   输入形状（用于确认是 prefill 图）
graph.hlo.gz         1.5 M   失败的 HLO（解压后 9.5 MB，可直接喂给 neuronx-cc 复现）
tp1_bs1.log           82 K   尝试 1 引擎日志
tp1_bs8_serial.log    80 K   尝试 2 引擎日志
```

### 最小复现（已验证 ✅）

不需要插件、不需要模型权重、**不需要 Neuron 设备**（编译是纯主机行为，日志中 0 处
设备引用），**2 分 44 秒**即可复现，免去插件流程里 35 分钟的 trace 阶段：

```bash
gunzip -c /mnt/nvme/bench_results/tp1_compile_failure/graph.hlo.gz > /tmp/graph.hlo
V=/opt/aws_neuronx_venv_pytorch_inference_vllm_0_21_0_1_0_0
PATH=$V/bin:$PATH neuronx-cc compile /tmp/graph.hlo --framework XLA --target trn2 \
  --output /tmp/out.neff --logfile /tmp/repro-neuron-cc.txt \
  --auto-cast=none --verbose=35 -O1 \
  '--internal-hlo2tensorizer-options=--modular-flow-mac-threshold=10 --experimental-unsafe-fp8e4m3fn-as-fp8e4m3' \
  --internal-backend-options=--enable-verifier=false
```

实测结果（2026-08-11 11:45–11:48）：

```
real    2m44.315s
EXIT=70
Fatal Python error: Segmentation fault
A process in the process pool was terminated abruptly ...
Current memory usage for neuronxcc is 1560144 Kilobytes      ← 约 1.5 GB，与插件流程一致
```

与插件流程内的失败完全一致（exit 70 / segfault / 内存约 1.5 GB）。
**这是给上游的最佳报告材料**：一个 9.5 MB 的 HLO + 一条命令 + 3 分钟。

> 上面的命令加了 `--logfile`（原始现场是插件传的），否则 verbose 输出只会打成一串
> `.....` 到 stdout，看不到失败在哪个 pass。

> 注：`neuronx-cc` 日志末尾提到 `Artifacts stored in: /mnt/nvme/vllm-neuron/neuronxcc-slx_5dra`，
> 该临时目录事后已不存在（进程退出时清理）。要保留它需要在复现时加
> `NEURON_CC_FLAGS="--save-temps"` 之类的选项。

## 10. 顺带发现：默认编译并行度在 12 vCPU 上过高

虽然与本 bug 无关，但串行化对照暴露了一个可优化点：

| 编译并行度 | 单个 HLO 编译耗时 |
|---|---|
| 默认 8 worker | **125.1 s** |
| 1 worker | **28.3 s** |

**4.4 倍差异。** `VLLM_NEURON_PARALLEL_COMPILE_WORKERS` 与
`VLLM_NEURON_PARALLEL_TRACE_WORKERS` 默认都是 8（`vllm_neuron/envs.py:35-38`），
而本机只有 12 vCPU，争抢明显。`envs.py` 的注释已标注该 TODO：

> `TODO: Determine the optimal forks-per-worker automatically based on underlying CPU, number of ranks, and buckets being compiled.`

建议在 vCPU 较少的实例上显式降低这两个值（未逐一实测最优值，2 可能是本机的平衡点）。

---

## Summary for upstream (English)

**Title:** neuronx-cc 2.26.6360.0 segfaults compiling the Qwen3-VL-8B prefill graph at TP=1

`neuronx-cc` dies with a segmentation fault (exit code 70) while compiling the
prefill graph of `Qwen/Qwen3-VL-8B-Instruct` when `tensor_parallel_size=1`. The
identical model and configuration compiles and runs fine at
`tensor_parallel_size=4`. Reproduced 2/2.

- **Instance:** trn2.3xlarge (4 logical NeuronCores, LNC=2, 96 GB HBM, 12 vCPU, 124 GB RAM)
- **Compiler:** neuronx-cc 2.26.6360.0+6f180f47, target trn2, `-O1`
- **Plugin:** vllm-neuron 0.21.0.1.0.0, vLLM 0.21.0, torch 2.11.0
- **Failing graph:** prefill, `max_num_batched_tokens=2048` (Input 0 = `(2048,) int64`), `graph.hlo` 9.5 MB
- **Failing pass:** `sg0000/Tensorizer/TensorInitialization`, ~6 s after entering it
- **Exception:** `concurrent.futures.process.BrokenProcessPool` — the compiler's own
  internal worker died; the top-level `neuronx-cc` process reports
  `Fatal Python error: Segmentation fault` inside `ProcessPoolExecutor._spawn_process`
  → `popen_fork._launch`
- **Not memory:** 124 GB host RAM with ~100 GB free, no OOM-killer entries, and the
  compiler's own diagnostic reports `Current memory usage for neuronxcc is 1578528 Kilobytes` (~1.5 GB)
- **Not the caller's compile parallelism:** setting both
  `VLLM_NEURON_PARALLEL_COMPILE_WORKERS=1` and `VLLM_NEURON_PARALLEL_TRACE_WORKERS=1`
  reproduces it identically
- **Time to failure:** ~35 minutes of compilation before the crash, both times
- **Minimal repro (verified):** the 9.5 MB `graph.hlo` alone reproduces it in
  **2m44s** with a single `neuronx-cc compile` invocation — no plugin, no model
  weights, and **no Neuron device** required (compilation is host-side only).
  Command and artifacts below.
- **Artifacts:** full HLO, per-pass dumps and compiler log preserved under
  `compile_cache/6dcc944f1be638ed5bb225055e4286f1/dev0_0.rank0/` (130 MB); the
  essential 2 MB subset (HLO + compiler log + both engine logs) is kept
  separately so it survives a compile-cache wipe.
