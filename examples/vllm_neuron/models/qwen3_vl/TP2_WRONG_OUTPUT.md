# Qwen3-VL-8B 在 TP=2 下静默输出乱码

**日期**: 2026-08-11 · **复现次数**: 3/3（100%）· **状态**: **根因已确认，插件侧已修复并验证**（真正的修法在 nkilib，见 §5.2）
**严重性**: 高 —— **静默的数值错误，无任何报错或警告**

一句话：`tensor_parallel_size=2` 时 Qwen3-VL-8B-Instruct 能正常编译、正常推理、
性能指标完全合理，但**生成的文本是乱码**。同样配置在 `tensor_parallel_size=4` 下正确。
**根因**：NKI QKV kernel 在 `fused_qkv_dim >= 3072` 时静默算错（相对误差 1.06），
而它自己的 SBUF 容量校验只在 ≥ 3584 才报错，3072 恰好漏过去。Qwen3-VL-8B 每 rank 的
`fused_qkv_dim` 在 TP=2 是 **3072**（错）、TP=4 是 **1536**（对）。
判据是绝对值而非与 hidden 的比例，所以原有的 `fused_qkv_dim > H` 守卫拦不住。
**已在插件侧修复并验证**（§5.3）。复现器：同目录 `qkv_kernel_isolation.py`，约 1 分钟。

> 这类失败比崩溃危险：崩溃会拦住你，静默算错不会。只看性能表会得出
> 「TP=2 × DP=2 比 TP=4 快 9.4%」的自信结论，而结果是错的。

---

## 1. 环境

与 [`TP1_COMPILE_FAILURE.md`](TP1_COMPILE_FAILURE.md) §1 完全相同：

| 项 | 值 |
|---|---|
| 实例 | `trn2.3xlarge`（i-0ad9fb51a3fed095e），4 逻辑 NeuronCore（LNC=2），96 GB HBM |
| 主机 | 12 vCPU / 124 GB RAM |
| Neuron 驱动 / neuronx-cc | 2.29.0.0 / **2.26.6360.0+6f180f47** |
| venv | `/opt/aws_neuronx_venv_pytorch_inference_vllm_0_21_0_1_0_0` |
| vllm / vllm-neuron / torch / transformers | 0.21.0 / 0.21.0.1.0.0 / 2.11.0 / 5.14.1 |
| 模型 | `Qwen/Qwen3-VL-8B-Instruct`，bf16 |

## 2. 证据矩阵

三个 TP=2 配置全部乱码，两个 TP=4 对照全部正确。`temperature=0`、
`on_device_sampling_config.all_greedy=True`，故输出是确定性的。

| # | TP | DP | 输入 | 生成文本（前 ~130 字符，原样） | 正确 |
|---|---|---|---|---|---|
| A | **4** | 1 | 视频 | `In the video, a young child with short, light-colored hair is sitting on a bed, wearing glasses and a light blue sleeveless top with pink pants...` | ✅ |
| B | **4** | 1 | 纯文本 | `a Trainium accelerator. The quick brown fox jumps over the lazy dog while the engineer measures decode latency across batch sizes...` | ✅ |
| C | **2** | 1 | 视频 | `\键0 numpy bl,true快手坐肚子肚子,True游ilit真真1111l把   � � �无无无无无 � � ... s s s s s s s ... ththhththhthth` | ❌ |
| D | **2** | 2 | 视频 | `\键0 numpy c自己,sende  <table \ \ \ \ \ \ \ \ \, \, �, < \ \ \ \ \ < \ < < < <, \ <, \, \, \, \, \, \, \, \,` | ❌ |
| E | **2** | 1 | 纯文本 | `\ \\b\b\b\l\l\l\l\l\l\l\l\l\l\l\l\l\l\l\l\l\l\l\l\l\l\l\l\l\l\l\l\l\l\l\l\l\l\l\l\l\l\l\l\l\l\l\l\l\l\l\l` | ❌ |

对照 B（TP=4 纯文本）正确续写了 prompt 里重复的填充句，说明纯文本通路本身没问题。
A 与 C 是**同一配置**（`max_model_len=2048`、batch 8、同一视频），只差 `tensor_parallel_size`。

完整文本样本另存在 `/mnt/nvme/bench_results/tp2_wrong_output/SAMPLE_OUTPUTS.md`。

### 由此得到的两个定位结论

- **E 乱码 → 与视觉塔无关。** 纯文本模式完全不构建 vision tower
  （脚本会省掉 `vision_neuron_config` 和 `limit_mm_per_prompt`），仍然乱码。
- **C 乱码 → 与请求级 DP 无关。** `data_parallel_size=1` 单副本同样乱码。

**结论：`tensor_parallel_size=2` 本身导致文本骨干算错。**

## 3. 复现步骤

```bash
V=/opt/aws_neuronx_venv_pytorch_inference_vllm_0_21_0_1_0_0

# C：视频，TP=2，DP=1 —— 乱码
PATH=$V/bin:$PATH VLLM_CACHE_ROOT=/mnt/nvme/cache/vllm NEURON_SKIP_EFA_AFFINITY=1 \
  $V/bin/python examples/vllm_neuron/models/qwen3_vl/benchmark_video_latency.py \
  --num-iters 3 --warmup-iters 1 --batch-sizes 8 \
  --tensor-parallel-size 2 --encoder-cache-num-blocks 48

# E：纯文本，TP=2 —— 乱码（最小复现，不涉及任何多模态代码）
PATH=$V/bin:$PATH VLLM_CACHE_ROOT=/mnt/nvme/cache/vllm NEURON_SKIP_EFA_AFFINITY=1 \
  $V/bin/python examples/vllm_neuron/models/qwen3_vl/benchmark_video_latency.py \
  --num-iters 3 --warmup-iters 1 --batch-sizes 8 --text-only \
  --tensor-parallel-size 2

# A：同上但 TP=4 —— 正确
... --tensor-parallel-size 4 --encoder-cache-num-blocks 48
```

脚本每次运行都会打印 `--- sample output (first 400 chars) ---`。
**E 是最小复现路径** —— 纯文本、单副本，只剩「模型 + TP=2」两个变量。

引擎配置（除 `tensor_parallel_size` 外与可用配置一致）：

```python
max_model_len          = 2048       # 自动取值
max_num_batched_tokens = 2048
max_num_seqs           = 8
tensor_parallel_size   = 2          # ← 触发条件
enable_prefix_caching  = False
additional_config = {"neuron_config": {
    "quantization": "bf16",
    "num_batched_tokens_buckets": [2048],
    "num_seqs_buckets": [8],
    "on_device_sampling_config": {"all_greedy": True},
}}
```

## 4. 为什么这个 bug 很难被发现

### 4.1 没有任何警告

引擎日志里唯一提到 TP 的地方就是配置回显 `tensor_parallel_size=2`，
之后一路正常：权重加载、编译、warmup、推理全部无异常。日志里与分片相关的
warning **一条都没有**（只有与本问题无关的 `NeuronAsyncScheduler`、
`Set seed for neuron device does not take effect` 等常规噪声）。

### 4.2 所有维度在 TP=2 下都能整除

不是「维度不匹配」这种能被 assert 拦住的问题：

| 维度 | 值 | / TP=2 | / TP=4 |
|---|---|---|---|
| Q heads | 32 | 16 ✅ | 8 ✅ |
| KV heads | 8 | 4 ✅ | 2 ✅ |
| MLP intermediate | 12288 | 6144 ✅ | 3072 ✅ |
| hidden_size | 4096 | — | — |
| head_dim | 128 | — | — |

（`mrope_section = [24, 20, 20]`，`head_dim=128`；视觉塔 16 heads / hidden 1152 /
intermediate 4304 / out_hidden 4096。TP=2 时视觉塔按 `resolve_tp_dp(2)` 解析为
`tp=1, dp=2`，即**视觉权重完全不分片** —— 又一条「问题不在视觉塔」的旁证。）

### 4.3 性能数字看起来完全合理

这是最容易上当的地方。TP=2 的 TPOT 是 TP=4 的 1.85 倍，与「核数减半」的物理预期
几乎完全吻合；DP=2 把两个副本凑到 16 并发后，吞吐甚至超过了 TP=4 的基线：

| 配置 | 总并发 | TTFT p50 | TPOT p50 | E2E p50 | tok/s | req/s |
|---|---|---|---|---|---|---|
| TP=4 × DP=1（**正确**） | 8 | 817 | 23.96 | 6905 | 284.1 | 1.11 |
| ~~TP=2 × DP=2~~（乱码） | 16 | 1006 | 44.08 | 12377 | ~~310.9~~ | ~~1.21~~ |
| ~~TP=2 × DP=1~~（乱码） | 8 | 1158 | 45.31 | 12711 | ~~158.4~~ | ~~0.62~~ |
| ~~TP=2 纯文本 × DP=1~~（乱码） | 8 | 773 | 43.88 | 11959 | ~~171.0~~ | ~~0.67~~ |

每个请求都恰好产出 256 token，10 轮迭代方差极小（TPOT p50 44.08 / p90 42.21）。
**从任何性能维度看都是"成功"的。** 唯一的破绽只有生成文本。

## 5. 根因：NKI QKV kernel 在 fused_qkv_dim >= 3072 时算错

用 `VLLM_NEURON_DISABLE_NKI_KERNELS=1`（该开关强制 `can_run_kernel()` 返回 False，
日志中 `kernel_call_count: 0` 可验证）做 2×2 对照，纯文本负载：

| TP | NKI kernels | 输出 | TTFT p50 | TPOT p50 | tok/s |
|---|---|---|---|---|---|
| 4 | 开 | ✅ | 577 | 23.01 | 317.0 |
| 4 | 关 | ✅ | 804 | 24.33 | 291.5 |
| **2** | **开** | **❌ 乱码** | 773 | 43.88 | 171.0 |
| **2** | **关** | **✅ 正确** | 1895 | 53.62 | 131.4 |

**关掉 NKI kernel 后 TP=2 输出正确。** 这一格是决定性的：

- **张量并行的分片逻辑、集合通讯、权重加载全部无罪** —— 否则关 kernel 也救不回来。
  （`fused_qkv_weight_loader` 与 head 分片数学也人工核对过，TP=2 下 16 Q / 4 KV per rank、
  `num_key_value_groups=4`，与 TP=4 一致且正确。）
- **故障在 kernel 层**：某个 NKI kernel 在 TP=2 的张量形状下产生错误结果。
- TP=4 关 kernel 也正确 → 各 PyTorch fallback 路径本身是对的。

### 5.1 确认：QKV kernel 在 `fused_qkv_dim >= 3072` 时算错

用 `qkv_kernel_isolation.py`（同目录）单独测这个 kernel —— 同一个 `NF.qkv_proj`，
CPU 上跑（`can_run_kernel` 返回 False → PyTorch 分支）当 golden，Neuron 上跑走 kernel，
直接比最大相对误差。整个 sweep 约 1 分钟。**T=512 与 T=2048 结果完全一致**，
所以阈值与序列长度无关。

| `fused_qkv_dim` | q/kv per rank | H | fused/H | rel 误差 | 判定 |
|---|---|---|---|---|---|
| 512 | 2/1 | 4096 | 0.125 | 0.0013 | ✅ |
| 768 | 4/1 | 4096 | 0.188 | 0.0028 | ✅ |
| **1536** | **8/2** | 4096 | 0.375 | 0.0011 | ✅ **(TP=4)** |
| 2048 | 8/4 | 4096 | 0.500 | 0.0021 | ✅ |
| 2560 | 16/2 | 4096 | 0.625 | 0.0021 | ✅ |
| 2688 | 15/3 | 4096 | 0.656 | 0.0026 | ✅ |
| 2816 | 16/3 | 4096 | 0.688 | 0.0021 | ✅ |
| **2944** | **17/3** | 4096 | 0.719 | 0.0046 | ✅ **最后一个正确值** |
| **3072** | **16/4** | 4096 | 0.750 | **1.0607** | ❌ **(TP=2)** |
| 3072 | 20/2 | 4096 | 0.750 | 1.0607 | ❌ 换 q/kv 拆分同样错 |
| 3072 | 16/4 | **8192** | **0.375** | 1.0051 | ❌ **比例小也错** |
| 1536 | 8/2 | **2048** | **0.750** | 0.0045 | ✅ **比例大但 fused 小就对** |
| 3584 | 20/4 | 4096 | 0.875 | — | 编译报错（见下） |
| 6144 | 32/8 | 4096 | 1.500 | 0.0039 | ✅ 被现有 `> H` 守卫拦下走 fallback |

**判据是 `fused_qkv_dim` 的绝对值，不是它与 H 的比例** —— 最后两行是关键：
3072 在 H=8192（比例 0.375）照样错，1536 在 H=2048（比例 0.75）照样对。
所以现有守卫 `fused_qkv_dim > H` **判据本身就不对**：它碰巧拦住 TP=1（6144 > 4096），
漏掉 TP=2（3072 < 4096），而且会错误放行 H=8192 下的 3072。

### 5.2 为什么 3072 会漏过去：kernel 自己的 SBUF 校验太乐观

`fused_qkv_dim = 3584` 时 kernel **明确报错**：

```
[NCC_INKI016] Kernel validation exception: SBUF budget exceeded even after
reducing weight buffers: sbuf_tile_space_non_buffered=229380, available=212984
```

所以 kernel 内部有 SBUF 容量校验，只是**阈值设得偏高**：

| `fused_qkv_dim` | 行为 |
|---|---|
| ≤ 2944 | 正确 |
| **3072** | **通过 SBUF 校验，但静默算错** ← bug |
| ≥ 3584 | SBUF 校验拦下，明确报错 |

按 3584 的数字线性缩放，3072 大约需要 196k，低于 212,984 的可用量，所以校验放行 ——
但生成的代码是错的。**根本问题在 nkilib 的 SBUF 会计在接近上限时过于乐观**，
那部分不在本仓库内，我们改不了。

### 5.3 修复（已实施并验证）

能在插件侧修：把守卫从「比例」改成「绝对值」，让坏区间退回 PyTorch。
`vllm_neuron/functional/attention/qkv.py`：

```python
_MIN_BROKEN_FUSED_QKV_DIM = 3072
...
if fused_qkv_dim >= _MIN_BROKEN_FUSED_QKV_DIM:
    return False
```

**验证**：全模型 TP=2 纯文本，打补丁后输出正确：

```
a Trainium accelerator. The quick brown fox jumps over the lazy dog while the
engineer measures decode latency across batch sizes on a Trainium accelerator...
```

**DP=2 下同样验证通过**：TP=2 × DP=2、16 并发、视频负载，输出正确
（"a young child ... reading a large, thick book"），每请求恰好 256 token。真实性能
286.1 tok/s / 1.12 req/s，对比 TP=4 batch 8 的 284.1 / 1.11 —— **打平**，而 E2E
13485 ms 是 TP=4 的 6905 ms 的 1.95 倍。乱码版本报的 310.9 tok/s（+9.4%）是假的。
**所以「小 TP + 多副本」这条提升吞吐的路，修好之后依然不划算。**

`kernel_call_count` 从 **109 降到 73**，正好少 36 个 = 每层一个 QKV kernel 退回 fallback
（与 §5.4 里 MLP 那 36 个是两套独立的，各自每层一个）。

代价（TP=2 纯文本，batch 8）：

| TP=2 配置 | 输出 | TTFT p50 | TPOT p50 | tok/s |
|---|---|---|---|---|
| 未修（kernel 全开） | ❌ 乱码 | 773 | 43.88 | 171.0 |
| **打补丁**（仅 QKV 退回） | ✅ **正确** | 1470 | 45.96 | **155.1** |
| 全关 kernel（诊断用） | ✅ 正确 | 1895 | 53.62 | 131.4 |
| *对照 TP=4（kernel 全开）* | ✅ 正确 | 577 | 23.01 | **317.0** |

修好的 TP=2 是 155.1 tok/s，仍只有 TP=4 的一半（核数减半，符合预期）。
**所以这个修复不改变「用 TP=4」的建议** —— 它的价值是让 TP=2 在必须使用时**算得对**，
以及给上游一个精确的修复位置。

> 该守卫是保守取值：只知道 2944 正确、3072 错误，边界就取 3072。真正的修法应该是
> nkilib 修正 SBUF 会计，之后这个守卫可以放宽或删除。

### 5.4 一个被排除的岔路：MLP kernel（记录以免他人重走）

`vllm_neuron/functional/attention/qkv.py` 的 `_can_use_qkv_kernel` 里有一段注释，
**明确承认这个 kernel 在某些尺寸比例下算错**：

> *"The kernel produces incorrect results when fused_qkv_dim is large relative to H
> (e.g., vision TP=1: H=1280, fused_qkv_dim=3\*H=3840). Validated configurations have
> fused_qkv_dim <= H. Fall back to PyTorch when this is exceeded until the kernel is
> validated for larger ratios."*

对应的守卫是 `if fused_qkv_dim > H: return False`。本模型 `H = 4096`（hidden_size，不分片），
`fused_qkv_dim = (num_q_heads + 2*num_kv_heads) * head_dim` 按 rank 计：

| TP | q/kv heads per rank | `fused_qkv_dim` | `fused_qkv_dim / H` | 守卫 | 实测输出 |
|---|---|---|---|---|---|
| 4 | 8 / 2 | 1536 | **0.375** | 通过 | ✅ |
| 2 | 16 / 4 | 3072 | **0.75** | 通过 | ❌ |

**两个都通过了 `<= H` 的守卫，但 TP=2 的比例是 TP=4 的两倍，紧贴边界。**
如果该 kernel 实际只在较小比例下被验证过，那么 `fused_qkv_dim <= H` 这个守卫**太松**，
TP=2 落进了未验证（且错误）的区间。这与注释里已知的失效模式（比例过大）方向一致。

**尚未确认**是不是 QKV kernel —— `VLLM_NEURON_DISABLE_NKI_KERNELS` 是全局开关，
无法只关某一个。要确认需要逐 kernel 开关，或用
`examples/vllm_neuron/accuracy/run_tensor_capture_qwen3_vl.py` 抓 TP=2 下 QKV 输出与
PyTorch 参考对比。



先前注意到 TP=2 的图比 TP=4 少 **36 个** NKI kernel 调用（145 → 109，`rewrite_count`
两者都是 189），而 36 正好等于文本层数，并追到了 `vllm_neuron/functional/mlp.py:196`
的 CTE PSUM 约束：`can_shard_on_i` 因 `H=4096 < _MIN_H_FOR_I_SHARDING=7168` 恒为 False，
故 `effective_i = I`，约束化为 `I <= 8*512 = 4096`：

| TP | 每 rank `I` = 12288/TP | `ceil(I/512)` | ≤ 8 |
|---|---|---|---|
| 4 | 3072 | 6 | ✅ 用 kernel |
| 2 | 6144 | 12 | ❌ 退回 PyTorch |

算术精确对上 36 这个差值。**但这不是正确性 bug 的原因** —— 上表「TP=4 关 kernel」一行
证明 PyTorch fallback 路径是对的。这个约束只造成 TP=2 下 prefill MLP 的**性能**损失。

### 5.5 全局关闭 kernel 的开关

`VLLM_NEURON_DISABLE_NKI_KERNELS=1` 能让 TP=2 输出正确，代价是性能：
TPOT 43.88 → 53.62 ms（+22%）、TTFT 773 → 1895 ms（+145%）、吞吐 171.0 → 131.4 tok/s（−23%）。

**但对本机没有实用价值**：修正后的 TP=2（131.4 tok/s）远低于 TP=4 开 kernel 的
317.0 tok/s。结论仍然是**用 TP=4**。这个开关的价值在于**诊断**，以及在只有
TP=2 可选的机器上作为临时正确性兜底。

## 6. 未做的定位工作

已定位到 kernel 层（§5），但**没有确认是哪一个 kernel**。`VLLM_NEURON_DISABLE_NKI_KERNELS`
是全局开关，无法逐个关闭。首要嫌疑是 QKV kernel 的尺寸比例（§5.1）。
仓库里有现成的工具可以继续往下查（都没跑）：

| 工具 | 用途 |
|---|---|
| `examples/vllm_neuron/accuracy/run_logit_validation_offline.py` | 与 golden logits 逐位置比对，最快确认从第几个 token 开始偏 |
| `examples/vllm_neuron/accuracy/compare_hf_vs_vllm_neuron.py` | 与 HF 参考实现对比 |
| `examples/vllm_neuron/accuracy/run_tensor_capture_qwen3_vl.py` | 逐层抓中间 tensor，二分定位发散的层 |
| `docs/model-dev/accuracy-debugging-guide.md` | 官方精度调试流程 |

建议的下一步：用 `run_tensor_capture_qwen3_vl.py` 在 **TP=2 开 kernel** 与
**TP=2 关 kernel**（后者已知正确，是天然的 golden 参考）下各抓一遍逐层输出，
二分找第一个发散的算子。这比与 TP=4 对比更干净 —— 两次运行的分片方式完全相同，
唯一变量就是 kernel，所以第一个发散点直接就是出错的 kernel。
若发散点在 QKV 投影，即可确认 §5.1 的假设，修法是把
`_can_use_qkv_kernel` 里 `fused_qkv_dim > H` 的守卫收紧到实际验证过的比例。

## 7. 影响

- **TP=4 是本机唯一可用配置**：TP=1 编不出来（[`TP1_COMPILE_FAILURE.md`](TP1_COMPILE_FAILURE.md)），
  TP=2 静默算错，TP=3 不能整除 head 数。
- **否掉了所有「小 TP + 多副本」提升吞吐的方案**（TP=1×4、TP=2×2）。
  吞吐天花板就是 TP=4 batch 8 的 284 tok/s。
- 与模型 recipe 只记录 TP=16（trn2.48xlarge）一致 —— **其他 TP 值上游似乎都未验证**。
  也就是说 TP=8、TP=2 这类值在 trn2.48xlarge 上同样值得先做正确性抽查。
- 生产不受影响（本机本来就只能用 TP=4），但**任何人在其他实例上调 TP 值都可能中招**。

## 8. 保留的现场

| 内容 | 路径 |
|---|---|
| C：TP=2 视频 DP=1 | `/tmp/tp2dp1_bs8.log`、`/mnt/nvme/bench_results/tp2dp1_bs8.json` |
| D：TP=2 视频 DP=2 | `/tmp/tp2dp2_bs16.log`、`/mnt/nvme/bench_results/tp2dp2_bs16.json` |
| E：TP=2 纯文本 DP=1 | `/tmp/tp2_textonly_bs8.log`、`/mnt/nvme/bench_results/tp2_textonly_bs8.json` |
| A：TP=4 视频（对照） | `/tmp/exp1_bs8.log`、`/mnt/nvme/bench_results/exp1_mml2048_bs8.json` |
| B：TP=4 纯文本（对照） | `/tmp/textonly_bs8.log`、`/mnt/nvme/bench_results/textonly_bs8.json` |

> JSON 里不含生成文本（只有时序指标），文本样本在 `.log` 的
> `--- sample output ---` 段。

---

## Summary for upstream (English)

**Title:** Qwen3-VL-8B silently produces garbage output at `tensor_parallel_size=2`

`Qwen/Qwen3-VL-8B-Instruct` compiles, runs, and reports entirely plausible
performance at `tensor_parallel_size=2`, but the generated text is garbage. No
error, no warning. The same configuration at `tensor_parallel_size=4` is correct.
Reproduced 3/3 with greedy sampling (`temperature=0`, on-device `all_greedy`).

```
TP=4, video:      "In this video, a young child is sitting on a bed, engrossed in reading a book..."
TP=4, text-only:  "a Trainium accelerator. The quick brown fox jumps over the lazy dog while..."
TP=2, video:      "\键0 numpy bl,true快手坐肚子肚子,True游ilit真真1111l把 ... s s s s s ... ththhthth"
TP=2, video, DP=2:"\键0 numpy c自己,sende  <table \ \ \ \ \ \ \, \, ..."
TP=2, text-only:  "\ \\b\b\b\l\l\l\l\l\l\l\l\l\l\l\l\l\l\l\l ..."
```

Localized with two controls:

- **text-only reproduces it** → the vision tower is not involved; the fault is in
  the text backbone's TP sharding. (At TP=2 the vision config resolves to
  `tp=1, dp=2`, so vision weights are not sharded at all.)
- **`data_parallel_size=1` reproduces it** → request-level DP is not involved.

Why it is easy to miss:

1. No warning of any kind; the only mention of TP in the log is the config echo.
2. Every dimension divides cleanly at TP=2 — Q 32/2=16, KV 8/2=4,
   MLP 12288/2=6144 — so no shape assertion fires.
3. The perf numbers look physically right: TP=2 TPOT is 44–45 ms vs TP=4's
   23.96 ms, almost exactly the 1.85x expected from halving the cores, and
   TP=2 × DP=2 at 16 concurrency reports 310.9 tok/s, *above* the TP=4 baseline.

- **Environment:** trn2.3xlarge (4 logical NeuronCores, LNC=2), neuronx-cc
  2.26.6360.0+6f180f47, vllm-neuron 0.21.0.1.0.0, vLLM 0.21.0, torch 2.11.0, bf16
- **Config:** `max_model_len=2048`, `max_num_batched_tokens=2048`,
  `max_num_seqs=8`, `num_seqs_buckets=[8]`, `enable_prefix_caching=False`
**Localized to the kernel layer.** With `VLLM_NEURON_DISABLE_NKI_KERNELS=1`
(verified: `kernel_call_count: 0`), **TP=2 produces correct output**:

| TP | NKI kernels | output | TPOT p50 | tok/s |
|---|---|---|---|---|
| 4 | on | correct | 23.01 | 317.0 |
| 4 | off | correct | 24.33 | 291.5 |
| 2 | on | **garbage** | 43.88 | 171.0 |
| 2 | **off** | **correct** | 53.62 | 131.4 |

That exonerates the TP sharding, the collectives and the weight loaders — none of
those change when kernels are disabled. The fault is an NKI kernel that computes
the wrong thing at TP=2 shapes. The PyTorch fallbacks are all correct.

**Root cause confirmed: the NKI QKV kernel is wrong from fused_qkv_dim=3072 up.**
Isolated with `qkv_kernel_isolation.py` (same directory, ~1 min): the same
`NF.qkv_proj` on CPU takes the PyTorch branch and serves as golden, on device it
takes the kernel. Identical results at T=512 and T=2048, so the threshold is
sequence-length independent.

| fused_qkv_dim | q/kv per rank | H | fused/H | rel err | verdict |
|---|---|---|---|---|---|
| 512 / 768 / 1536 / 2048 / 2560 / 2688 / 2816 / 2944 | — | 4096 | .125–.719 | .0011–.0068 | OK |
| **3072** | 16/4 | 4096 | 0.750 | **1.0607** | **WRONG (TP=2)** |
| 3072 | 20/2 | 4096 | 0.750 | 1.0607 | WRONG — split-independent |
| 3072 | 16/4 | **8192** | **0.375** | 1.0051 | **WRONG — ratio-independent** |
| 1536 | 8/2 | **2048** | **0.750** | 0.0045 | OK — ratio-independent |
| 3584 | 20/4 | 4096 | 0.875 | — | loud compile error, see below |

**The criterion is absolute fused_qkv_dim, not its ratio to H.** The last three
rows settle it: 3072 is wrong at H=8192 (ratio 0.375) and 1536 is fine at H=2048
(ratio 0.75). So the existing `fused_qkv_dim > H` guard uses the wrong criterion —
it happens to catch TP=1 (6144 > 4096) but misses TP=2 (3072 < 4096), and it would
wrongly admit 3072 at H=8192.

**Why 3072 slips through: the kernel's own SBUF check is too optimistic.** At
fused_qkv_dim=3584 the kernel fails loudly:

```
[NCC_INKI016] Kernel validation exception: SBUF budget exceeded even after
reducing weight buffers: sbuf_tile_space_non_buffered=229380, available=212984
```

So there is an SBUF capacity check; its threshold is just set too high. Scaling
229380 by 3072/3584 gives ~196k against 212,984 available, so 3072 passes
validation and then computes garbage. The real fix belongs in nkilib's SBUF
accounting, which is outside this repo.

**Plugin-side fix (implemented and verified).** Replace the ratio guard with an
absolute bound in `vllm_neuron/functional/attention/qkv.py`:

```python
_MIN_BROKEN_FUSED_QKV_DIM = 3072
...
if fused_qkv_dim >= _MIN_BROKEN_FUSED_QKV_DIM:
    return False
```

Verified on the full model at TP=2, text-only: output is correct, and
`kernel_call_count` drops 109 → 73, exactly 36 = num_hidden_layers, i.e. one QKV
kernel per layer now taking the PyTorch path.

| TP=2 config | output | TTFT p50 | TPOT p50 | tok/s |
|---|---|---|---|---|
| unpatched (kernels on) | garbage | 773 | 43.88 | 171.0 |
| **patched** (QKV falls back) | **correct** | 1470 | 45.96 | **155.1** |
| all kernels off (diagnostic) | correct | 1895 | 53.62 | 131.4 |
| *TP=4 reference (kernels on)* | correct | 577 | 23.01 | **317.0** |

Fixed TP=2 still runs at half of TP=4, as expected from halving the cores, so this
does not change the recommendation to use TP=4 — its value is making TP=2 correct
where TP=2 is the only option, and giving upstream an exact location. The bound is
conservative: 2944 is verified good and 3072 verified bad, so the guard sits at
3072.

**Superseded hypothesis (kept so nobody re-walks it): the QKV kernel's size ratio.** `_can_use_qkv_kernel` in
`vllm_neuron/functional/attention/qkv.py` already documents that this kernel
"produces incorrect results when fused_qkv_dim is large relative to H ...
Validated configurations have fused_qkv_dim <= H", and guards with
`fused_qkv_dim > H`. With H=4096 (unsharded hidden):

| TP | q/kv heads per rank | fused_qkv_dim | ratio to H | guard | output |
|---|---|---|---|---|---|
| 4 | 8 / 2 | 1536 | 0.375 | passes | correct |
| 2 | 16 / 4 | 3072 | **0.75** | passes | **garbage** |

Both pass the `<= H` guard, but TP=2 sits at twice the ratio. If the kernel was
only validated at smaller ratios, that guard is too loose. Not confirmed — the
disable switch is global, so a single kernel cannot be isolated with it.

**Ruled out along the way** (recorded so nobody re-walks it): TP=2 traces 36
fewer NKI kernel calls than TP=4 (145 → 109, 36 = num_hidden_layers), which
traces to the CTE PSUM constraint in `vllm_neuron/functional/mlp.py:196` —
`can_shard_on_i` is always False here since H=4096 < 7168, so the constraint is
`I <= 8*512 = 4096`, and per-rank I is 3072 at TP=4 (passes) but 6144 at TP=2
(falls back). The arithmetic matches the count exactly, but the "TP=4, kernels
off" row proves the fallbacks are correct, so this is a *performance* effect
only, not the correctness bug.

**Workaround:** `VLLM_NEURON_DISABLE_NKI_KERNELS=1` makes TP=2 correct at a cost
(TPOT +22%, TTFT +145%, throughput −23%). Not useful in practice here — corrected
TP=2 at 131.4 tok/s is well below TP=4 with kernels at 317.0 tok/s — but it is a
clean golden reference for bisecting, since sharding is identical between the two
runs and the kernel is the only variable.

**Best next step:** `run_tensor_capture_qwen3_vl.py` at TP=2 with kernels on vs
off, and bisect the first diverging op. Same sharding both sides, so the first
divergence is the faulty kernel directly.
- **Note:** the Qwen3-VL model recipe documents only TP=16, so TP values other
  than 16 may be unvalidated in general — worth a correctness spot-check at TP=8
  and TP=2 on trn2.48xlarge as well.
