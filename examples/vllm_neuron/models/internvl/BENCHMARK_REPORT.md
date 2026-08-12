# InternVL3-8B on trn2.3xlarge — 延时测量报告

分支 `model/InternVL3-8B`。测量脚本：`benchmark_latency.py`（同目录）。

结论先说：**视觉编码器是延时的全部瓶颈，而且随 tile 数呈约二次增长**。解码侧
（TPOT）干净、可预测，与图像大小几乎无关。

## 1. 配置

| 项 | 值 |
|---|---|
| 模型 | `OpenGVLab/InternVL3-8B-Instruct`（本地 `/mnt/nvme/models/InternVL3-8B-Instruct`）|
| 硬件 | trn2.3xlarge，1 Neuron device，4 逻辑 core（LNC=2），96GB HBM |
| 并行 | TP=4（文本）；视觉 tp=1 / dp=4，为 `resolve_tp_dp` 默认 |
| 精度 | bf16，on-device greedy sampling |
| 输出 | 固定 256 token（`ignore_eos=True`，否则 TPOT 不可比）|
| `vision_attention_block_size` | 1024 = 1 个 tile |
| 迭代 | 每配置 1 轮 warmup + 3~5 轮测量；同配置轮间抖动 < 0.1% |

**每个请求的图像都是字节唯一的**（改 3 个像素）。否则 `EncoderCacheManager` 按
`mm_hash` 命中缓存、整个视觉编码器被跳过 —— 在 Qwen3-VL 上这个陷阱让 TTFT 少算
了 42%。下面 "cache hit" 那一列是故意打开复用测的，用来把视觉编码器的开销单独
剥出来。

## 2. Batch 扫描（7 tiles/图，1846 prompt token）

| batch | TTFT mean | TTFT p99 | TPOT mean | E2E mean | 整批 wall | 吞吐 |
|---|---|---|---|---|---|---|
| 1 | 1003 ms | 1006 ms | 12.54 ms | 4200 ms | 4201 ms | 0.24 req/s |
| 2 | 1474 | 1953 | 14.54 | 5181 | 5208 | 0.38 |
| 4 | 2387 | 3865 | 23.44 | 8365 | 8479 | 0.47 |
| 8 | 4046 | 7118 | 34.21 | 12770 | 13238 | 0.60 |

吞吐随 batch 提升（0.24 → 0.60 req/s），但 **p99 TTFT 从 1.0s 恶化到 7.1s**。

TTFT 近似线性增长的原因不是解码：视觉 bucket 是按**单张图**尺寸配的，
`_cap_encoder_budget_to_vision_bucket` 于是每个 prefill step 只放行一张图，8 张图
的 prefill 完全串行。这是合理的生产配置（按 batch×tiles 配 bucket 会让编译体积和
显存爆掉），代价就写在这张表里。

## 3. Tile 扫描（batch=1）——视觉开销单独剥出

| tiles | 补齐后 tiles | prompt tok | max_model_len | TTFT 冷 | TTFT 缓存命中 | **视觉编码** | TPOT |
|---|---|---|---|---|---|---|---|
| 1 | 4 | 310 | 768 | 363 ms | 53 ms | **310 ms** | 12.24 ms |
| 5 | 8 | 1334 | 1792 | 973 | 93 | **880 ms** | 12.52 |
| 7 | 8 | 1846 | 2304 | 1003 | 115 | **888 ms** | 12.54 |
| 10 | 12 | 2614 | 3072 | 2521 | 154 | **2366 ms** | 12.25 |
| 13 | 16 | 3382 | 3840 | 4733 | 187 | **4546 ms** | 12.87 |

「补齐后 tiles」是 `ceil(ceil(bucket / vision_attention_block_size) / dp) * dp`，
dp=4，所以实际计算量按 4 的倍数向上取整。

两个直接可用的结论：

**(a) 计算量的单位是补齐后的 tile 数，不是真实 tile 数。** 5 tiles 和 7 tiles 都
补齐到 8，视觉耗时 880 vs 888 ms —— 实测相同。也就是说 tile 数落在 4 的倍数之间
时，多出来的 tile 是免费的；反过来，从 8 涨到 9 个 tile 会跳一整档。

**(b) 视觉耗时 ≈ 补齐 tile 数的二次函数。** 归一化到 4 tiles：

| 补齐 tiles | 4 | 8 | 12 | 16 |
|---|---|---|---|---|
| 实测倍数 | 1.0 | 2.85 | 7.6 | 14.7 |
| 线性预期 | 1 | 2 | 3 | 4 |
| 二次预期 | 1 | 4 | 9 | 16 |

13 tiles 时视觉占 TTFT 的 **96%**（4546 / 4733 ms）。

## 4. 二次增长的排查

FLOPs 应该是线性的：tower 全程保持 `[tiles, s, hidden]`，attention 是
`[tiles, heads, s, s]`（每个 tile 独立，InternViT 本来就该这样），MLP 对 token 逐点
运算。所以这是运行时/编译器行为，不是建模错误。已排除的：

- **主机侧图像处理**：单独测 dynamic tiling + `to_tensor`，1/7/13 tiles 分别
  0.9 / 10.6 / 19.7 ms。完全不是瓶颈。
- **encoder cache buffer 大小**：一开始我把 `encoder_cache_num_blocks` 写成 tile 数
  的函数，buffer 和 tile 数是耦合的。固定 buffer 重测：13 tiles 在 22 blocks 下
  4734 ms、34 blocks 下 4733 ms；7 tiles 在两种 buffer 下都是 ~1005 ms。图内
  `index_put_` scatter 无关。
- **跨 tile attention**：确认 attention 是 per-tile 的，没有把所有 tile 拼成一条序列。

**最可能的原因，也是首选优化项**：`InternVisionAttention.forward` 用的是朴素
attention，而且 softmax 走 fp32：

    attn = torch.matmul(q * self.scale, k.transpose(-2, -1))
    attn = torch.softmax(attn.float(), dim=-1).to(self.dtype)

视觉 tp=1，所以每个 rank 拿全部 16 个 head，那个 fp32 中间张量是
`[tiles, 16, 1025, 1025] × 4B`，**每个 tile 67MB**：16 tiles 时单层瞬时 1.08GB，再加
bf16 副本 538MB。24 层都要走一遍。这种量级的临时张量正是编译器会 tile 得很差、
或者直接 spill 的地方 —— 和实测的二次曲线吻合。

对照 `qwen3_vl/vision_encoder_bf16.py:333`：它的视觉 attention 用
`NF.flash_attention(q, k, v, scale=..., causal_mask=False, tp_q=True, tp_k=True)`，
布局 `[num_blocks * heads, block_size, head_dim]`，根本不物化 `s×s` 矩阵。

InternViT 换过去比 Qwen3-VL 还简单：**没有 RoPE，没有 bounds masking**，就是 tile
内的双向全 attention。唯一的麻烦是 s = 1025（1024 patch + CLS token）不是 128 的
倍数，需要像 qwen3_vl 那样把 attention 序列 pad 到 1152 并 mask 掉尾部。**这一项
没做**，是留下的最大一块收益。

## 5. TPOT

batch=1 时 **12.24 ~ 12.87 ms，跨 1→13 tiles 几乎不变**。解码代价来自 KV 窗口和
batch，与图像大小无关（图像 token 在 prefill 就已经并入 KV）。

batch 的影响是真实的：12.54 → 14.54 → 23.44 → 34.21 ms（batch 1/2/4/8）。

`max_model_len` 按工作负载贴合取值，取 256 的倍数上取整而**不是 2 的幂**。7 tiles
需要约 2100 token，取 2 的幂会得到 4096 —— 解码每个 token 都要在整个窗口上做
attention，Qwen3-VL 上这一项曾造成 3.1x 的 TPOT 退化。2304 这种非 2 的幂值实测可
用，这一点之前并不确定。

## 6. 复现

```sh
V=/opt/aws_neuronx_venv_pytorch_inference_vllm_0_21_0_1_0_0
export PATH=$V/bin:$PATH PYTHONPATH=/mnt/nvme/vllm-neuron

# batch 扫描
for BS in 1 2 4 8; do
  python examples/vllm_neuron/models/internvl/benchmark_latency.py \
    --tiles 7 --batch-size $BS --output /tmp/internvl_latency.json
done

# tile 扫描 + 视觉开销剥离（--reuse-image 让 encoder cache 命中）
for T in 1 5 7 10 13; do
  for F in "" "--reuse-image"; do
    python examples/vllm_neuron/models/internvl/benchmark_latency.py \
      --tiles $T --batch-size 1 $F --output /tmp/internvl_latency.json
  done
done
```

每个配置一个独立进程、一个独立 engine：单个 engine 传多个 `num_seqs_buckets` 会在
第一个请求上挂死（记录在 Qwen3-VL 分支），而在同一进程里反复建/拆 engine 在这个
plugin 上没有验证过。

## 7. 没做的

- **视觉 attention 换 `NF.flash_attention`**（见 §4）—— 首选优化项。
- **batch > 1 与多 tile 组合**：batch 扫描固定在 7 tiles。
- **视觉 bucket 按 batch 放大**，让一个 prefill step 编码多张图，用编译体积和显存
  换 batch TTFT。
- **精度评测**：只确认了输出文本合理（含一个 tile 顺序可检验的用例），没跑基准集。
- **多图 prompt**：`limit_mm_per_prompt={"image": 1}`。
