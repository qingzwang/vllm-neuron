# InternVL3-8B on trn2.3xlarge — 延时测量与优化报告

分支 `model/InternVL3-8B`。脚本：`benchmark_latency.py`（延时）、`compare_vs_hf.py`
（与 HF 参照对照）。

两条结论：

1. **输出是对的** —— 与 HF 参照实现在真实照片上逐 token 对照，事实内容完全一致。
2. **视觉 attention 换成 `NF.flash_attention` 后，13 tiles 的 TTFT 从 4733ms 降到
   562ms（8.4×）**，视觉编码本身 12.1×，而且原来的二次增长变成线性。

## 1. 配置

| 项 | 值 |
|---|---|
| 模型 | `OpenGVLab/InternVL3-8B-Instruct`（本地 `/mnt/nvme/models/InternVL3-8B-Instruct`）|
| 硬件 | trn2.3xlarge，1 Neuron device，4 逻辑 core（LNC=2），96GB HBM |
| 并行 | TP=4（文本）；视觉 tp=1 / dp=4（默认，且实测最优，见 §5）|
| 精度 | bf16，on-device greedy sampling |
| 输出 | 固定 256 token（`ignore_eos=True`，否则 TPOT 不可比）|
| `vision_attention_block_size` | 1024 = 1 个 tile |
| 迭代 | 每配置 1 轮 warmup + 3~5 轮测量；同配置轮间抖动 < 0.5% |

**每个请求的图像都是字节唯一的**（改 3 个像素）。否则 `EncoderCacheManager` 按
`mm_hash` 命中缓存、整个视觉编码器被跳过 —— 在 Qwen3-VL 上这个陷阱让 TTFT 少算了
42%。下表「缓存命中」列是故意打开复用测的，用来把视觉编码器开销单独剥出来：
**视觉编码 = 冷 TTFT − 缓存命中 TTFT**。缓存命中时视觉图整个不跑，所以这一列只含文本
prefill；优化前后都实测过，确认它不受视觉侧改动影响（13 tiles：187.2 → 187.6 ms），
减法才成立。

## 2. 当前性能（flash attention 之后）

### Batch 扫描（7 tiles/图，1846 prompt token）

| batch | TTFT mean | TTFT p99 | TPOT mean | E2E mean | 整批 wall | 吞吐 |
|---|---|---|---|---|---|---|
| 1 | 284 ms | 284 ms | 12.53 ms | 3480 ms | 3481 ms | 0.29 req/s |
| 2 | 403 | 516 | 13.14 | 3753 | 3790 | 0.53 |
| 4 | 624 | 949 | 19.09 | 5492 | 5621 | 0.71 |
| 8 | 1024 | 1458 | 23.68 | 7061 | 7506 | **1.07** |

### Tile 扫描（batch=1）

| tiles | 补齐后 | 每 rank | prompt tok | max_model_len | TTFT | 缓存命中 | **视觉编码** | TPOT |
|---|---|---|---|---|---|---|---|---|
| 1 | 4 | 1 | 310 | 768 | 137 ms | 53 ms | **84 ms** | 12.24 ms |
| 5 | 8 | 2 | 1334 | 1792 | 255 | 93 | **162** | 12.52 |
| 7 | 8 | 2 | 1846 | 2304 | 284 | 119 | **165** | 12.53 |
| 10 | 12 | 3 | 2614 | 3072 | 411 | 154 | **257** | 12.25 |
| 13 | 16 | 4 | 3382 | 3840 | 562 | 187 | **375** | 12.87 |

「补齐后」是 `ceil(ceil(bucket / vision_attention_block_size) / dp) * dp`，dp=4；
「每 rank」= 补齐后 / dp，这才是决定视觉耗时的量。

**视觉耗时现在是线性的**：每 rank 1/2/3/4 个 tile → 84/165/257/375 ms，
归一化 1 : 1.97 : 3.07 : 4.48。

**计算量的单位仍是补齐后的 tile 数**：5 tiles 和 7 tiles 都补齐到 8，视觉耗时
162 vs 169 ms。tile 数落在 4 的倍数之间时多出来的 tile 基本免费，跨过一个倍数则跳
一整档。

## 2.5 Video

Video 在这个实现里**不需要单独的代码路径**：vLLM 的 video processor 里有
`assert len(pil_frame) == 1`，每帧固定 1 个 tile / 256 embed token，没有 temporal
merging、没有时间戳文本（不像 Qwen3-VL）。所以一帧就是一张独立的单 tile 图，
`embed_multimodal` 只需把 `pixel_values_flat_video` / `video_num_patches` 折到 image
路径上。帧序由 `video_smoke_test.py` 用红→绿→蓝三段 clip 验证（帧序错了照样能产出流畅
文字，必须专门验）。

### 16 帧 448×448（1 秒 @16fps），256 输出 token

| batch | TTFT mean | TTFT p99 | TPOT | E2E | 整批 wall | 吞吐 |
|---|---|---|---|---|---|---|
| 1 | 571 ms | 578 ms | 12.68 ms | 3804 ms | 3806 ms | 0.26 req/s |
| 2 | 806 | 1037 | 13.79 | 4322 | 4360 | 0.46 |
| 4 | 1310 | 2008 | 20.60 | 6563 | 6684 | 0.60 |
| 8 | 2228 | 3745 | 27.75 | 9305 | 9781 | **0.82** |

视觉编码剥离（帧数即 tile 数，dp=4 下 8 和 16 都无需补齐）：

| 帧数 | 每 rank | TTFT 冷 | 缓存命中 | **视觉编码** |
|---|---|---|---|---|
| 8 | 2 | 281 ms | 126 ms | **155 ms** |
| 16 | 4 | 571 | 228 | **343 ms** |

与 image 侧一致：16 帧的 343 ms 对应 13 tiles 图的 375 ms（两者补齐后都是 16 tiles）。

**500 token 文本 padding 几乎不影响**：TTFT 571 → 591 ms，TPOT 12.68 → 12.36。成本全在
视觉 token，文本 token 便宜。

### 达到 1 req/s 的配置

16 帧 + 256 输出在 batch 8 只有 0.82 req/s，**不达标**。两个可行配置：

| 配置 | max_model_len | TTFT | TPOT | E2E | 吞吐 |
|---|---|---|---|---|---|
| 16 帧, batch 8, **128** 输出 | 4608 | 2232 ms | 34.89 ms | 6663 ms | 1.12 req/s |
| **8 帧, batch 8, 256 输出** | 2560 | **1096 ms** | 21.89 ms | 6678 ms | **1.16 req/s** |

**推荐 8 帧（1 秒 clip 降到 8fps）**：吞吐相同，但 TTFT 只有一半（1096 vs 2232 ms），
而且保留完整 256 输出 token。视觉 token 从 4096 砍到 2048，`max_model_len` 跟着从 4608
降到 2560 —— 后者对 TPOT 的影响比前者更大。

### 与 Qwen3-VL-8B 的对比：差距是架构性的

同一台机器、同样 16 帧 448×448、256 输出 token（Qwen3-VL 数据见
`benchmark/Qwen3-VL-8B` 分支）：

| | Qwen3-VL-8B | InternVL3-8B |
|---|---|---|
| 视觉 token / clip | ~1300 | **4096**（16 × 256）|
| prompt token | ~1651 | ~4150+ |
| 视觉 bucket | 8192 raw patch | **16384** |
| max_model_len | 2048 | 4608 |
| TTFT (bs=1) | 294 ms | 571 ms |
| TPOT (bs=1) | 11.35 ms | 12.68 ms |
| batch 8 吞吐 | 1.11 req/s | 0.82 req/s |

根因是 **InternVL3 没有 temporal merging**。Qwen3-VL 的 `temporal_patch_size=2` 把 16 帧
合成 T=8 再做 2×2 空间 merge；InternVL3 每帧独立成 256 token，**同一个 clip 用掉约 2.5 倍
视觉 token**。

拖累的不是视觉编码器 —— flash attention 之后 343 ms 已经很快，比 Qwen3-VL 的视觉侧只慢
一点。真正的代价是这 4096 个视觉 token 要流过文本骨干：prefill 更长，更关键的是
`max_model_len` 被顶到 4608 而解码每个 token 都要在整个窗口上做 attention。所以 batch 8
的 TPOT 反而更高（27.75 vs 23.96 ms），尽管视觉更快。

**结论：做 1 秒实时视频，InternVL3 先天比 Qwen3-VL 贵**，达标要靠减帧数或减输出长度。

## 3. 优化前后对比

| tiles | 每 rank | 视觉编码（原始） | 中间步：bf16 softmax | **flash attention** | 总加速 |
|---|---|---|---|---|---|
| 1 | 1 | 310 ms | — | **84 ms** | 3.7× |
| 7 | 2 | 888 | 834 | **165** | 5.4× |
| 13 | 4 | 4546 | 3582 | **375** | **12.1×** |

**归因：最终数字全部来自 flash attention，bf16 softmax 那一步没有贡献。** 它是个诊断性
的中间实验（单独有 −21%，并且证明了主因不是中间张量的体积），但 flash 改动把整个
`torch.softmax` 代码路径删掉了 —— 现在 softmax 发生在 kernel 内部，fp32/bf16 的选择在
当前代码里已不存在。

而且 flash attention 是**带着额外开销**取得这个结果的：序列从 1025 pad 到 1152 意味着多
算 12.4% 的 attention，还要构造并按 head 复制 bounds。这些都是成本，所以 kernel 本身的
收益比表里的净值更大。

端到端 TTFT：

| tiles | 原始 | 现在 | |
|---|---|---|---|
| 1 | 363 ms | **137 ms** | 2.7× |
| 7 | 1003 | **284** | 3.5× |
| 13 | 4733 | **562** | 8.4× |

Batch（7 tiles）：

| batch | TTFT 原始 → 现在 | TPOT 原始 → 现在 | 吞吐 原始 → 现在 |
|---|---|---|---|
| 1 | 1003 → **284** ms | 12.54 → 12.53 ms | 0.24 → **0.29** req/s |
| 2 | 1474 → **403** | 14.54 → 13.14 | 0.38 → **0.53** |
| 4 | 2387 → **624** | 23.44 → 19.09 | 0.47 → **0.71** |
| 8 | 4046 → **1024** | 34.21 → **23.68** | 0.60 → **1.07** |

13 tiles 时视觉占 TTFT 的比例从 **96% 降到 67%** —— 瓶颈第一次不在视觉侧。

注意 batch>1 时 TPOT 也降了（batch 8：34.21 → 23.68 ms），但**这不是解码变快了**。
TPOT 按 `(e2e − ttft) / (n−1)` 算，batch 8 下 request 0 解码期间会被 request 1~7 的
prefill（含视觉编码）打断，所以 TPOT 里混着 prefill 干扰。视觉快了，干扰就小了。
batch=1 的 TPOT 才是纯解码，改动前后一致（12.5 ms）—— 符合预期，改的只是 prefill
里的视觉图。

## 4. 二次增长的定位与修复

原始实现的视觉耗时对每 rank tile 数近似二次（归一化 1 : 2.85 : 7.6 : 14.7），而 FLOPs
是线性的：tower 全程保持 `[tiles, s, hidden]`，attention 是 `[tiles, heads, s, s]`，
每个 tile 独立（InternViT 本来就该这样），MLP 对 token 逐点运算。

逐一排除：

- **主机侧图像处理**：单独测 dynamic tiling + `to_tensor`，1/7/13 tiles 分别
  0.9 / 10.6 / 19.7 ms。不是瓶颈。
- **encoder cache buffer 大小**：最初把 `encoder_cache_num_blocks` 写成了 tile 数的
  函数，buffer 和 tile 数耦合。固定 buffer 重测：13 tiles 在 22 blocks 下 4734ms、
  34 blocks 下 4733ms；7 tiles 两种 buffer 都是 ~1005ms。图内 `index_put_` 无关。
- **跨 tile attention**：确认 attention 是 per-tile 的，没把所有 tile 拼成一条序列。
- **fp32 softmax 上转**：原来写的是 `torch.softmax(attn.float(), ...)`，而 HF 自己的
  `_naive_attn` 是 `attn.softmax(dim=-1)`，**没有上转** —— 也就是说这个 fp32 比参照
  还精确但并不必要。去掉后 13 tiles 视觉从 4546 → 3582ms（−21%），**但二次曲线还在**
  （每 rank 2→4 tiles 时 834→3582，4.3 倍）。

所以主因不是中间张量的体积，而是**物化 `s×s` 这件事本身**。

### 修复：`NF.flash_attention`

照 `qwen3_vl/vision_encoder_bf16.py:333` 的用法改写 `InternVisionAttention`：

1. q/k/v 折成 `[tiles * heads, s, head_dim]`，`causal_mask=False`（InternViT 是双向）
2. **序列 pad 到 128 的倍数**。kernel 按 `_ATTN_SEQ_ALIGN` 的整块读 bounds，s 不对齐
   会把越界 DMA 编进 NEFF、运行时加载直接拒绝。InternViT 的
   s = 1 + grid_size² = **1025，永远不对齐**，所以 tower 里无条件 pad 到 1152。
3. **bounds 只用来遮 padding**，不承担 packing 职责（这点和 qwen3_vl 不同）：真实行
   `bound_min=0, bound_max=1025`（全序列双向），pad 行 `bound_min=bound_max=0` 谁都不
   attend，而真实行的 `bound_max` 停在 1025 所以没人 attend pad 行。
4. 出 tower 时一并切掉 CLS 和 padding：`hidden_states[:, 1:real_s, :]`。

比 Qwen3-VL 那版简单：InternViT 没有 RoPE，也不需要按 frame 划 block。

## 5. 试过但更差的：视觉 TP

| 视觉并行 | 13 tiles TTFT（flash 之前测的）|
|---|---|
| tp=1 / dp=4（默认）| 4733 ms |
| tp=4 / dp=1 | **20089 ms（慢 4.2×）** |
| tp=2 / dp=2 | **设备执行超时，`FATAL-RT-UNDEFINED-STATE`** |

tp=4 时每 rank 的 `s×s` 张量其实和默认差不多大（219MB vs 269MB），所以慢 4 倍不是
内存 —— 是每层两次 all-reduce，加上分片后矩阵变小、利用率下降。tp=2/dp=2 直接把视觉图
跑挂，设备需要清场。**结论：保持默认 tp=1 / dp=4。**

## 6. 正确性

`compare_vs_hf.py`，checkpoint 自带的真实照片，与 HF 参照实现（float32 CPU）逐 token
对照。参照侧用 float32 是有意的：这一侧是标尺，自身噪声要尽可能小。

| 输入 | HF 参照 | Neuron TP=4 + flash | 一致程度 |
|---|---|---|---|
| image1，13 tiles，详细描述 | red panda / wooden platform / reddish-brown coat / fluffy tail | 同 | 前 140 字符、28 token 完全一致 |
| image1，单词回答 | `Red panda` | `Red panda` | **逐 token 完全一致** |
| image2，10 tiles，详细描述 | giant panda / bamboo / black patches around eyes, ears, limbs | 同 | 前 111 字符、20 token 完全一致 |

两处分叉都是单词级同义替换（"face markings" vs "facial markings"、"black and white
fur" vs "black and white fur pattern"），出现在模型本身不确定的位置；物种、姿态、毛色、
眼周黑斑、在吃竹子等事实内容全部一致。float32 CPU 参照 vs bf16 + NKI kernel，这个程度
的后段分叉是预期的 —— 两个候选 logit 一旦落进 bf16 噪声内，argmax 就可能合理地不同，
而一个 token 不同会改变它之后的全部。**前几个 token 分叉、或者描述错对象**才是 bug。

这同时验证了之前没验过的路径：TP=4 文本分片、13 tiles 的多 tile 视觉路径、encoder
cache 往返。flash attention 改动前后对照结果同级（image2 的分叉点完全相同）。

另外 CPU 侧 `validate_vision_encoder.py` 在 flash attention 之后仍然
MATCH：tower rel 9.7e-06、extract_feature rel 3.5e-06。注意它走的是
`NF.flash_attention` 的 **torch fallback**，所以它验证的是 padding/bounds 的逻辑，
NKI kernel 本身只能靠设备上的 HF 对照来验。

HF 侧要绕一个坑：checkpoint 的 remote code 没调 `post_init()`，新版 transformers 加载
时会因缺 `all_tied_weights_keys` 直接挂；脚本里打了个类级默认值的补丁。

## 7. 复现

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

# 正确性：两侧分开跑，避免争抢
python examples/vllm_neuron/models/internvl/compare_vs_hf.py --side hf     --out /tmp/hf_ref.json
python examples/vllm_neuron/models/internvl/compare_vs_hf.py --side neuron --out /tmp/neuron_out.json
python examples/vllm_neuron/models/internvl/compare_vs_hf.py --side compare
```

每个配置一个独立进程、一个独立 engine：单个 engine 传多个 `num_seqs_buckets` 会在第一个
请求上挂死（记录在 Qwen3-VL 分支），而在同一进程里反复建/拆 engine 在这个 plugin 上没有
验证过。

`max_model_len` 按工作负载贴合取值，取 **256 的倍数上取整而不是 2 的幂**：7 tiles 需要
约 2100 token，取 2 的幂会得到 4096，而解码每个 token 都要在整个窗口上做 attention ——
Qwen3-VL 上这一项曾造成 3.1× 的 TPOT 退化。2304 这种非 2 的幂实测可用。

## 8. 没做的

- **视觉耗时在每 rank 4 tiles 处仍略微超线性**（4.48× vs 4×），没进一步追。
- **batch > 1 与多 tile 组合**：batch 扫描固定在 7 tiles。
- **视觉 bucket 按 batch 放大**，让一个 prefill step 编码多张图。目前 bucket 按单张图
  配，`_cap_encoder_budget_to_vision_bucket` 于是每个 prefill step 只放行一张图，batch
  的 prefill 串行 —— 这是 batch TTFT 随 batch 增长的原因。
- **精度评测**：只做了 HF 对照，没跑基准集。
- **多图 prompt**：`limit_mm_per_prompt` 固定为 1。NxDI 上那个参考负载是
  **15 张图 × 258 token + 500 文本 ≈ 4390 token**，和 16 帧 video（~4150）token 量级
  接近但**不是同一场景** —— 多图每张最多 13 tile 且各自独立走 dynamic tiling，video 每帧
  强制 1 tile。要和 NxDI 的数直接对比，得把 `limit_mm_per_prompt` 放开到 15 再测一轮。
- **video 的 tile 上限**：每帧固定 1 tile 是 vLLM processor 的选择
  (`assert len(pil_frame) == 1`)，没试过让帧走多 tile。
