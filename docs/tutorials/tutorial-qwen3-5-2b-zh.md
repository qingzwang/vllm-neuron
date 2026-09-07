# 教程:在 Trainium 上用 vLLM 跑 Qwen3.5-2B(中文上手指南)

<!-- meta: description: Step-by-step Chinese tutorial for running Qwen3.5-2B
(hybrid gated-DeltaNet + full attention, vision-language) on Trn2 with the vLLM
Neuron plugin: environment pinning, CPU-side numerical checks, text generation at
TP=4, latency, accuracy against HuggingFace, the vision tower, and troubleshooting. -->

<!-- meta: keywords: vLLM, Neuron, Trainium, trn2, Qwen3.5-2B, DeltaNet, linear
attention, hybrid model, MambaSpec, mRoPE, vision language, tensor parallel, 教程, 中文 -->

<!-- meta: date_updated: 2026-09-07 -->

<!-- Content type: procedural-tutorial -->

这篇是**动手复现**用的:把 `Qwen/Qwen3.5-2B` 在这个插件上从环境搭起、跑通文本和图文两条路,
并且知道每个数字该是多少。模型移植过程中的设计决策、踩过的坑和完整测量矩阵见
[`vllm_neuron/model/qwen3_5/HANDOFF.md`](../../vllm_neuron/model/qwen3_5/HANDOFF.md);
这篇只讲怎么做。

所有命令都在 **trn2.3xlarge(4 个逻辑核)** 上实际跑过(2026-09-07),贴出来的输出都是真实输出。

---

## 0. 先建立四个概念

**① 这个模型的 24 层里只有 6 层是注意力。** `layer_types` 是
`[linear, linear, linear, full] × 6`,完整注意力只在第 3、7、11、15、19、23 层。
其余 18 层是门控 DeltaNet——一个循环层,状态大小**和序列长度无关**,所以它没有 paged KV
cache、没有 block table。这一条决定了后面所有观测:插件必须同时管两种 cache
(注意力的 KV 页 + DeltaNet 的 conv/recurrent 状态),而 decode 特别快、prefill 相对慢。

**② 版本必须钉死,而且不是"越新越好"。** 这个分支基于 **vLLM 0.21**,而 DLAMI 里可能是
0.24;编译器要用 **neuronx-cc 2.26.6360.0**,2.27 会在编译时报
`[NCC_ISMP902] Simplifier error`。第 1 节给出实测可用的组合。

**③ 必须让仓库代码生效。** venv 里的 `vllm_neuron` 是通过一张静态模块表做的 editable
安装,**装的时候还不存在的子包(`vllm_neuron.model.qwen3_5`)看不见**。所以每条命令都要带
`PYTHONPATH=<repo>`,不是可选项。

**④ 先在 CPU 上验数值,再上设备。** 一次引擎启动要几分钟,而且失败时只会告诉你"整个模型
不对"。这个移植带了两个纯 CPU 的对照脚本(第 3 节),几秒钟就能把 DeltaNet 和注意力两种层
逐段和 HuggingFace 比一遍。先跑它们。

---

## 1. 环境

### 1.1 机器

| 需要什么 | 本文用的 |
|---|---|
| 实例 | **trn2.3xlarge**(12 vCPU / 124 GB 内存 / 4 个逻辑核) |
| 逻辑核配置 | 默认 `logical-neuroncore-config: 2` → 4 个逻辑核,所以 **TP=4 是上限** |
| 磁盘 | ≥ 40 GB 空余(venv 约 9 GB,checkpoint 4.3 GB,编译缓存若干) |

```bash
/opt/aws/neuron/bin/neuron-ls        # 看 NEURON CORES 一列,就是 TP 的上限
```

### 1.2 建 venv(关键:版本组合)

DLAMI 自带的 vLLM venv 很可能是别的版本(本文这台是 0.24),**不能直接用**——这个分支的
`requirements/core.txt` 写的是 `vllm==0.21.0`。自己建一个:

```bash
python3 -m venv /mnt/nvme/venv-vllm021
V=/mnt/nvme/venv-vllm021
$V/bin/pip install --upgrade pip

# 插件本体:会带来 vllm 0.21.0、torch、transformers
$V/bin/pip install "vllm-neuron==0.21.*" \
    --extra-index-url https://pip.repos.neuron.amazonaws.com

# Neuron 后端和编译器要单独装,而且版本要配对
$V/bin/pip install "libtorch-neuronx-lite==2.11.0.1.0.696" "torch-xla==2.11.0" \
    "neuronx-cc==2.26.6360.0" \
    --extra-index-url https://pip.repos.neuron.amazonaws.com
```

实测装出来、并且**能跑通全文**的组合:

```
vllm 0.21.0 | vllm-neuron 0.21.0.1.0.0 | libtorch-neuronx-lite 2.11.0.1.0.696
torch 2.11.0 | torch-xla 2.11.0 | neuronx-cc 2.26.6360.0 | nki 0.5.0
transformers 5.16.1 | Python 3.12
```

> **为什么钉 neuronx-cc 2.26.6360.0。** 先用 2.27.5334.0 试过,DeltaNet 的图编译不过:
> ```
> [INTERNAL_ERROR] [NCC_ISMP902] Simplifier error: is_subset(): incompatible function arguments
> ...
> RuntimeError: Worker failed with error 'neuronx-cc compilation failed with 70. Check compiler logs.'
> ```
> 换成 2.26.6360.0 之后同一条命令直接通过。`libtorch-neuronx-lite` 也要取
> `...696` 那一支(和 2.26 同代),不要取 `...1284`。

> **transformers 是这条栈的一个优势。** NxDI 那边的参考实现用的 transformers 4.57.6
> 不认识 `qwen3_5` 架构,所以它要**另建一个 venv** 才能拿 HF 当精度参照。这里的 5.16.1
> 原生支持 `qwen3_5`,所以第 5 节的对照可以在同一个 venv 里跑。

### 1.3 拿代码

```bash
git clone https://github.com/qingzwang/vllm-neuron.git
cd vllm-neuron
git checkout model/Qwen3.5-2B
```

**不需要 `pip install -e .`**,但**每条命令都必须带 `PYTHONPATH`**(概念 ③):

```bash
export PYTHONPATH=$PWD                      # 或者仓库的绝对路径
```

### 1.4 每条命令都要的环境变量

```bash
V=/mnt/nvme/venv-vllm021
export PATH=$V/bin:/opt/aws/neuron/bin:$PATH   # 插件用 shutil.which("neuronx-cc") 找编译器
export PYTHONPATH=/path/to/vllm-neuron         # 见概念 ③
export VLLM_CACHE_ROOT=/mnt/nvme/cache/vllm021 # 编译缓存,放在大盘上
export HF_HOME=/mnt/nvme/hf-cache
export NEURON_SKIP_EFA_AFFINITY=1
export NEURON_CC_FLAGS="--temp-dir=/tmp/neuroncc_tmp"
mkdir -p /mnt/nvme/cache/vllm021 /tmp/neuroncc_tmp
```

`PATH` 里少了 venv 的 `bin`,报的是"找不到 neuronx-cc";`PYTHONPATH` 少了仓库,报的是
`ModuleNotFoundError: vllm_neuron.model.qwen3_5`。

### 1.5 下载 checkpoint

```bash
python -c "from huggingface_hub import snapshot_download; \
  snapshot_download('Qwen/Qwen3.5-2B', local_dir='/mnt/nvme/models/Qwen3.5-2B')"
```

13 个文件、**4.3 GB**、单个 safetensors 分片。

---

## 2. 这个模型和插件里其他模型有什么不一样

不看这一节也能跑通,但出问题时会看不懂。三句话:

1. **两种 cache 共存。** 6 个注意力层要 KV 页,18 个 DeltaNet 层要固定大小的
   `conv` + `recurrent` 状态。这个移植接的是 vLLM 自己的 `MambaSpec`(而不是自己开一块
   静态 buffer),所以 prefix caching / 抢占的语义不会被悄悄破坏。实测每个序列每个 rank
   **4.88 MB**(18 层 × 0.271 MB),TP=4 下总量很小。
2. **归一化有两种。** `Qwen3_5RMSNorm` 乘的是 `1 + weight`,checkpoint 里的 norm 张量是
   围绕 0 分布的;写成常见的 `weight * x` 会把激活乘成接近 0。但 DeltaNet 的输出 norm
   是 `Qwen3_5RMSNormGated`,用的是普通 `weight`。
3. **注意力带门控、RoPE 只转一部分。** `q_proj` 每头输出 `[query | gate]`,输出要乘
   `sigmoid(gate)`;`head_dim=256` 里只有前 64 维参与旋转(`partial_rotary_factor=0.25`),
   mRoPE 分段 `[11, 11, 10]` 加起来是 `rotary_dim/2`。

---

## 3. 先跑 CPU 校验(不占设备,几秒钟)

这两个脚本把每一层的每一段和 `transformers` 的实现逐个比:

```bash
python examples/vllm_neuron/models/qwen3_5/check_deltanet_vs_hf.py \
    --model /mnt/nvme/models/Qwen3.5-2B
python examples/vllm_neuron/models/qwen3_5/check_attention_vs_hf.py \
    --model /mnt/nvme/models/Qwen3.5-2B
```

实测输出(节选):

```
1. kernels vs transformers reference
  PASS  chunk output: max|d|=4.470e-08 rel=6.305e-07
  PASS  chunk final state: max|d|=7.749e-07 rel=9.088e-07
  PASS  step output: max|d|=0.000e+00 rel=0.000e+00
2. module prefill vs HF module
  PASS  prefill, exact length: rel=1.133e-07
  PASS  prefill, padded to bucket: rel=1.133e-07
3. decode replay vs prefill
  PASS  decode outputs / conv state carry / recurrent state carry
4. TP=4 sharding vs TP=1
  PASS  summed rank outputs / concatenated rank states
ALL CHECKS PASSED
```

```
1. RMSNorm (zero-centred weight)      PASS  (max|d|=0)
2. partial interleaved mRoPE          PASS  cos/sin 位级相同
3. attention prefill vs HF module     PASS  (max|d|=0)
4. MLP                                PASS  (max|d|=0)
5. attention decode replay vs prefill PASS  rel=7.629e-07
6. TP=4 sharding vs TP=1              PASS  rel=4.962e-07
ALL CHECKS PASSED
```

值得注意的地方:

- **`padded to bucket` 和 `exact length` 完全相同**。prefill 会把 prompt pad 到桶宽,
  而 DeltaNet 是循环的——pad 位如果不处理,状态就被多推了几步。这里的做法是从
  `positions` 反推哪些是真 token(`positions[i] - positions[0] == i`),把 pad 位的
  `g`/`beta` 归零,让它们成为精确的 no-op。这一行 PASS 就是这件事对了的证据。
- **`decode replay vs prefill`** 验的是"逐 token 解码"和"一次 prefill"给出同样的结果,
  包括两种状态的传递。DeltaNet 的状态传递错了,输出仍然流畅但内容会漂。
- **`TP=4 vs TP=1`** 走的是真实的权重加载器,所以切分错(比如 conv 状态转置了)会在这里
  就暴露。

改了这两层的任何代码,先跑这两个脚本——几秒钟,而一次引擎启动是几分钟。

---

## 4. 文本推理(TP=4)

```bash
python examples/vllm_neuron/models/qwen3_5/run.py \
    --model /mnt/nvme/models/Qwen3.5-2B --tensor-parallel-size 4
```

首次要编译(本文这台机器全程约 6 分钟,之后命中 `VLLM_CACHE_ROOT` 缓存)。实测输出:

```
Generated: ' Paris.\nA. True\nB. True\nC.\n\n\n...'
Generated: ' 6 7 8 9 10 11 12 13 14 15 16 17'
Generated: '\n    if n <= 0:\n        return []\n    elif n == 1:\n        return [0]\n    elif n == '
Generated: ' little boy named Tom. Tom loved to play with his toys, especially those that could
            make him feel happy. One day, he found a special box that could'
```

四条都是该有的样子:首都答对、计数从 6 接着数、fibonacci 的 base case 链完整、故事连贯。
**这一步的意义不只是"能出字"**:DeltaNet 的状态如果传错,输出仍然是流畅的英文,只是内容
和 prompt 无关——所以要看第 2 条(计数必须接着上文)和第 3 条(代码必须是有效的分支链)。

---

## 5. 和 HuggingFace 对精度

同一个 venv 里就能做(第 1.2 节的说明)。分两侧跑,脚本自己合并结果:

```bash
B=examples/vllm_neuron/models/qwen3_5
python $B/check_generation_vs_hf.py --side neuron --model /mnt/nvme/models/Qwen3.5-2B --tokens 32
python $B/check_generation_vs_hf.py --side hf     --model /mnt/nvme/models/Qwen3.5-2B --tokens 32
```

实测(设备 bf16 对 CPU float32,贪心 32 个 token):

```
  prefix 9    9/32 tokens   'The capital of France is'
  EXACT      32/32 tokens   'I am gonna keep counting forever, 1 2 3 4 5'
  EXACT      32/32 tokens   'def fibonacci(n):'
  prefix 6    6/32 tokens   'Once upon a time, there was a'
  EXACT      32/32 tokens   'The three primary colours are'

total 111/160 tokens (69.4%), 3/5 prompts exact
```

**怎么读这个数**:

- **没有任何 prompt 在第一个 token 就分叉**——脚本对这种情况会明确失败,因为那是 bug 的
  形状,不是数值误差的形状。
- 两处分歧都是"前缀一致,然后在一个近似平局上抛硬币":`' Paris.\nA. True\nB. '` 之后
  是 `True` 还是 `False`;`' little boy named Tom. Tom '` 之后是 `loved to play` 还是
  `was very curious`。bf16 对独立的 float32 实现就是这个样子,贪心解码把差异放大。
- NxDI 那边同模型公布的是 53/80 = 66%、3/5 完全一致,量级一致。**这个指标本身有几个百分点
  的抖动,别当回归门槛用**;要严格就比 logits,不要比贪心 token 序列。

---

## 6. 延时(TP=4)

```bash
python examples/vllm_neuron/models/qwen3_5/benchmark_latency.py \
    --model /mnt/nvme/models/Qwen3.5-2B --tensor-parallel-size 4 \
    --max-num-seqs 1 --input-tokens 896 --output-tokens 128 --iterations 3
```

896 个输入 token + 128 个输出 token(合起来正好是 `max_model_len` 1024),丢弃一轮预热后
取 3 轮中位数,用 `AsyncLLM` 流式所以 TTFT 是真正的首 token 时间:

```
results over 3 rounds x 1 requests
  TTFT     median   110.67 ms    (min 110.13  max 111.19)
  TPOT     median     3.72 ms    (min  3.72   max  3.73)
  E2E      median   582.71 ms
  per-stream decode  268.65 tok/s
```

和 NxDI 参考实现的对比。它公布的 TP=4 数字是 TTFT 42.2 ms / TPOT 4.75 ms / 210 tok/s;
**这台机器上我把它也跑了一遍**(`qwen3.5-2b-hybrid-deltanet` 分支,`run_benchmark.py`,
TP=4、seq_len 1024),得到 TTFT 42.9 ms / TPOT 4.70–4.78 ms / 209–213 tok/s——它的公布值
在这台 3xlarge 上复现得很好,所以下面这张表是同一台机器上的两个实现:

| | 这里(vllm-neuron) | NxDI 参考(实测) | |
|---|---|---|---|
| TTFT | 110.7 ms | **42.9 ms** | 参考快 **2.6×** |
| TPOT | **3.72 ms** | 4.70–4.78 ms | 这里快 **1.27×** |
| 单流解码 | **268.7 tok/s** | 209–213 tok/s | 这里快 1.27× |

**这个分裂是有原因的,而且不是猜的**:

- **decode 赢**在 DeltaNet 的一步循环很小、是内存受限的,torch 实现足够;
- **prefill 输**在两处:6 个注意力层的 `head_dim=256` 超过了插件 flash kernel 的
  `MAX_HEAD_DIM=128`(`functional/attention/attention_cte.py`),所以它们退回 torch,
  prefill 要物化 `[heads, 1024, 1024]` 的分数矩阵;而分块 delta rule 也是纯 torch,
  参考实现那边是 NKI kernel。
- 更重要的是:HANDOFF 里把这个差距拆开过,**参考实现的 text prefill 拟合是
  `9.9 ms + 39.9 µs/token`,这里是 `69 ms + 39.0 µs/token`——每 token 的斜率只差 2%,
  差的全是固定开销**。所以要追这 68 ms,方向不是模型代码。

---

## 7. 图文(VL)

### 7.1 先跑通

```bash
python examples/vllm_neuron/models/qwen3_5/run_vl.py \
    --model /mnt/nvme/models/Qwen3.5-2B \
    --image-size 224 --vision-bucket 256 --max-tokens 64
```

实测输出:

```
vision: 1 image(s), block_size=256, bucket=256

Question:  'Describe this image in detail.'
Generated: 'This is a vibrant, vertically oriented photograph that captures a striking
            contrast between natural beauty and urban architecture.

            **Foreground:**
            The image is dominated by branches of cherry blossom trees (sakura) in
            full bloom. The ...'
```

这是 vLLM 自带的 `cherry_blossom` 测试图,224×224 经过处理器变成 16×16 的 patch 网格
(256 个原始 patch → 64 个合并后的视觉 token)。描述里出现了樱花**和**背后的建筑,说明视觉
塔真的在看图,而不是从提示词里编。

**视觉这一半是复用的,不是重写的。** HF 的 `Qwen3_5VisionModel` 就是
`Qwen3VLVisionModel` 去掉 deepstack merger(`deepstack_visual_indexes` 是空的),
checkpoint 里 `model.visual.*` 的张量名也完全一样,所以 `vl.py` 直接复用了这个插件
Qwen3-VL 的 ViT、权重加载、`embed_multimodal` 和打包工具。文本解码器才是 Qwen3.5 独有的那部分。

**文本-only 和 VL 是两个不同的类**,由工厂按"引擎有没有配图像/视频"来选(
`limit_mm_per_prompt` 全为 0 时平台跳过视觉桶解析,runner 就不会给 `VisionNeuronConfig`)。
所以纯文本启动**不会**付视觉塔的权重和编译时间。

### 7.2 两个坑,都会让你量出一个假的好数字

**坑一:vLLM 会缓存视觉编码器的输出。** 缓存键是图片的哈希,所以如果你每轮发**同一张
字节相同**的图,预热之后每一轮都从缓存拿,视觉塔根本没跑。HANDOFF 记录过这个错误的量级:
1024×1024 下"每次换图" 325.23 ms vs "重复同一张图" 164.21 ms——**低估了一半**。
`benchmark_latency.py` 现在默认每次请求都改动图片一个 4×4 的角(改哈希、不改 token 数),
`--reuse-image` 才是量缓存。

**坑二:视觉塔按整个填充块收费,而不是按图片。** `vision_attention_block_size` 比实际
需要大的话,代价几乎按整块算——HANDOFF 实测 512×512 塞进 4096 的块,视觉开销是贴合块
(1024)的 **4.08×**。所以**块要按你真的会发的图片尺寸来定**,非 2 的幂也没问题。

### 7.3 视觉编码器默认是**不分片**的

这是这个移植里"改一个配置就有的最大提速":`VisionNeuronConfig.resolve_tp_dp` 的规则是
"两个都留默认(1)→ `tp_size=1, dp_size=world_size`"。也就是说默认下视觉塔在每个 rank 上
**各存一份完整的 16 头**,而视觉的数据并行是**按 item 切**的——单图请求下 4 个 rank 有 3 个
在闲着。

把它打开就是最便宜的一次提速。三个配置都在 `max_model_len 2048`、batch 1、128 个输出
token、每次请求换图、丢一轮预热取 3 轮中位数下实测:

```bash
# 默认(tp=1 / dp=4,视觉塔不分片)
python examples/vllm_neuron/models/qwen3_5/benchmark_latency.py \
    --model /mnt/nvme/models/Qwen3.5-2B --max-model-len 2048 \
    --input-tokens 896 --output-tokens 128 --iterations 3 \
    --vision-bucket 4096 --image-size 1024

# 视觉塔切到 4 个核
... 同上 --vision-tp 4

# 同桶宽的纯文本基线(不给 --vision-bucket,就不建视觉塔)
python examples/vllm_neuron/models/qwen3_5/benchmark_latency.py \
    --model /mnt/nvme/models/Qwen3.5-2B --max-model-len 2048 \
    --input-tokens 896 --output-tokens 128 --iterations 3
```

| 配置 | TTFT | 视觉那一半 | TPOT |
|---|---|---|---|
| 纯文本,不建视觉塔 | 149.77 ms | *基线* | 3.99 ms |
| 1024×1024,默认(tp=1 / dp=4) | 310.09 ms | +160.3 ms | 3.99 ms |
| 1024×1024,**`--vision-tp 4`** | **226.49 ms** | **+76.7 ms** | 4.00 ms |

- 视觉那一半快了 **2.09×**,端到端 TTFT 降了 **27%**(−83.6 ms)。不到 4× 是因为分片给
  24 层每层加了两次 all-reduce。
- **TPOT 三行都是 3.99–4.00 ms**,一点没动——decode 不会再跑视觉塔。这也是判断"你到底
  有没有在量视觉塔"的一个旁证。
- 输出内容不变(措辞会变,因为 bf16 的求和顺序变了)。

（这一组和 HANDOFF 里之前记录的 325.23 → 230.40 ms、视觉开销 176.3 → 81.4 ms 是同一结论,
两次独立测量都落在 −27%~−29% / 2.1~2.2× 上。)


规则:**单图低延时用 `tp_size=4`,多图高吞吐用默认的 `dp`**。`run_vl.py` 和
`benchmark_latency.py` 都有 `--vision-tp`。

---

## 8. 常见问题

| 现象 | 原因 | 处理 |
|---|---|---|
| `ModuleNotFoundError: No module named 'vllm_neuron.model.qwen3_5'` | venv 的 editable 安装走静态模块表,看不到新子包 | 每条命令都带 `PYTHONPATH=<repo>`(概念 ③) |
| `neuronx-cc compilation failed with 70` + `[NCC_ISMP902] Simplifier error: is_subset()` | neuronx-cc 2.27 | 钉 `neuronx-cc==2.26.6360.0`,`libtorch-neuronx-lite` 取 `...696` 那一支(1.2 节) |
| 报找不到 `neuronx-cc` 可执行文件 | 插件用 `shutil.which` 找编译器 | `PATH` 里加 venv 的 `bin` |
| 引擎起不来,报 vLLM 内部 API 不存在 | 装的是 vllm 0.24 而这个分支要 0.21 | 用 1.2 节的 venv;别用 DLAMI 里 0.24 的那个 |
| 首次跑很慢 | 编译。`VLLM_CACHE_ROOT` 命中后就快了 | 把它指到大盘,并且**不要**在换 TP / max_model_len / VL 开关后期待复用 |
| 想用 TP=8 | 逻辑核只有 4 个 | `neuron-ls` 数核;这台机器上限就是 4 |
| VL 输出正确但 TTFT 好得离谱 | vLLM 的多模态缓存按图片哈希缓存了**编码器输出**,重复发同一张图就不再跑视觉塔 | 每次请求换一张图(benchmark 脚本默认已经这么做);`--reuse-image` 才是测缓存 |
| 大图 TTFT 远超预期 | `vision_attention_block_size` 比图片实际需要的大,视觉塔按整个填充块收费 | 把 block 调到贴合实际图片尺寸(第 7 节) |

---

## 9. 继续读

- [`vllm_neuron/model/qwen3_5/HANDOFF.md`](../../vllm_neuron/model/qwen3_5/HANDOFF.md)
  —— 移植过程的完整记录:两个只在设备上出现的 bug(`Tensor.split` 静默编译错误、
  `--modular-flow-mac-threshold=10` 破坏 codegen)、UT 变换求逆为什么不能用 Neumann 级数、
  NKI kernel 移植为什么反而更慢、以及和 NxDI 参考实现的逐项对比
- `vllm_neuron/model/qwen3_5/model.py` —— 文本解码器(DeltaNet 层在 `deltanet.py`)
- `vllm_neuron/model/qwen3_5/vl.py` —— 复用 Qwen3-VL 视觉塔的那一层胶水
- `examples/vllm_neuron/models/qwen3_5/probe_device_ops.py` /
  `probe_device_model.py` —— 在设备上按 op / 按模块二分定位问题,秒级
