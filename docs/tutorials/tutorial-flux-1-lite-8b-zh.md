# 教程：在 Trainium 上跑 FLUX.1-lite-8B（中文上手指南）

<!-- meta: description: Step-by-step Chinese tutorial for running FLUX.1-lite-8B
text-to-image generation on Trn2 with the vLLM Neuron plugin, covering
environment setup, NeuronCore planning, tensor parallelism for the diffusion
transformer, measured latency, and troubleshooting. -->

<!-- meta: keywords: vLLM, Neuron, FLUX, FLUX.1-lite, flux.1-lite-8B, 文生图,
diffusion, text-to-image, tensor parallelism, 张量并行, NeuronCore, LNC, Trn2,
Trainium, tutorial, 中文, 教程 -->

<!-- meta: date_updated: 2026-09-03 -->

<!-- Content type: procedural-tutorial -->

这篇是给**没用过 Trainium 的人**写的：从一台干净的 trn2 实例开始，一步步跑通
FLUX.1-lite-8B 出图，再把 transformer 用张量并行切到多个 NeuronCore 上，把每步延时
砍掉一半。所有命令都在 **trn2.3xlarge** 上实际跑过（2026-09-03），贴出来的输出也是
真实输出。

模型本身的特性、精度和限制见
[FLUX.1-lite-8B 模型说明](../model-recipes/flux-1-lite-8b.md)，这里只讲怎么动手。

---

## 0. 先建立四个概念

**① Neuron 是「先编译、再执行」的。** 模型不是直接跑 PyTorch：`torch.compile` 把
计算图交给 `neuronx-cc` 编译成 NEFF（设备可执行文件），再加载到核上。所以形状必须
是静态的——这条 pipeline 把**分辨率**和**prompt 长度预算**在创建时就固定下来，之后
每一步去噪都重放同一个图。改这两个值、改并行度，都要重新编译。首次编译
1024×1024 大约 4 分钟，之后命中本地编译缓存。

**② 一个进程只能驱动一个逻辑核，而且核是独占的。** 编译后端把每个 NEFF 都加载到
本进程自己的核上，所以一个进程用不了两个核；反过来，一个核也只能被一个进程持有。
实测第二个进程连 runtime 都起不来：

```
ERROR NRT:nrt_infodump  Visible cores: 0
RuntimeError: The PyTorch Neuron Runtime could not be initialized.
```

这条约束决定了后面所有的「核预算」计算。

**③ HBM 按物理核对分区，不是按逻辑核。** trn2 的 96 GB HBM 分成四块约 22 GiB 的
分区，每块归一对物理核。`logical-neuroncore-config`（LNC）决定一对是 1 个逻辑核
（LNC=2，默认，共 4 个逻辑核）还是 2 个（LNC=1，共 8 个，单核算力减半）。所以
「换个核」并不等于「换一块显存」。

**④ FLUX 不走 `vllm serve`。** vLLM 0.24 没有文生图的请求路径（它的
`DiffusionConfig` 说的是离散扩散*语言*模型）。FLUX 走的是本插件里的独立 pipeline
`vllm_neuron.model.flux.NeuronFluxPipeline`，它复用这个插件的编译栈和 NKI 注意力
kernel，但不用它的 model runner。

---

## 1. 确认机器和环境

```bash
export PATH=/opt/aws/neuron/bin:$PATH
neuron-ls          # 能列出设备 = 驱动和 runtime 正常
```

DLAMI 自带的插件环境在 `/opt/aws_neuronx_venv_pytorch_inference_vllm_0_24_0_1_1_0`。
它**通常是只读的**（root 所有），所以装额外依赖有两条路：

```bash
V=/opt/aws_neuronx_venv_pytorch_inference_vllm_0_24_0_1_1_0

# A. venv 可写：直接装
$V/bin/pip install -r requirements/flux.txt

# B. venv 只读（DLAMI 默认）：装到一个目录，用 PYTHONPATH 挂进去
$V/bin/pip install --target /mnt/nvme/pyext -r requirements/flux.txt
export PYTHONPATH=/mnt/nvme/pyext:$PYTHONPATH
```

**PATH 一定要包含 venv 的 `bin`**，否则编译时会失败在找不到编译器上——这是个很容易
踩的坑，因为如果之前已经编译过、命中了缓存，它不会报错：

```bash
export PATH=$V/bin:/opt/aws/neuron/bin:$PATH
```

验证过的版本组合：

```
vllm 0.24.0 | vllm-neuron 0.24.0.1.1.0 | libtorch-neuronx-lite 2.11.0.1.0.1284
torch 2.11.0 | neuronx-cc 2.27.5334.0 | diffusers 0.40.0 | Python 3.12
```

---

## 2. 第一次出图

```bash
python examples/vllm_neuron/models/flux/generate.py \
    --model-checkpoint Freepik/flux.1-lite-8B \
    --prompt "A close-up photo of a red panda wearing tiny round glasses, reading a leather-bound book in a cozy library" \
    --steps 28 \
    --output flux_output.png
```

![FLUX.1-lite-8B 在 trn2 上的输出：戴圆眼镜看书的小熊猫](../model-recipes/images/flux-1-lite-8b-sample.png)

*1024×1024、28 步、guidance 3.5、seed 42，就是上面那条命令。为了这个页面缩过。*

首次运行会编译四个组件（transformer、VAE、CLIP、T5），几分钟；之后命中缓存，从启动
到第一张图约 55 秒，大部分时间在从磁盘读权重。

如果你自己写代码，注意**核的固定必须发生在 `import vllm_neuron` 之前**——runtime 是
在那次 import 里起来的，可见核的选择在那一刻就锁定了：

```python
import os

# 必须在 import 之前：给 T5 的 worker 留下 1 号核
os.environ["NEURON_RT_VISIBLE_CORES"] = "0"

from vllm_neuron.model.flux import FluxNeuronConfig, NeuronFluxPipeline

config = FluxNeuronConfig(height=1024, width=1024, max_sequence_length=512)
with NeuronFluxPipeline.from_pretrained("Freepik/flux.1-lite-8B", config) as pipeline:
    pipeline.compile()
    image, timing = pipeline.generate("a red panda reading a book", num_inference_steps=28, seed=42)
image.save("out.png")
print(timing.as_dict())
```

**为什么必须固定核？** 一个什么都不设的进程会在 runtime 起来时把**所有**核都claim
掉，于是 T5 的 worker 没核可用，pipeline 会退回到 CPU 跑 T5 并打印原因——每个请求
多花约 1.5 秒。

---

## 3. 四个网络分别跑在哪

| 组件 | 跑在 | BF16 大小 |
|---|---|---|
| `transformer`（FluxTransformer2DModel） | Neuron，本进程的核 | 15.2 GiB |
| `vae`（AutoencoderKL 解码器） | 同上 | 0.15 GiB |
| `text_encoder`（CLIP-L） | 同上 | 0.22 GiB |
| `text_encoder_2`（T5-XXL） | Neuron，另一个核（子进程） | 8.9 GiB |

T5 必须单独一个核，两个原因各自都成立：一块 HBM 分区只有约 22 GiB，而
15.2 + 8.9 已经超了；而且一个进程驱动不了第二个核。所以它跑在一个用
`NEURON_RT_VISIBLE_CORES` 固定的子进程里，由 pipeline 自己管理生命周期（用 `with`
或者记得 `close()`）。

这么做的收益：prompt 编码从 CPU eager 的 1617 ms 降到 98 ms，16 倍。

---

## 4. 张量并行：把 transformer 切开

一个请求的成本几乎全在 transformer 上（28 步里它跑 28 次，别的组件各跑 1 次），而
它同时也是 15.2 GiB 的大头。把它切到多个核上，两件事一起解决。

### 原理一句话

只切**注意力头**和**FFN 的中间维**；残差流保持全宽、每个 rank 上都一样。所以
norm、调制投影、embedder 和最后的 `proj_out` 都是复制的，注意力 kernel 也不用改
——它从张量形状里推头数，列并行的 `to_q` 给它更少的头就行。

唯一不规整的地方是单流块的 `proj_out`：它吃的是 `cat([注意力输出, MLP 输出])`，两个
生产者都是列并行，所以每个 rank 手上是 `[注意力维/tp + MLP维/tp]`，这**不是**全局
输入的连续切片。实现里把它的权重按两半分别切、两个偏积相加之后只做一次 all-reduce。

### 怎么开

```python
config = FluxNeuronConfig(
    height=1024, width=1024,
    tp_degree=2,              # 把 transformer 切到 2 个核
    tp_core_ids=(0, 1),       # 两个 rank 各自的物理核
    worker_device_index=2,    # T5 的核
)
# 本进程（CLIP + VAE）自己占一个核，import 之前固定：
#   os.environ["NEURON_RT_VISIBLE_CORES"] = "3"
```

`tp_degree` 必须是 2 的幂，而且要能整除 24 个注意力头，所以可选 1、2、4、8。
`tp_degree>1` 时 transformer 搬到 `tp_degree` 个 worker 进程里，本进程只负责调度。

### 核预算（这是这台机器上的硬约束）

每个 rank 一个核，T5 一个核，本进程还要一个核：

| tp_degree | transformer | T5 | 本进程 | 合计 | trn2.3xlarge（4 核）放得下？ |
|---|---|---|---|---|---|
| 1 | 1 | 1 | 与 transformer 共用 | 2 | ✅ |
| 2 | 2 | 1 | 1 | 4 | ✅ 刚好 |
| 4 | 4 | 1 | 1 | 6 | ❌ 要更大的实例 |

**在 trn2.3xlarge 上，`tp_degree=2` 是能全部跑在设备上的最大配置。** `tp_degree=4`
要 6 个核，更大的 trn2 实例都有。

有人会想到用 `NEURON_LOGICAL_NC_CONFIG=1` 把 4 个核变成 8 个——**对这么大的模型不要
这么做**：LNC=1 的每个核是单个物理 NeuronCore-v3 而不是融合的一对，算力减半，实测
1227 ms/step 对 LNC=2 的 791 ms/step。而且这条 pipeline 目前在 LNC=1 下根本不对：用
`--lnc=1` 编出来的 transformer 图从第一步去噪就返回 NaN（512 和 1024 都一样），而同
一次运行里 text encoder 的输出和初始 latent 都是正常的。LNC=1 是给小模型用的——小到
一个融合核会闲着。

如果你确实要在本机上量 `tp_degree=4` 的每步延时，可以把其余组件放到 CPU 上，这样 4
个核全给 rank：

```bash
python examples/vllm_neuron/models/flux/benchmark.py --tp 4 --on-device transformer \
    --sizes 1024 --steps 28 --iterations 2
```

这么跑端到端时间没意义（CPU 上的 VAE 要 7.5 秒），但每步延时和 latent 都是真的
——实测 `tp_degree=1` 用同样的方式跑是 791.3 ms/step，和全设备的 790.9 ms/step 一致，
说明 encoder 放哪儿不影响去噪那一步。

### 实测

1024×1024、28 步、512 token 预算、同一台 trn2.3xlarge（LNC=2，默认）：

| tp_degree | ms/step | 加速比 | 端到端（28 步） |
|---|---|---|---|
| 1 | 791 | 1.00× | 22.24 s |
| 2 | 392 | 2.02× | 11.07 s |
| 4 | 214 | 3.70× | 见上面的说明 |

几乎线性。这是合理的：1024×1024 下一步要做 46 次 4608 token 的注意力加上前馈，每个
rank 的活儿足够多，每个 block 那两次 all-reduce 占不了主导。

启动的代价在另一头：每个 rank 都要先把完整 checkpoint 读进来才留下自己那一片，而且
为了不让主机内存爆掉，它们是拿锁排队读的。所以 `tp_degree=2` 冷启动约 200 秒，而
单进程是约 60 秒。每步的额外开销倒是很小（约 1 ms）：请求不变的东西（prompt 
embedding、pooled、guidance、RoPE 表）只在开头发一次，latent 整个循环留在 worker 里，
每步只传三个标量。

### 正确性

切分是精确的，不是近似，但它确实改变了求和顺序，bf16 下这会体现出来。同一个种子、
同一个 prompt，比最终 latent：

| | 步数 | cos | max\|d\| | latent 标准差 |
|---|---|---|---|---|
| 512×512，TP=1 vs TP=2 | 4 | 0.999815 | 0.39 | 1.18 |
| 1024×1024，TP=1 vs TP=4 | 4 | 0.999962 | 0.96 | 1.51 |
| 1024×1024，TP=1 vs TP=2 | 28 | 0.998331 | 2.65 | 1.29 |

28 步那一行低一些，是因为重结合误差在链上累积，不是因为切得更多。切错的话根本不是
这个量级——会掉到 cos 0.9 以下，图上一眼就能看出来。

![tp_degree=2 的输出：同一只小熊猫](../model-recipes/images/flux-1-lite-8b-tp2.png)

*`tp_degree=2`，其余设置和第 2 节那张完全一样（那张是 `tp_degree=1`）。两张图每个通道
平均差 1.35/255，肉眼看不出区别。*

### 为什么这里 TP=1 能跑，以及它的代价

如果你在 NxD Inference 里跑过 FLUX，会发现那边 TP=1 根本跑不起来，报显存不够。原因是
那边**四个组件都按同一个 TP 度数切**，TP=1 就等于把四个模型全压在一个核上：
transformer 15.20 GiB + T5 8.87 GiB + CLIP/VAE 0.37 GiB = 24.44 GiB，而一个核可用的
HBM 只有约 22 GiB。它能编译、也能加载，然后在要激活空间时死掉
（`NRT_RESOURCE in nrt_tensor_allocate`）。所以那边 TP=2 是下限。

这边 `tp_degree=1` 能跑，**但不是因为它更省显存**。两种排法都占了**两个逻辑核、也就是
两块约 22 GiB 的分区**，24.44 GiB 在任何一块单独的分区里都放不下。区别在于第二个核在
干什么：

- NxD Inference 的 TP=2：两个核各持每个模型的一半，每个阶段两个核都在算。
- 这边的 `tp_degree=1`：一个核整个装下 transformer 并独自跑完所有去噪，另一个核只装
  T5，请求的其余时间都闲着——28 步 1024×1024 的一次请求里它只干 0.098 秒的活，而第一个
  核干了 22.6 秒。

所以放置式拆分并没有把这两个核用好。同样的核数下，同 checkpoint、同分辨率、同步数：

| 核数 | 这条 pipeline | ms/step | NxD Inference | ms/step |
|---|---|---|---|---|
| 2 | `tp_degree=1`，T5 占第二个核 | 791 | TP=2，全部切开 | 406 |
| 4 | `tp_degree=2`，再加 T5 和本进程 | 392 | TP=4，全部切开 | 217 |

放置式拆分换来的是「transformer 不切也能跑起来」，加上一个比 CPU 快 16 倍的 T5；代价是
每个核的吞吐大约只有一半。两套实现对「切本身能快多少」的结论倒是一致的——4 个 rank 是
214 对 217 ms/step，2 个 rank 是 392 对 406——这本身是个交叉验证。

为什么会是这个排法：一个进程只能驱动一个核，所以不同进程里的组件没法共用一个核
——CLIP 和 VAE 在 pipeline 进程里，T5 在它的 worker 里，各占一个核。把四个都搬进张量
并行的 worker（T5、CLIP、VAE 作为 rank 0 上的额外图）能腾出两个核，那样
`tp_degree=4` 就能在 trn2.3xlarge 上全跑在设备里。目前没做——放置式拆分比张量并行先
出现在这个仓库里。

---

## 5. 还想更快

- **减步数。** 延时和步数成正比，FLUX.1-lite 退化得很平缓：8 步还能把主体、材质和
  光照解出来，4 步可以当预览。
- **降分辨率。** 512×512 每步比 1024×1024 快 2.8 倍：联合注意力序列从 4608 降到
  1536 个 token。
- **缩短 prompt 预算。** `--max-sequence-length 256` 把每一步每一个注意力都少算
  256 个 token。超过预算的 prompt 会被截断。
- **`-O2` / `-O3`。** `--optimization-level` 直接对应 `neuronx-cc -O`。编译更慢，跑
  得是否更快取决于负载。
- **张量并行**，见上一节。

---

## 6. 踩坑清单

**`RuntimeError: neuronx-cc compiler binary does not exist`**
venv 的 `bin` 不在 PATH 上。编译器是用 `shutil.which("neuronx-cc")` 找的。注意如果
NEFF 已经在缓存里，这个错不会出现——所以它常常在你换了分辨率或并行度时才突然冒出来。

**`The PyTorch Neuron Runtime could not be initialized`**
你要的核被别的进程占着（核是独占的）。用 `neuron-ls` 或 `neuron-top` 看谁在占；上一次
跑挂了留下的 worker 子进程是最常见的原因，所以 pipeline 要用 `with` 或显式 `close()`。

**worker 报 NRT 相关的失败，还带一句 fork-safety 的诊断**
`tp_degree>1` 的 worker 是 fork 出来的，而 fork 之前本进程如果已经初始化过 NRT，
子进程继承到的就是一个死掉的 NRT 句柄。pipeline 里 TP 的启动被特意排在本进程碰
Neuron **之前**；如果你自己改了 `compile()` 的顺序，就会看到这个错。

**切到 LNC=1 之后所有组件都退回 CPU**
NEFF 和 runtime 的 LNC 不匹配。用 `FluxNeuronConfig` 默认的编译参数（它会把
`NEURON_LOGICAL_NC_CONFIG` 映射成 `--lnc`），不要手动覆盖 `compiler_args` 把它丢掉。
不过对这个模型来说，LNC=1 本身就不该用，见第 4 节。

**日志里说 T5 留在了 CPU**
本进程 claim 了所有核，或者 `worker_device_index` 指的核不在另一块 HBM 分区里。回到
第 2 节的核固定；注意 LNC=1 时 2k 和 2k+1 是同一块分区，所以 0 和 1 不行，0 和 2 行。

**`tp_degree=3` 被拒**
必须是 2 的幂。24 个头确实能被 3 整除，但 Neuron 的 replica group、以及 LLM 那条路径
支持的所有 TP 度数都是 2 的幂，所以这里直接拒掉，而不是让它在更深的地方失败。

**`This process has already initialized the Neuron runtime`**
TP 的 worker 是 fork 出来的，一个进程只能启动一次、而且必须在自己碰设备之前启动。
所以一个进程里跑多个 tp_degree>1 的配置是不行的——一个配置一个进程。

**`tp_degree>1 requires fuse_scheduler_step=True`**
TP 下 latent 整个去噪循环都留在 worker 里，host 侧的 scheduler 看不到它。想用 host
scheduler 交叉验证融合更新的话，把 `tp_degree` 调回 1。

**显存不够（load 阶段就挂）**
先算：分片后的 transformer（15.2 GiB / tp_degree）+ 激活，要装进一块约 22 GiB 的
分区，而**同一对物理核共享这一块**。LNC=1 时把 rank 分散到不同的核对上，或者降分辨率。

---

## 7. 继续读

- [FLUX.1-lite-8B 模型说明](../model-recipes/flux-1-lite-8b.md) —— 特性、放置、完整延时和精度数据
- `vllm_neuron/model/flux/README.md` —— 为什么它不是一个注册进 vLLM 的模型，以及为 Neuron 改了什么
- `examples/vllm_neuron/models/flux/generate.py` / `benchmark.py` —— 出图和压测脚本
