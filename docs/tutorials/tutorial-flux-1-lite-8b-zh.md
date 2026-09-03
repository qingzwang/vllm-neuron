# 教程：在 Trainium 上跑 FLUX.1-lite-8B（中文上手指南）

<!-- meta: description: Step-by-step Chinese tutorial for running FLUX.1-lite-8B
text-to-image generation on Trn2 with the vLLM Neuron plugin: environment setup,
tensor parallelism across NeuronCores, measured latency at tp_degree 2 and 4, and
troubleshooting. -->

<!-- meta: keywords: vLLM, Neuron, FLUX, FLUX.1-lite, flux.1-lite-8B, 文生图,
diffusion, text-to-image, tensor parallelism, 张量并行, NeuronCore, Trn2,
Trainium, tutorial, 中文, 教程 -->

<!-- meta: date_updated: 2026-09-03 -->

<!-- Content type: procedural-tutorial -->

这篇是给**没用过 Trainium 的人**写的：从一台干净的 trn2 实例开始，一步步跑通
FLUX.1-lite-8B 出图，并理解它是怎么切到多个 NeuronCore 上的。所有命令都在
**trn2.3xlarge** 上实际跑过（2026-09-03），贴出来的输出也是真实输出。

模型本身的特性、精度和限制见
[FLUX.1-lite-8B 模型说明](../model-recipes/flux-1-lite-8b.md)，这里只讲怎么动手。

---

## 0. 先建立四个概念

**① Neuron 是「先编译、再执行」的。** 模型不是直接跑 PyTorch：`torch.compile` 把
计算图交给 `neuronx-cc` 编译成 NEFF（设备可执行文件），再加载到核上。所以形状必须
是静态的——这条 pipeline 把**分辨率**、**prompt 长度预算**和**并行度**在创建时就固定
下来，之后每一步去噪都重放同一个图。改这三个值都要重新编译。首次编译约 4 分钟，
之后命中本地编译缓存。

**② 一个进程只能驱动一个逻辑核，而且核是独占的。** 编译后端把每个 NEFF 都加载到
本进程自己的核上，所以一个进程用不了两个核；反过来，一个核也只能被一个进程持有。
实测第二个进程连 runtime 都起不来：

```
ERROR NRT:nrt_infodump  Visible cores: 0
RuntimeError: The PyTorch Neuron Runtime could not be initialized.
```

**所以「用 N 个核」在实现上就等于「起 N 个进程」**，这也是这条 pipeline 的结构：模型
张量并行地切到 `tp_degree` 个核上，每个核一个 rank 进程，四个网络全都住在这些进程
里；而你写代码所在的这个进程**一个核都不占**，只负责分词、驱动去噪循环、把结果转成
图片。

**③ 这个模型放不进一个核。** 四个组件加起来是 **24.44 GiB** 的 BF16 权重，而一个核的
HBM 分区只有约 22 GiB。所以**没有 `tp_degree=1`**，2 是下限；trn2.3xlarge 有 4 个逻辑
核（默认 `logical-neuroncore-config: 2`），所以上限是 4。

**④ FLUX 不走 `vllm serve`。** vLLM 0.24 没有文生图的请求路径（它的 `DiffusionConfig`
说的是离散扩散*语言*模型）。FLUX 走的是本插件里的独立 pipeline
`vllm_neuron.model.flux.NeuronFluxPipeline`，它复用这个插件的编译栈、NKI 注意力
kernel 和张量并行层，但不用它的 model runner。

---

## 1. 环境配置

### 1.1 机器和 AMI

| 需要什么 | 本文用的 | 说明 |
|---|---|---|
| 实例 | **trn2.3xlarge** | 1 颗 Trainium2，默认 4 个逻辑核；`tp_degree` 2 和 4 都够用 |
| AMI | Deep Learning AMI Neuron (Ubuntu 22.04/24.04) | 驱动、runtime、工具、插件 venv 都已经装好 |
| 磁盘 | ≥ 100 GB 可用 | checkpoint 约 25 GB，编译缓存约 1 GB，另外留出空间给 HF 缓存 |
| 主机内存 | ≥ 64 GB（本文 124 GB） | 每个 rank 要先把完整 checkpoint 读进内存才留下自己的分片 |

用 DLAMI Neuron 的话，驱动、运行时、工具和插件 venv 都是现成的，可以直接跳到 1.3 去
确认版本。用干净的 Ubuntu 就看下一小节。

### 1.2 如果 AMI 里什么都没有：自己装驱动和运行时

上一小节假设你用的是 DLAMI。如果是一台干净的 Ubuntu（本文验证的是 **Ubuntu 24.04
noble**，Trainium 上另一个常见选择是 22.04 jammy，把下面的 `noble` 换成 `jammy`），
整套要自己装。分三层：**驱动 → 运行时和工具 → Python 栈**。

**第 0 步：内核头文件。** 驱动是 DKMS 模块，装的时候要现场编译，所以必须先有和当前内核
匹配的头文件，否则装完不会生成模块、`/dev/neuron0` 也不会出现：

```bash
sudo apt-get update
sudo apt-get install -y linux-headers-$(uname -r) dkms
```

**第 1 步：加 Neuron 的 apt 源。**

```bash
# 公钥
curl -fsSL https://apt.repos.neuron.amazonaws.com/GPG-PUB-KEY-AMAZON-AWS-NEURON.PUB \
  | sudo gpg --dearmor -o /usr/share/keyrings/neuron-keyring.gpg

# 源（noble = Ubuntu 24.04；22.04 用 jammy）
echo "deb [signed-by=/usr/share/keyrings/neuron-keyring.gpg] \
https://apt.repos.neuron.amazonaws.com noble main" \
  | sudo tee /etc/apt/sources.list.d/neuron.list
sudo apt-get update
```

**第 2 步：驱动、运行时、工具。**

```bash
sudo apt-get install -y \
    aws-neuronx-dkms \
    aws-neuronx-runtime-lib \
    aws-neuronx-collectives \
    aws-neuronx-tools
```

四个包各自的作用：`dkms` 是内核模块（驱动），`runtime-lib` 是运行时，
`collectives` 是跨核集合通信（**张量并行必须有它**，只跑单核不会报错，一上 TP 就报），
`tools` 提供 `neuron-ls` / `neuron-top` / `neuron-monitor`。

装完确认三件事：

```bash
lsmod | grep neuron          # 模块已加载
ls /dev/neuron0              # 设备节点存在
/opt/aws/neuron/bin/neuron-ls
```

模块没加载就 `sudo modprobe neuron`；还是不行就说明 DKMS 没编译成功，回去看第 0 步的
头文件版本是否和 `uname -r` 一致（换过内核就要重装 `aws-neuronx-dkms`）。

**第 3 步：Python 栈。** 用一个干净的 venv，别装进系统 Python：

```bash
sudo apt-get install -y python3-venv python3-pip
python3 -m venv /mnt/nvme/venv-vllm-neuron
V=/mnt/nvme/venv-vllm-neuron
$V/bin/pip install --upgrade pip

# 插件本体。它会把 vllm、libtorch-neuronx-lite、torch、torch-xla、transformers 一起带来
$V/bin/pip install "vllm-neuron==0.24.*" \
    --extra-index-url https://pip.repos.neuron.amazonaws.com

# 编译器要单独装：vllm-neuron 和 libtorch-neuronx-lite 都没有把它写进依赖
$V/bin/pip install "neuronx-cc==2.27.*" \
    --extra-index-url https://pip.repos.neuron.amazonaws.com
```

> **编译器版本要钉。** 这条栈（`libtorch-neuronx-lite`）用 2.27 是好的；但如果你同时
> 也在用 NxD Inference 那条栈，注意它在 2.27 上编译 FLUX 会崩
> （`[NCC_ISMP902] Simplifier error`），要钉 2.26.6360.0。两条栈各自一个 venv，各钉各的。

自己装出来的 venv 和 DLAMI 里那个是一回事，后面 1.5 起的步骤照做即可——把 `V` 指向你
这个 venv，而且它是**可写的**，所以 1.5 和 1.6 都可以走 `pip install` 那条路，不必用
`--target` + `PYTHONPATH`。

### 1.3 确认驱动和 runtime

```bash
export PATH=/opt/aws/neuron/bin:$PATH

neuron-ls          # 列出设备；PID 列为空说明没有别的进程占着核
dpkg -l | grep -i neuron | awk '{print $2, $3}'
```

本文验证过的版本：

```
aws-neuronx-dkms          2.30.2.0       内核模块（驱动）
aws-neuronx-runtime-lib   2.34.10.0      运行时
aws-neuronx-collectives   2.34.10.0      跨核集合通信（张量并行要用）
aws-neuronx-tools         2.32.28.0      neuron-ls / neuron-top / neuron-monitor
```

`neuron-ls` 的输出里有两件重要的事：表头的 `logical-neuroncore-config: 2`（默认，也是
本文唯一支持的设置），和表里的 `NEURON CORES` 一列（可用的逻辑核数，就是 `tp_degree`
的上限）。

### 1.4 选对 Python 环境

DLAMI 里有两个 Neuron 的 venv，**互相冲突，不要混用**：

| venv | 栈 | 用途 |
|---|---|---|
| `aws_neuronx_venv_pytorch_inference_vllm_0_24_0_1_1_0` | `libtorch-neuronx-lite` | **本文用这个**（vLLM Neuron 插件） |
| `aws_neuronx_venv_jax_0_10` | JAX | 与本文无关 |

如果你还用过 NxD Inference（`torch-neuronx` + `neuronx-distributed`），那是**第三套**
栈，和这里的 `libtorch-neuronx-lite` 在 `torch`、`torch-xla`、`transformers` 上都冲突，
必须各自一个 venv，不要试图合并。

```bash
V=/opt/aws_neuronx_venv_pytorch_inference_vllm_0_24_0_1_1_0
$V/bin/python -V                       # Python 3.12
$V/bin/pip list | grep -iE "^(vllm|vllm-neuron|libtorch-neuronx-lite|torch|neuronx-cc) "
```

验证过的版本组合：

```
vllm 0.24.0 | vllm-neuron 0.24.0.1.1.0 | libtorch-neuronx-lite 2.11.0.1.0.1284
torch 2.11.0 | neuronx-cc 2.27.5334.0 | transformers 5.15.0 | Python 3.12
```

### 1.5 装 diffusers（FLUX 的额外依赖）

FLUX 不走 vLLM 引擎，所以 diffusers 没有装在插件环境里，要自己加。DLAMI 的 venv
**通常是只读的**（root 所有），两条路选一条：

```bash
# A. venv 可写：直接装
$V/bin/pip install -r requirements/flux.txt

# B. venv 只读（DLAMI 默认）：装到一个目录，用 PYTHONPATH 挂进去
$V/bin/pip install --target /mnt/nvme/pyext -r requirements/flux.txt
export PYTHONPATH=/mnt/nvme/pyext:$PYTHONPATH
```

`requirements/flux.txt` 要求 `diffusers>=0.40.0`（本文用 0.40.0）。

### 1.6 拿到这个仓库的代码

插件是通过 vLLM 的 platform plugin 入口发现的（启动时会打印
`Platform plugin neuron is activated`）。如果你要用仓库里的代码而不是 pip 装的那份：

```bash
git clone <this repo> /mnt/nvme/vllm-neuron
cd /mnt/nvme/vllm-neuron
git checkout model/flux1-lite-8B

# venv 可写：editable 安装
$V/bin/pip install -e . --no-deps
# venv 只读：放进 PYTHONPATH 的最前面
export PYTHONPATH=/mnt/nvme/vllm-neuron:$PYTHONPATH
```

确认生效（应该指向你的工作目录，而不是 site-packages）：

```bash
$V/bin/python -c "import vllm_neuron; print(vllm_neuron.__file__)"
```

### 1.7 必须设的环境变量

```bash
V=/opt/aws_neuronx_venv_pytorch_inference_vllm_0_24_0_1_1_0

# 1) 编译器要能被找到。少了这条，编译时报
#    "neuronx-cc compiler binary does not exist"——而且如果 NEFF 已在缓存里就不报，
#    等你换了分辨率或并行度才突然炸。
export PATH=$V/bin:/opt/aws/neuron/bin:$PATH

# 2) 代码和依赖（按 1.5 / 1.6 选择的方式）
export PYTHONPATH=/mnt/nvme/vllm-neuron:/mnt/nvme/pyext:$PYTHONPATH

# 3) 把 25 GB 的 checkpoint 放到大盘上，别塞进根盘
export HF_HOME=/mnt/nvme/hf-cache
```

可选但值得知道的：

| 变量 | 默认 | 什么时候动 |
|---|---|---|
| `NEURON_LIBTORCH_CACHE_ROOT` | `~/.cache/neuron_libtorch` | 编译缓存换盘。缓存约 1 GB，命中它能省掉 4 分钟编译 |
| `NEURON_LIBTORCH_COMPILATION_TIMEOUT` | 例子脚本里设成 3600 | 编译慢的机器上调大；VAE 那五段图最耗时 |
| `NEURON_RT_VISIBLE_CORES` | 不要设 | 由 rank 自己设置。手动设反而会和 rank 的核抢 |
| `NEURON_LOGICAL_NC_CONFIG` | 不要设（= LNC 2） | 见第 4 节，这个模型不该用 LNC=1 |

### 1.8 下载模型

```bash
$V/bin/pip install "huggingface_hub[cli]"      # DLAMI 里一般已有
hf download Freepik/flux.1-lite-8B             # 约 25 GB，落在 $HF_HOME
```

`Freepik/flux.1-lite-8B` 不是 gated 仓库，不用登录。要跑 FLUX.1-dev 的话它是 gated 的，
需要先在网页上同意协议再 `hf auth login`。

### 1.9 验一遍环境

跑完这一段没报错，就可以进第 2 节了：

```bash
$V/bin/python - <<'EOF'
import torch, vllm_neuron
from vllm_neuron.model.flux import FluxNeuronConfig, NeuronFluxPipeline
print("vllm_neuron:", vllm_neuron.__file__)
import diffusers; print("diffusers:", diffusers.__version__)
print("config ok:", FluxNeuronConfig(height=512, width=512, tp_degree=2).tp_core_ids)
# 一个最小的设备计算：能过就说明驱动、runtime、编译器、核都正常
x = torch.ones(8, 8).to(torch.device("neuron", 0))
print("device matmul:", float((x @ x).cpu().sum()))
EOF
```

最后一行应该打印 `512.0`。这一小段自己会占用 0 号核，跑完就释放；如果它报
`The PyTorch Neuron Runtime could not be initialized`，说明有别的进程占着核，见第 7 节。

---

## 2. 第一次出图

```bash
python examples/vllm_neuron/models/flux/generate.py \
    --model-checkpoint Freepik/flux.1-lite-8B \
    --tp 4 \
    --prompt "A close-up photo of a red panda wearing tiny round glasses, reading a leather-bound book in a cozy library" \
    --steps 28 \
    --output flux_output.png
```

![FLUX.1-lite-8B 在 trn2 上的输出：戴圆眼镜看书的小熊猫](../model-recipes/images/flux-1-lite-8b-sample.png)

*1024×1024、28 步、guidance 3.5、seed 42、`tp_degree=4`，就是上面那条命令。为了这个
页面缩过。*

`--tp 4` 就是用 4 个核（rank 占 0-3 号核）。`--tp 2` 用 2 个核，慢一些但省两个核。
**不需要自己安排核的绑定**——这个进程不碰设备，rank 自己会绑。

首次运行会编译四个组件（transformer、T5、CLIP、VAE），约 4 分钟；之后命中缓存，rank
起来要 100–140 秒（主要在从磁盘读权重）。另外，一个进程里**第一张要解码的图**会多花约
12 秒，那是把 VAE 的 NEFF 加载到设备上，从第二张开始才是稳定延时。

代码里这么写：

```python
from vllm_neuron.model.flux import FluxNeuronConfig, NeuronFluxPipeline

config = FluxNeuronConfig(height=1024, width=1024, tp_degree=4)
# with 会在退出时把 rank 占的核释放掉
with NeuronFluxPipeline.from_pretrained("Freepik/flux.1-lite-8B", config) as pipeline:
    pipeline.compile()
    image, timing = pipeline.generate("a red panda reading a book",
                                      num_inference_steps=28, seed=42)
image.save("out.png")
print(timing.as_dict())
```

---

## 3. 四个网络是怎么切的

| 组件 | BF16 权重 | 跨 rank | 为什么 |
|---|---|---|---|
| `transformer` | 15.20 GiB | **切开** | 一次请求里跑 28 次，成本几乎全在这 |
| `text_encoder_2`（T5-XXL） | 8.87 GiB | **切开** | 大，而且 64 个头能整除 |
| `text_encoder`（CLIP-L） | 0.22 GiB | 复制 | 切它要加 collective，省不下什么 |
| `vae`（解码器） | 0.15 GiB | 复制 | 卷积结构，一次请求只跑一次 |

和 NxD Inference 对同一个模型的切法一致，所以两边可以直接比（见模型说明里的对比）。

只有**注意力头**和**FFN 中间维**被切开；残差流保持全宽、每个 rank 上都一样，所以
norm、调制投影、embedder、最后的 `proj_out` 都是复制的——注意力 kernel 也因此完全不用
改，它从张量形状里推头数，而不是读配置。

有两个地方不能照套标准模式：

* **单流块的 `proj_out`** 吃的是 `cat([注意力输出, MLP 输出])`。两个生产者都是列并行，
  所以每个 rank 手上是 `[注意力维/tp + MLP维/tp]`，这**不是**全局输入的连续切片，用普通
  行并行会乘错权重。实现里把权重按两半分别切，两个偏积相加之后只做一次 all-reduce。
* **T5 的相对位置偏置**是只有第 0 块持有的 `Embedding(32 桶, 64 头)`，算一次之后穿给
  后面每一块。它按头切，同时把 `T5Attention.n_heads` 和 `inner_dim` 一起改小——T5 是
  按这两个属性 reshape 的，不像 FLUX 那样从张量推。

---

## 4. 核怎么算

一个 rank 一个核，除此之外**不需要额外的核**：

| tp_degree | 核数 | 每核权重 | trn2.3xlarge（4 核）放得下？ |
|---|---|---|---|
| 2 | 2 | 12.22 GiB | ✅ |
| 4 | 4 | 6.11 GiB | ✅ 刚好 |
| 8 | 8 | 3.06 GiB | ❌ 要更大的实例 |

用默认的 `logical-neuroncore-config: 2`（LNC=2），**逻辑核就是计量单位**。有人会想到用
LNC=1 把 4 个核变成 8 个——**对这么大的模型不要这么做**：LNC=1 的每个核是单个物理
NeuronCore 而不是融合的一对，算力减半（实测同一个未切分的 transformer 是 1227 对
791 ms/step）；而且这条 pipeline 目前在 LNC=1 下根本不对，用 `--lnc=1` 编出来的
transformer 图从第一步去噪就返回 NaN，而同一次运行里 encoder 的输出和初始 latent 都是
正常的。

顺带解释一个容易困惑的点：trn2 的 96 GB HBM 是**按物理核对**分成四块约 22 GiB 的分区，
不是按逻辑核。LNC=2 时一个逻辑核就是一对物理核，所以它独占一块分区——上面那张表里
「每核权重」才是真正要和 22 GiB 比的数。

---

## 5. 实测延时

trn2.3xlarge、LNC=2、BF16、batch 1、`neuronx-cc -O1`、512 token 预算。中位数，跑之前
先丢弃一次预热请求。

| tp_degree | 分辨率 | 步数 | ms/step | 去噪 | prompt 编码 | VAE 解码 | **端到端** |
|---|---|---|---|---|---|---|---|
| 2 | 512×512 | 4 | 134 | 0.54 s | 0.06 s | 0.14 s | **0.74 s** |
| 4 | 512×512 | 4 | 80 | 0.32 s | 0.03 s | 0.14 s | **0.51 s** |
| 2 | 1024×1024 | 28 | 392 | 11.0 s | 0.06 s | 0.56 s | **11.65 s** |
| 4 | 1024×1024 | 28 | 214 | 6.0 s | 0.03 s | 0.57 s | **6.62 s** |

rank 数翻倍，每步快 1.84 倍，整个请求快 1.76 倍。每步延时在不同步数下是平的、抖动远
小于 1%——每步跑的是同一个图、静态形状，host 上只剩一次一元素的栅栏读取。

1024×1024、`tp_degree=4` 时去噪占一次请求的 91%。其余部分放在 Neuron 上省了多少（对比
同样权重在 CPU eager 上跑）：

| 阶段 | Neuron（tp=4） | CPU eager | 倍数 |
|---|---|---|---|
| prompt 编码（CLIP + T5，512 token） | 0.03 s | 1.62 s | 54× |
| VAE 解码（1024×1024，五段图） | 0.57 s | 7.55 s | 13× |

**启动的代价在另一头**：每个 rank 都要先把完整 checkpoint 读进来才留下自己那几片，而且
为了不让主机内存爆掉，它们是拿锁排队读的。所以 rank 数越多启动越慢（100–140 秒）。
每步的额外开销很小（约 1 ms）：embedding 和 latent 整个请求都留在 rank 里，每步只发三个
标量、收一个一元素栅栏。

### 正确性

切分是精确的，不是近似（每个 rank 的偏积之和等于稠密层的结果，这一条在 CPU 上逐层验
过），但它确实改变了求和顺序，bf16 下会体现出来。同种子同 prompt，比最终 latent：

| | 步数 | cos | max\|d\| | latent 标准差 |
|---|---|---|---|---|
| 512×512，tp=2 vs tp=4 | 4 | 0.999545 | 0.74 | 1.18 |
| 1024×1024，tp=2 vs tp=4 | 28 | 0.998092 | 2.27 | 1.29 |

28 步那一行低一些，是因为重结合误差沿链累积，不是因为切得更多。解码出来的两张
1024×1024 图每个通道平均差 1.71/255：

![tp_degree=2 的输出](../model-recipes/images/flux-1-lite-8b-tp2.png)

*`tp_degree=2`，其余设置和第 2 节那张（`tp_degree=4`）完全一样。*

切错的话根本不是这个量级——会掉到 cos 0.9 以下，图上一眼就能看出来。

---

## 6. 还想更快

- **加 rank。** `tp_degree` 从 2 到 4，每步快 1.84 倍，代价是多两个核。
- **减步数。** 延时和步数成正比，FLUX.1-lite 退化得很平缓：8 步还能把主体、材质和
  光照解出来，4 步可以当预览。
- **降分辨率。** 512×512 每步比 1024×1024 快 2.7 倍：联合注意力序列从 4608 降到
  1536 个 token。
- **缩短 prompt 预算。** `--max-sequence-length 256` 把每一步每一个注意力都少算
  256 个 token。超过预算的 prompt 会被截断。
- **`-O2` / `-O3`。** `--optimization-level` 直接对应 `neuronx-cc -O`。编译更慢，跑
  得是否更快取决于负载。

压测：

```bash
python examples/vllm_neuron/models/flux/benchmark.py \
    --tp 4 --sizes 512,1024 --steps 8,28 --iterations 2 --json flux_latency.json
```

---

## 7. 踩坑清单

**`RuntimeError: neuronx-cc compiler binary does not exist`**
venv 的 `bin` 不在 PATH 上（编译器是用 `shutil.which` 找的）。注意 NEFF 已在缓存里时
这个错不会出现，所以它常常在你换了分辨率或并行度时才突然冒出来。

**`The PyTorch Neuron Runtime could not be initialized`**
要用的核被别的进程占着——核是独占的。最常见的原因是上一次跑挂了、留下了 rank 进程：

```bash
neuron-ls                 # PID 列会告诉你谁在占
pkill -f your_script.py   # 清掉残留
```

正常退出路径上不会有残留：pipeline 用 `with`（或显式 `close()`）会杀掉 rank，rank
加载失败时也会主动把它们清掉。

**`tp_degree=1 is not supported` / `tp_degree=3 is not supported`**
只支持 2、4、8。没有 1 是因为四个组件加起来 24.44 GiB，装不进一个核的 ~22 GiB 分区；
不支持 3 是因为 Neuron 的 replica group、以及 LLM 那条路径支持的所有 TP 度数都是 2 的幂。

**`This process has already initialized the Neuron runtime`**
rank 是 fork 出来的，而 fork 之前本进程如果已经初始化过 NRT，子进程继承到的就是一个
死掉的 NRT 句柄。所以一个进程里只能起一套 rank——**一个配置一个进程**，压测多个
`--tp` 时分开跑。

**启动很慢（100 秒以上）**
正常。每个 rank 都要读完整的 checkpoint 才留下自己的分片，而且是排队读的（否则主机
内存放不下）。编译缓存只省编译，不省这个。

**第一张图特别慢（多 12 秒）**
把 VAE 的 NEFF 加载到设备上的一次性代价，第二张开始就正常了。压测脚本默认会先丢弃一次
预热请求。

**切到 LNC=1 之后出的图全是噪声 / latent 是 NaN**
LNC=1 目前不支持，见第 4 节。用默认的 LNC=2。

**改了分辨率 / prompt 预算 / tp_degree 之后要重新编译**
这是设计如此：形状和并行度都进了编译缓存的 key。

---

## 8. 继续读

- [FLUX.1-lite-8B 模型说明](../model-recipes/flux-1-lite-8b.md) —— 特性、切分细节、完整延时和精度数据
- `vllm_neuron/model/flux/README.md` —— 为什么它不是一个注册进 vLLM 的模型，以及为 Neuron 改了什么
- `vllm_neuron/model/flux/parallel.py` —— 逐层的切分规则
- `examples/vllm_neuron/models/flux/generate.py` / `benchmark.py` —— 出图和压测脚本
