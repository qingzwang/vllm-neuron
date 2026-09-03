# 教程：在 Trainium 上跑 FLUX.1-dev（中文上手指南）

<!-- meta: description: Step-by-step Chinese tutorial for running FLUX.1-dev
text-to-image generation on Trn2 with the vLLM Neuron plugin: environment setup,
tensor parallelism across NeuronCores, dynamic LoRA, measured latency at tp_degree 2
and 4, and troubleshooting. -->

<!-- meta: keywords: vLLM, Neuron, FLUX, FLUX.1-dev, 文生图, LoRA,
diffusion, text-to-image, tensor parallelism, 张量并行, NeuronCore, Trn2,
Trainium, tutorial, 中文, 教程 -->

<!-- meta: date_updated: 2026-09-03 -->

<!-- Content type: procedural-tutorial -->

这篇是给**没用过 Trainium 的人**写的：从一台干净的 trn2 实例开始，一步步跑通
FLUX.1-dev 出图，理解它是怎么切到多个 NeuronCore 上的，以及怎么在运行时换 LoRA。所有命令都在
**trn2.3xlarge** 上实际跑过（2026-09-03），贴出来的输出也是真实输出。

模型本身的特性、精度和限制见
[FLUX.1-dev 模型说明](../model-recipes/flux-1-dev.md)，这里只讲怎么动手。

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

**③ 这个模型要 4 个核。** 四个组件加起来是 **31.42 GiB** 的 BF16 权重（transformer 自己
就 22.17 GiB），而一个核的 HBM 分区只有约 22 GiB：`tp_degree=1` 直接被配置拒掉；
`tp_degree=2` 权重勉强放得下（每核 15.71 GiB），但 1024×1024 的激活放不下、rank 加载时
报 `Allocation Failure`。所以**这个 checkpoint 用 `tp_degree=4`**（每核 7.86 GiB），而
trn2.3xlarge 正好有 4 个逻辑核。

**④ FLUX 不走 `vllm serve`。** vLLM 0.24 没有文生图的请求路径（它的 `DiffusionConfig`
说的是离散扩散*语言*模型）。FLUX 走的是本插件里的独立 pipeline
`vllm_neuron.model.flux.NeuronFluxPipeline`，它复用这个插件的编译栈、NKI 注意力
kernel 和张量并行层，但不用它的 model runner。

---

## 1. 环境配置

### 1.1 机器和 AMI

| 需要什么 | 本文用的 | 说明 |
|---|---|---|
| 实例 | **trn2.3xlarge** | 1 颗 Trainium2，默认 4 个逻辑核——正好是这个模型需要的 `tp_degree=4` |
| AMI | Deep Learning AMI Neuron (Ubuntu 22.04/24.04) | 驱动、runtime、工具、插件 venv 都已经装好 |
| 磁盘 | ≥ 100 GB 可用 | checkpoint 约 31 GB，编译缓存约 1 GB，另外留出空间给 HF 缓存 |
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
栈，和这里的 `libtorch-neuronx-lite` 冲突，必须各自一个 venv——机器上只有 NxDI 环境时
怎么加这一条栈，见本节末尾。

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

#### 已经有 NxDI 环境、想再加 vllm-neuron

这是很常见的情形：驱动装好了，机器上也已经有一个 NxD Inference 的 venv，现在要把
vllm-neuron 也跑起来。

**好消息是系统层不用动。** 驱动、运行时、集合通信、工具都是 apt 装的系统级组件，**两条
栈共用同一份**——两边的 Python 库链的都是同一个 `libnrt.so.1`：

```bash
# vllm-neuron 这条栈
ldd .../site-packages/libtorch_neuronx_lite/lib/libtorchneuron.so | grep nrt
#   libnrt.so.1 => /opt/aws/neuron/lib/libnrt.so.1
# NxDI 那条栈
ldd .../site-packages/libneuronxla/libneuronpjrt.so | grep nrt
#   libnrt.so.1 => /opt/aws/neuron/lib/libnrt.so.1
```

所以「驱动有了」就等于系统层这一半已经就绪，要加的只有 Python 层。

**必须新建一个 venv，不要往 NxDI 那个里装。** 两条栈在四个包上都要换版本，装一起等于把
先装的那个弄坏。本机两个 venv 的实测对照：

| | vllm-neuron 栈 | NxDI 栈 |
|---|---|---|
| Neuron 层 | `libtorch-neuronx-lite` 2.11.0.1.0.1284 | `torch-neuronx` 2.9.0.2.15 |
| torch | 2.11.0 | 2.9.1 |
| torch-xla | 2.11.0 | 2.9.0 |
| transformers | 5.15.0 | 4.57.6 |
| neuronx-cc | 2.27.5334.0 | **2.26.6360.0** |

```bash
python3 -m venv /mnt/nvme/venv-vllm-neuron       # 和 NxDI 的 venv 平级、互不影响
V=/mnt/nvme/venv-vllm-neuron
$V/bin/pip install --upgrade pip
$V/bin/pip install "vllm-neuron==0.24.*" "neuronx-cc==2.27.*" \
    --extra-index-url https://pip.repos.neuron.amazonaws.com
```

之后照 1.5（diffusers）和 1.6（仓库代码）做完就行。

**要特别小心 PATH。** 两个 venv 各自带一个 `neuronx-cc` 可执行文件，而且版本不同；编译器
是用 `shutil.which` 找的，**谁在 PATH 前面就用谁**。用 NxDI 的 2.26 去编 vllm-neuron 的图
（或者反过来）会得到很难联想到原因的失败。每次进环境先确认一遍：

```bash
export PATH=$V/bin:/opt/aws/neuron/bin:$PATH
which neuronx-cc          # 应该指向 $V/bin
neuronx-cc --version      # 应该是 2.27.x（NxDI 那边应该是 2.26.6360.0）
```

不要在同一个 shell 里同时 export 两个 venv 的 `PATH`/`PYTHONPATH`。分两个 shell，或者写
两个小的 `env-vllm.sh` / `env-nxdi.sh` 分别 source。

**运行时版本得够新。** 系统那份 runtime 是共用的，如果它比 1.3 的表更老，先升：

```bash
sudo apt-get update
sudo apt-get install --only-upgrade \
    aws-neuronx-runtime-lib aws-neuronx-collectives aws-neuronx-tools
# 驱动也要升的话（要重新加载模块，最省事是重启）
sudo apt-get install --only-upgrade aws-neuronx-dkms
```

注意这一步对 NxDI 那条栈同样生效——它们共用，所以升级前留意一下那边是否有版本要求。

**编译缓存互不干扰**，不用管：这条栈默认写 `~/.cache/neuron_libtorch`，NxDI 那条写
`/var/tmp/neuron-compile-cache`，两个目录各自独立。

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
git checkout feature/flux-dynamic-lora

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
# FLUX.1-dev 是 gated 仓库：先在网页上同意协议，再登录
hf auth login
hf download black-forest-labs/FLUX.1-dev       # 约 31 GB，落在 $HF_HOME
```

任何 diffusers 格式、`guidance_embeds=True` 的 FLUX checkpoint 都走同一条路（块数、头数
都是从 checkpoint 里读的），蒸馏版会按它砍掉的块数成比例变快。

### 1.9 验一遍环境

跑完这一段没报错，就可以进第 2 节了：

```bash
$V/bin/python - <<'EOF'
import torch, vllm_neuron
from vllm_neuron.model.flux import FluxNeuronConfig, NeuronFluxPipeline
print("vllm_neuron:", vllm_neuron.__file__)
import diffusers; print("diffusers:", diffusers.__version__)
print("config ok:", FluxNeuronConfig(height=512, width=512, tp_degree=4).tp_core_ids)
# 一个最小的设备计算：能过就说明驱动、runtime、编译器、核都正常
x = torch.ones(8, 8).to(torch.device("neuron", 0))
print("device matmul:", float((x @ x).cpu().sum()))
EOF
```

最后一行应该打印 `512.0`。这一小段自己会占用 0 号核，跑完就释放；如果它报
`The PyTorch Neuron Runtime could not be initialized`，说明有别的进程占着核，见第 8 节。

---

## 2. 第一次出图

```bash
python examples/vllm_neuron/models/flux/generate.py \
    --model-checkpoint black-forest-labs/FLUX.1-dev \
    --tp 4 \
    --prompt "A close-up photo of a red panda wearing tiny round glasses, reading a leather-bound book in a cozy library" \
    --steps 28 \
    --output flux_output.png
```

![FLUX.1-dev 在 trn2 上的输出：戴圆眼镜看书的小熊猫](../model-recipes/images/flux-1-dev-sample.png)

*1024×1024、28 步、guidance 3.5、seed 42、`tp_degree=4`，就是上面那条命令。为了这个
页面缩过。*

`--tp 4` 就是用 4 个核（rank 占 0-3 号核），这个 checkpoint 只能用 4，原因见概念 ③。
**不需要自己安排核的绑定**——这个进程不碰设备，rank 自己会绑。

首次运行会编译四个组件（transformer、T5、CLIP、VAE），约 4 分钟；之后命中缓存，rank
起来要 100–140 秒（主要在从磁盘读权重）。另外，一个进程里**第一张要解码的图**会多花约
12 秒，那是把 VAE 的 NEFF 加载到设备上，从第二张开始才是稳定延时。

代码里这么写：

```python
from vllm_neuron.model.flux import FluxNeuronConfig, NeuronFluxPipeline

config = FluxNeuronConfig(height=1024, width=1024, tp_degree=4)
# with 会在退出时把 rank 占的核释放掉
with NeuronFluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", config) as pipeline:
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
| `transformer` | 22.17 GiB | **切开** | 一次请求里跑 28 次，成本几乎全在这 |
| `text_encoder_2`（T5-XXL） | 8.87 GiB | **切开** | 大，而且 64 个头能整除 |
| `text_encoder`（CLIP-L） | 0.23 GiB | 复制 | 切它要加 collective，省不下什么 |
| `vae`（解码器） | 0.16 GiB | 复制 | 卷积结构，一次请求只跑一次 |

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

一个 rank 一个核，除此之外**不需要额外的核**。这个 checkpoint 的选择只有一个：

| tp_degree | 每核权重 | 结果 |
|---|---|---|
| 1 | 31.42 GiB | 配置直接拒掉——transformer 自己就 22.17 GiB |
| 2 | 15.71 GiB | 权重放得下，但 1024×1024 的激活放不下，加载时 `Allocation Failure` |
| **4** | **7.86 GiB** | **可用，还有空间放 LoRA 槽位** |
| 8 | 3.93 GiB | 要比 trn2.3xlarge 更大的实例 |

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

全部是 `tp_degree=4`：

| 分辨率 | 步数 | ms/step | 去噪 | prompt 编码 | VAE 解码 | **端到端** |
|---|---|---|---|---|---|---|
| 512×512 | 4 | 105 | 0.42 s | 0.04 s | 0.14 s | **0.61 s** |
| 1024×1024 | 4 | 276 | 1.10 s | 0.03 s | 0.60 s | **1.77 s** |
| 1024×1024 | 8 | 274 | 2.20 s | 0.03 s | 0.58 s | **2.85 s** |
| 1024×1024 | 28 | 274 | 7.69 s | 0.03 s | 0.60 s | **8.37 s** |

每步延时在不同步数下是平的、抖动远小于 1%——每步跑的是同一个图、静态形状，host 上只剩
一次一元素的栅栏读取。1024×1024、28 步时去噪占一次请求的 92%。其余部分放在 Neuron 上省
了多少（对比同样权重在 CPU eager 上跑）：

| 阶段 | Neuron（tp=4） | CPU eager | 倍数 |
|---|---|---|---|
| prompt 编码（CLIP + T5，512 token） | 0.03 s | 1.62 s | 54× |
| VAE 解码（1024×1024，五段图） | 0.60 s | 7.55 s | 13× |

**启动的代价在另一头**：每个 rank 都要先把完整 checkpoint 读进来才留下自己那几片，而且
为了不让主机内存爆掉，它们是拿锁排队读的（130–180 秒）。
每步的额外开销很小（约 1 ms）：embedding 和 latent 整个请求都留在 rank 里，每步只发三个
标量、收一个一元素栅栏。

### 正确性

切分是精确的，不是近似（每个 rank 的偏积之和等于稠密层的结果，这一条在 CPU 上逐层验
过），但它确实改变了求和顺序，bf16 下会体现出来。同种子同 prompt，比最终 latent：

| | 步数 | cos | max\|d\| | latent 标准差 |
|---|---|---|---|---|
| 512×512，tp=2 vs tp=4 | 4 | 0.999554 | 0.74 | 1.22 |

只有 512×512 这一行，因为 `tp_degree=2` 在 1024×1024 下装不下。差异来自 bf16 下求和顺序
变了（注意力和前馈被切开）。切错的话根本不是这个量级——会掉到 cos 0.9 以下，图上一眼就能
看出来。

---

## 6. 动态 LoRA

adapter 可以在**运行时**加载到设备槽位里、按请求切换，不重新编译任何东西，而且**切换只要
不到一毫秒**：

```python
config = FluxNeuronConfig(height=1024, width=1024, tp_degree=4,
                          lora_slots=2, lora_max_rank=64)
with NeuronFluxPipeline.from_pretrained(CKPT, config) as pipeline:
    pipeline.compile()
    pipeline.load_lora("realism", "/adapters/xlabs-realism")
    pipeline.load_lora("superreal", "/adapters/super-realism.safetensors")

    pipeline.set_lora("realism")
    image_a, _ = pipeline.generate(prompt, num_inference_steps=28)
    pipeline.set_lora("superreal")     # 约 0.6 ms
    image_b, _ = pipeline.generate(prompt, num_inference_steps=28)
    pipeline.set_lora(None)            # 回到未改的模型
```

命令行：

```bash
python examples/vllm_neuron/models/flux/generate.py --tp 4 \
    --lora realism=/adapters/xlabs-realism \
    --lora superreal=/adapters/super-realism.safetensors
```

给了 `--lora` 而没给 `--use-lora` 时，它会把 base 和每个 adapter 各出一张图——因为切换
几乎不要钱，多出的图只花自己那点去噪时间。

| 底模 | [XLabs realism](https://huggingface.co/XLabs-AI/flux-RealismLora)（r=16） | [kohya super-realism](https://huggingface.co/strangerzonehf/Flux-Super-Realism-LoRA)（r=64） |
|---|---|---|
| ![](../model-recipes/images/flux-lora-base.png) | ![](../model-recipes/images/flux-lora-xlabs.png) | ![](../model-recipes/images/flux-lora-kohya.png) |

<sub>同一个编译好的模型、同一个 prompt、同一个种子；两个 adapter 都是运行时加载的，切换
用的是亚毫秒的一次索引写。512×512，28 步，`tp_degree=4`。</sub>

`lora_slots=0`（默认）时图和原来完全一样，不用 adapter 的部署不付任何代价。
diffusers/PEFT、kohya、XLabs 三种格式都能读（文件交给
`FluxPipeline.lora_state_dict`，让 diffusers 自己的转换器处理格式）。

### 为什么不用重新编译

靠两条实测出来的性质：

1. **原地写设备张量，已经编译好的图能看到新值。** 往 Parameter、buffer 或普通张量属性里
   `copy_` 之后，下一次调用返回的就是新值，而 Dynamo 报告没有新图。所以 adapter 的权重
   可以放在 NEFF 直接读的设备张量里。
2. **选择用的索引也可以是设备张量。** 每个被适配的层读的是**同一个**一元素张量，在槽位维
   上做 `index_select`。所以切 adapter 写的是 4 个字节，不是搬权重。

第 2 条才是切换便宜的原因。一个完整的 adapter 在每个 rank 上是几百 MB、散在一千多个小
张量里，搬一次要几百毫秒到几秒；槽位存在的意义就是让这个代价**每个 adapter 付一次**，而
不是每次切换都付。

### 实测（tp=4，512px，2 个槽位，rank 64）

| 操作 | 耗时 |
|---|---|
| `set_lora(...)` 在已加载的 adapter 之间切 | **0.5–1.0 ms** |
| `load_lora(...)`，22 MiB 的 adapter（152 个模块） | 0.13–0.14 s |
| `load_lora(...)`，585 MiB 的 adapter（494 个模块） | 0.44–0.58 s |
| 每步延时（adapter 生效时） | 116.3–117.4 ms，对比底模的 115.3 ms |
| 槽位显存 | 385 MB / 槽 / rank |

adapter 生效时的每步开销测不出来——多出来的是每个适配层两个很薄的矩阵乘，对比的是
4608 token 的注意力。槽位显存和 `lora_max_rank` 成正比，所以按你真正会用的最大 rank 设。

### 正确性

adapter 的切分必须和它所适配的层的切分对齐，而**行并行的层，delta 必须加在 all-reduce
之前**：那里 `x` 是切开的，每个 rank 只能算出部分的 `A @ x`，加在 reduce 之后会让每个
rank 得到各自不同的错误结果。四种情况（列并行、行并行、单流块被拆开的 `proj_out`、普通
层）都在 CPU 上和稠密的 `W x + B (A x)` 精确对比过。

设备上，每个槽位都和**每一个** adapter 的 CPU **float32** 参考对比，而不只是和它自己那个。
提示词 `"a photo of a red panda reading a book"`、distilled guidance 3.5、true CFG 关闭
（这条 pipeline 根本没有负向那一遍）——和 NxD Inference 的 LoRA 测试用的是同一套设置，所以
两边的数字可以直接比。单步、512×512、两边喂同一份初始 latent，比最终 latent 的余弦：

| 槽位 | cpu-base | cpu-xlabs | cpu-kohya |
|---|---|---|---|
| base（槽位 0） | **0.999772** | 0.997536 | 0.992593 |
| xlabs（槽位 1） | 0.997322 | **0.999820** | 0.991455 |
| kohya（槽位 2） | 0.993000 | 0.991381 | **0.999797** |

每一行的最大值都在对角线上，而且对角线贴着 ~0.9998——也就是这个模型 bf16 对 fp32 的底噪
——非对角是 0.991～0.998。如果某个槽位读错了权重、或者 adapter 被套到了错误的模块上，最大
值就会跑到对角线外面去。NxD Inference 那边同 checkpoint、同两个 adapter 的对应检查，对角线
也在 0.9992～0.9996 这个量级；它比的是直接的 velocity、256×256、两边喂同一份 fp32 的 prompt
embedding，而这边是各自编码自己的 prompt。

切换本身也必须是无损的：切走再切回**逐位相同**，回到槽位 0 也逐位相同。

### adapter 必须和 checkpoint 匹配

adapter 是按模块名指定目标的，所以对着别的 FLUX 变体训出来的 adapter
会点名 19 个双流块，而砍过块数的蒸馏版没有那么多。把不匹配的 adapter 加载进来，只有确实
存在的层会被适配——这不是 adapter 作者的本意——并且会打印掉了多少：

```
WARNING Adapter targets 266 modules that are not adapted here, e.g. [...]
```

这是警告而不是报错，因为"只适配一部分"本身是合法的用法；但如果你加载的是现成的 adapter
还看到它，那问题就在 checkpoint 上。除此之外这篇文档的其他内容和跑哪个 checkpoint 无关
——块数是从 checkpoint 里读的。

---

## 7. 还想更快

- **减步数。** 延时和步数成正比，但 FLUX.1-dev 没有做步数蒸馏，所以退化得比蒸馏版快：
  8 步（2.85 s）还能把主体、材质和光照解出来，4 步（1.77 s）就偏暗偏糊了，不只是细节少。
- **降分辨率。** 512×512 每步比 1024×1024 快 2.6 倍（105 对 274 ms）：联合注意力序列从
  4608 降到 1536 个 token。
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

## 8. 踩坑清单

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

## 9. 继续读

- [FLUX.1-dev 模型说明](../model-recipes/flux-1-dev.md) —— 特性、切分细节、完整延时和精度数据
- `vllm_neuron/model/flux/README.md` —— 为什么它不是一个注册进 vLLM 的模型，以及为 Neuron 改了什么
- `vllm_neuron/model/flux/parallel.py` —— 逐层的切分规则
- `examples/vllm_neuron/models/flux/generate.py` / `benchmark.py` —— 出图和压测脚本
