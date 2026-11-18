# H100 Hy3-FP8 profiling 结果

测试时间:2026-11-18 02:58–05:30 UTC,p5.48xlarge(8× H100 80GB HBM3,NVLink 全互联 NV18)。

## 环境
- GPU / 驱动 / CUDA:8× NVIDIA H100 80GB HBM3,驱动 595.112.07,SM 最高 1980 MHz,显存最高 2619 MHz;宿主机 Ubuntu 26.04.1,CUDA 13.0(nvcc V13.0.88,DLAMI 自带的 `/opt/pytorch/cuda`)。详见 `env.txt`。
- vLLM 容器:`vllm/vllm-openai:v0.29.0`:torch 2.13.0+cu130,CUDA 13.0,vLLM 0.29.0,flashinfer 0.6.18;`torch.cuda.nccl.version()` = 2.29.7,pip 包 `nvidia-nccl-cu13` = 2.30.7。
- nccl-tests:在**宿主机**上编译(nccl-tests 2.21.1 @ afd59ab),链接宿主机 DLAMI pip 里的 NCCL 2.30.7(`/opt/pytorch/lib/python3.13/site-packages/nvidia/cu13`;输出开头是 `nccl-headers=23007 nccl-library=23007`)。没有设置任何 `NCCL_*` 环境变量,所以 `NCCL_ALGO`、`NCCL_PROTO` 都没设(`nccl/nccl_env.txt` 是空文件)。
- 用的镜像:server 直接用 `vllm/vllm-openai:v0.29.0`(单机不用 Mooncake/EFA,不需要 `hy3-efa-029`)。benchmark client 用 `hy3-bench-client`:同一个镜像只加了 `pandas`、`datasets`(`vllm bench serve` 需要的 `vllm[bench]` 依赖,原镜像里没有),并把 `huggingface_hub` 固定回镜像原来的 1.30.0。Dockerfile 在 `client_img/`。
- 模型:`tencent/Hy3-FP8`,不需要 token,从 HF 下载到 `/opt/dlami/nvme/models/Hy3-FP8`(280 GB,101 个 shard 都核对过)。
- 代码:`qingzwang/vllm-neuron`,分支 `hy3-fp8-contrib` @ e8910540。
- serve 配置:A 和 B 都测了,完整命令见 `serve_cmd_A.txt`、`serve_cmd_B.txt`(即 `serve_A.sh`、`serve_B.sh`)。
  - **A**:`gpu_serve.sh` 的参数(TP=8,BF16 KV,不开 MTP,`--tool-call-parser hy_v3 --enable-auto-tool-choice --reasoning-parser hy_v3`,`VLLM_FLASHINFER_ALLREDUCE_BACKEND=trtllm`,max_model_len 87040),去掉了 PD 相关的部分,见第 6 节。
  - **B**:`gpu_single_serve.sh`(你确认过的),`MAX_LEN=87040`,不开 offload:TP=8,`--kv-cache-dtype fp8`,`--speculative-config '{"method":"mtp","num_speculative_tokens":3}'`,其余参数和 A 一样。
  - 两个配置都加了 `--profiler-config '{"profiler": "torch", "torch_profiler_dir": "/workspace/prof_decode"}'`。扫描、profile、prefill 都在同一个 server 进程里做(每个配置一个进程)。
  - vLLM 自动选的后端(见 `server_A.log` / `server_B.log`):attention 用 FLASH_ATTN(FA3,sm90);MoE 用 FLASHINFER_CUTLASS FP8;TP all-reduce 按 FLASHINFER(trtllm)→ CUSTOM → SYMM_MEM → PYNCCL 的顺序选;启用了 `allreduce_rms` 融合。KV 容量:A 是 896,640 token,B 是 1,690,656 token。
- benchmark 数据:`bench_data/prefix/in85000_p78336_c8.jsonl.gz`,md5 `118365ca9028896f922288e59aeb4d7a`(仓库里的文件,没有重新生成)。**README 没有给出 md5 的具体值**(只说"md5-checked"),所以没法和参考值比对,只把本机的 md5 记在 `bench_data_md5.txt`。

## 1. nccl-tests(8 卡,bfloat16,单位 µs;CUDA Graph 那组单独一列)
取 out-of-place 的 time 列,`-w 20 -n 100`;graph 列加了 `-G 20`。AG 和 RS 的"大小"是 nccl-tests 的 size 列(总字节数)。完整输出见 `nccl/nccl_h100.txt`。

| 大小 | AR | AR(graph) | AG | AG(graph) | RS | RS(graph) |
|---|---|---|---|---|---|---|
| 16 KB | 49.24 | 15.67 | 48.82 | 10.13 | 48.52 | 9.80 |
| 32 KB | 49.69 | 17.74 | 49.28 | 12.34 | 48.79 | 10.60 |
| 64 KB | 50.08 | 17.94 | 50.80 | 12.45 | 49.96 | 10.72 |
| 128 KB | 51.07 | 18.64 | 52.77 | 12.62 | 51.68 | 10.84 |
| 256 KB | 53.56 | 19.99 | 55.35 | 12.68 | 53.10 | 11.10 |
| 512 KB | 56.01 | 20.08 | 58.39 | 13.12 | 54.66 | 12.44 |
| 1 MB | 59.24 | 28.40 | 60.30 | 17.01 | 58.25 | 15.95 |
| 2 MB | 51.19 | 41.76 | 61.14 | 22.32 | 58.21 | 20.34 |
| 16 MB | 126.04 | 122.45 | 71.95 | 69.37 | 75.53 | 72.32 |
| 64 MB | 325.81 | 322.47 | 209.70 | 198.60 | 202.69 | 199.27 |
| 256 MB | 1122.35 | 1119.39 | 727.52 | 701.16 | 723.62 | 719.97 |

256 MB 时的 busbw(GB/s,1e9 字节,原样记录):AR 418.6 / 419.7(graph),AG 322.9 / 335.0,RS 324.6 / 326.2。

对照 Trn2 nccom-test 的 AR(64 rank / 8 rank,µs):16 KB 22.2 / 10.3,64 KB 22.8 / 18.0,256 KB 27.2 / 24.8,2 MB 78.0 / 50.6,16 MB 259.6,64 MB 657.2,256 MB 1,726.4。

## 2. HBM 带宽
- 8 张卡的 D2D copy 都是 **3.03 TB/s**(读+写,4 GiB buffer,30 次取平均,宿主机 torch 2.14.1+cu130)。见 `hbm/hbm_h100.txt`。

## 3. decode 随并发的变化
D-only 测法:先把 80 个 prompt 各发一次(输出 1 个 token,C=8)存进前缀缓存;然后每个 C 用同一个 `cache_salt` 重放前 10×C 个请求。输出 1200 token,`--ignore-eos`,temperature 0.9,seed 0,2 次预热,`--request-rate inf`。完整输出、JSON 和请求前后的 `/metrics` 快照都在 `decode_sweep/<A|B>/`,表格在 `decode_sweep/summary.md`。
- 表里的"前缀命中率"是这一轮 `vllm:prefix_cache_hits / queries` 的增量。预热那一轮是 91.0%(前缀在第一个请求之后才进缓存)。
- MTP 每步 token 数 = 1 + accepted / drafts(`vllm:spec_decode_*` 的增量);接受率 = accepted / draft_tokens。
- C=16 只发了 80 个请求(c8 文件总共 80 条,10×16 被截到 80)。

### 配置 A(BF16 KV,不开 MTP)
| C | TTFT 均值/P99 | TPOT 均值/P99 | ITL 中位数/P99 | 输出 tok/s | MTP 每步 token | 前缀命中率 |
|---|---|---|---|---|---|---|
| 1 | 272 / 286 ms | 11.10 / 11.11 ms | 11.12 / 11.56 ms | 88 | 1(无 MTP) | 100.0% |
| 4 | 304 / 676 ms | 12.97 / 13.01 ms | 12.99 / 13.83 ms | 302 | 1 | 100.0% |
| 8 | 340 / 1262 ms | 17.14 / 17.23 ms | 17.16 / 24.60 ms | 458 | 1 | 100.0% |
| 16 | 531 / 2538 ms | 22.85 / 23.00 ms | 22.86 / 68.52 ms | 683 | 1 | 100.0% |

### 配置 B(FP8 KV + MTP k=3)
| C | TTFT 均值/P99 | TPOT 均值/P99 | ITL 中位数/P99 | 输出 tok/s | MTP 每步 token(接受率) | 前缀命中率 |
|---|---|---|---|---|---|---|
| 1 | 293 / 313 ms | 6.20 / 7.42 ms | 14.06 / 14.67 ms | 155 | 2.28(42.6%) | 100.0% |
| 4 | 310 / 688 ms | 8.26 / 9.94 ms | 18.77 / 23.02 ms | 458 | 2.28(42.6%) | 100.0% |
| 8 | 361 / 1295 ms | 11.16 / 13.88 ms | 25.02 / 66.39 ms | 675 | 2.24(41.3%) | 100.0% |
| 16 | 583 / 2573 ms | 14.76 / 18.77 ms | 33.71 / 78.56 ms | 999 | 2.26(42.1%) | 100.0% |

- B 在 C=8 时 TPOT 11.16 ms、ITL 中位数 25.0 ms,和你同事的 11.0 / 约 25 ms 一致。A 在 C=8 时 TPOT 17.14 ms,和 README 里 H100 1P1D 的 D-only 17.09 ms 一致。
- 开了 MTP 以后,`vllm bench serve` 的 ITL 是按流式返回的每个 chunk 算的(一个 chunk 里有多个 token),所以 ITL 中位数 ≈ 一个 decode step(1 次 target 前向 + 3 次 draft 前向)。
- MTP 接受数是 2.24–2.28 token/步(temperature 0.9)。README 里 H100 1P1D 写的是 3.12–3.15,两者的测法或条件可能不同,我没有深究,只原样记录。

## 4. decode C=8 每步拆分
方法 A(vLLM 自带的 torch profiler)可以用:graph 内部的 kernel 都记录下来了(A 每个 rank 38.8 万个 kernel、178 次 `cudaGraphLaunch`;B 每个 rank 30.9 万个 kernel、363 次),所以没有用 nsys。抓取方式:C=8 的 D-only 重放进入稳定状态后(8 个请求都在跑,再等 30 秒),start_profile → sleep 3 → stop_profile。

- 用的 step kernel:
  - **A**:`^_gumbel_sample_kernel$`(采样,graph 外,每步 1 次)。抓到 178 步,窗口 3093.5 ms / 178 = 17.38 ms,ITL 中位数 17.16 ms,吻合。
  - **B**:`^_rejection_kernel$`(target 前向之后的拒绝采样,每步 1 次)。抓到 121 步,窗口 3078.0 ms / 121 = 25.44 ms,ITL 中位数 25.02 ms,吻合。所以 B 的"每步"= 1 次 target 前向 + 3 次 draft 前向 + 采样和拒绝采样。
- 8 个 rank 的结果差别在 ±2% 以内(见 `prof_decode/<A|B>/summary.txt`)。下表用 rank0。
- 表里有两套分类:"脚本分类"是 `prof_summary.py` 按名字正则分的原始结果;"细分"是 `prof_detail.py` 的补充分类,规则如下:
  - `tensorrt_llm::kernels::cutlass_kernels::*` 和 `cutlass GemmUniversal<GroupProblemShape>`(grouped GEMM)归到 moe。依据:log 里写着 `Using FLASHINFER_CUTLASS Fp8 MoE backend`,而且这些 kernel 每次前向正好 79 次,和 MoE 层数一致(attention 是 80 次,所以 80 层里有 1 层是 dense)。脚本分类把它们算进了 gemm,MoE 被明显低估。
  - 只靠名字分不清的 kernel 单独列出来,没有硬分:`cutlass_80_simt_sgemm_*` + `cublasLt::splitKreduce_kernel`(FP32 SIMT GEMM,每个 MoE 层 1 次,**可能**是 router gate,没验证);以及 Inductor 生成的 `triton_*_fused_*`(融合了 to_copy、clamp、reciprocal、silu、add 等)。

### 配置 A(C=8,8 个 token/步)
| 类别 | 脚本分类 每步 ms(占 kernel 总和) | 细分 每步 ms(占比) | 说明 |
|---|---|---|---|
| attention | 5.90(32.7%) | 5.90(32.7%) | FA3 decode 80 次/步,平均 67.0 µs,外加 FlashAttnFwdCombine 和 reshape_and_cache |
| moe | 0.87(4.8%) | **6.71(37.1%)** | grouped GEMM 2×79 次/步,共 5.24 ms;topk 0.61 ms;expand/strides/activation/expert-maps 约 0.86 ms |
| gemm(非 MoE) | 8.41(46.6%) | 1.55(8.6%) | 细分:`cutlass_3x_gemm_sm90_fp8` 320 次/步(qkv、o_proj、共享专家等)1.50 ms,lm_head(nvjet)0.05 ms |
| comm | 1.70(9.4%) | 1.70(9.4%) | trtllm oneshot lamport AR 161 次/步,中位数 9.9 µs;NCCL AllGather(RING_LL)1 次/步,22.5 µs |
| sampling | 0.03 | 0.03 | |
| elementwise | 0.65(3.6%) | 0.13 | |
| other | 0.49(2.7%) | 0.22 | |
| unclear: FP32 SIMT sgemm(可能是 router) | — | 0.91(5.1%) | 79 次/步,平均 9.0 µs,外加 splitKreduce 2.6 µs |
| unclear: Inductor triton fused | — | 0.90(5.0%) | |
| GPU 空闲 | 0.59 ms(窗口的 3.4%) | | 1 − busy(busy 97%) |
| 合计 / ITL | kernel 总和 18.07 ms/步;窗口 17.38 ms/步;ITL 中位数 17.16 ms(不开 profiler) | | kernel 总和比窗口长,说明有多个 stream 的 kernel 在并行 |

- A 每步只有 **1 个 CUDA graph**(2150 个 kernel,17.90 ms);graph 外的 eager kernel(lm_head、AllGather、采样)每步只有 0.17 ms。
- 开 profiler 时:ITL 中位数 17.16 ms,P99 25.35 ms;按 trace 算是 17.38 ms/步。**但 TPOT 均值 61.0 ms、P99 455.6 ms 不能用**:`stop_profile` 要等 trace 写完,用了约 8.7 分钟,这段时间 engine 是停住的,所有正在跑的请求都卡在那里。完整输出见 `decode_sweep/A_prof/`。

### 配置 B(C=8,MTP k=3:target 每次前向 8×4 = 32 个 token)
| 类别 | 脚本分类 每步 ms(占 kernel 总和) | 细分 每步 ms(占比) | 说明 |
|---|---|---|---|
| attention | 6.70(25.8%) | 6.70(25.8%) | FA3 83 次/步(80 次 target + 3 次 draft),平均 71.7 µs;FP8 KV |
| moe | 0.84(3.2%) | **11.46(44.1%)** | grouped GEMM 80 + 82 次/步(两个 kernel 变体),共 9.87 ms(target 79 层 + draft 3 次) |
| gemm(非 MoE) | 13.69(52.7%) | 1.94(7.5%) | |
| comm | 2.64(10.2%) | 2.64(10.2%) | target:AR 161 次/步,中位数 14.1 µs;draft:AR 9 次/步;AllGather 4 次/步 |
| sampling | 0.13 | 0.13 | |
| elementwise | 1.36(5.2%) | 0.73 | |
| other | 0.62(2.4%) | 0.29 | |
| unclear: FP32 SIMT sgemm(可能是 router) | — | 1.02(3.9%) | |
| unclear: Inductor triton fused | — | 1.08(4.1%) | |
| GPU 空闲 | 0.89 ms(窗口的 3.5%) | | busy 97% |
| 合计 / ITL | kernel 总和 25.99 ms/步;窗口 25.44 ms/步;ITL 中位数 25.02 ms(不开 profiler) | | |

**target 和 draft 分开**(按 kernel 的 `graph id` 拆,rank0,每步 ms):

| 部分 | graph | kernel 数/次 | 每步 ms | 细分 |
|---|---|---|---|---|
| target 前向(80 层,32 个 token) | g25226,每步 1 次 | 2150 | **23.96** | moe 11.15,attention 6.45,comm 2.44,unclear 2.02,gemm 1.59,other 0.20,elementwise 0.10 |
| draft 前向 1(32 个 token,推断) | g25373,每步 1 次 | 123 | 0.66 | AR 3 次,中位数 13.4 µs;AG 1 次 |
| draft 前向 2+3(每次 8 个 token,推断) | g25529,每步 1 次 | 240 | 1.09 | AR 6 次,中位数 7.1 µs;AG 2 次 |
| graph 外(采样、拒绝采样、准备输入) | eager | — | 0.29 | |

"draft 1 处理 32 个 token、draft 2/3 各 8 个 token"是根据 kernel 数和 AR 耗时推断的(13.4 µs 和 target 的 256 KB AR 相近,7.1 µs 接近 64 KB 的耗时),没有从代码上确认。

- 开 profiler 时:ITL 中位数 25.02 ms,P99 67.98 ms;按 trace 算是 25.44 ms/步;MTP 2.244 token/步(这是整轮 benchmark 的值)。TPOT 均值 57.0 ms、P99 471.5 ms 同样被 `stop_profile` 的停顿(约 9.2 分钟)污染了,不能用。见 `decode_sweep/B_prof/`。

### comm kernel 中位数 vs nccl-tests 同大小(µs)
| | 消息大小 | trace 里的 kernel | trace 中位数 | nccl AR | nccl AR(graph) | Trn2 AR 64 rank / 8 rank |
|---|---|---|---|---|---|---|
| A,C=8 | 8 × 4096 × 2 B = 64 KB | flashinfer trtllm `allreduce_fusion_kernel_oneshot_lamport` | **9.9** | 50.08 | 17.94 | 22.8 / 18.0 |
| B,C=8 target | 32 × 4096 × 2 B = 256 KB | 同上 | **14.1** | 53.56 | 19.99 | 27.2 / 24.8 |

- decode 的 AR **没有走 NCCL**,走的是 flashinfer 的 trtllm one-shot lamport kernel,而且**和 RMSNorm 融合在一起**:每步有 79 次 `AllReduceFusionPattern 1`(AR + 残差加 + RMSNorm)、81 次 `pattern 2`(AR + 残差加 + RMSNorm + FP8 量化)、1 次 `pattern 0`(纯 AR)。依据是 `flashinfer/comm/trtllm_ar.py` 里的枚举定义。所以 trace 里的 comm 时间**包含了 RMSNorm 和 FP8 量化**,比纯通信要多一点。
- 每步只有 lm_head 输出的 AllGather 走 NCCL(`ncclDevKernel_AllGather_RING_LL`),每次约 22 µs。
- 同样大小下,trtllm 融合 AR 比 nccl-tests 的 graph 模式还快约 1.4–1.8 倍,比普通模式快 4–5 倍。nccl-tests 普通模式的 50 µs 主要是 kernel 启动和同步的开销。

### top 20 kernel(rank0,按每步 ms 排序;完整的 top 60 见 `prof_decode/<A|B>/detail_rank0.txt`,脚本原版的 top 40 见 `summary_pass1.txt`)
#### 配置 A(rank0)
| # | 每步次数 | 每步 ms | 平均 µs | graph id | 脚本分类 | 细分分类 | kernel(截断) |
|---|---|---|---|---|---|---|---|
| 1 | 80.00 | 5.363 | 67.0 | g24935 | attention | attention | `cutlass::device_kernel<flash::enable_sm90_or_later<flash::FlashAttnFwdSm90<flash::CollectiveMainloop` |
| 2 | 79.00 | 2.950 | 37.3 | g24935 | gemm | moe | `cutlass GemmUniversal<GroupProblemShape,...> (grouped GEMM)` |
| 3 | 79.00 | 2.291 | 29.0 | g24935 | gemm | moe | `cutlass GemmUniversal<GroupProblemShape,...> (grouped GEMM)` |
| 4 | 81.00 | 0.870 | 10.7 | g24935 | comm | comm | `flashinfer::trtllm_allreduce_fusion::allreduce_fusion_kernel_oneshot_lamport<(flashinfer::trtllm_all` |
| 5 | 159.00 | 0.832 | 5.2 | g24935 | gemm | gemm | `cutlass::device_kernel<vllm::cutlass_3x_gemm_sm90_fp8<cutlass::float_e4m3_t, cutlass::bfloat16_t, vl` |
| 6 | 79.00 | 0.803 | 10.2 | g24935 | comm | comm | `flashinfer::trtllm_allreduce_fusion::allreduce_fusion_kernel_oneshot_lamport<(flashinfer::trtllm_all` |
| 7 | 79.00 | 0.708 | 9.0 | g24935 | gemm | unclear(fp32 simt sgemm, 1/M | `cutlass::Kernel2<cutlass_80_simt_sgemm_64x64_8x5_tn_align1>(cutlass_80_simt_sgemm_64x64_8x5_tn_align` |
| 8 | 161.00 | 0.668 | 4.1 | g24935 | gemm | gemm | `cutlass::device_kernel<vllm::cutlass_3x_gemm_sm90_fp8<cutlass::float_e4m3_t, cutlass::bfloat16_t, vl` |
| 9 | 79.00 | 0.613 | 7.8 | g24935 | moe | moe | `vllm::moe::single_group_topk::detail::single_group_topk_warp_kernel<float, float, int, (vllm::moe::S` |
| 10 | 80.00 | 0.360 | 4.5 | g24935 | attention | attention | `cutlass::device_kernel<flash::FlashAttnFwdCombine<cute::tuple<cute::C<8>, cute::C<128> >, 5, 256, 1,` |
| 11 | 158.00 | 0.288 | 1.8 | g24935 | elementwise | unclear(triton fused, Induct | `triton_poi_fused__to_copy_clamp_mul_reciprocal_0` |
| 12 | 79.00 | 0.261 | 3.3 | g24935 | moe | moe | `tensorrt_llm::kernels::cutlass_kernels::fusedBuildExpertMapsSortFirstTokenKernel<32, 8, 8>(int const` |
| 13 | 79.00 | 0.206 | 2.6 | g24935 | gemm | unclear(fp32 simt sgemm, 1/M | `cublasLt::splitKreduce_kernel<32, 16, int, float, float, float, float, false, float, float, float, t` |
| 14 | 79.00 | 0.205 | 2.6 | g24935 | gemm | moe | `tensorrt_llm::kernels::cutlass_kernels::expandInputRowsKernel<__nv_fp8_e4m3, __nv_fp8_e4m3, (tensorr` |
| 15 | 79.00 | 0.204 | 2.6 | g24935 | gemm | moe | `tensorrt_llm::kernels::cutlass_kernels::computeStridesTmaWarpSpecializedKernel<__nv_fp8_e4m3, __nv_f` |
| 16 | 79.00 | 0.200 | 2.5 | g24935 | other | other | `vllm::scaled_fp8_quant_kernel_strided_group_shape<c10::BFloat16, c10::Float8_e4m3fn, true, true>(c10` |
| 17 | 79.00 | 0.188 | 2.4 | g24935 | gemm | moe | `tensorrt_llm::kernels::cutlass_kernels::doActivationKernel<__nv_fp8_e4m3, __nv_bfloat16, __nv_bfloat` |
| 18 | 80.00 | 0.177 | 2.2 | g24935 | attention | attention | `vllm::reshape_and_cache_flash_kernel<__nv_bfloat16, __nv_bfloat16, (vllm::Fp8KVCacheDataType)0>(__nv` |
| 19 | 78.00 | 0.147 | 1.9 | g24935 | other | unclear(triton fused, Induct | `triton_red_fused_3` |
| 20 | 79.00 | 0.135 | 1.7 | g24935 | elementwise | unclear(triton fused, Induct | `triton_poi_fused_mul_silu_slice_0` |

#### 配置 B(rank0)
| # | 每步次数 | 每步 ms | 平均 µs | graph id | 脚本分类 | 细分分类 | kernel(截断) |
|---|---|---|---|---|---|---|---|
| 1 | 83.00 | 5.948 | 71.7 | g25226,25373, | attention | attention | `cutlass::device_kernel<flash::enable_sm90_or_later<flash::FlashAttnFwdSm90<flash::CollectiveMainloop` |
| 2 | 80.00 | 5.641 | 70.5 | g25226,25373 | gemm | moe | `cutlass GemmUniversal<GroupProblemShape,...> (grouped GEMM)` |
| 3 | 82.00 | 4.225 | 51.5 | g25226,25373, | gemm | moe | `cutlass GemmUniversal<GroupProblemShape,...> (grouped GEMM)` |
| 4 | 81.00 | 1.329 | 16.4 | g25226 | comm | comm | `flashinfer::trtllm_allreduce_fusion::allreduce_fusion_kernel_oneshot_lamport<(flashinfer::trtllm_all` |
| 5 | 79.00 | 1.096 | 13.9 | g25226 | comm | comm | `flashinfer::trtllm_allreduce_fusion::allreduce_fusion_kernel_oneshot_lamport<(flashinfer::trtllm_all` |
| 6 | 161.00 | 0.900 | 5.6 | g25226,25373 | gemm | gemm | `cutlass::device_kernel<vllm::cutlass_3x_gemm_sm90_fp8<cutlass::float_e4m3_t, cutlass::bfloat16_t, vl` |
| 7 | 163.00 | 0.713 | 4.4 | g25226,25373 | gemm | gemm | `cutlass::device_kernel<vllm::cutlass_3x_gemm_sm90_fp8<cutlass::float_e4m3_t, cutlass::bfloat16_t, vl` |
| 8 | 80.00 | 0.685 | 8.6 | g25226,25373 | gemm | unclear(fp32 simt sgemm, 1/M | `cutlass::Kernel2<cutlass_80_simt_sgemm_128x64_8x5_tn_align1>(cutlass_80_simt_sgemm_128x64_8x5_tn_ali` |
| 9 | 82.00 | 0.562 | 6.9 | g25226,25373, | moe | moe | `vllm::moe::single_group_topk::detail::single_group_topk_warp_kernel<float, float, int, (vllm::moe::S` |
| 10 | 83.00 | 0.398 | 4.8 | g25226,25373, | attention | attention | `cutlass::device_kernel<flash::FlashAttnFwdCombine<cute::tuple<cute::C<8>, cute::C<128> >, 5, 256, 1,` |
| 11 | 83.00 | 0.350 | 4.2 | g25226,25373, | attention | attention | `vllm::reshape_and_cache_flash_kernel<__nv_bfloat16, unsigned char, (vllm::Fp8KVCacheDataType)1>(__nv` |
| 12 | 82.00 | 0.313 | 3.8 | g25226,25373, | gemm | unclear(fp32 simt sgemm, 1/M | `cublasLt::splitKreduce_kernel<32, 16, int, float, float, float, float, false, float, float, float, t` |
| 13 | 170.00 | 0.285 | 1.7 | g25226,25373, | elementwise | unclear(triton fused, Induct | `triton_poi_fused__to_copy_clamp_mul_reciprocal_0` |
| 14 | 82.00 | 0.279 | 3.4 | g25226,25373, | moe | moe | `tensorrt_llm::kernels::cutlass_kernels::fusedBuildExpertMapsSortFirstTokenKernel<32, 8, 8>(int const` |
| 15 | 82.00 | 0.239 | 2.9 | g25226,25373, | gemm | moe | `tensorrt_llm::kernels::cutlass_kernels::expandInputRowsKernel<__nv_fp8_e4m3, __nv_fp8_e4m3, (tensorr` |
| 16 | 82.00 | 0.233 | 2.8 | g25226,25373, | elementwise | unclear(triton fused, Induct | `triton_poi_fused_mul_silu_slice_0` |
| 17 | 82.00 | 0.226 | 2.8 | g25226,25373, | gemm | moe | `tensorrt_llm::kernels::cutlass_kernels::computeStridesTmaWarpSpecializedKernel<__nv_fp8_e4m3, __nv_f` |
| 18 | 82.00 | 0.211 | 2.6 | g25226,25373, | gemm | moe | `tensorrt_llm::kernels::cutlass_kernels::doActivationKernel<__nv_fp8_e4m3, __nv_bfloat16, __nv_bfloat` |
| 19 | 82.00 | 0.208 | 2.5 | g25226,25373, | other | other | `vllm::scaled_fp8_quant_kernel_strided_group_shape<c10::BFloat16, c10::Float8_e4m3fn, true, true>(c10` |
| 20 | 78.00 | 0.196 | 2.5 | g25226 | other | unclear(triton fused, Induct | `triton_red_fused_3` |

## 5. prefill(可选,只在配置 B 上做了)
同一个 B server,C=1,每次 1 个请求,输出 1 个 token,请求格式和 benchmark 一样(chat、temperature 0.9、seed 0、带 `cache_salt`),脚本是 `prefill_req.py`。TTFT 由客户端从流式响应里测。

| 请求 | 前缀命中 | TTFT(不开 profiler) | TTFT(开 profiler) | trace GPU 窗口(rank0) | 主要类别(脚本分类,rank0,ms) |
|---|---|---|---|---|---|
| 85k,92% 前缀命中(salt `prefill-B`,先发 row 1 存前缀,再测 row 2 / row 3) | 78,320 / 85,015 = 92.1% | 791.7 ms(row 2) | 808.6 ms(row 3) | 554.2 ms,busy 97%,3125 个 kernel,只有 1 个 chunk | attention 326.7,gemm 130.2(其中 MoE grouped GEMM 72.4),comm 37.6,elementwise 27.9,other 9.3,moe 4.2 |
| 85k 冷启动(新 salt,row 4) | 0% | 4674.7 ms(row 1,新 salt 第一次发,等于一次冷启动) | 5098.3 ms | 4857.4 ms,busy 99%,11 个 chunk(8192 token/chunk) | attention 1852.1,gemm 1548.2(其中 MoE grouped GEMM 827.5,FP32 xmma sgemm 263.4),comm 880.4,elementwise 353.6,other 116.8,moe 50.0 |

- prefill 的 AR 走的是 `multimem_all_reduce_kernel`(SYMM_MEM / NVLS),不是 trtllm 的 kernel。92% 命中时每次中位数约 225 µs(约 6.7k token × 8 KB ≈ 55 MB,164 次);冷启动时 1804 次,中位数 262 µs,P90 1207 µs。
- 92% 命中时,TTFT 808 ms 比 GPU 窗口 554 ms 多出约 250 ms,这部分在 GPU 之外(85k token 的 tokenize、调度、HTTP)。
- 汇总文件:`prof_prefill/summary.txt`、`prof_prefill_cold/summary.txt`;请求日志 `requests.log` 里有前后的命中计数。
- 冷启动那次的 `profiler_out_*.txt`(key_averages 表)没有出现在目录里,trace 文件是完整的,不影响汇总。

## 6. 问题和偏差
- **没完成的项目**:无。配置 A、B 都做完了扫描(C=1/4/8/16)和 C=8 的 profile;prefill 只在 B 上做。方法 A 就能看到 graph 内部的 kernel,所以没有用 nsys。
- **和本说明不一致的地方**:
  1. 配置 B 的参数没有另外收到,经你确认用的是仓库里的 `scripts/pd/gpu/gpu_single_serve.sh`(MAX_LEN=87040,IMAGE=vllm/vllm-openai:v0.29.0,不开 OFFLOAD)。
  2. 配置 A 是单机,不做 PD:去掉了 `gpu_serve.sh` 里的 `--kv-transfer-config`(MooncakeConnector)、`VLLM_MOONCAKE_BOOTSTRAP_PORT`、`FI_*`、`/dev/gdrdrv`;端口改成 8000;镜像用的是 `vllm/vllm-openai:v0.29.0`,不是 `hy3-efa-029`。保留了 `--privileged`(B 和 `gpu_single_serve.sh` 一样没有这个参数)。
  3. 除了 profiler 参数,两个配置的 docker 命令还额外挂了 `-v /opt/dlami/nvme/vllm_cache:/root/.cache`(只用来缓存编译结果)和 profiler 目录。没有改 vLLM 的任何参数。
  4. D-only 请求数按你的要求用 10×C。原脚本 `pd_bench_dcust.sh` 是 max(40, 10×C),所以 C=1 只发了 10 个请求(原脚本是 40 个)。C=16 被截到 80 个。benchmark 用的是改成单 server 版本的 `bench_donly.sh`(P_URL = D_URL = PROXY = :8000),`vllm bench serve` 的参数和原脚本一样,另外额外保存了完整的 stdout 和 `/metrics` 快照。
  5. `prof_summary.py` 原文按类别排序那一行有一处语法错误 `sorted(tot.items(, key=...)`,改成了 `sorted(tot.items(), key=...)`,其他没动。另外加了补充脚本 `prof_detail.py`(按 graph 拆、细分 MoE),没有替代原脚本的输出。
  6. 第 7 节 prefill 的 profile 目录:server 的 profiler 目录一直是 `/workspace/prof_decode`,每次抓完再把 trace 移到 `prof_prefill/`、`prof_prefill_cold/`,没有重启 server。
  7. nccl-tests 用的是宿主机的 NCCL 2.30.7,和 vLLM 镜像里 pip 装的 `nvidia-nccl-cu13` 是同一个版本;torch 编译时用的是 2.29.7。decode 的 AR 本来就不走 NCCL。
  8. 开着 profiler 的那两轮,TPOT 均值和 P99 被 `stop_profile` 停住 engine 的那段时间(写 trace 用了 8.7 分钟 / 9.2 分钟)污染了,只有 ITL 中位数和 trace 里的每步时间可以用。
  9. 第一次跑 A 的扫描时,client 镜像缺 `vllm[bench]` 依赖,全部失败(`decode_sweep/sweep_A_failed_missing_bench_deps.log`)。修好后重新跑了全部,没有数据丢失。
- trace 原始文件:`~/h100_bench_traces.tar.gz`(1.7 GB,36 个 `*.pt.trace.json.gz`,解压后路径保持 `h100_bench/...`);`~/h100_bench/` 里也还留着一份。

## 文件索引
`env.txt`,`serve_cmd_A.txt`,`serve_cmd_B.txt`,`server_A.log`,`server_B.log`,`nccl/`(`nccl_h100.txt`、`nccl_table.md`、`busbw.txt`),`hbm/hbm_h100.txt`,`decode_sweep/`(`A/`、`B/`、`A_prof/`、`B_prof/`、`summary.md`、`sweep_*.log`),`prof_decode/{A,B}/`(`summary_pass1.txt`、`summary.txt`、`detail_rank0.txt`、`profiler_out_*.txt`、`prof_*_run.log`),`prof_decode/top20.md`,`prof_prefill/`,`prof_prefill_cold/`,脚本(`bench_donly.sh`、`prof_decode_run.sh`、`prof_summary.py`、`prof_detail.py`、`prefill_req.py`、`sweep_table.py`、`serve_A.sh`、`serve_B.sh`)。

**测试已完成,请停止 p5 实例。**(按小时计费。当前 `hy3-single` 容器里还跑着配置 B 的 server。)
