
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
