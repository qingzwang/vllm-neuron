import json, os, sys
base = os.path.expanduser("~/h100_bench/decode_sweep")
for cfg in sys.argv[1:]:
    print(f"\n### 配置 {cfg}\n| C | 完成 | TTFT 均值/P99 (ms) | TPOT 均值/P99 (ms) | ITL 中位数/P99 (ms) | 输出 tok/s | MTP 每步 token | MTP 接受率 | 前缀命中率 |\n|---|---|---|---|---|---|---|---|---|")
    for C in (1, 4, 8, 16):
        p = f"{base}/{cfg}/c{C}.json"
        if not os.path.exists(p): continue
        d = json.load(open(p))
        acc = f"{d['spec_accept_rate']:.1%}" if 'spec_accept_rate' in d else "-"
        print(f"| {C} | {d['completed']}/{d['num_prompts']} | {d['mean_ttft_ms']:.0f} / {d['p99_ttft_ms']:.0f} | {d['mean_tpot_ms']:.2f} / {d['p99_tpot_ms']:.2f} | "
              f"{d['median_itl_ms']:.2f} / {d['p99_itl_ms']:.2f} | {d['output_throughput']:.0f} | {d['spec_tokens_per_step']:.2f} | {acc} | {d['prefix_hit_rate']:.1%} |")
