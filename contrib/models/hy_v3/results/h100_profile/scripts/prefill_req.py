"""Send one chat request (same shape as vllm bench serve --backend openai-chat --skip-chat-template:
prompt as the user message, temperature 0.9, seed 0, max_tokens 1, cache_salt), streaming; print TTFT.
Usage: prefill_req.py <row index in in85000_p78336_c8 (1 = warm-up row, 2.. = measured)> <cache_salt> [profile]
With 'profile': start_profile before, stop_profile after the response is complete."""
import gzip, json, sys, time, http.client, os
row, salt = int(sys.argv[1]), sys.argv[2]
prof = len(sys.argv) > 3 and sys.argv[3] == "profile"
f = os.path.expanduser("~/vllm-neuron/contrib/models/hy_v3/bench_data/prefix/in85000_p78336_c8.jsonl.gz")
rows = [json.loads(l) for l in gzip.open(f, "rt")]
prompt = rows[0]["prefix"] + rows[row]["suffix"]
body = json.dumps({"model": "hy3", "messages": [{"role": "user", "content": prompt}], "max_tokens": 1,
                   "max_completion_tokens": 1, "temperature": 0.9, "seed": 0, "stream": True,
                   "stream_options": {"include_usage": True}, "ignore_eos": True, "cache_salt": salt})
def post(path, b=None, timeout=3600):
    c = http.client.HTTPConnection("127.0.0.1", 8000, timeout=timeout)
    c.request("POST", path, body=b, headers={"Content-Type": "application/json"})
    return c, c.getresponse()
if prof:
    t = time.time(); _, r = post("/start_profile"); print("start_profile", r.status, f"{time.time()-t:.2f}s", flush=True)
t0 = time.perf_counter(); ttft = None; usage = None
c, r = post("/v1/chat/completions", body)
for line in r:
    line = line.strip()
    if not line.startswith(b"data:") or line == b"data: [DONE]": continue
    d = json.loads(line[5:])
    if ttft is None and d.get("choices") and (d["choices"][0].get("delta", {}).get("content") or d["choices"][0].get("delta", {}).get("reasoning_content") or d["choices"][0].get("finish_reason")):
        ttft = time.perf_counter() - t0
    if d.get("usage"): usage = d["usage"]
e2e = time.perf_counter() - t0
print(f"row {row} salt {salt}: TTFT {ttft*1e3:.1f} ms, e2e {e2e*1e3:.1f} ms, usage {usage}", flush=True)
if prof:
    t = time.time(); _, r = post("/stop_profile"); print("stop_profile", r.status, f"{time.time()-t:.1f}s", flush=True)
