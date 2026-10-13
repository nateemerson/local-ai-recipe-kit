#!/usr/bin/env python3
"""sweep.py's campaign protocol against an OpenAI-compatible server (vLLM) instead of SGLang's native API.

Same measurements, same prompt shapes, same JSON layout as sweep.py (see its docstring and PROTOCOL.md); only the
transport differs:
  - context length from GET /v1/models (max_model_len) instead of /server_info
  - prompts are still token ids: prefill uses random ids; decode prompts are the sweep's topic requests rendered with
    the MODEL'S OWN chat template via POST /tokenize {"messages": ...} (sweep.py hard-codes a Qwen template)
  - requests go to POST /v1/completions with "prompt": [ids], streamed; completion tokens come from the final usage
    chunk (stream_options.include_usage), so multi-token chunks (speculative decoding) are counted correctly
  - max_tokens = remaining context, as in sweep.py (uncapped: completions run to natural EOS)
"""
import argparse, json, os, random, statistics, sys, threading, time, urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sweep import TOPICS, Power, med  # noqa: E402


def rand_ids(n, seed):
    # sweep.py draws ids from 1000-150000; some vocabularies are smaller (Nemotron 131,072, LFM2 65,536), so stay below 60,000
    rng = random.Random(seed)
    return [rng.randrange(1000, 60000) for _ in range(n)]

MODEL = None


def _req(url, path, payload, timeout):
    return urllib.request.urlopen(urllib.request.Request(url + path, data=json.dumps(payload).encode(),
                                                         headers={"Content-Type": "application/json"}), timeout=timeout)


def post(url, ids, max_tokens, first_only=False, timeout=7200):
    payload = {"model": MODEL, "prompt": ids, "max_tokens": max_tokens, "temperature": 0, "stream": True,
               "stream_options": {"include_usage": True}}
    t0 = time.perf_counter(); times = []; usage = None; finish = None
    with _req(url, "/v1/completions", payload, timeout) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            body = line[5:].strip()
            if body == "[DONE]":
                break
            ev = json.loads(body)
            if ev.get("error") or ev.get("object") == "error":
                raise RuntimeError(f"server error in stream: {str(ev)[:300]}")
            if ev.get("usage"):
                usage = ev["usage"]
            for ch in ev.get("choices") or []:
                # a 1-token reply can arrive as a chunk with empty text (special/reasoning token): it still marks TTFT
                if ch.get("text") or not times:
                    times.append(time.perf_counter())
                if ch.get("finish_reason"):
                    finish = ch["finish_reason"]
            if first_only and times:
                break
    n = (usage or {}).get("completion_tokens", len(times))
    return t0, times, n, finish


NONCE = __import__("secrets").token_hex(16)
PROMPT_HASHES = []


def chat_ids(url, i):
    msg = (f"[cache-bust nonce: {NONCE}-{i}]\nWrite a detailed, well-structured explanation of "
           f"{TOPICS[i % len(TOPICS)]} (variant {i}).")
    PROMPT_HASHES.append(__import__("hashlib").sha256(msg.encode()).hexdigest())
    with _req(url, "/tokenize", {"model": MODEL, "messages": [{"role": "user", "content": msg}],
                                 "add_generation_prompt": True}, 60) as r:
        return json.loads(r.read())["tokens"]


def main():
    global MODEL
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--card", required=True)
    ap.add_argument("--config", required=True, help="label: image digest + knobs")
    ap.add_argument("--prefill", type=int, nargs="*", default=[8192, 16384, 32768, 65536])
    ap.add_argument("--conc", type=int, nargs="*", default=[1, 2, 3, 4])
    ap.add_argument("--high", type=int, nargs="*", default=[])
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--dec-reps", type=int, default=2)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    m = json.loads(urllib.request.urlopen(a.url + "/v1/models", timeout=30).read())["data"][0]
    MODEL, ctx_len = m["id"], int(m.get("max_model_len") or 131072)
    res = {"card": a.card, "config": a.config, "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "ctx_len": ctx_len,
           "protocol": "bench/sweep_oai.py v1 (sweep.py v1 protocol over the OpenAI API)", "model": MODEL,
           "prefill": {}, "decode": {}, "status": "RUNNING"}

    def save():
        json.dump(res, open(a.out, "w"), indent=1)

    seed = int(time.time())
    post(a.url, rand_ids(512, seed), 1, True)
    for n in a.prefill:
        if n + 64 > ctx_len:
            res["prefill"][str(n)] = {"skipped": f"> context {ctx_len}"}; continue
        rates = []
        for rep in range(a.reps):
            t0, times, _, _ = post(a.url, rand_ids(n, seed + 1000 * n + rep), 1, True)
            rates.append(round(n / (times[0] - t0), 1))
            print(f"prefill {n}: {rates[-1]} tok/s", flush=True)
        res["prefill"][str(n)] = med(rates); save()

    def run_conc(c, rnd, prefix=None):
        out = [None] * c
        def one(i):
            ids = (prefix or []) + chat_ids(a.url, rnd * 100 + i)
            t0, times, n, finish = post(a.url, ids, max(64, (ctx_len - len(ids)) // max(c, 1) - 64))
            out[i] = {"t0": t0, "first": times[0], "last": times[-1], "tokens": n, "finish": finish}
        ths = [threading.Thread(target=one, args=(i,)) for i in range(c)]
        [t.start() for t in ths]; [t.join() for t in ths]
        tot = sum(o["tokens"] for o in out)
        span = max(o["last"] for o in out) - min(o["first"] for o in out)
        per = [(o["tokens"] - 1) / (o["last"] - o["first"]) for o in out if o["tokens"] > 1 and o["last"] > o["first"]]
        return {"aggregate": round(tot / span, 2), "per_stream_mean": round(statistics.mean(per), 2),
                "per_stream_min": round(min(per), 2), "tokens": tot,
                "finish": sorted({o["finish"] for o in out if o["finish"]}), "ttft_max": round(max(o["first"] - o["t0"] for o in out), 2)}

    for c in list(a.conc) + list(a.high):
        with Power() as pw:
            rounds = [run_conc(c, r) for r in range(a.dec_reps)]
        agg = [r["aggregate"] for r in rounds]
        res["decode"][f"C{c}"] = {"aggregate": med(agg), "per_stream_mean": med([r["per_stream_mean"] for r in rounds]),
                                   "rounds": rounds, "gpu_power_w_mean": pw.mean()}
        print(f"decode C{c}: aggregate {statistics.median(agg):.2f} tok/s, per-stream "
              f"{statistics.median([r['per_stream_mean'] for r in rounds]):.2f}", flush=True)
        save()
    pre = rand_ids(32768, seed + 99) if ctx_len > 40000 else None
    if pre:
        r = [run_conc(1, 90 + i, prefix=pre) for i in range(a.dec_reps)]
        res["decode"]["C1@32k"] = {"aggregate": med([x["aggregate"] for x in r]), "rounds": r}
        print(f"decode C1@32k: {statistics.median([x['aggregate'] for x in r]):.2f} tok/s", flush=True)
    res["prompt_sha256"] = PROMPT_HASHES; res["nonce_prefix"] = NONCE
    res["status"] = "DONE"; res["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z"); save()
    print("SWEEP DONE", a.out, flush=True)


if __name__ == "__main__":
    main()
