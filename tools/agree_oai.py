#!/usr/bin/env python3
"""Backend-agreement panel for one-knob A/B checks on the SAME model behind an OpenAI-compatible server (vLLM).

Use it when you change something that should not change the model's outputs (MoE backend, attention backend, FP8 KV
cache, speculative decoding, CUDA graphs on/off, TP vs PP, a new image) and want to know how much it did.

  # config A (the baseline) running on :8000
  python3 tools/agree_oai.py --url http://127.0.0.1:8000 --record work/agree-A.json
  # relaunch with ONE knob changed, then
  python3 tools/agree_oai.py --url http://127.0.0.1:8000 --compare work/agree-A.json --out work/agree-B-vs-A.json

How it works
  1. Each prompt (reference/quality/agree_prompts.jsonl, 64 mixed-domain chats) is rendered with the model's own chat
     template through POST /tokenize {"messages": ...}, so A and B see identical token ids (the ids are stored in the
     record file and re-used verbatim by --compare; B's tokenizer is only checked, never trusted).
  2. --record: greedy generation of the first K (=128) tokens via /v1/completions with prompt = token ids, logprobs 20.
     Then A's own tokens are teacher-forced (prompt = prompt ids + A's ids, prompt_logprobs = 20, max_tokens 1) and
     those position-aligned top-20 distributions become the reference. (Teacher-forced vs decode-path agreement of
     A against itself is reported as `self_check`: it is the floor from batching / kernel paths alone.)
  3. --compare: B teacher-forces A's tokens the same way, so every position is compared under an identical prefix,
     even after B would have generated something different. Reported, like tools/score_ref_panel.py:
       top1_agreement   B argmax == A argmax at the same position
       mean_kl_top20    KL(A || B) on A's top-20 ids, both renormalised over those ids
       greedy_match     B argmax == A's generated token
       first_divergence position where B's greedy continuation would first leave A's (median over prompts)
     B only returns its own top-20; an id from A's top-20 that is missing there gets B's 20th logprob as an upper
     bound (counted in missing_support_frac; the KL is then a slight underestimate). Launch both configs with
     --max-logprobs 64 and pass --top 64 to shrink that if it matters.

Teacher forcing needs vLLM's `prompt_logprobs` (completions extra parameter; prefix caching is skipped for those
requests automatically). Servers without it (llama.cpp, most others) fail with a clear message; `--tf stepwise`
falls back to one max_tokens=1 request per position (slow: K requests per prompt, but works on any server that
returns completion logprobs).

The record file is {"_meta": ..., "panel": [...]}; each panel entry has the same keys as the kit's reference panel
(prompt, prompt_len, tokens, completion, top20_ids, top20_lp), so `jq .panel` gives a score_ref_panel.py input.
"""
import argparse, concurrent.futures as cf, json, math, os, statistics, sys, time, urllib.error, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
PROMPTS = os.path.join(HERE, "..", "reference", "quality", "agree_prompts.jsonl")
VERSION = "agree_oai v1"


class ServerError(RuntimeError):
    pass


def post(url, path, payload, timeout=600):
    req = urllib.request.Request(url + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        raise ServerError(f"HTTP {e.code} on {path}: {e.read()[:500].decode(errors='replace')}") from None


def tokenize_chat(url, model, messages, tkw=None):
    """Prompt ids via the server's chat template. vLLM: /tokenize with messages. llama.cpp: /apply-template + /tokenize."""
    try:
        r = post(url, "/tokenize", {"model": model, "messages": messages, "add_generation_prompt": True,
                                    **({"chat_template_kwargs": tkw} if tkw else {})})
        if isinstance(r.get("tokens"), list) and r["tokens"]:
            return r["tokens"], "vllm /tokenize messages"
    except ServerError:
        pass
    try:
        txt = post(url, "/apply-template", {"messages": messages, **({"chat_template_kwargs": tkw} if tkw else {})})[
            "prompt"]
        r = post(url, "/tokenize", {"content": txt, "add_special": False, "parse_special": True})
        return [t if isinstance(t, int) else t["id"] for t in r["tokens"]], "llama.cpp /apply-template + /tokenize"
    except (ServerError, KeyError) as e:
        raise ServerError(f"server cannot tokenize chat messages (needs vLLM /tokenize with messages, or llama.cpp "
                          f"/apply-template): {e}")


def _tid(key):
    """vLLM with return_tokens_as_token_ids returns 'token_id:123'; prompt_logprobs keys are '123'."""
    if isinstance(key, int):
        return key
    return int(str(key).rsplit(":", 1)[-1])


def parse_completion_logprobs(choice, top):
    """[(generated id, [(id, lp)] sorted desc)] from a /v1/completions choice; handles vLLM and llama.cpp layouts."""
    lp = choice.get("logprobs")
    if not lp:
        raise ServerError("server returned no completion logprobs (request had logprobs=%d)" % top)
    out = []
    if lp.get("content") is not None:  # llama.cpp (chat-style layout on /v1/completions)
        for e in lp["content"]:
            alts = sorted(((a["id"], a["logprob"]) for a in e.get("top_logprobs") or []), key=lambda x: -x[1])[:top]
            out.append((e["id"], alts))
        return out
    for tok, tl in zip(lp["tokens"], lp["top_logprobs"]):  # vLLM legacy completions layout
        if tl is None:
            raise ServerError("missing top_logprobs entry")
        alts = sorted(((_tid(k), v) for k, v in tl.items()), key=lambda x: -x[1])[:top]
        out.append((_tid(tok), alts))
    return out


def generate(url, model, ids, k, top, timeout):
    r = post(url, "/v1/completions", {"model": model, "prompt": ids, "max_tokens": k, "temperature": 0,
                                      "logprobs": top, "return_tokens_as_token_ids": True}, timeout)
    ch = r["choices"][0]
    return parse_completion_logprobs(ch, top), ch.get("text", ""), ch.get("finish_reason")


def teacher_force(url, model, prompt_ids, gen_ids, top, mode, timeout):
    """Per-position top-k [(id, lp)] for positions predicting gen_ids[0..n-1] given prompt + gen_ids[:t]."""
    if mode == "prompt_logprobs":
        r = post(url, "/v1/completions", {"model": model, "prompt": prompt_ids + gen_ids, "max_tokens": 1,
                                          "temperature": 0, "prompt_logprobs": top}, timeout)
        pl = r["choices"][0].get("prompt_logprobs")
        if pl is None:
            raise ServerError(
                "server ignored `prompt_logprobs` (not vLLM?). Teacher forcing in one request needs vLLM's completions "
                "`prompt_logprobs` (and --max-logprobs >= --top). Re-run with `--tf stepwise` for the slow "
                "per-position fallback that works on any server with completion logprobs.")
        P = len(prompt_ids)
        out = []
        for t in range(len(gen_ids)):
            d = pl[P + t] or {}
            vals = []
            for key, v in d.items():
                vals.append((_tid(key), v["logprob"] if isinstance(v, dict) else float(v)))
            # the dict holds the top-k plus the actual token; keep the top-k by logprob
            out.append(sorted(vals, key=lambda x: -x[1])[:top])
        return out
    out = []
    for t in range(len(gen_ids)):  # stepwise fallback
        steps, _, _ = generate(url, model, prompt_ids + gen_ids[:t], 1, top, timeout)
        out.append(steps[0][1])
    return out


def compare_dists(ref_ids, ref_lp, srv):
    """Metrics for one position: A's top-k (ids, lps) vs B's [(id, lp)]."""
    srv_map = dict(srv)
    floor = min(srv_map.values()) if srv_map else -50.0
    pr = [math.exp(x) for x in ref_lp]
    zr = sum(pr)
    missing = sum(1 for t in ref_ids if t not in srv_map)
    qs = [math.exp(srv_map.get(t, floor)) for t in ref_ids]
    zq = sum(qs)
    kl = sum((p / zr) * (math.log(p / zr) - math.log(max(q / zq, 1e-30))) for p, q in zip(pr, qs) if p > 0)
    srv_arg = max(srv, key=lambda e: e[1])[0] if srv else None
    return srv_arg, max(kl, 0.0), missing


def score(panel, dists_by_item, top):
    tot = {"pos": 0, "top1": 0, "greedy": 0, "kl": 0.0, "missing": 0}
    per, firsts = [], []
    for item, dists in zip(panel, dists_by_item):
        P = item["prompt_len"]
        gen = item["tokens"][P:]
        s = {"pos": 0, "top1": 0, "greedy": 0, "kl": 0.0, "missing": 0}
        first = None
        for j, srv in enumerate(dists):
            arg, kl, miss = compare_dists(item["top20_ids"][j], item["top20_lp"][j], srv)
            s["pos"] += 1
            s["top1"] += int(arg == item["top20_ids"][j][0])
            s["greedy"] += int(arg == gen[j])
            s["kl"] += kl
            s["missing"] += miss
            if first is None and arg != gen[j]:
                first = j
        firsts.append(first if first is not None else len(dists))
        per.append({"id": item["id"], "positions": s["pos"], "top1": round(s["top1"] / max(1, s["pos"]), 4),
                    "kl": round(s["kl"] / max(1, s["pos"]), 6), "greedy_match": round(s["greedy"] / max(1, s["pos"]), 4),
                    "first_divergence": firsts[-1]})
        for k in tot:
            tot[k] += s[k]
    n = max(1, tot["pos"])
    return {"positions": tot["pos"], "top1_agreement": round(tot["top1"] / n, 5), "mean_kl_top20": round(tot["kl"] / n, 6),
            "greedy_match": round(tot["greedy"] / n, 5), "missing_support_frac": round(tot["missing"] / (n * top), 5),
            "first_divergence_median": statistics.median(firsts) if firsts else None,
            "prompts_fully_agreeing": sum(1 for p in per if p["first_divergence"] == p["positions"]),
            "per_prompt": per}


def server_model(url, model):
    ms = json.loads(urllib.request.urlopen(url + "/v1/models", timeout=30).read())["data"]
    m = next((x for x in ms if x["id"] == model), None) if model else ms[0]
    if m is None:
        sys.exit(f"model {model} not served; have {[x['id'] for x in ms]}")
    return m["id"]


def run_pool(fn, n, conc):
    out = [None] * n
    def one(i):
        out[i] = fn(i)
    with cf.ThreadPoolExecutor(max(1, conc)) as ex:
        for f in [ex.submit(one, i) for i in range(n)]:
            f.result()  # re-raise the first error
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", help="served model id (default: first of /v1/models)")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--record", metavar="A.json", help="record config A's reference panel to this file")
    g.add_argument("--compare", metavar="A.json", help="score this server (config B) against a recorded panel")
    ap.add_argument("--out", help="--compare: result JSON (default: print only)")
    ap.add_argument("--prompts", default=PROMPTS)
    ap.add_argument("--n", type=int, default=64, help="number of prompts (record)")
    ap.add_argument("--k", type=int, default=128, help="generated positions per prompt (record)")
    ap.add_argument("--top", type=int, default=20, help="top-k logprobs (server --max-logprobs must allow it)")
    ap.add_argument("--tf", choices=["prompt_logprobs", "stepwise"], default="prompt_logprobs",
                    help="teacher-forcing method (stepwise = any server, K requests per prompt)")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--timeout", type=float, default=600)
    ap.add_argument("--label", default="", help="config label: image digest + knobs")
    ap.add_argument("--chat-template-kwargs", help='JSON for the chat template at record time, e.g. '
                    '\'{"enable_thinking": false}\' (stored in the record and reused for the template check)')
    a = ap.parse_args()
    model = server_model(a.url, a.model)
    tkw = json.loads(a.chat_template_kwargs) if a.chat_template_kwargs else None
    t0 = time.perf_counter()

    if a.record:
        rows = [json.loads(l) for l in open(a.prompts) if l.strip()]
        rows = [r for r in rows if "_meta" not in r][:a.n]
        print(f"record: {len(rows)} prompts x {a.k} tokens, model {model}", flush=True)

        def rec(i):
            r = rows[i]
            ids, how = tokenize_chat(a.url, model, r["messages"], tkw)
            steps, text, finish = generate(a.url, model, ids, a.k, a.top, a.timeout)
            gen = [s[0] for s in steps]
            dec = [s[1] for s in steps]
            try:
                tf = teacher_force(a.url, model, ids, gen, a.top, a.tf, a.timeout)
                ref, path = tf, a.tf
            except ServerError as e:
                if a.tf == "stepwise":
                    raise
                print(f"  [{r['id']}] {e}\n  -> recording the decode-path distributions as reference", flush=True)
                ref, path = dec, "decode"
            print(f"  [{r['id']}] prompt {len(ids)} tok, generated {len(gen)} ({finish}), ref={path}", flush=True)
            return {"id": r["id"], "domain": r.get("domain"), "prompt": r["messages"], "prompt_len": len(ids),
                    "tokens": ids + gen, "completion": text,
                    "top20_ids": [[t for t, _ in d] for d in ref], "top20_lp": [[round(v, 6) for _, v in d] for d in ref],
                    "decode_top_ids": [[t for t, _ in d] for d in dec], "decode_top_lp": [[round(v, 6) for _, v in d] for d in dec],
                    "ref_path": path, "tokenize": how, "finish": finish}

        panel = run_pool(rec, len(rows), a.concurrency)
        # self-check: A's decode-path distributions vs A's teacher-forced ones (noise floor of the method itself)
        self_check = None
        if all(p["ref_path"] != "decode" for p in panel):
            self_check = score(panel, [[list(zip(i, l)) for i, l in zip(p["decode_top_ids"], p["decode_top_lp"])]
                                       for p in panel], a.top)
            self_check.pop("per_prompt")
        meta = {"tool": VERSION, "model": model, "label": a.label, "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "n": len(panel), "k": a.k, "top": a.top, "tf": a.tf, "ref_path": sorted({p["ref_path"] for p in panel}),
                "prompts_file": os.path.basename(a.prompts), "chat_template_kwargs": tkw, "seconds": round(time.perf_counter() - t0, 1),
                "self_check_decode_vs_teacher_forced": self_check}
        json.dump({"_meta": meta, "panel": panel}, open(a.record, "w"))
        print(json.dumps({k: v for k, v in meta.items()}), flush=True)
        print("RECORDED", a.record, flush=True)
        return

    rec = json.load(open(a.compare))
    meta, panel = rec["_meta"], rec["panel"]
    if meta["model"] != model:
        print(f"warning: panel recorded on model {meta['model']!r}, server serves {model!r}", flush=True)
    top = meta["top"]
    # tokenizer/template drift check (ids themselves are always A's)
    try:
        ids_b, _ = tokenize_chat(a.url, model, panel[0]["prompt"], meta.get("chat_template_kwargs"))
        tmpl_same = ids_b == panel[0]["tokens"][:panel[0]["prompt_len"]]
    except ServerError:
        tmpl_same = None
    print(f"compare: {len(panel)} prompts vs {a.compare} (A: {meta.get('label')}), tf={a.tf}, "
          f"chat template ids identical: {tmpl_same}", flush=True)

    def cmp(i):
        p = panel[i]
        P = p["prompt_len"]
        d = teacher_force(a.url, model, p["tokens"][:P], p["tokens"][P:], top, a.tf, a.timeout)
        print(f"  [{p['id']}] {len(d)} positions", flush=True)
        return d

    try:
        dists = run_pool(cmp, len(panel), a.concurrency)
    except ServerError as e:
        sys.exit(f"compare failed: {e}")
    res = score(panel, dists, top)
    res = {"tool": VERSION, "model": model, "a": {k: meta.get(k) for k in ("label", "created", "ref_path", "n", "k", "top")},
           "b": {"label": a.label, "tf": a.tf, "chat_template_ids_identical": tmpl_same},
           "a_self_check": meta.get("self_check_decode_vs_teacher_forced"), "seconds": round(time.perf_counter() - t0, 1),
           **res}
    if meta.get("ref_path") == ["decode"] or "decode" in (meta.get("ref_path") or []):
        res["note"] = ("A's reference came from the decode path (server had no prompt_logprobs at record time); B is "
                       "teacher-forced, so part of any disagreement is decode-vs-prefill numerics, not the knob.")
    print(json.dumps({k: v for k, v in res.items() if k != "per_prompt"}), flush=True)
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
