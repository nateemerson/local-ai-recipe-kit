#!/usr/bin/env python3
"""Long-context needle check against an OpenAI-compatible server: filler text of ~N tokens with one fact inserted at a
given depth, then a question about it (chat, thinking off, temperature 0, completion runs to EOS).

  python3 tools/needle.py --url http://127.0.0.1:30100 --tokens 131072 --depth 0.6
"""
import argparse
import json
import random
import sys
import time
import urllib.error
import urllib.request

WORDS = ("river mountain lantern harbor orchard meadow copper violet thunder glacier canyon ember willow falcon "
         "compass quartz maple tide prairie beacon cobalt saffron marble lagoon summit basalt heron juniper").split()

RESERVE = 256  # tokens kept free below max context for the completion


def post_json(url, payload, timeout=60):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def bases(url):
    """vLLM serves /tokenize and /v1/models-style routes at the server root, so try the root before the URL as given."""
    url = url.rstrip("/")
    root = url[:-3] if url.endswith("/v1") else url
    return [root] if root == url else [root, url]


def make_counter(url, model):
    """Find a working tokenize endpoint. Returns (count(text) -> prompt tokens, max_model_len or None, label), or None.
    vLLM chat form counts the real chat prompt (template included); vLLM prompt / llama.cpp forms count raw text, so
    those add a margin for the chat template."""
    forms = [("vllm-chat", 16, lambda t: {"model": model, "messages": [{"role": "user", "content": t}],
                                           "add_generation_prompt": True,
                                           "chat_template_kwargs": {"enable_thinking": False}}, "/tokenize"),
             ("vllm-prompt", 128, lambda t: {"model": model, "prompt": t, "add_special_tokens": False}, "/tokenize"),
             ("llama.cpp", 128, lambda t: {"content": t}, "/tokenize")]
    for base in bases(url):
        for label, margin, mk, path in forms:
            def count(text, base=base, mk=mk, path=path, margin=margin):
                r = post_json(base + path, mk(text))
                return (r["count"] if isinstance(r.get("count"), int) else len(r["tokens"])) + margin, r
            try:
                _, r = count("hello world")
            except Exception:
                continue
            return (lambda text: count(text)[0]), r.get("max_model_len"), label
    return None


def server_max_len(url):
    for base in bases(url):
        try:
            return json.loads(urllib.request.urlopen(base + "/v1/models", timeout=30).read())["data"][0]["max_model_len"]
        except Exception:
            continue
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:30100")
    ap.add_argument("--tokens", type=int, default=131072, help="target prompt tokens (clamped below the server's max context)")
    ap.add_argument("--depth", type=float, default=0.6)
    ap.add_argument("--model", default="flashnext", help="served model name (vLLM checks it)")
    ap.add_argument("--out")
    a = ap.parse_args()
    a.url = a.url.rstrip("/")
    rng = random.Random(7)
    code = f"{rng.randrange(1000, 9999)}-{rng.randrange(1000, 9999)}"
    needle = f"Important: the vault access code for the Kestrel project is {code}."
    question = "\n\nQuestion: What is the vault access code for the Kestrel project? Reply with the code only."
    pool = []  # sentences of 12 words, grown on demand

    def build(n):
        while len(pool) < n:
            pool.append(" ".join(rng.choice(WORDS) for _ in range(12)).capitalize() + ".")
        sents = pool[:n]
        sents.insert(int(len(sents) * a.depth), needle)
        return " ".join(sents) + question

    tok = make_counter(a.url, a.model)
    max_len = (tok[1] if tok else None) or server_max_len(a.url)
    target = a.tokens
    if max_len and target > max_len - RESERVE:
        target = max_len - RESERVE
        print(f"needle: --tokens {a.tokens} exceeds max context {max_len}; using {target} (reserve {RESERVE})",
              file=sys.stderr)
    if tok is None:
        print(f"needle: WARNING no working /tokenize endpoint at {a.url}; falling back to a ~1.3 tokens/word estimate, "
              "which can overshoot the server's context on some tokenizers", file=sys.stderr)
        n_sent = int(target / 1.3 / 13)
    else:
        count, _, label = tok
        base = count(build(0))  # needle + question + chat template
        if base >= target:
            sys.exit(f"needle: target {target} tokens leaves no room for filler (overhead is {base})")
        probe = 256
        per = (count(build(probe)) - base) / probe  # tokens per sentence
        n_sent = int((target - base) / per * 0.995)
        for _ in range(6):  # verify with the real count; rescale until within [97%, 100%] of the target
            total = count(build(n_sent))
            if 0.97 * target <= total <= target:
                break
            n_sent = max(0, int(n_sent * (target - base) / max(total - base, 1) * 0.998))
        else:
            total = count(build(n_sent))
        if total > target:
            sys.exit(f"needle: could not fit the prompt under {target} tokens (last count {total})")
        print(f"needle: tokenizer={label} counted {total} prompt tokens (target {target}, max context {max_len})",
              file=sys.stderr)
    msg = build(n_sent)
    payload = {"model": a.model, "messages": [{"role": "user", "content": msg}], "temperature": 0,
               "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(bases(a.url)[0] + "/v1/chat/completions", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=3600) as r:
            out = json.loads(r.read())
    except urllib.error.HTTPError as e:
        sys.exit(f"needle: HTTP {e.code} from chat completions: {e.read().decode(errors='replace')[:500]}")
    dt = time.time() - t0
    ans = out["choices"][0]["message"]["content"]
    res = {"prompt_tokens": out["usage"]["prompt_tokens"], "depth": a.depth, "code": code, "answer": ans,
           "pass": code in (ans or ""), "seconds": round(dt, 1)}
    print(json.dumps(res), flush=True)
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
