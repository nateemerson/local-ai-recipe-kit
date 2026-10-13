#!/usr/bin/env python3
"""Task-based quality score for any chat model behind an OpenAI-compatible server (vLLM, SGLang, llama.cpp, ...).

Fixed, versioned suites (data cached under reference/quality/, see SOURCES.md there); temperature 0, no max_tokens
(completions run to natural EOS; a per-request timeout guards against runaway generations):

  math       GSM8K test subset (100)            final number after "Answer:" (fallbacks: \\boxed{}, last number)
  knowledge  MMLU-Pro subset (100, by category) letter after "Answer:" (A-J)
  ifeval     IFEval subset (50)                 strict verifiable instruction checks (prompt-level pass)
  tools      25 function-calling cases          OpenAI `tools`, tool_choice auto: name + argument values; 5 negatives
  needle     long-context retrieval             needle.py's filler + code, 8K / 32K / near-max x depth 0.1/0.5/0.9
  code       HumanEval subset (20)              pass@1, executed in a sandboxed subprocess (no network, rlimits)

Reasoning models: the answer is message.content with any <think>...</think> block stripped; the reasoning text
(reasoning_content / reasoning / think block) is kept in the per-item record but never scored.

  python3 tools/quality_oai.py --url http://127.0.0.1:8000 --out work/quality.json
  python3 tools/quality_oai.py --url ... --suites math,tools --limit 10 --out work/q-smoke.json

Output: JSON with per-item results, per-suite score + Wilson 95% CI + wall time + tokens, and a markdown summary line
(also printed). Exit code 0 even when items fail; non-zero only for setup errors.
"""
import argparse, concurrent.futures as cf, hashlib, json, math, os, random, re, resource, shutil, signal, subprocess
import sys, tempfile, time, urllib.error, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
DATA = os.path.join(HERE, "..", "reference", "quality")
VERSION = "quality_oai v1 (suites q1)"
ALL_SUITES = ["math", "knowledge", "ifeval", "tools", "needle", "code"]


# ---------------------------------------------------------------- transport
def post(url, path, payload, timeout):
    req = urllib.request.Request(url + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code} on {path}: {e.read()[:400].decode(errors='replace')}") from None


THINK_RE = re.compile(r"<think>.*?</think>", re.S)


def split_reasoning(msg):
    """(answer, reasoning) from a chat message, for servers that parse reasoning and those that don't."""
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
    if "</think>" in content:  # unparsed think block (or a template that opens <think> in the prompt)
        head, _, content = content.rpartition("</think>")
        reasoning = (reasoning + "\n" + head.replace("<think>", "")).strip()
    content = THINK_RE.sub("", content).strip()
    return content, reasoning


class Client:
    def __init__(self, a):
        self.url, self.model, self.timeout = a.url, a.model, a.timeout
        self.extra = json.loads(a.extra_body) if a.extra_body else {}
        if a.chat_template_kwargs:
            self.extra["chat_template_kwargs"] = json.loads(a.chat_template_kwargs)
        if a.reasoning_effort:
            self.extra["reasoning_effort"] = a.reasoning_effort

    def chat(self, messages, **kw):
        payload = {"model": self.model, "messages": messages, "temperature": 0, **self.extra, **kw}
        t0 = time.perf_counter()
        r = post(self.url, "/v1/chat/completions", payload, self.timeout)
        ch = r["choices"][0]
        msg = ch.get("message") or {}
        answer, reasoning = split_reasoning(msg)
        u = r.get("usage") or {}
        return {"answer": answer, "reasoning_chars": len(reasoning), "reasoning_tail": reasoning[-300:],
                "tool_calls": msg.get("tool_calls") or [], "finish": ch.get("finish_reason"),
                "prompt_tokens": u.get("prompt_tokens", 0), "completion_tokens": u.get("completion_tokens", 0),
                "seconds": round(time.perf_counter() - t0, 2)}


def load(name):
    rows = [json.loads(l) for l in open(os.path.join(DATA, name)) if l.strip()]
    return rows[0]["_meta"], rows[1:]


def sha(name):
    return hashlib.sha256(open(os.path.join(DATA, name), "rb").read()).hexdigest()[:16]


def wilson(k, n, z=1.96):
    if n == 0:
        return [0.0, 0.0]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0.0, c - h), 4), round(min(1.0, c + h), 4)]


# ---------------------------------------------------------------- math (GSM8K)
MATH_PROMPT = ("Solve the following math problem. Reason step by step, then give the final answer on the last line "
               "in the form \"Answer: <number>\" (just the number, no units).\n\n{q}")
NUM = r"-?\$?\s*\d[\d,]*(?:\.\d+)?"


def to_num(s):
    s = s.replace(",", "").replace("$", "").replace(" ", "").rstrip(".")
    try:
        return float(s)
    except ValueError:
        return None


def extract_number(text):
    m = re.findall(r"(?i)answer\s*(?:is)?\s*[:：]?\s*\**\s*(" + NUM + ")", text)
    if m:
        return to_num(m[-1]), "answer-line"
    m = re.findall(r"\\boxed\{\s*(" + NUM + r")\s*\}", text)
    if m:
        return to_num(m[-1]), "boxed"
    m = re.findall(NUM, text)
    if m:
        return to_num(m[-1]), "last-number"
    return None, "none"


def run_math(cl, item):
    r = cl.chat([{"role": "user", "content": MATH_PROMPT.format(q=item["question"])}])
    got, how = extract_number(r["answer"])
    ok = got is not None and abs(got - float(item["answer"])) < 1e-6
    return ok, {"expected": item["answer"], "got": got, "extracted_by": how}, r


# ---------------------------------------------------------------- knowledge (MMLU-Pro)
LETTERS = "ABCDEFGHIJ"
MC_PROMPT = ("Answer the following multiple-choice question. Think it through, then finish with a final line of the "
             "form \"Answer: X\" where X is the letter of the correct option.\n\nQuestion: {q}\n\nOptions:\n{opts}")


def extract_letter(text):
    m = re.findall(r"(?i:answer)\s*(?:(?i:is))?\s*[:：]?\s*[\*\(\[]*\s*([A-J])(?=[ \t]*(?:$|[\n\.\)\]\*,:;\-]))", text,
                   re.M)
    if m:
        return m[-1], "answer-line"
    m = re.findall(r"\\boxed\{\s*\(?([A-J])\)?\s*\}", text)
    if m:
        return m[-1], "boxed"
    m = re.fullmatch(r"\s*[\*\(\[]*([A-J])[\)\]\.\*]*\s*", text)
    if m:
        return m.group(1), "bare-letter"
    return None, "none"


def run_knowledge(cl, item):
    opts = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(item["options"]))
    r = cl.chat([{"role": "user", "content": MC_PROMPT.format(q=item["question"], opts=opts)}])
    got, how = extract_letter(r["answer"])
    return got == item["answer"], {"expected": item["answer"], "got": got, "extracted_by": how,
                                    "category": item["category"]}, r


# ---------------------------------------------------------------- IFEval (strict)
def _cmp(n, relation, target):
    return n < target if relation == "less than" else n >= target


def _words(t):
    return re.findall(r"\w+", t)


def _paragraphs_stars(t):
    ps = re.split(r"\s?\*\*\*\s?", t)
    n = len(ps)
    for i, p in enumerate(ps):
        if not p.strip():
            if i == 0 or i == len(ps) - 1:
                n -= 1
            else:
                return None
    return n


def ifeval_check(iid, kw, resp, prompt):
    t = resp
    if iid == "punctuation:no_comma":
        return "," not in t
    if iid == "length_constraints:number_words":
        return _cmp(len(_words(t)), kw["relation"], kw["num_words"])
    if iid == "keywords:forbidden_words":
        return not any(re.search(r"\b" + re.escape(w) + r"\b", t, re.I) for w in kw["forbidden_words"])
    if iid == "detectable_format:number_highlighted_sections":
        n = sum(1 for h in re.findall(r"\*[^\n\*]*\*", t) if h.strip("*").strip())
        n += sum(1 for h in re.findall(r"\*\*[^\n\*]*\*\*", t) if h.removeprefix("**").removesuffix("**").strip())
        return n >= kw["num_highlights"]
    if iid == "keywords:frequency":
        return _cmp(len(re.findall(re.escape(kw["keyword"]), t, re.I)), kw["relation"], kw["frequency"])
    if iid == "combination:repeat_prompt":
        return t.strip().lower().startswith(kw["prompt_to_repeat"].strip().lower())
    if iid == "startend:quotation":
        s = t.strip()
        return len(s) > 1 and s[0] == '"' and s[-1] == '"'
    if iid == "change_case:english_lowercase":
        return t.islower()
    if iid == "keywords:existence":
        return all(re.search(re.escape(k), t, re.I) for k in kw["keywords"])
    if iid == "detectable_format:title":
        return any(x.lstrip("<").rstrip(">").strip() for x in re.findall(r"<<[^\n]+>>", t))
    if iid == "keywords:letter_frequency":
        return _cmp(t.lower().count(kw["letter"].lower()), kw["let_relation"], kw["let_frequency"])
    if iid == "detectable_format:number_bullet_lists":
        n = len(re.findall(r"^\s*\*[^\*].*$", t, re.M)) + len(re.findall(r"^\s*-.*$", t, re.M))
        return n == kw["num_bullets"]
    if iid == "detectable_content:number_placeholders":
        return len(re.findall(r"\[.*?\]", t)) >= kw["num_placeholders"]
    if iid == "length_constraints:number_paragraphs":
        return _paragraphs_stars(t) == kw["num_paragraphs"]
    if iid == "startend:end_checker":
        return t.strip().strip('"').lower().endswith(kw["end_phrase"].strip().lower())
    if iid == "detectable_content:postscript":
        mk = kw["postscript_marker"].strip()
        if mk.lower() == "p.p.s":
            pat = r"\s*p\.\s?p\.\s?s.*$"
        elif mk.lower() == "p.s.":
            pat = r"\s*p\.\s?s\..*$"
        else:
            pat = r"\s*" + re.escape(mk.lower()) + r".*$"
        return bool(re.findall(pat, t.lower(), re.M))
    if iid == "change_case:english_capital":
        return t.isupper()
    if iid == "change_case:capital_word_frequency":
        n = sum(1 for w in re.findall(r"\b\w+\b", t) if w.isupper())
        return _cmp(n, kw["capital_relation"], kw["capital_frequency"])
    if iid == "detectable_format:json_format":
        s = t.strip()
        for p in ("```json", "```Json", "```JSON", "```"):
            s = s.removeprefix(p)
        s = s.removesuffix("```").strip()
        try:
            json.loads(s)
            return True
        except ValueError:
            return False
    if iid == "detectable_format:multiple_sections":
        parts = re.split(r"\s?" + re.escape(kw["section_spliter"]) + r"\s?\d+\s?", t)
        return len(parts) - 1 >= kw["num_sections"]
    if iid == "detectable_format:constrained_response":
        return any(o in t.strip() for o in ("My answer is yes.", "My answer is no.", "My answer is maybe."))
    if iid == "combination:two_responses":
        parts = t.split("******")
        valid = []
        for i, p in enumerate(parts):
            if not p.strip():
                if i not in (0, len(parts) - 1):
                    return False
            else:
                valid.append(p)
        return len(valid) == 2 and valid[0].strip() != valid[1].strip()
    if iid == "length_constraints:nth_paragraph_first_word":
        ps = re.split(r"\n\n", t)
        n = sum(1 for p in ps if p.strip())
        nonempty = [p for p in ps if p.strip()]
        if kw["nth_paragraph"] > len(nonempty) or n != kw["num_paragraphs"]:
            return False
        words = nonempty[kw["nth_paragraph"] - 1].strip().split()
        if not words:
            return False
        w = words[0].lstrip("'").lstrip('"')
        first = ""
        for ch in w:
            if ch in ".,?!'\"":
                break
            first += ch.lower()
        return first == kw["first_word"].lower()
    raise KeyError(f"IFEval instruction {iid} not implemented")


def run_ifeval(cl, item):
    r = cl.chat([{"role": "user", "content": item["prompt"]}])
    res = [bool(ifeval_check(i, k, r["answer"], item["prompt"])) for i, k in zip(item["instruction_id_list"], item["kwargs"])]
    return all(res), {"instructions": dict(zip(item["instruction_id_list"], res))}, r


# ---------------------------------------------------------------- tool calling
TOOL_CATALOG = None


def match_arg(m, v):
    if "eq" in m:
        e = m["eq"]
        return (isinstance(v, str) and isinstance(e, str) and v.strip().lower() == e.lower()) or v == e
    if "num" in m:
        try:
            return abs(float(v) - float(m["num"])) < 1e-6
        except (TypeError, ValueError):
            return False
    if "contains" in m:
        return isinstance(v, str) and m["contains"].lower() in v.lower()
    if "any_of" in m:
        return any(match_arg({"eq": e}, v) for e in m["any_of"])
    raise KeyError(m)


def run_tools(cl, item):
    tools = [TOOL_CATALOG[n] for n in item["tools"]]
    r = cl.chat(item["messages"], tools=tools, tool_choice="auto")
    calls = r.pop("tool_calls")
    rec = {"calls": [{"name": (c.get("function") or {}).get("name"), "arguments": (c.get("function") or {}).get("arguments")}
                     for c in calls]}
    exp = item["expect"]
    if exp is None:
        rec["kind"] = "negative"
        return not calls, rec, r
    rec["kind"] = "positive"
    if not calls:
        rec["why"] = "no tool call"
        return False, rec, r
    c = rec["calls"][0]
    if c["name"] != exp["name"]:
        rec["why"] = f"called {c['name']}"
        return False, rec, r
    try:
        args = c["arguments"] if isinstance(c["arguments"], dict) else json.loads(c["arguments"] or "{}")
    except ValueError:
        rec["why"] = "arguments are not valid JSON"
        return False, rec, r
    bad = [k for k, m in exp["args"].items() if k not in args or not match_arg(m, args[k])]
    if bad:
        rec["why"] = f"wrong/missing args {bad}"
    return not bad, rec, r


# ---------------------------------------------------------------- long-context needle
from needle import WORDS  # noqa: E402  (same filler vocabulary as tools/needle.py)


def count_tokens(cl, text):
    """Token count via the server's /tokenize (vLLM: prompt -> count; llama.cpp: content -> tokens)."""
    for payload in ({"model": cl.model, "prompt": text}, {"content": text}):
        try:
            r = post(cl.url, "/tokenize", payload, 120)
            if r.get("count") is not None:
                return r["count"]
            if r.get("tokens") is not None:
                return len(r["tokens"])
        except Exception:
            continue
    return None


def needle_items(cl, ctx_len, reserve):
    rng = random.Random(7)
    sample = " ".join(" ".join(rng.choice(WORDS) for _ in range(12)).capitalize() + "." for _ in range(150))
    n = count_tokens(cl, sample)
    tok_per_word = (n / 1950) if n else 1.3 / 0.9  # 150 sentences x 13 words; conservative fallback
    near_max = ctx_len - reserve
    lengths = [L for L in (8192, 32768) if L <= near_max]
    if not lengths or near_max > lengths[-1] * 1.1:
        lengths.append(near_max)
    items = []
    for L in lengths:
        for d in (0.1, 0.5, 0.9):
            items.append({"id": f"needle-{L}-d{d}", "tokens": L, "depth": d, "tok_per_word": tok_per_word})
    return items, {"ctx_len": ctx_len, "reserve": reserve, "lengths": lengths, "tokens_per_word": round(tok_per_word, 4),
                   "calibrated": n is not None}


def run_needle(cl, item):
    rng = random.Random(f"{item['tokens']}-{item['depth']}")
    n_sent = int((item["tokens"] - 200) / item["tok_per_word"] / 13)
    sents = [" ".join(rng.choice(WORDS) for _ in range(12)).capitalize() + "." for _ in range(n_sent)]
    code = f"{rng.randrange(1000, 9999)}-{rng.randrange(1000, 9999)}"
    sents.insert(int(len(sents) * item["depth"]), f"Important: the vault access code for the Kestrel project is {code}.")
    msg = (" ".join(sents) + "\n\nQuestion: What is the vault access code for the Kestrel project? "
           "Reply with the code only.")
    r = cl.chat([{"role": "user", "content": msg}])
    return code in r["answer"], {"code": code, "got": r["answer"][:200], "target_tokens": item["tokens"],
                                 "depth": item["depth"]}, r


# ---------------------------------------------------------------- code (HumanEval subset)
CODE_PROMPT = ("Complete the following Python function. Reply with the complete function (signature, body, and any "
               "imports it needs) in a single ```python code block.\n\n```python\n{p}```")
NET_ISOLATION = None


def sandbox_prefix():
    global NET_ISOLATION
    if NET_ISOLATION is None:
        u = shutil.which("unshare")
        ok = False
        if u:
            try:
                ok = subprocess.run([u, "-rn", "true"], capture_output=True, timeout=10).returncode == 0
            except Exception:
                ok = False
        NET_ISOLATION = [u, "-rn"] if ok else []
    return NET_ISOLATION


def _limits(cpu):
    def f():
        os.setsid()
        resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu + 1))
        resource.setrlimit(resource.RLIMIT_AS, (4 << 30, 4 << 30))
        resource.setrlimit(resource.RLIMIT_FSIZE, (16 << 20, 16 << 20))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    return f


def extract_code(text, entry):
    blocks = re.findall(r"```(?:python|py|Python)?[ \t]*\n(.*?)```", text, re.S)
    if not blocks and "```" in text:  # unterminated block
        blocks = [text.split("```", 1)[1].split("\n", 1)[-1]]
    for b in blocks:
        if re.search(r"def\s+" + re.escape(entry) + r"\s*\(", b):
            return b
    return blocks[0] if blocks else text


def build_program(item, code):
    if re.search(r"def\s+" + re.escape(item["entry_point"]) + r"\s*\(", code):
        # keep the prompt's imports and helper functions, then the model's full definition
        head = item["prompt"][:re.search(r"def\s+" + re.escape(item["entry_point"]) + r"\s*\(", item["prompt"]).start()]
        prog = head + "\n" + code
    else:  # body-only answer: continue the prompt
        prog = item["prompt"] + code
    return prog + "\n\n" + item["test"] + f"\n\ncheck({item['entry_point']})\n"


def exec_sandboxed(prog, timeout=15):
    d = tempfile.mkdtemp(prefix="qoai-")
    try:
        p = os.path.join(d, "prog.py")
        open(p, "w").write(prog)
        cmd = sandbox_prefix() + [sys.executable, "-I", "-B", p]
        try:
            pr = subprocess.Popen(cmd, cwd=d, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.DEVNULL,
                                  env={"PATH": "/usr/bin:/bin", "HOME": d, "PYTHONHASHSEED": "0"},
                                  preexec_fn=_limits(timeout))
            try:
                out, err = pr.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                os.killpg(pr.pid, signal.SIGKILL)
                pr.communicate()
                return False, "timeout"
            return pr.returncode == 0, err.decode(errors="replace")[-400:]
        except OSError as e:
            return False, f"exec error {e}"
    finally:
        shutil.rmtree(d, ignore_errors=True)


def run_code(cl, item):
    r = cl.chat([{"role": "user", "content": CODE_PROMPT.format(p=item["prompt"])}])
    code = extract_code(r["answer"], item["entry_point"])
    ok, err = exec_sandboxed(build_program(item, code))
    return ok, {"stderr_tail": err if not ok else "", "network_isolated": bool(sandbox_prefix())}, r


SUITES = {
    "math": ("gsm8k.jsonl", run_math),
    "knowledge": ("mmlu_pro.jsonl", run_knowledge),
    "ifeval": ("ifeval.jsonl", run_ifeval),
    "tools": ("tool_calls.jsonl", run_tools),
    "needle": (None, run_needle),
    "code": ("humaneval.jsonl", run_code),
}


def main():
    global TOOL_CATALOG
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", help="served model id (default: first entry of /v1/models)")
    ap.add_argument("--suites", default=",".join(ALL_SUITES), help=f"comma list of {ALL_SUITES}")
    ap.add_argument("--limit", type=int, help="first N items per suite (smoke test; scores are then not comparable)")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--needle-concurrency", type=int, default=1, help="long prompts are sent this many at a time")
    ap.add_argument("--needle-reserve", type=int, default=4096, help="near-max length = max_model_len - this")
    ap.add_argument("--ctx-len", type=int, help="override context length (default: /v1/models max_model_len, "
                                               "else llama.cpp /props n_ctx)")
    ap.add_argument("--timeout", type=float, default=1200, help="per-request seconds")
    ap.add_argument("--chat-template-kwargs", help='JSON, e.g. \'{"enable_thinking": false}\'')
    ap.add_argument("--reasoning-effort", help="low|medium|high (gpt-oss and others)")
    ap.add_argument("--extra-body", help="JSON merged into every chat request")
    ap.add_argument("--label", default="", help="config label for the summary line (image digest + knobs)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    models = json.loads(urllib.request.urlopen(a.url + "/v1/models", timeout=30).read())["data"]
    m = next((x for x in models if x["id"] == a.model), None) if a.model else models[0]
    if m is None:
        sys.exit(f"model {a.model} not served; have {[x['id'] for x in models]}")
    a.model = m["id"]
    ctx_len = a.ctx_len or m.get("max_model_len")
    if not ctx_len:
        try:
            ctx_len = json.loads(urllib.request.urlopen(a.url + "/props", timeout=30).read())[
                "default_generation_settings"]["n_ctx"]
        except Exception:
            ctx_len = None
    cl = Client(a)
    suites = [s.strip() for s in a.suites.split(",") if s.strip()]
    for s in suites:
        if s not in SUITES:
            sys.exit(f"unknown suite {s}")
    res = {"tool": VERSION, "model": a.model, "url_port": a.url.rsplit(":", 1)[-1], "label": a.label,
           "ctx_len": ctx_len, "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "settings": {"temperature": 0, "max_tokens": None, "timeout_s": a.timeout, "concurrency": a.concurrency,
                        "limit": a.limit, "extra": cl.extra}, "suites": {}, "status": "RUNNING"}

    def save():
        json.dump(res, open(a.out, "w"), indent=1)

    t_all = time.perf_counter()
    for s in suites:
        fname, fn = SUITES[s]
        if s == "needle":
            if not ctx_len:
                res["suites"][s] = {"skipped": "context length unknown; pass --ctx-len"}
                continue
            items, meta = needle_items(cl, int(ctx_len), a.needle_reserve)
            meta["source"] = "generated (tools/needle.py filler, seed per length/depth)"
            conc = a.needle_concurrency
        else:
            meta, items = load(fname)
            meta = {k: v for k, v in meta.items() if k != "catalog"} | {"sha256_16": sha(fname)}
            if s == "tools":
                TOOL_CATALOG = load(fname)[0]["catalog"]
            conc = a.concurrency
        if a.limit:
            items = items[:a.limit]
        t0 = time.perf_counter()
        out = [None] * len(items)

        def one(i):
            it = items[i]
            try:
                ok, rec, r = fn(cl, it)
                out[i] = {"id": it["id"], "pass": bool(ok), **rec, **r}
            except Exception as e:  # timeouts, HTTP errors: count as a fail, keep going
                out[i] = {"id": it["id"], "pass": False, "error": f"{type(e).__name__}: {e}"[:500]}
            print(f"  {s} {it['id']}: {'PASS' if out[i]['pass'] else 'fail'}"
                  f"{' (' + out[i]['error'][:80] + ')' if 'error' in out[i] else ''}", flush=True)

        print(f"[{s}] {len(items)} items, concurrency {conc}", flush=True)
        with cf.ThreadPoolExecutor(max(1, conc)) as ex:
            list(ex.map(one, range(len(items))))
        k, n = sum(o["pass"] for o in out), len(out)
        sr = {"n": n, "pass": k, "score": round(k / n, 4) if n else None, "ci95": wilson(k, n),
              "wall_s": round(time.perf_counter() - t0, 1),
              "prompt_tokens": sum(o.get("prompt_tokens", 0) for o in out),
              "completion_tokens": sum(o.get("completion_tokens", 0) for o in out),
              "errors": sum(1 for o in out if "error" in o),
              "truncated": sum(1 for o in out if o.get("finish") == "length"), "data": meta, "items": out}
        if s == "ifeval":
            flat = [v for o in out for v in (o.get("instructions") or {}).values()]
            sr["instruction_level"] = round(sum(flat) / len(flat), 4) if flat else None
        if s == "tools":
            pos = [o for o in out if o.get("kind") == "positive"]
            neg = [o for o in out if o.get("kind") == "negative"]
            sr["positive"] = f"{sum(o['pass'] for o in pos)}/{len(pos)}"
            sr["negative"] = f"{sum(o['pass'] for o in neg)}/{len(neg)}"
        res["suites"][s] = sr
        print(f"[{s}] {k}/{n} = {sr['score']} (95% CI {sr['ci95'][0]:.2f}-{sr['ci95'][1]:.2f}), "
              f"{sr['wall_s']} s, {sr['completion_tokens']} completion tokens, {sr['errors']} errors", flush=True)
        save()

    res["wall_s"] = round(time.perf_counter() - t_all, 1)
    res["completion_tokens"] = sum(v.get("completion_tokens", 0) for v in res["suites"].values())
    cells = []
    for s in suites:
        v = res["suites"].get(s, {})
        cells.append(f"{s} {v['pass']}/{v['n']} ({v['ci95'][0]:.2f}-{v['ci95'][1]:.2f})" if "n" in v else f"{s} skipped")
    res["summary_md"] = (f"| {a.model} | {a.label} | " + " | ".join(cells) +
                         f" | {res['wall_s']:.0f} s | {res['completion_tokens']} tok |" +
                         (f" limit {a.limit} |" if a.limit else ""))
    res["status"] = "DONE"
    res["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    save()
    print(res["summary_md"], flush=True)


if __name__ == "__main__":
    main()
