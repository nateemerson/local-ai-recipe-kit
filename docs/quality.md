# Quality checks for any model: `quality_oai.py` and `agree_oai.py`

`tools/score_ref_panel.py` only works for Qwen3.8-Flash-Next (its panel is in that model's token ids, scored through
SGLang). These two tools work for any chat model behind an OpenAI-compatible server (vLLM, SGLang, llama.cpp):

| tool | question it answers | compares |
|---|---|---|
| `tools/quality_oai.py` | Is this model and config any good at real tasks? | the model against fixed answers |
| `tools/agree_oai.py` | Did my one-knob change alter the model's outputs? | config B against config A of the **same** model |

Both are stdlib-only Python and talk to the server through the API, so the server's own performance is never touched.
Run them only on a server you are allowed to load: a quality run sends hundreds of requests.

## quality_oai.py: task suites

Fixed, versioned data in `reference/quality/` (sources, revisions, and licenses are in `SOURCES.md` there). Every
request uses temperature 0 and **no `max_tokens`**, so completions run to their natural end. A per-request `--timeout`
(default 1200 s) stops runaway generations; a timeout counts as a fail and is listed under `errors`.

| suite | n | what passes |
|---|---|---|
| `math` | 100 GSM8K test | the number after `Answer:` equals the reference (fallbacks: `\boxed{}`, then the last number in the reply) |
| `knowledge` | 100 MMLU-Pro (7-8 per category, 10 options) | the letter after `Answer:` (fallbacks: `\boxed{X}`, a bare letter). Chance is about 10 % |
| `ifeval` | 50 IFEval | every instruction in the prompt passes its strict check (prompt-level). `instruction_level` is also reported |
| `tools` | 20 + 5 negative | first tool call has the right function and the expected argument values; negatives pass when no tool is called. `positive` and `negative` are reported separately |
| `needle` | 3 lengths x 3 depths | the filler-and-code check from `tools/needle.py`, at 8K, 32K, and near-max (`max_model_len` − 4096), depths 0.1, 0.5, and 0.9 |
| `code` | 20 HumanEval | the reply's function passes the HumanEval tests (pass@1). Runs in a subprocess with `python3 -I`, no network (`unshare -rn`), rlimits (CPU 15 s, 4 GB address space, 16 MB files), and a temporary working directory |

Reasoning models: the scored answer is `message.content` with any `<think>…</think>` removed. Reasoning that the
server returns in `reasoning_content` or `reasoning` (or inline before `</think>`) is never scored; its length and
last 300 characters are kept per item. Use `--chat-template-kwargs '{"enable_thinking": false}'` or
`--reasoning-effort low|medium|high` to pin the mode, and **always compare runs made in the same mode**. Tool
calling needs the server's tool parser enabled (vLLM: `--enable-auto-tool-choice --tool-call-parser <parser>`;
llama.cpp: `--jinja`). Otherwise the `tools` suite fails, and that failure is a real finding about the recipe.

```bash
# full run (default model: first entry of /v1/models; concurrency 8)
python3 tools/quality_oai.py --url http://127.0.0.1:8000 --label "<image digest> <knobs>" --out work/quality.json
# quick smoke test (scores NOT comparable with full runs)
python3 tools/quality_oai.py --url http://127.0.0.1:8000 --suites math,tools --limit 10 --out work/q-smoke.json
# a thinking model with thinking off
python3 tools/quality_oai.py --url ... --chat-template-kwargs '{"enable_thinking": false}' --out ...
```

The output JSON holds per-item records (answer, extraction method, finish reason, tokens, seconds), and per suite
the `score`, the Wilson 95 % `ci95`, `wall_s`, prompt and completion tokens, `errors`, `truncated` (finish = length),
and the data hash. It also holds a one-line `summary_md` table row, which is printed at the end:

```
| <model> | <label> | math 87/100 (0.79-0.92) | knowledge 61/100 (0.51-0.70) | ifeval 41/50 (0.69-0.90) | tools 24/25 (0.80-0.99) | needle 9/9 (0.70-1.00) | code 18/20 (0.70-0.97) | 412 s | 210334 tok |
```

### Runtime

There are 304 requests in all: 295 short ones plus 9 long-context ones. Estimates from item counts for a fast GPU
MoE (about 100-150 tok/s per stream, concurrency 8):

- **Non-reasoning, or thinking off:** about 300-500 completion tokens per item (measured on the CPU dev model: 330
  per item for math, 50-100 for MCQ letters). That is about 1.5-2 min for the short suites. Add the needle prefills:
  about 0.1M tokens at 8K + 32K, plus 3 × near-max (about 0.6M tokens at 200K). At 5-10K tok/s that is about 1-3
  min. **Total: about 3-5 min.**
- **Reasoning on:** typically 1-4K tokens per item, about 0.3-1.2M tokens in all, so **about 10-25 min**. A model
  that loops at temperature 0 burns a full `--timeout` per looping item; lower `--timeout` for screening runs.

### Reading the scores

The n values are small on purpose. The point is to catch broken configs (a bad quantization, a wrong chat template,
a broken tool parser, a kernel bug), not to rank close models. Read every score with its interval:

| n | observed | Wilson 95 % CI |
|---|---|---|
| 100 | 80 % | 71-87 % |
| 50 | 80 % | 67-89 % |
| 20 | 80 % | 58-92 % |
| 9 | 9/9 | 70-100 % |

- **Same model, two configs:** overlapping intervals mean "no evidence of a difference". A one-knob change that
  moves `math` from 88 to 84 is noise. A drop larger than the interval (for example 88 → 60) or a suite collapsing
  (tools 24 → 3, needle 9 → 2) is a real regression. Look at the per-item `answer` fields before you believe it,
  because a changed output format can break extraction. For fine-grained same-model comparisons use `agree_oai.py`,
  which measures differences that are far too small for these suites.
- **Different models:** treat scores within about 10 points (n=100) as ties. The scores are not comparable with
  published numbers (different prompts, extraction, and subsets). They are comparable across runs of this tool with
  the same `tool` version and suite hashes.
- `errors` greater than 0 or `truncated` greater than 0 deserve a look: they usually mean timeouts or a server
  problem, not model quality.
- `--limit` runs are smoke tests only.

## agree_oai.py: backend agreement for one-knob changes

Use it whenever a knob should not change the math much: MoE backend, attention backend, FP8 KV cache, speculative
decoding (should be lossless at temperature 0), a new image or vLLM version, TP vs PP, or CUDA graphs.

```bash
# 1. baseline config A
python3 tools/agree_oai.py --url http://127.0.0.1:8000 --label "A: <digest> default MoE" --record work/agree-A.json
# 2. relaunch with ONE knob changed (same weights, same --max-logprobs), then
python3 tools/agree_oai.py --url http://127.0.0.1:8000 --label "B: --kv-cache-dtype fp8" \
    --compare work/agree-A.json --out work/agree-B-vs-A.json
# 3. noise floor: relaunch A unchanged and compare it with its own recording too
python3 tools/agree_oai.py --url http://127.0.0.1:8000 --label "A again" --compare work/agree-A.json --out work/agree-A2-vs-A.json
```

How it works:

1. Each of the 64 prompts is rendered with the model's own chat template through vLLM `POST /tokenize {"messages"}`.
   The resulting **token ids** are stored and reused verbatim, so B sees exactly A's ids even if the template changed.
   A mismatch is reported as `chat_template_ids_identical: false`.
2. `--record` generates K=128 greedy tokens (`/v1/completions`, prompt = ids, `logprobs: 20`). It then
   teacher-forces those tokens (`prompt` = prompt ids + A's ids, `prompt_logprobs: 20`, `max_tokens: 1`) and stores
   the per-position top-20 as the reference. It also stores the decode-path top-20 and reports
   `self_check_decode_vs_teacher_forced`, which is how much A disagrees with itself between its decode and prefill
   kernels.
3. `--compare` teacher-forces A's tokens on B, so every position is compared under an identical prefix, even after
   B's own greedy output would have diverged. vLLM skips prefix-cache reads for `prompt_logprobs` requests, so the
   numbers come from a real forward pass.

Metrics, in the same style as `score_ref_panel.py`:

- `top1_agreement`: B's argmax equals A's argmax.
- `mean_kl_top20`: KL(A‖B) on A's top-20 ids, both renormalised.
- `greedy_match`: B's argmax equals A's generated token.
- `first_divergence_median` and `prompts_fully_agreeing`: where B's greedy path would first leave A's.
- Per-prompt breakdowns.

**How to read it:** always measure the noise floor first (step 3: A relaunched against A). Batch order alone makes
vLLM non-bit-exact.

| result vs the A-vs-A floor | verdict |
|---|---|
| top-1 within about 0.5 pt, KL within about 2x | the knob is numerically neutral |
| top-1 1-3 pt lower, KL 2-10x higher | expected for lossy knobs (FP8 KV, a different quant kernel); confirm with `quality_oai.py` that task scores hold |
| top-1 more than 5 pt lower, or KL more than 10x higher, or many prompts diverging in the first few tokens | something is wrong (kernel bug, wrong template, a broken spec-decode acceptance path). Do not tune speed on it |

Speculative decoding at temperature 0 should match A as closely as the floor. If it does not, treat that as a bug.

Caveats:

- **`prompt_logprobs` is vLLM-only.** In vLLM 0.31 it is a `/v1/completions` extra parameter, capped by the server's
  `--max-logprobs` (default 20). It is rejected with `--kv-sharing-fast-prefill`. Other servers ignore it, and
  `--compare` then stops with a clear message. `--tf stepwise` is a slow fallback that sends one `max_tokens: 1`
  request per position (K requests per prompt) and works on any server that returns completion logprobs. On such
  servers `--record` falls back to decode-path references, and the result carries a note saying so.
- B only returns its own top-20. An id in A's top-20 that is missing from B's gets B's 20th logprob as an upper
  bound, counted in `missing_support_frac`. That makes the KL a slight underestimate. To tighten it, launch both
  configs with `--max-logprobs 64` and pass `--top 64`.
- The record file is `{"_meta", "panel"}`. Each panel entry has the reference-panel keys (`prompt`, `prompt_len`,
  `tokens`, `completion`, `top20_ids`, `top20_lp`), so `jq .panel A.json` is a valid `score_ref_panel.py` panel for
  an SGLang server running the same model.
- Runtime on a fast GPU model: record is 64 × 128-token generations plus 64 teacher-forced prefills, and compare is
  64 prefills. Each takes well under a minute. Stepwise mode is 8192 requests.

## What was tested (2026-10-12)

Both tools were tested on llama.cpp b11561 running on CPU (`-t 6`, Qwen3.5-0.8B Q8_0, 32K context, `--jinja`). The
outputs are in `work/2026-10-12-quality-dev/`:

- `quality_oai.py`: all six suites at `--limit 5` with thinking off (`smoke-nothink.json`, 325 s on CPU). Every
  failure was checked by hand and was a real model error, not an extraction bug. A thinking-on run timed out on most
  items: the 0.8B model loops at temperature 0. That run did confirm that `reasoning_content` is split from the
  scored answer (`smoke-think.json`, partial). The code sandbox was checked separately: the canonical solutions
  pass 20/20, stubs fail, network is blocked, and the timeout fires.
- `agree_oai.py`: `--tf stepwise` record (16 prompts × 64 tokens) and compare. A vs A gave top-1 0.997 and KL
  2.0e-4. B (`--cache-type-k/v q8_0`) vs A gave top-1 0.992 and KL 6.0e-4. The default `prompt_logprobs` path on
  llama.cpp records decode-path references and stops `--compare` with the fallback hint.

Not tested, because no GPU was available and vLLM was not run:

- the `prompt_logprobs` path, checked only against the vLLM 0.31 source (`completion/protocol.py`, prompt_logprobs
  dict keyed by token id)
- `return_tokens_as_token_ids` parsing (`token_id:N` keys)
- the `/v1/models` `max_model_len` path
- vLLM `/tokenize` with messages, and vLLM tool parsers

Run the A-vs-A check on the first vLLM model before you rely on it.
