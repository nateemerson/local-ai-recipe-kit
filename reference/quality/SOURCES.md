# Quality-suite data (suites q1)

Cached, fixed subsets read by `tools/quality_oai.py` and `tools/agree_oai.py`. Do not edit in place: a changed file
is a new suite version (bump `VERSION` in `quality_oai.py`; each run records the file hashes). `build.py` regenerates
the four sampled files from the pinned upstream revisions (seed 20261012). Each JSONL's first line is `_meta` with the
source, revision, license, and how the subset was chosen.

| file | n | source (revision) | license | subset |
|---|---|---|---|---|
| `gsm8k.jsonl` | 100 | [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k) `main` test (740312ad) | MIT | random sample |
| `mmlu_pro.jsonl` | 100 | [TIGER-Lab/MMLU-Pro](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro) test (b189ec76) | MIT | 7-8 per category, 14 categories |
| `ifeval.jsonl` | 50 | [google/IFEval](https://huggingface.co/datasets/google/IFEval) (966cd895) | Apache-2.0 | random sample of the prompts whose checks are all implemented (no language-detection or sentence-count checks) |
| `humaneval.jsonl` | 20 | [openai/openai_humaneval](https://huggingface.co/datasets/openai/openai_humaneval) (7dce6050) | MIT | random sample (canonical solutions kept for the sandbox self-test) |
| `tool_calls.jsonl` | 25 | hand-written for this kit | this repo's license | 20 positive (name + argument values), 5 negative (no call expected) |
| `agree_prompts.jsonl` | 64 | hand-written for this kit | this repo's license | 8 math, 10 code, 12 knowledge, 8 writing, 8 reasoning, 8 instructions, 10 multilingual |
