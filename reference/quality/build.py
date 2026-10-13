#!/usr/bin/env python3
"""Rebuild the cached quality-suite JSONL files in this folder from the upstream datasets (pinned revisions, fixed seed).

You do not need to run this: the JSONL files are committed and tools/quality_oai.py only reads them. It exists so the
subsets are reproducible and auditable. Needs pyarrow and the `hf` CLI:

  for r in openai/gsm8k TIGER-Lab/MMLU-Pro google/IFEval openai/openai_humaneval; do
    hf download $r --repo-type dataset --revision <rev below> --local-dir /tmp/qds/${r//\\//_}; done
  uv run --no-project --with pyarrow python reference/quality/build.py /tmp/qds

Sources (all permissive; license noted in each JSONL's _meta line and in SOURCES.md):
"""
import json, os, random, sys, collections

SEED = 20261012
SOURCES = {
    "gsm8k": ("openai/gsm8k", "740312add88f781978c0658806c59bc2815b9866", "MIT", "main/test-00000-of-00001.parquet"),
    "mmlu_pro": ("TIGER-Lab/MMLU-Pro", "b189ec765aa7ed75c8acfea42df31fdae71f97be", "MIT", "data/test-00000-of-00001.parquet"),
    "ifeval": ("google/IFEval", "966cd89545d6b6acfd7638bc708b98261ca58e84", "Apache-2.0", "ifeval_input_data.jsonl"),
    "humaneval": ("openai/openai_humaneval", "7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544", "MIT",
                  "openai_humaneval/test-00000-of-00001.parquet"),
}
# IFEval instruction ids whose checkers tools/quality_oai.py implements (language detection and sentence splitting
# need langdetect/nltk, so prompts using them are not sampled).
IFEVAL_IMPLEMENTED = {
    "punctuation:no_comma", "length_constraints:number_words", "keywords:forbidden_words",
    "detectable_format:number_highlighted_sections", "keywords:frequency", "combination:repeat_prompt",
    "startend:quotation", "change_case:english_lowercase", "keywords:existence", "detectable_format:title",
    "keywords:letter_frequency", "detectable_format:number_bullet_lists", "detectable_content:number_placeholders",
    "length_constraints:number_paragraphs", "startend:end_checker", "detectable_content:postscript",
    "change_case:english_capital", "change_case:capital_word_frequency", "detectable_format:json_format",
    "detectable_format:multiple_sections", "detectable_format:constrained_response", "combination:two_responses",
    "length_constraints:nth_paragraph_first_word",
}


def path(root, key):
    repo, _, _, f = SOURCES[key]
    return os.path.join(root, repo.replace("/", "_"), f)


def meta(key, n, note):
    repo, rev, lic, f = SOURCES[key]
    return {"_meta": {"source": f"https://huggingface.co/datasets/{repo}", "revision": rev, "file": f,
                      "license": lic, "seed": SEED, "n": n, "selection": note}}


def write(name, head, rows):
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), name), "w") as fh:
        fh.write(json.dumps(head) + "\n")
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(name, len(rows))


def main():
    import pyarrow.parquet as pq
    root = sys.argv[1]
    rng = random.Random(SEED)

    g = pq.read_table(path(root, "gsm8k")).to_pylist()
    idx = sorted(rng.sample(range(len(g)), 100))
    rows = [{"id": f"gsm8k-{i}", "question": g[i]["question"],
             "answer": g[i]["answer"].split("####")[-1].strip().replace(",", "")} for i in idx]
    write("gsm8k.jsonl", meta("gsm8k", 100, "random.Random(seed).sample of test indices, sorted"), rows)

    m = pq.read_table(path(root, "mmlu_pro")).to_pylist()
    by = collections.defaultdict(list)
    for r in m:
        by[r["category"]].append(r)
    cats = sorted(by)
    per = {c: 100 // len(cats) for c in cats}
    for c in rng.sample(cats, 100 - sum(per.values())):
        per[c] += 1
    rows = []
    for c in cats:
        for r in rng.sample(by[c], per[c]):
            rows.append({"id": f"mmlupro-{r['question_id']}", "category": c, "question": r["question"],
                         "options": list(r["options"]), "answer": r["answer"]})
    write("mmlu_pro.jsonl", meta("mmlu_pro", len(rows), f"{per} per category, random.Random(seed).sample"), rows)

    ife = [json.loads(l) for l in open(path(root, "ifeval"))]
    ok = [r for r in ife if set(r["instruction_id_list"]) <= IFEVAL_IMPLEMENTED]
    pick = sorted(rng.sample(range(len(ok)), 50))
    rows = [{"id": f"ifeval-{ok[i]['key']}", "prompt": ok[i]["prompt"], "instruction_id_list": ok[i]["instruction_id_list"],
             "kwargs": [{k: v for k, v in kw.items() if v is not None} for kw in ok[i]["kwargs"]]} for i in pick]
    write("ifeval.jsonl", meta("ifeval", 50, f"random sample of the {len(ok)}/{len(ife)} prompts whose instructions "
                                             "are all implemented (no language / sentence-count checks)"), rows)

    h = pq.read_table(path(root, "humaneval")).to_pylist()
    pick = sorted(rng.sample(range(len(h)), 20))
    rows = [{"id": h[i]["task_id"], "prompt": h[i]["prompt"], "entry_point": h[i]["entry_point"], "test": h[i]["test"],
             "canonical_solution": h[i]["canonical_solution"]} for i in pick]
    write("humaneval.jsonl", meta("humaneval", 20, "random sample of 20 of 164 problems"), rows)


if __name__ == "__main__":
    main()
