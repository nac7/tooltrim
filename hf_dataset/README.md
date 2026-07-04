---
license: mit
task_categories:
- question-answering
- text-retrieval
language:
- en
tags:
- llm-agents
- tool-use
- context-compression
- faithfulness
- long-context
- evaluation
pretty_name: Tool-Output Faithfulness Benchmark (TOFB)
size_categories:
- n<1K
configs:
- config_name: default
  data_files:
  - split: test
    path: data/tofb.jsonl
dataset_info:
  features:
  - name: id
    dtype: string
  - name: content_type
    dtype: string
  - name: category
    dtype: string
  - name: question
    dtype: string
  - name: gold
    dtype: string
  - name: all_of
    sequence: string
  - name: must_not
    sequence: string
  - name: tool_output
    dtype: string
  - name: tool_output_tokens
    dtype: int64
  splits:
  - name: test
    num_examples: 62
---

# Tool-Output Faithfulness Benchmark (TOFB)

**TOFB** measures whether an LLM agent can still recover a needed fact after a
large, realistic **tool output** (HTML page, JSON API response, server logs, CSV
export, long prose) is compressed before re-entering the model's context. It is
the evaluation dataset behind [**tooltrim**](https://github.com/nac7/tooltrim),
a drop-in library for compressing agent tool outputs.

Each case plants one distinctive fact (a "needle") inside a bloated,
type-appropriate tool output and pairs it with a question. A method passes a
case if the model recovers the gold fact from the (possibly compressed) output
it was given. Filler is generated deterministically (seeded), so the dataset is
fully reproducible from source.

## Why this dataset

Tool outputs are the largest and most-ignored token sink in agent loops. The
usual "fix" — blind truncation — silently drops the fact the agent needs. TOFB
isolates that failure mode: it rewards methods that **keep the answer** while
cutting tokens, and penalizes those that keep bytes but lose meaning. Because
every case has a single checkable gold string (plus optional multi-fact and
distractor variants), scoring is unambiguous and works with both an offline
lexical judge and a real-LLM judge.

## Composition (62 cases)

| Category | Count | What it tests |
|---|---:|---|
| `single` | 50 | One fact buried in bloat; question overlaps the needle. |
| `multi` | 6 | Two facts placed far apart — both must be recovered (`all_of`). |
| `distractor` | 6 | A similar-but-wrong value sits far from the right one; the answer must appear and the trap (`must_not`) must not. |

Content types: text (18), json (14), html (10), logs (10), tabular (10).
Mean tool-output size ≈ 6.6k tokens/case (≈411k tokens total).

## Schema

| Field | Type | Description |
|---|---|---|
| `id` | string | Stable case id, e.g. `text-01`, `multi-03`, `distractor-05`. |
| `content_type` | string | `text` / `html` / `json` / `logs` / `tabular`. |
| `category` | string | `single` / `multi` / `distractor`. |
| `question` | string | The agent's goal; also the relevance query for compression. |
| `gold` | string | The fact that must be recovered. |
| `all_of` | list[string] | Additional facts that must *also* appear (multi). |
| `must_not` | list[string] | Distractor values that must *not* appear. |
| `tool_output` | string | The full, bloated tool output shown to the model. |
| `tool_output_tokens` | int | Token count of `tool_output` (cl100k). |

## Usage

```python
from datasets import load_dataset

ds = load_dataset("nac7/tool-output-faithfulness", split="test")
ex = ds[0]
print(ex["question"], "->", ex["gold"])
print("output tokens:", ex["tool_output_tokens"])
```

Scoring a case (single-fact): the model's answer passes if it contains `gold`
(case-insensitive, whitespace-normalized). For `multi`, every string in `gold` +
`all_of` must appear. For `distractor`, `gold` must appear and no string in
`must_not` may appear. The reference implementation of the judge, the offline
BM25 control model, and real-LLM adapters live in the
[tooltrim `eval/`](https://github.com/nac7/tooltrim/tree/main/eval) package.

## Reproducing / extending

The dataset is generated from source (no scraping, no PII):

```bash
git clone https://github.com/nac7/tooltrim && cd tooltrim
python hf_dataset/export.py   # regenerates data/tofb.jsonl deterministically
```

## Citation

```bibtex
@misc{lele2026tofb,
  title  = {Tool-Output Faithfulness Benchmark (TOFB)},
  author = {Lele, Nachiket},
  year   = {2026},
  howpublished = {\url{https://huggingface.co/datasets/nac7/tool-output-faithfulness}},
  note   = {Companion dataset to tooltrim}
}
```

## License

MIT. The tool outputs are synthetic and deterministically generated; the dataset
contains no personal or copyrighted source material.
