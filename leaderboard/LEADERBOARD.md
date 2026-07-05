# TOFB Leaderboard

Faithful, query-aware compression of LLM agent tool outputs. Tracks (a) answer faithfulness/retention, (b) downstream-extractability (does the compressed output still parse in code), and (c) end-to-end agent-task success (parse + arithmetic, no LLM judge).

## agent_tasks

End-to-end task success: the agent must parse the compressed output and count/sum/list/lookup over it. Success decided by real parsing + arithmetic, no LLM judge.

| method | model | budget | metric | value | n | p vs RAG | source |
|:--|:--|--:|:--|--:|--:|:--:|:--|
| `tooltrim` | — | 256 | task_success | 100% | 10 | 0.013* | `run_agent_tasks.py` |
| `rag-topk` | — | 256 | task_success | 20% | 10 | — | `run_agent_tasks.py` |
| `truncate-head` | — | 256 | task_success | 0% | 10 | — | `run_agent_tasks.py` |

## downstream_extractability

Fraction of compressed JSON/tabular outputs that parse in code AND still contain the gold fact. Deterministic, judge-independent.

| method | model | budget | metric | value | n | p vs RAG | source |
|:--|:--|--:|:--|--:|--:|:--:|:--|
| `tooltrim` | — | 256 | extractable_rate | 57% | 66 | — | `run_frontier.py` |
| `rag-topk` | — | 256 | extractable_rate | 39% | 66 | — | `run_frontier.py` |

## faithfulness

Answer accuracy retention vs full context; offline BM25 judge and real-LLM judge. n=66.

| method | model | budget | metric | value | n | p vs RAG | source |
|:--|:--|--:|:--|--:|--:|:--:|:--|
| `tooltrim` | claude-opus-4-8 | 256 | accuracy | 100% | 66 | — | `run_frontier.py` |
| `tooltrim` | claude-haiku-4-5 | 256 | accuracy | 100% | 66 | — | `run_frontier.py` |
| `tooltrim` | gpt-4o-mini | 256 | accuracy | 97% | 66 | — | `run_frontier.py` |
| `tooltrim` | claude-sonnet-5 | 256 | accuracy | 92% | 66 | — | `run_frontier.py` |
| `truncate-head` | — | 256 | retention | 4% | 66 | — | `run_baselines.py` |

\* significant (paired McNemar, p<0.05). Regenerate: `python leaderboard/validate.py --render`.
