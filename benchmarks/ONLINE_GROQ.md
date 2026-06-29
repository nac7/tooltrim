# tooltrim online — hosted LLM, real rate limits (Groq)

Run against a live hosted model (Groq, `llama-3.1-8b-instant`) on **2026-06-28**.
This is the *online* counterpart to the offline/local faithfulness runs: it shows
tooltrim working in front of a real provider, and surfaces an effect you only see
online — **provider rate limits**.

## The finding: raw tool outputs don't fit; compressed ones do

Groq's free tier caps a single request at **6,000 tokens per minute**. Real tool
outputs routinely exceed that, so the call is rejected outright (HTTP 413) before
the model ever runs. tooltrim compresses each tool result first, so the same call
fits.

Over the 62-case benchmark dataset (deterministic, no API needed to reproduce):

| | fits Groq free-tier 6,000-TPM request cap |
|---|---:|
| **raw** tool outputs | **28 / 62 (45%)** |
| **tooltrim-compressed** (@400) | **62 / 62 (100%)** |

> Without tooltrim, **34 of 62** tool-result calls are rejected by the free tier.
> With tooltrim, every call is admissible.

Reproduce:

```bash
python - <<'PY'
from eval import default_cases
from tooltrim import ToolCompressor, count_tokens
TPM = 6000
cs = default_cases()
raw = sum(1 for x in cs if count_tokens(x.tool_output) <= TPM)
tc = ToolCompressor(max_tokens=400, add_footer=False)
comp = sum(1 for x in cs if tc.compress(x.tool_output, query=x.question).compressed_tokens <= TPM)
print(f"raw fits:        {raw}/{len(cs)}")
print(f"compressed fits: {comp}/{len(cs)}")
PY
```

## Live proxy A/B (same call, same hosted model)

The [proxy](../run_proxy.py) in front of Groq, with a 14,415-token tool output:

```
raw tool output: 14,415 tokens   (free-tier per-request cap = 6,000)

A) DIRECT to Groq         -> HTTP 413: "Request too large ... Limit 6000, Requested 14484"
B) THROUGH tooltrim proxy -> compressed in flight: 14,415 -> 26 tokens (saved 14,389)
                          -> Groq accepts; reply: "customer 4417 received a refund ... $420."
```

Start the proxy and reproduce:

```bash
python run_proxy.py --upstream https://api.groq.com/openai/v1 --max-tokens 400
# then send any OpenAI-compatible request whose role:"tool" message is large;
# the direct call 413s, the proxied call succeeds.
```

## Why this matters for enterprises

The token/cost story (96–98% fewer tokens, see [COMPARISON.md](COMPARISON.md)) is
the same online. But online adds a hard constraint the offline runs can't show:
**provider rate limits and per-request size caps**. Compression converts calls
that are *rejected* (or that consume an outsized share of a TPM/TPD quota) into
calls that fit — so the same quota serves more traffic, and a drop-in proxy in
front of any team's traffic keeps every app under the cap with no code change.
