"""Push the Tool-Output Faithfulness Benchmark to the HuggingFace Hub.

Prerequisites:
    pip install huggingface_hub
    export HF_TOKEN=hf_...          # a write token from https://hf.co/settings/tokens
    python hf_dataset/export.py     # (re)generate data/tofb.jsonl first

Then:
    python hf_dataset/upload.py --repo nac7/tool-output-faithfulness

This creates the dataset repo if needed and uploads the card + data. Idempotent:
re-running updates the files in place.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="nac7/tool-output-faithfulness",
                    help="target dataset repo id (owner/name)")
    ap.add_argument("--private", action="store_true",
                    help="create the repo as private")
    ap.add_argument("--token", default=os.environ.get("HF_TOKEN"),
                    help="HF write token (defaults to $HF_TOKEN, then to the "
                         "token saved by `huggingface-cli login`)")
    args = ap.parse_args()

    # Fall back to a persisted login (~/.cache/huggingface/token) so the token
    # need not be re-supplied every run. token=None lets HfApi use the cache.
    if not args.token:
        try:
            from huggingface_hub import get_token
            args.token = get_token()
        except Exception:
            args.token = None
    if not args.token:
        print("error: no token. Set HF_TOKEN, pass --token, or run "
              "`huggingface-cli login`.", file=sys.stderr)
        return 2

    data = HERE / "data" / "tofb.jsonl"
    if not data.exists():
        print(f"error: {data} missing — run `python hf_dataset/export.py` first.",
              file=sys.stderr)
        return 2

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("error: pip install huggingface_hub", file=sys.stderr)
        return 2

    api = HfApi(token=args.token)
    api.create_repo(repo_id=args.repo, repo_type="dataset",
                    private=args.private, exist_ok=True)
    api.upload_folder(
        repo_id=args.repo,
        repo_type="dataset",
        folder_path=str(HERE),
        allow_patterns=["README.md", "data/*.jsonl"],
        commit_message="Upload Tool-Output Faithfulness Benchmark (TOFB)",
    )
    print(f"done: https://huggingface.co/datasets/{args.repo}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
