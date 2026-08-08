"""Reproducibility manifest + raw-results persistence for publishable runs.

A point estimate in a markdown table is not a scientific artifact. To be
citable (and to survive an artifact-evaluation review), every run must emit:

  * a *manifest* pinning exactly what produced the numbers — tau-bench commit,
    the resolved model snapshot (not the floating alias), seed, package
    versions, timestamp, and the full CLI args; and
  * the *raw* per-task / per-trial rewards and compression accounting, so any
    statistic (CIs, McNemar, budget curves) can be re-derived offline without
    re-spending on the API.

Both are plain JSON so they diff cleanly in git and drop straight into a paper's
supplementary material.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
import threading
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple


# Floating aliases resolve to a dated snapshot server-side; pinning the snapshot
# is what makes a run reproducible months later. Extend as models are added.
MODEL_SNAPSHOTS = {
    "gpt-4o-mini": "gpt-4o-mini-2024-07-18",
    "gpt-4o": "gpt-4o-2024-08-06",
}


def resolve_model_snapshot(model: str) -> str:
    """Map a floating model alias to its pinned dated snapshot, if known."""
    return MODEL_SNAPSHOTS.get(model, model)


def _pkg_version(name: str) -> str:
    try:
        return metadata.version(name)
    except Exception:
        return "unknown"


def _taubench_commit() -> str:
    """Best-effort tau-bench commit SHA (editable installs); else version."""
    try:
        import tau_bench  # type: ignore

        pkg_dir = Path(tau_bench.__file__).resolve().parent.parent
        sha = subprocess.run(
            ["git", "-C", str(pkg_dir), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if sha.returncode == 0 and sha.stdout.strip():
            return sha.stdout.strip()
    except Exception:
        pass
    return _pkg_version("tau-bench")


def build_manifest(args: Dict[str, Any], *, seed: int | None = None) -> Dict[str, Any]:
    """Assemble the reproducibility manifest for a run from its resolved args."""
    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "args": dict(args),
        "models": {
            "agent": resolve_model_snapshot(args.get("agent_model", "")),
            "user": resolve_model_snapshot(args.get("user_model", "")),
        },
        "versions": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "tooltrim": _pkg_version("tooltrim"),
            "tau_bench_commit": _taubench_commit(),
            "litellm": _pkg_version("litellm"),
            "openai": _pkg_version("openai"),
        },
    }


def _statobj(st: Any) -> Dict[str, Any]:
    if is_dataclass(st) and not isinstance(st, type):
        return asdict(st)
    # duck-typed fallback
    keys = ("tool_calls", "compressed", "raw_tokens", "context_tokens")
    return {k: getattr(st, k, None) for k in keys}


def save_results(path: str | Path, *, manifest: Dict[str, Any],
                 rewards_by_method: Dict[str, Dict[int, Any]],
                 stats_by_method: Dict[str, Any],
                 budget: int) -> Path:
    """Persist manifest + raw rewards + compression accounting as JSON.

    ``rewards_by_method[m][task_id]`` may be a single float or a list of floats
    (one per trial); both are serialized verbatim so no information is lost.
    Task-id keys are stringified for JSON.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "manifest": manifest,
        "budget": budget,
        "rewards": {
            m: {str(i): v for i, v in tasks.items()}
            for m, tasks in rewards_by_method.items()
        },
        "stats": {m: _statobj(st) for m, st in stats_by_method.items()},
    }
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path


# Sentinel budget for budget-independent conditions (e.g. `none`): they never
# compress, so one solve is valid at every budget. Keying them with this constant
# makes resume match regardless of which budget list the run was invoked with.
_BUDGET_INDEPENDENT_KEY = -1

Key = Tuple[str, int, int, int]  # (method, budget, task, trial)


class Checkpoint:
    """Append-only, per-solve checkpoint that makes a run resumable.

    Every completed ``(method, budget, task, trial)`` solve is written as one
    JSON line with its reward and compression-stat deltas. On a later run with
    the same checkpoint file, already-done solves are skipped and their
    contributions reloaded instead of re-spending on the API — so an interrupted
    sweep (rate-limit wall, Ctrl-C, sleep) continues from exactly where it
    stopped. Writes are flushed per line and guarded by a lock, so a fresh
    process (or a concurrent worker) sees a consistent, torn-write-tolerant file.

    ``budget_independent`` names methods keyed with a sentinel budget so, e.g.,
    ``none`` resumes regardless of the budget list.
    """

    def __init__(self, path: str | Path, *, budget_independent: Iterable[str] = ()):
        self.path = Path(path)
        self.budget_independent = frozenset(budget_independent)
        self._lock = threading.Lock()
        self._done: Dict[Key, Dict[str, Any]] = {}

    def key(self, method: str, budget: int, task: int, trial: int) -> Key:
        b = _BUDGET_INDEPENDENT_KEY if method in self.budget_independent else int(budget)
        return (method, b, int(task), int(trial))

    def load(self) -> "Checkpoint":
        """Read any existing checkpoint into memory (tolerates a torn last line)."""
        self._done = {}
        if not self.path.exists():
            return self
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue  # a partially-written final line from a hard kill; skip it
            k: Key = (rec["method"], int(rec["budget"]), int(rec["task"]), int(rec["trial"]))
            self._done[k] = rec
        return self

    def __len__(self) -> int:
        return len(self._done)

    def has(self, key: Key) -> bool:
        return key in self._done

    def get(self, key: Key) -> Dict[str, Any]:
        return self._done[key]

    def record(self, method: str, budget: int, task: int, trial: int,
               reward: float, stats: Any) -> None:
        """Append one completed solve and flush it to disk immediately."""
        k = self.key(method, budget, task, trial)
        rec = {"method": method, "budget": k[1], "task": int(task),
               "trial": int(trial), "reward": float(reward), "stats": _statobj(stats)}
        with self._lock:
            self._done[k] = rec
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, default=str) + "\n")
                f.flush()
