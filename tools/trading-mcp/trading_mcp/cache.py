"""Persistent cache of dependency verdicts.

A sweep over thousands of market pairs is dominated by LLM cost, and the same
pairs recur on every run. Verdicts are keyed by market text rather than by
market id, so a pair stays cached across restarts and across id churn, and the
file is rewritten atomically so a crash mid-write cannot corrupt it.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any


def pair_key(question_a: str, question_b: str) -> str:
    """Return an order-independent key for a market pair.

    Sorting before hashing means (A, B) and (B, A) share one entry — the
    dependency relation is symmetric, and paying an LLM twice for it is waste.
    """
    first, second = sorted((question_a.strip(), question_b.strip()))
    digest = hashlib.sha256(f"{first}\x00{second}".encode())
    return digest.hexdigest()[:32]


class DependencyCache:
    """A JSON-backed store of dependency verdicts, loaded once and written through."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._entries: dict[str, dict[str, Any]] = {}
        self.hits = 0
        self.misses = 0
        self._load()

    def _load(self) -> None:
        """Read the cache file, tolerating absence and corruption.

        A damaged cache is a performance problem, not a correctness one: start
        empty and let the sweep repopulate rather than failing the run.
        """
        if not self.path.exists():
            return
        try:
            loaded = json.loads(self.path.read_text())
        except (json.JSONDecodeError, OSError):
            return
        if isinstance(loaded, dict):
            self._entries = {k: v for k, v in loaded.items() if isinstance(v, dict)}

    def get(self, question_a: str, question_b: str) -> dict[str, Any] | None:
        """Return a stored verdict, or None on a miss."""
        entry = self._entries.get(pair_key(question_a, question_b))
        if entry is None:
            self.misses += 1
            return None
        self.hits += 1
        return entry["verdict"]

    def put(self, question_a: str, question_b: str, verdict: dict[str, Any]) -> None:
        """Store a verdict and flush the cache to disk."""
        self._entries[pair_key(question_a, question_b)] = {
            "questions": [question_a, question_b],
            "verdict": verdict,
            "stored_at": time.time(),
        }
        self._flush()

    def _flush(self) -> None:
        """Write the cache atomically so an interrupted run leaves a valid file."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle, temp_path = tempfile.mkstemp(dir=self.path.parent, suffix=".tmp")
        try:
            with os.fdopen(handle, "w") as stream:
                json.dump(self._entries, stream)
            os.replace(temp_path, self.path)
        except BaseException:
            Path(temp_path).unlink(missing_ok=True)
            raise

    def stats(self) -> dict[str, Any]:
        """Return cache size and this session's hit/miss counts."""
        total = self.hits + self.misses
        return {
            "path": str(self.path),
            "entries": len(self._entries),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hits / total if total else None,
        }

    def __len__(self) -> int:
        return len(self._entries)
