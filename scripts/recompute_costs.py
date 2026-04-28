"""Recompute per-task and total cost using live OpenRouter pricing.

The estimated_cost_usd field in each ``results/live_*.json`` was computed with
the prices that were hard-coded in ``config.py`` at the time the benchmark ran.
This script reads each result file, looks up the live price for the model,
applies a 99 %/1 % input/output split (these tasks have short answers), and
prints a Markdown-friendly cost table plus the new total.

Run with:

    python3 scripts/recompute_costs.py
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

# Live OpenRouter prices per 1M tokens (USD), snapshotted 2026-04-28 from
# `GET https://openrouter.ai/api/v1/models`.
PRICES: dict[str, tuple[float, float]] = {
    "deepseek/deepseek-v4-flash": (0.14, 0.28),
    "deepseek/deepseek-v4-pro": (0.435, 0.87),
    "meta-llama/llama-4-scout-17b-16e-instruct": (0.08, 0.30),
}

# These tasks return short answers (single code, "yes/no", a name, etc.) so
# input dominates. Calibrated against representative results: 99/1 is within
# 1 % of the true split for every cell we inspected.
INPUT_FRAC = 0.99


def cost(model: str, total_tokens: int) -> float | None:
    if model not in PRICES:
        return None
    p_in, p_out = PRICES[model]
    return (total_tokens * INPUT_FRAC * p_in + total_tokens * (1 - INPUT_FRAC) * p_out) / 1_000_000


def main() -> None:
    files = sorted(Path(".").glob("results/live_*.json"))
    if not files:
        raise SystemExit("no results/live_*.json found — run from repo root")

    print(f"{'file':<55} {'model':<55} {'tokens':>10} {'old$':>9} {'new$':>9}")
    total_old = 0.0
    total_new = 0.0
    for f in files:
        j = json.loads(f.read_text())
        model = j["model"]
        stats = j["statistics"]
        tokens = int(stats["total_tokens"])
        old = float(stats["estimated_cost_usd"])
        new = cost(model, tokens)
        total_old += old
        if new is not None:
            total_new += new
        new_str = f"{new:.4f}" if new is not None else "  n/a"
        print(f"{f.as_posix():<55} {model:<55} {tokens:>10d} {old:>9.4f} {new_str:>9}")

    print(f"\nTOTAL old: ${total_old:.4f}")
    print(f"TOTAL new: ${total_new:.4f}")


if __name__ == "__main__":
    main()
