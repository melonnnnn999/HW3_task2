#!/usr/bin/env python3
"""Create a compact Markdown summary from task2 evaluation JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_results(results_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(results_dir.glob("act_calvin_*.json")):
        with path.open("r", encoding="utf-8") as handle:
            rows.append(json.load(handle))
    return rows


def fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path, default=Path("results/task2_summary.md"))
    args = parser.parse_args()

    rows = load_results(args.results_dir)
    if not rows:
        raise RuntimeError(f"No evaluation JSON files found in {args.results_dir}")

    lines = [
        "# Task 2 Evaluation Summary",
        "",
        "| Run | Split | Mean L1 | Median L1 | Std L1 | Success Proxy | Samples | Episodes |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {run_name} | {split} | {mean_l1} | {median_l1} | {std_l1} | {success_proxy} | {num_samples} | {num_episodes} |".format(
                run_name=row["run_name"],
                split=row.get("split", "d_test"),
                mean_l1=fmt(row["mean_l1"]),
                median_l1=fmt(row["median_l1"]),
                std_l1=fmt(row["std_l1"]),
                success_proxy=fmt(row["success_proxy"]),
                num_samples=row["num_samples"],
                num_episodes=row.get("num_episodes", ""),
            )
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
