#!/usr/bin/env python3
"""Plot training losses exported from WandB/SwanLab as CSV files."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def find_column(frame: pd.DataFrame, candidates: tuple[str, ...]) -> str:
    normalized = {column.lower().strip(): column for column in frame.columns}
    for candidate in candidates:
        if candidate in normalized:
            return normalized[candidate]
    for column in frame.columns:
        lowered = column.lower()
        if any(candidate in lowered for candidate in candidates):
            return column
    raise KeyError(f"Could not find any of columns: {candidates}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--single-env-csv", type=Path, required=True)
    parser.add_argument("--multi-env-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("plots/task2_loss_curve.png"))
    args = parser.parse_args()

    single = pd.read_csv(args.single_env_csv)
    multi = pd.read_csv(args.multi_env_csv)

    step_single = find_column(single, ("step", "_step", "global_step"))
    loss_single = find_column(single, ("train/action_l1_loss", "action_l1_loss", "loss"))
    step_multi = find_column(multi, ("step", "_step", "global_step"))
    loss_multi = find_column(multi, ("train/action_l1_loss", "action_l1_loss", "loss"))

    plt.figure(figsize=(7.2, 4.4))
    plt.plot(single[step_single], single[loss_single], label="ACT trained on A")
    plt.plot(multi[step_multi], multi[loss_multi], label="ACT trained on A+B+C")
    plt.xlabel("Training step")
    plt.ylabel("Action L1 loss")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=200)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
