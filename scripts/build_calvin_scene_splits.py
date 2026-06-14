#!/usr/bin/env python3
"""Build CALVIN A/B/C episode splits from official task_ABC_D metadata."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


def scene_for_range(scene_info: dict[str, list[int]], start: int, end: int) -> str:
    for name, raw_range in scene_info.items():
        scene_start, scene_end = int(raw_range[0]), int(raw_range[1])
        if max(start, scene_start) <= min(end, scene_end):
            return name.rsplit("_", 1)[-1]
    raise ValueError(f"No CALVIN scene matched range {start}-{end}.")


def split_indices(indices: list[int], val_ratio: float, rng: np.random.Generator) -> tuple[list[int], list[int]]:
    shuffled = np.array(indices, dtype=np.int64)
    rng.shuffle(shuffled)
    val_count = max(1, int(round(len(shuffled) * val_ratio)))
    val = sorted(int(x) for x in shuffled[:val_count])
    train = sorted(int(x) for x in shuffled[val_count:])
    return train, val


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta-dir", type=Path, default=Path("data/raw_meta/task_ABC_D/training"))
    parser.add_argument("--output", type=Path, default=Path("data/splits/calvin_scene_splits.json"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    args = parser.parse_args()

    scene_info = np.load(args.meta_dir / "scene_info.npy", allow_pickle=True).item()
    annotations = np.load(args.meta_dir / "lang_annotations" / "auto_lang_ann.npy", allow_pickle=True).item()
    ranges = np.asarray(annotations["info"]["indx"])

    by_scene: dict[str, list[int]] = {"A": [], "B": [], "C": []}
    for episode_index, (start, end) in enumerate(ranges):
        scene = scene_for_range(scene_info, int(start), int(end))
        by_scene.setdefault(scene, []).append(episode_index)

    rng = np.random.default_rng(args.seed)
    splits: dict[str, object] = {
        "source": {
            "dataset_repo_id": "Traly/calvin_abc_d-lerobot",
            "scene_metadata_source": "official CALVIN task_ABC_D training metadata",
            "metadata": str(args.meta_dir),
            "seed": args.seed,
            "val_ratio": args.val_ratio,
        },
        "counts": dict(sorted(Counter({scene: len(values) for scene, values in by_scene.items()}).items())),
        "scenes": {},
    }

    train_abc: list[int] = []
    val_abc: list[int] = []
    for scene in sorted(by_scene):
        train, val = split_indices(by_scene[scene], args.val_ratio, rng)
        splits["scenes"][scene] = {
            "all": by_scene[scene],
            "train": train,
            "val": val,
        }
        train_abc.extend(train)
        val_abc.extend(val)

    splits["abc"] = {
        "all": sorted(train_abc + val_abc),
        "train": sorted(train_abc),
        "val": sorted(val_abc),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(splits, indent=2), encoding="utf-8")

    print(f"wrote {args.output}")
    print("scene counts:", splits["counts"])
    print("A train/val:", len(splits["scenes"]["A"]["train"]), len(splits["scenes"]["A"]["val"]))
    print("ABC train/val:", len(splits["abc"]["train"]), len(splits["abc"]["val"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
