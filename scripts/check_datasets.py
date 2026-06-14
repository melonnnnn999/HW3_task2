#!/usr/bin/env python3
"""Check that the configured CALVIN LeRobot datasets are reachable."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def import_dataset_classes() -> tuple[Any, Any | None]:
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

        return LeRobotDataset, LeRobotDatasetMetadata
    except ImportError:
        try:
            from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

            return LeRobotDataset, LeRobotDatasetMetadata
        except ImportError:
            try:
                from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

                return LeRobotDataset, None
            except ImportError as exc:
                raise RuntimeError("LeRobot dataset classes are unavailable. Install LeRobot first.") from exc


def configure_cache(config: dict[str, Any]) -> None:
    cache_root = Path(config["project"].get("cache_dir", ".cache")).resolve()
    train_root = Path(config["dataset"].get("train_root", "data/lerobot")).resolve().parent
    os.environ["HF_HOME"] = str(cache_root / "huggingface")
    os.environ["HF_HUB_CACHE"] = str(cache_root / "huggingface" / "hub")
    os.environ["XDG_CACHE_HOME"] = str(cache_root)
    os.environ["HF_LEROBOT_HOME"] = str(train_root)
    for key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "all_proxy"):
        os.environ.pop(key, None)


def load_splits(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def describe_dataset(
    label: str,
    repo_id: str,
    root: str | None,
    revision: str,
    action_key: str,
    episodes: list[int],
) -> None:
    if "your-hf-username" in repo_id:
        print(f"[{label}] placeholder repo id: {repo_id}")
        return

    try:
        LeRobotDataset, LeRobotDatasetMetadata = import_dataset_classes()
    except RuntimeError as exc:
        print(f"[{label}] cannot inspect {repo_id}: {exc}")
        return
    if LeRobotDatasetMetadata is not None:
        metadata = LeRobotDatasetMetadata(repo_id, root=root, revision=revision)
        features = getattr(metadata, "features", {})
        print(f"[{label}] {repo_id}")
        print(f"  root: {root}")
        print(f"  revision: {revision}")
        print(f"  total episodes/frames: {metadata.total_episodes}/{metadata.total_frames}")
        print(f"  features: {', '.join(sorted(features.keys()))}")

    dataset = LeRobotDataset(repo_id, root=root, revision=revision, episodes=episodes, video_backend="pyav")
    print(f"  episodes/samples: {len(dataset)} samples")
    sample = dataset[0]
    print(f"  sample keys: {', '.join(sorted(sample.keys()))}")
    if action_key not in sample:
        raise KeyError(f"`{action_key}` is missing from {label} sample.")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/task2_act_calvin.yaml"))
    args = parser.parse_args()

    config = load_yaml(args.config)
    configure_cache(config)
    dataset_config = config["dataset"]
    action_key = dataset_config["action_key"]
    splits = load_splits(dataset_config["splits_path"])
    describe_dataset(
        "train_a",
        dataset_config["train_repo_id"],
        dataset_config.get("train_root"),
        dataset_config.get("train_revision", "main"),
        action_key,
        list(splits["scenes"]["A"]["train"][:1]),
    )
    describe_dataset(
        "train_abc",
        dataset_config["train_repo_id"],
        dataset_config.get("train_root"),
        dataset_config.get("train_revision", "main"),
        action_key,
        list(splits["abc"]["train"][:1]),
    )
    describe_dataset(
        "test_d",
        dataset_config["test_d_repo_id"],
        dataset_config.get("test_d_root"),
        dataset_config.get("test_d_revision", "main"),
        action_key,
        [int(dataset_config.get("test_d_episode_start", 0))],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
