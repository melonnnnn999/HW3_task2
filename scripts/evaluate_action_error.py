#!/usr/bin/env python3
"""Evaluate ACT chunk-level action L1 error on CALVIN splits."""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader


@dataclass
class EvalStats:
    mean_l1: float
    median_l1: float
    std_l1: float
    success_proxy: float
    num_samples: int
    num_episodes: int


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def configure_environment(config: dict[str, Any]) -> None:
    project = config["project"]
    dataset = config["dataset"]
    cache_root = Path(project.get("cache_dir", ".cache")).resolve()
    lerobot_root = Path(dataset.get("train_root", "data/lerobot")).resolve().parent
    os.environ["HF_HOME"] = str(cache_root / "huggingface")
    os.environ["HF_HUB_CACHE"] = str(cache_root / "huggingface" / "hub")
    os.environ["XDG_CACHE_HOME"] = str(cache_root)
    os.environ["HF_LEROBOT_HOME"] = str(lerobot_root)
    for key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "all_proxy"):
        os.environ.pop(key, None)


def split_episodes(config: dict[str, Any], split: str) -> list[int]:
    if split == "d_test":
        limit = config["evaluation"].get("d_episode_limit")
        if limit is None:
            return []
        start = int(config["dataset"].get("test_d_episode_start", 0))
        return list(range(start, start + int(limit)))

    splits = load_json(Path(config["dataset"]["splits_path"]))
    if split == "a_val":
        episodes = list(splits["scenes"]["A"]["val"])
    elif split == "abc_val":
        episodes = list(splits["abc"]["val"])
    else:
        raise ValueError(f"Unknown split: {split}")
    limit = config["evaluation"].get("val_episode_limit")
    if limit:
        episodes = episodes[: int(limit)]
    return episodes


def move_to_device(batch: Any, device: torch.device) -> Any:
    if torch.is_tensor(batch):
        return batch.to(device, non_blocking=True)
    if isinstance(batch, dict):
        return {key: move_to_device(value, device) for key, value in batch.items()}
    if isinstance(batch, list):
        return [move_to_device(value, device) for value in batch]
    if isinstance(batch, tuple):
        return tuple(move_to_device(value, device) for value in batch)
    return batch


def map_batch(batch: dict[str, Any], config: dict[str, Any]) -> dict[str, torch.Tensor]:
    dataset = config["dataset"]
    image = batch[dataset["image_key"]].float()
    state = batch[dataset["state_key"]].float()
    action = batch[dataset["action_key"]].float()
    pad_key = f"{dataset['action_key']}_is_pad"
    action_is_pad = batch.get(pad_key)
    if action_is_pad is None:
        action_is_pad = torch.zeros(action.shape[:-1], dtype=torch.bool, device=action.device)

    mean = torch.tensor([0.485, 0.456, 0.406], device=image.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=image.device).view(1, 3, 1, 1)
    image = (image - mean) / std

    return {
        "observation.images.agentview": image,
        "observation.state": state,
        "action": action,
        "action_is_pad": action_is_pad.bool(),
    }


def load_policy(checkpoint: Path, device: torch.device) -> Any:
    from lerobot.policies.act.modeling_act import ACTPolicy

    policy = ACTPolicy.from_pretrained(str(checkpoint))
    policy.to(device)
    policy.eval()
    return policy


def load_dataset(config: dict[str, Any], split: str, episodes: list[int]) -> Any:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = config["dataset"]
    training = config["training"]
    fps = int(dataset.get("test_d_fps" if split == "d_test" else "train_fps", 10))
    delta_timestamps = {
        dataset["action_key"]: [i / fps for i in range(int(training["chunk_size"]))],
    }

    if split == "d_test":
        return LeRobotDataset(
            dataset["test_d_repo_id"],
            root=dataset.get("test_d_root"),
            revision=dataset.get("test_d_revision", "main"),
            episodes=episodes or None,
            delta_timestamps=delta_timestamps,
            video_backend="pyav",
        )

    return LeRobotDataset(
        dataset["train_repo_id"],
        root=dataset.get("train_root"),
        revision=dataset.get("train_revision", "main"),
        episodes=episodes,
        delta_timestamps=delta_timestamps,
        video_backend="pyav",
    )


def iter_l1_errors(
    policy: Any,
    loader: DataLoader,
    config: dict[str, Any],
    device: torch.device,
    max_batches: int | None,
) -> Iterable[np.ndarray]:
    with torch.inference_mode():
        for batch_index, raw_batch in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            batch = map_batch(move_to_device(raw_batch, device), config)
            prediction = policy.predict_action_chunk(batch)
            target = batch["action"].float()
            mask = ~batch["action_is_pad"].bool()
            abs_error = torch.mean(torch.abs(prediction - target), dim=-1)
            masked = (abs_error * mask.float()).sum(dim=1) / mask.float().sum(dim=1).clamp_min(1)
            yield masked.detach().cpu().numpy()


def bootstrap_ci(values: np.ndarray, samples: int, seed: int) -> tuple[float, float]:
    if samples <= 0 or len(values) == 0:
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        draw = rng.choice(values, size=len(values), replace=True)
        means[index] = float(np.mean(draw))
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/task2_act_calvin.yaml"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--split", choices=["a_val", "abc_val", "d_test"], default="d_test")
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    config = load_yaml(args.config)
    configure_environment(config)
    eval_config = config["evaluation"]
    training_config = config["training"]
    episodes = split_episodes(config, args.split)

    device_name = training_config.get("device", "cuda")
    if device_name == "cuda" and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)

    dataset = load_dataset(config, args.split, episodes)
    loader = DataLoader(
        dataset,
        batch_size=int(eval_config["batch_size"]),
        shuffle=False,
        num_workers=int(eval_config.get("num_workers", 0)),
        pin_memory=device.type == "cuda",
    )
    policy = load_policy(args.checkpoint, device)

    chunks = list(iter_l1_errors(policy, loader, config, device, eval_config.get("max_batches")))
    if not chunks:
        raise RuntimeError("No evaluation batches were produced.")

    errors = np.concatenate(chunks)
    threshold = float(eval_config["success_threshold_l1"])
    ci_low, ci_high = bootstrap_ci(errors, int(eval_config["bootstrap_samples"]), int(config["project"]["seed"]))
    stats = EvalStats(
        mean_l1=float(np.mean(errors)),
        median_l1=float(np.median(errors)),
        std_l1=float(np.std(errors)),
        success_proxy=float(np.mean(errors <= threshold)),
        num_samples=int(errors.size),
        num_episodes=len(episodes) if episodes else int(getattr(dataset, "num_episodes", 0)),
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "run_name": args.run_name,
        "split": args.split,
        "checkpoint": str(args.checkpoint),
        "success_proxy_threshold_l1": threshold,
        "mean_l1_ci95": [ci_low, ci_high],
        **stats.__dict__,
    }
    output_path = args.output_dir / f"{args.run_name}_{args.split}.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
