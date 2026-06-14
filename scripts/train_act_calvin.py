#!/usr/bin/env python3
"""Train LeRobot ACT on CALVIN scene splits.

The training loop uses LeRobotDataset for CALVIN data loading and ACTPolicy
for the policy implementation, while selecting the assignment-specific
environment A and mixed A/B/C episode splits.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm


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
    os.environ.setdefault("WANDB_DIR", str((cache_root / "wandb").resolve()))
    for key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "all_proxy"):
        os.environ.pop(key, None)


def select_episodes(config: dict[str, Any], run_kind: str) -> list[int]:
    splits = load_json(Path(config["dataset"]["splits_path"]))
    per_scene_limit = config["training"].get("train_episode_limit_per_scene")
    if run_kind == "a":
        episodes = list(splits["scenes"]["A"]["train"])
        if per_scene_limit:
            episodes = episodes[: int(per_scene_limit)]
    elif per_scene_limit:
        episodes = []
        for scene in ("A", "B", "C"):
            episodes.extend(list(splits["scenes"][scene]["train"][: int(per_scene_limit)]))
        episodes = sorted(episodes)
    else:
        episodes = list(splits["abc"]["train"])

    limit = config["training"].get("train_episode_limit")
    if limit:
        episodes = episodes[: int(limit)]
    return episodes


def make_policy(config: dict[str, Any], device: torch.device) -> torch.nn.Module:
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy

    training = config["training"]
    dataset = config["dataset"]
    policy_config = ACTConfig(
        input_features={
            "observation.images.agentview": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 200, 200)),
            "observation.state": PolicyFeature(
                type=FeatureType.STATE,
                shape=(int(dataset.get("state_dim", 15)),),
            ),
        },
        output_features={
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(int(dataset.get("action_dim", 7)),)),
        },
        chunk_size=int(training["chunk_size"]),
        n_action_steps=int(training["n_action_steps"]),
        optimizer_lr=float(training["learning_rate"]),
        optimizer_weight_decay=float(training["weight_decay"]),
        optimizer_lr_backbone=float(training["learning_rate"]),
        device=str(device),
    )
    policy = ACTPolicy(policy_config)
    return policy.to(device)


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


def make_dataset(config: dict[str, Any], episodes: list[int]) -> Any:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = config["dataset"]
    training = config["training"]
    fps = int(dataset.get("train_fps", 10))
    delta_timestamps = {
        dataset["action_key"]: [i / fps for i in range(int(training["chunk_size"]))],
    }
    return LeRobotDataset(
        dataset["train_repo_id"],
        root=dataset.get("train_root"),
        revision=dataset.get("train_revision", "main"),
        episodes=episodes,
        delta_timestamps=delta_timestamps,
        video_backend="pyav",
    )


def save_policy(policy: torch.nn.Module, output_dir: Path, step: int, is_last: bool = False) -> None:
    checkpoint_dir = output_dir / "checkpoints" / f"{step:06d}" / "pretrained_model"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(checkpoint_dir)

    if is_last:
        last_dir = output_dir / "checkpoints" / "last"
        if last_dir.exists() or last_dir.is_symlink():
            if last_dir.is_symlink() or last_dir.is_file():
                last_dir.unlink()
            else:
                shutil.rmtree(last_dir)
        last_dir.mkdir(parents=True)
        shutil.copytree(checkpoint_dir, last_dir / "pretrained_model")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def maybe_init_wandb(config: dict[str, Any], run_kind: str, output_dir: Path) -> Any | None:
    training = config["training"]
    if training.get("tracker") != "wandb":
        return None
    try:
        import wandb

        return wandb.init(
            project=config["project"]["name"],
            name=f"act_calvin_{run_kind}",
            mode=training.get("wandb_mode", "offline"),
            dir=os.environ.get("WANDB_DIR"),
            config=config,
            notes=f"outputs: {output_dir}",
        )
    except Exception as exc:
        print(f"wandb init failed; continuing with local JSONL logs only: {exc}")
        return None


def train(config: dict[str, Any], run_kind: str) -> Path:
    configure_environment(config)
    seed = int(config["project"]["seed"])
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    device_name = config["training"].get("device", "cuda")
    if device_name == "cuda" and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)

    episodes = select_episodes(config, run_kind)
    output_dir = Path(config["project"]["output_dir"]) / f"act_calvin_{run_kind}"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (output_dir / "episodes.json").write_text(json.dumps(episodes), encoding="utf-8")

    dataset = make_dataset(config, episodes)
    loader = DataLoader(
        dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=True,
        num_workers=int(config["training"]["num_workers"]),
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    iterator = iter(loader)

    policy = make_policy(config, device)
    optimizer = torch.optim.AdamW(
        policy.get_optim_params(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    wandb_run = maybe_init_wandb(config, run_kind, output_dir)

    metrics_path = output_dir / "train_metrics.jsonl"
    if metrics_path.exists():
        metrics_path.unlink()

    steps = int(config["training"]["steps"])
    log_freq = int(config["training"]["log_freq"])
    save_freq = int(config["training"]["save_freq"])
    start_time = time.time()
    progress = tqdm(
        range(1, steps + 1),
        desc=f"act_calvin_{run_kind}",
        disable=os.environ.get("TQDM_ENABLE") != "1",
    )

    for step in progress:
        try:
            raw_batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            raw_batch = next(iterator)

        batch = map_batch(move_to_device(raw_batch, device), config)
        policy.train()
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
            loss, loss_dict = policy(batch)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), 10.0)
        scaler.step(optimizer)
        scaler.update()

        if step % log_freq == 0 or step == 1:
            row = {
                "step": step,
                "loss": float(loss.detach().cpu()),
                "action_l1_loss": float(loss_dict["l1_loss"]),
                "grad_norm": float(grad_norm.detach().cpu()),
                "lr": optimizer.param_groups[0]["lr"],
                "elapsed_s": time.time() - start_time,
                "num_train_episodes": len(episodes),
            }
            append_jsonl(metrics_path, row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
            if wandb_run:
                wandb_run.log(row, step=step)
            progress.set_postfix(loss=f"{row['loss']:.4f}", l1=f"{row['action_l1_loss']:.4f}")

        if step % save_freq == 0:
            save_policy(policy, output_dir, step)

    save_policy(policy, output_dir, steps, is_last=True)
    if wandb_run:
        wandb_run.finish()
    return output_dir


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/task2_act_calvin.yaml"))
    parser.add_argument("--run-kind", choices=["a", "abc"], required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_yaml(args.config)
    episodes = select_episodes(config, args.run_kind)
    print(f"run_kind={args.run_kind}")
    print(f"episodes={len(episodes)}")
    print(f"output_dir={Path(config['project']['output_dir']) / f'act_calvin_{args.run_kind}'}")
    if args.dry_run:
        return 0

    output_dir = train(config, args.run_kind)
    print(f"finished: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
