from __future__ import annotations

import json
import re
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from .logging_utils import create_summary_writer, make_progress_bar, stage_log
from .manifest import append_csv, write_json
from .networks import build_network
from .optional_deps import require_dependency


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "architecture"


def load_sweep_config(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text())
    architectures = payload.get("architectures")
    assert isinstance(architectures, list) and architectures, "Sweep config requires a non-empty 'architectures' list."
    for item in architectures:
        _validate_architecture_spec(item)
    split = payload.get("split", {"train": 0.8, "val": 0.1, "test": 0.1})
    total = float(split.get("train", 0.0)) + float(split.get("val", 0.0)) + float(split.get("test", 0.0))
    assert total > 0.0, "Split ratios must sum to a positive value."
    payload["split"] = {
        "train": float(split.get("train", 0.8)) / total,
        "val": float(split.get("val", 0.1)) / total,
        "test": float(split.get("test", 0.1)) / total,
    }
    return payload


def _validate_architecture_spec(spec: dict[str, Any]) -> None:
    for key in ("name", "type", "learning_rate", "batch_size", "epochs", "seed"):
        assert key in spec, f"Architecture spec missing required field: {key}"
    network_type = str(spec["type"])
    if network_type == "mlp":
        assert "hidden_sizes" in spec, "mlp architecture requires hidden_sizes."
    elif network_type == "residual_mlp":
        assert "width" in spec and "blocks" in spec, "residual_mlp requires width and blocks."
    elif network_type == "cnn1d":
        for key in ("channels", "kernel_sizes", "embedding_hidden_size", "param_hidden_sizes", "head_hidden_sizes"):
            assert key in spec, f"cnn1d architecture requires {key}."
    elif network_type == "hybrid_cnn_mlp":
        for key in ("channels", "kernel_sizes", "param_hidden_sizes", "fusion_hidden_sizes"):
            assert key in spec, f"hybrid_cnn_mlp architecture requires {key}."
    elif network_type in {"rnn", "gru", "lstm"}:
        assert "hidden_size" in spec, f"{network_type} architecture requires hidden_size."
    else:
        raise ValueError(f"Unsupported architecture type: {network_type}")


def _load_metadata(dataset_path: Path) -> dict[str, Any]:
    metadata_path = dataset_path.with_name("metadata.json")
    if metadata_path.exists():
        return json.loads(metadata_path.read_text())
    return {}


def _split_indices(
    row_count: int,
    split: dict[str, float],
    seed: int,
    *,
    group_ids: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    if group_ids is not None:
        groups = rng.permutation(np.unique(group_ids))
        train_end = int(round(groups.shape[0] * split["train"]))
        val_end = train_end + int(round(groups.shape[0] * split["val"]))
        train_end = min(max(train_end, 1), groups.shape[0])
        val_end = min(max(val_end, train_end), groups.shape[0])
        split_groups = {
            "train": groups[:train_end],
            "val": groups[train_end:val_end],
            "test": groups[val_end:],
        }
        if split_groups["val"].size == 0:
            split_groups["val"] = split_groups["train"]
        if split_groups["test"].size == 0:
            split_groups["test"] = split_groups["val"]
        return {
            name: np.flatnonzero(np.isin(group_ids, values))
            for name, values in split_groups.items()
        }
    indices = rng.permutation(row_count)
    train_end = int(round(row_count * split["train"]))
    val_end = train_end + int(round(row_count * split["val"]))
    train_end = min(max(train_end, 1), row_count)
    val_end = min(max(val_end, train_end), row_count)
    result = {
        "train": indices[:train_end],
        "val": indices[train_end:val_end],
        "test": indices[val_end:],
    }
    if result["val"].size == 0:
        result["val"] = result["train"]
    if result["test"].size == 0:
        result["test"] = result["val"]
    return result


def _action_features(action_count: int, param_count: int, action_step: float) -> np.ndarray:
    rows = []
    denominator = max(1, int(param_count) - 1)
    for action in range(int(action_count)):
        parameter_index = action // 2
        direction = 1.0 if action % 2 == 0 else -1.0
        rows.append([float(parameter_index / denominator), float(direction * action_step)])
    return np.asarray(rows, dtype=np.float32)


def _prepare_supervised_arrays(
    dataset,
    metadata: dict[str, Any],
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], np.ndarray | None, np.ndarray | None]:
    observations = np.asarray(dataset["observations"], dtype=np.float32)
    rewards = np.asarray(dataset["action_rewards"], dtype=np.float32)
    assert observations.ndim == 2, f"Expected 2D observations, got shape {observations.shape}."
    assert rewards.ndim == 2, f"Expected 2D action_rewards, got shape {rewards.shape}."
    target = str(config.get("target", "all_action_rewards"))
    if target == "all_action_rewards":
        return observations, rewards, metadata, None, None
    if target != "action_reward_as_feature_change_proxy":
        raise ValueError(f"Unsupported sweep target: {target}")

    param_count = int(metadata["param_count"])
    action_count = int(rewards.shape[1])
    action_step = float(metadata.get("action_step", metadata.get("args", {}).get("action_step", 0.05)))
    row_mask = np.ones((observations.shape[0],), dtype=bool)
    if bool(config.get("exclude_failed_rows", True)):
        if "failed_action_counts" in dataset.files:
            row_mask &= np.asarray(dataset["failed_action_counts"], dtype=np.int32) == 0
        if "state_skipped" in dataset.files:
            row_mask &= np.asarray(dataset["state_skipped"], dtype=np.int32) == 0
    observations = observations[row_mask]
    rewards = rewards[row_mask]
    assert observations.shape[0] > 0, "No rows remain after action-conditioned dataset filtering."

    features = _action_features(action_count, param_count, action_step)
    expanded_observations = np.repeat(observations, action_count, axis=0)
    expanded_features = np.tile(features, (observations.shape[0], 1))
    expanded_rewards = rewards.reshape(-1, 1).astype(np.float32)
    group_ids = np.repeat(np.arange(observations.shape[0], dtype=np.int32), action_count)
    action_ids = np.tile(np.arange(action_count, dtype=np.int32), observations.shape[0])
    max_expanded_rows = int(config.get("max_expanded_rows", 0) or 0)
    if max_expanded_rows > 0 and expanded_rewards.shape[0] > max_expanded_rows:
        rng = np.random.default_rng(int(config.get("seed", 7)))
        groups = np.unique(group_ids)
        group_budget = max(1, max_expanded_rows // action_count)
        selected_groups = rng.choice(groups, size=min(group_budget, groups.shape[0]), replace=False)
        keep = np.isin(group_ids, selected_groups)
        expanded_observations = expanded_observations[keep]
        expanded_features = expanded_features[keep]
        expanded_rewards = expanded_rewards[keep]
        group_ids = group_ids[keep]
        action_ids = action_ids[keep]
    action_conditioned_observations = np.concatenate([expanded_observations, expanded_features], axis=1).astype(np.float32)
    prepared_metadata = dict(metadata)
    prepared_metadata["param_count"] = int(param_count + features.shape[1])
    prepared_metadata["action_conditioned"] = True
    prepared_metadata["action_feature_layout"] = {
        "parameter_index_normalized": observations.shape[1],
        "signed_delta": observations.shape[1] + 1,
    }
    prepared_metadata["source_row_count"] = int(row_mask.shape[0])
    prepared_metadata["used_source_row_count"] = int(observations.shape[0])
    prepared_metadata["expanded_row_count"] = int(action_conditioned_observations.shape[0])
    return action_conditioned_observations, expanded_rewards, prepared_metadata, group_ids, action_ids


def _batch_ranges(count: int, batch_size: int):
    for start in range(0, count, batch_size):
        yield start, min(start + batch_size, count)


def _evaluate(
    torch,
    network,
    observations,
    rewards,
    indices: np.ndarray,
    batch_size: int,
    *,
    group_ids: np.ndarray | None = None,
    action_ids: np.ndarray | None = None,
) -> dict[str, float]:
    network.eval()
    predictions = []
    targets = []
    with torch.no_grad():
        for start, stop in _batch_ranges(len(indices), batch_size):
            batch_idx = indices[start:stop]
            obs = observations[batch_idx]
            target = rewards[batch_idx]
            predictions.append(network(obs).detach().cpu().numpy())
            targets.append(target.detach().cpu().numpy())
    pred = np.concatenate(predictions, axis=0)
    true = np.concatenate(targets, axis=0)
    errors = pred - true
    result = {
        "mse": float(np.mean(np.square(errors))),
        "mae": float(np.mean(np.abs(errors))),
        "sign_accuracy": float(np.mean(np.sign(pred) == np.sign(true))),
    }
    if true.shape[1] > 1:
        true_best = np.argmax(true, axis=1)
        pred_best = np.argmax(pred, axis=1)
        top_k = min(5, true.shape[1])
        pred_top_k = np.argsort(pred, axis=1)[:, -top_k:]
        chosen_true_reward = true[np.arange(true.shape[0]), pred_best]
        best_true_reward = true[np.arange(true.shape[0]), true_best]
        result.update(
            {
                "top1_accuracy": float(np.mean(pred_best == true_best)),
                "top5_accuracy": float(np.mean([true_best[index] in pred_top_k[index] for index in range(true.shape[0])])),
                "mean_regret": float(np.mean(best_true_reward - chosen_true_reward)),
            }
        )
        return result
    if group_ids is not None and action_ids is not None:
        selected_groups = group_ids[indices]
        selected_actions = action_ids[indices]
        pred_values = pred.reshape(-1)
        true_values = true.reshape(-1)
        top1_hits = []
        top5_hits = []
        regrets = []
        for group in np.unique(selected_groups):
            mask = selected_groups == group
            group_true = true_values[mask]
            group_pred = pred_values[mask]
            group_actions = selected_actions[mask]
            true_best_index = int(np.argmax(group_true))
            pred_best_index = int(np.argmax(group_pred))
            true_best_action = int(group_actions[true_best_index])
            pred_best_action = int(group_actions[pred_best_index])
            top_k = min(5, group_pred.shape[0])
            pred_top_actions = set(int(group_actions[index]) for index in np.argsort(group_pred)[-top_k:])
            top1_hits.append(pred_best_action == true_best_action)
            top5_hits.append(true_best_action in pred_top_actions)
            regrets.append(float(group_true[true_best_index] - group_true[pred_best_index]))
        result.update(
            {
                "top1_accuracy": float(np.mean(top1_hits)) if top1_hits else 0.0,
                "top5_accuracy": float(np.mean(top5_hits)) if top5_hits else 0.0,
                "mean_regret": float(np.mean(regrets)) if regrets else 0.0,
            }
        )
    else:
        result.update({"top1_accuracy": 0.0, "top5_accuracy": 0.0, "mean_regret": 0.0})
    return result


def _train_one(
    spec: dict[str, Any],
    *,
    observations_np: np.ndarray,
    rewards_np: np.ndarray,
    split_indices: dict[str, np.ndarray],
    metadata: dict[str, Any],
    group_ids: np.ndarray | None,
    action_ids: np.ndarray | None,
    out_dir: Path,
    progress: bool,
    tensorboard: bool,
) -> dict[str, Any]:
    torch = require_dependency("torch", "ml")
    torch.manual_seed(int(spec["seed"]))
    observations = torch.tensor(observations_np, dtype=torch.float32)
    rewards = torch.tensor(rewards_np, dtype=torch.float32)
    observation_size = int(observations_np.shape[1])
    action_size = int(rewards_np.shape[1])
    network = build_network(
        spec,
        observation_size=observation_size,
        action_size=action_size,
        param_count=metadata.get("param_count"),
        embedding_size=metadata.get("embedding_size"),
    )
    optimizer = torch.optim.Adam(
        network.parameters(),
        lr=float(spec["learning_rate"]),
        weight_decay=float(spec.get("weight_decay", 0.0)),
    )
    loss_fn = torch.nn.MSELoss()
    batch_size = int(spec["batch_size"])
    epochs = int(spec["epochs"])
    arch_dir = out_dir / _slugify(str(spec["name"]))
    arch_dir.mkdir(parents=True, exist_ok=True)
    writer = create_summary_writer(tensorboard, arch_dir / "tensorboard")
    losses: list[dict[str, float | int | str]] = []
    started = perf_counter()
    progress_bar = make_progress_bar(total=epochs, desc=f"train {spec['name']}", enabled=progress)
    train_indices = split_indices["train"]
    rng = np.random.default_rng(int(spec["seed"]))
    for epoch in range(1, epochs + 1):
        network.train()
        epoch_indices = rng.permutation(train_indices)
        batch_losses: list[float] = []
        for start, stop in _batch_ranges(len(epoch_indices), batch_size):
            batch_idx = epoch_indices[start:stop]
            prediction = network(observations[batch_idx])
            loss = loss_fn(prediction, rewards[batch_idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.item()))
        train_loss = float(np.mean(batch_losses)) if batch_losses else float("nan")
        val_metrics = _evaluate(
            torch,
            network,
            observations,
            rewards,
            split_indices["val"],
            batch_size,
            group_ids=group_ids,
            action_ids=action_ids,
        )
        row = {"epoch": epoch, "train_loss": train_loss, **{f"val_{k}": v for k, v in val_metrics.items()}}
        losses.append(row)
        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/val_mse", val_metrics["mse"], epoch)
        writer.add_scalar("metrics/val_regret", val_metrics["mean_regret"], epoch)
        progress_bar.set_postfix({"loss": f"{train_loss:.4f}", "regret": f"{val_metrics['mean_regret']:.4f}"})
        progress_bar.update(1)
    progress_bar.close()
    metrics = {
        "name": str(spec["name"]),
        "type": str(spec["type"]),
        "training_seconds": float(perf_counter() - started),
        "train": _evaluate(
            torch,
            network,
            observations,
            rewards,
            split_indices["train"],
            batch_size,
            group_ids=group_ids,
            action_ids=action_ids,
        ),
        "val": _evaluate(
            torch,
            network,
            observations,
            rewards,
            split_indices["val"],
            batch_size,
            group_ids=group_ids,
            action_ids=action_ids,
        ),
        "test": _evaluate(
            torch,
            network,
            observations,
            rewards,
            split_indices["test"],
            batch_size,
            group_ids=group_ids,
            action_ids=action_ids,
        ),
        "checkpoint": str(arch_dir / "checkpoint.pt"),
    }
    torch.save({"state_dict": network.state_dict(), "spec": spec, "metadata": metadata}, arch_dir / "checkpoint.pt")
    write_json(arch_dir / "metrics.json", metrics)
    write_json(arch_dir / "resolved_config.json", spec)
    append_csv(arch_dir / "loss.csv", list(losses[0].keys()) if losses else [], losses)
    writer.flush()
    writer.close()
    return metrics


def compare_architectures(
    dataset_path: Path,
    config_path: Path,
    out_dir: Path,
    *,
    progress: bool = True,
    tensorboard: bool = False,
) -> dict[str, Any]:
    dataset_path = Path(dataset_path)
    assert dataset_path.exists(), f"Dataset file does not exist: {dataset_path}"
    config_path = Path(config_path)
    assert config_path.exists(), f"Sweep config file does not exist: {config_path}"
    stage_log(f"Loading action dataset from {dataset_path}.")
    dataset = np.load(dataset_path)
    metadata = _load_metadata(dataset_path)
    config = load_sweep_config(config_path)
    observations, rewards, metadata, group_ids, action_ids = _prepare_supervised_arrays(dataset, metadata, config)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "sweep_config.json", config)
    split_indices = _split_indices(
        observations.shape[0],
        config["split"],
        int(config.get("seed", 7)),
        group_ids=group_ids,
    )
    stage_log(
        f"Comparing {len(config['architectures'])} architecture(s). "
        f"Rows={observations.shape[0]} obs={observations.shape[1]} actions={rewards.shape[1]}."
    )
    leaderboard: list[dict[str, Any]] = []
    arch_progress = make_progress_bar(total=len(config["architectures"]), desc="architectures", enabled=progress)
    for spec in config["architectures"]:
        stage_log(f"Training architecture '{spec['name']}' ({spec['type']}).")
        metrics = _train_one(
            spec,
            observations_np=observations,
            rewards_np=rewards,
            split_indices=split_indices,
            metadata=metadata,
            group_ids=group_ids,
            action_ids=action_ids,
            out_dir=out_dir,
            progress=progress,
            tensorboard=tensorboard,
        )
        leaderboard.append(
            {
                "name": metrics["name"],
                "type": metrics["type"],
                "val_mse": metrics["val"]["mse"],
                "val_mae": metrics["val"]["mae"],
                "val_top1_accuracy": metrics["val"]["top1_accuracy"],
                "val_top5_accuracy": metrics["val"]["top5_accuracy"],
                "val_mean_regret": metrics["val"]["mean_regret"],
                "test_mse": metrics["test"]["mse"],
                "test_mean_regret": metrics["test"]["mean_regret"],
                "training_seconds": metrics["training_seconds"],
                "checkpoint": metrics["checkpoint"],
            }
        )
        arch_progress.update(1)
    arch_progress.close()
    leaderboard = sorted(leaderboard, key=lambda item: (float(item["val_mean_regret"]), float(item["val_mse"])))
    payload = {
        "dataset_path": str(dataset_path),
        "config_path": str(config_path),
        "row_count": int(observations.shape[0]),
        "action_count": int(rewards.shape[1]),
        "leaderboard": leaderboard,
        "best": leaderboard[0] if leaderboard else None,
    }
    write_json(out_dir / "leaderboard.json", payload)
    append_csv(out_dir / "leaderboard.csv", list(leaderboard[0].keys()) if leaderboard else [], leaderboard)
    stage_log(f"Architecture comparison complete. Leaderboard written to {out_dir / 'leaderboard.json'}.")
    return payload
