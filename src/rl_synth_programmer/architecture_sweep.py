from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import yaml

from .logging_utils import make_progress_bar, stage_log
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
    cv_folds = int(payload.get("cv_folds", 1) or 1)
    assert cv_folds >= 1, "cv_folds must be an integer >= 1."
    payload["cv_folds"] = cv_folds
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


def _dataset_label(dataset_path: Path) -> str:
    parts = Path(dataset_path).parts
    if len(parts) >= 3 and parts[-2] == "action_dataset":
        return str(parts[-3])
    return Path(dataset_path).stem


def _tensorboard_context(
    dataset_path: Path,
    config: dict[str, Any],
    metadata: dict[str, Any],
    observations: np.ndarray,
    rewards: np.ndarray,
) -> dict[str, Any]:
    return {
        "dataset_name": _dataset_label(dataset_path),
        "dataset_path": str(dataset_path),
        "plugin_path": str(metadata.get("plugin_path", "")),
        "synth": Path(str(metadata.get("plugin_path", ""))).stem if metadata.get("plugin_path") else "",
        "reward_mode": str(metadata.get("reward_mode", "")),
        "target": str(config.get("target", "all_action_rewards")),
        "action_step_mode": str(metadata.get("action_step_mode", "fixed")),
        "row_count": int(observations.shape[0]),
        "observation_size": int(observations.shape[1]),
        "action_count": int(rewards.shape[1]),
        "embedding_size": int(metadata.get("embedding_size", 0) or 0),
        "param_count": int(metadata.get("param_count", 0) or 0),
        "expanded_row_count": int(metadata.get("expanded_row_count", observations.shape[0]) or observations.shape[0]),
        "source_row_count": int(metadata.get("source_row_count", observations.shape[0]) or observations.shape[0]),
    }


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


def _cross_validation_splits(
    row_count: int,
    folds: int,
    seed: int,
    *,
    group_ids: np.ndarray | None = None,
) -> list[dict[str, np.ndarray]]:
    assert folds >= 2, "Cross validation requires cv_folds >= 2."
    rng = np.random.default_rng(seed)
    if group_ids is not None:
        units = rng.permutation(np.unique(group_ids))
        assert folds <= units.shape[0], (
            f"cv_folds={folds} requires at least {folds} source groups, found {units.shape[0]}."
        )
        fold_units = np.array_split(units, folds)
        splits = []
        for fold_index in range(folds):
            test_groups = fold_units[fold_index]
            val_groups = fold_units[(fold_index + 1) % folds]
            train_parts = [
                fold_units[index] for index in range(folds) if index not in {fold_index, (fold_index + 1) % folds}
            ]
            train_groups = np.concatenate(train_parts) if train_parts else np.asarray([], dtype=units.dtype)
            if train_groups.size == 0:
                train_groups = np.concatenate([fold_units[index] for index in range(folds) if index != fold_index])
            splits.append(
                {
                    "train": np.flatnonzero(np.isin(group_ids, train_groups)),
                    "val": np.flatnonzero(np.isin(group_ids, val_groups)),
                    "test": np.flatnonzero(np.isin(group_ids, test_groups)),
                }
            )
        return splits

    units = rng.permutation(row_count)
    assert folds <= units.shape[0], f"cv_folds={folds} requires at least {folds} rows, found {units.shape[0]}."
    fold_units = np.array_split(units, folds)
    splits = []
    for fold_index in range(folds):
        test_indices = fold_units[fold_index]
        val_indices = fold_units[(fold_index + 1) % folds]
        train_parts = [
            fold_units[index] for index in range(folds) if index not in {fold_index, (fold_index + 1) % folds}
        ]
        train_indices = np.concatenate(train_parts) if train_parts else np.asarray([], dtype=units.dtype)
        if train_indices.size == 0:
            train_indices = np.concatenate([fold_units[index] for index in range(folds) if index != fold_index])
        splits.append({"train": train_indices, "val": val_indices, "test": test_indices})
    return splits


def _action_features(action_count: int, param_count: int, action_step: float, action_deltas: list[float] | None = None) -> np.ndarray:
    rows = []
    denominator = max(1, int(param_count) - 1)
    if action_deltas is not None:
        assert len(action_deltas) == int(action_count), (
            f"metadata action_deltas length {len(action_deltas)} does not match action_count {action_count}."
        )
    for action in range(int(action_count)):
        parameter_index = action // 2
        signed_delta = float(action_deltas[action]) if action_deltas is not None else float((1.0 if action % 2 == 0 else -1.0) * action_step)
        rows.append([float(parameter_index / denominator), signed_delta])
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
    action_deltas = metadata.get("action_deltas")
    if action_deltas is not None:
        action_deltas = [float(value) for value in action_deltas]
    row_mask = np.ones((observations.shape[0],), dtype=bool)
    if bool(config.get("exclude_failed_rows", True)):
        if "failed_action_counts" in dataset.files:
            row_mask &= np.asarray(dataset["failed_action_counts"], dtype=np.int32) == 0
        if "state_skipped" in dataset.files:
            row_mask &= np.asarray(dataset["state_skipped"], dtype=np.int32) == 0
    observations = observations[row_mask]
    rewards = rewards[row_mask]
    assert observations.shape[0] > 0, "No rows remain after action-conditioned dataset filtering."

    features = _action_features(action_count, param_count, action_step, action_deltas)
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


def _metrics_from_predictions(
    pred: np.ndarray,
    true: np.ndarray,
    *,
    group_ids: np.ndarray | None = None,
    action_ids: np.ndarray | None = None,
) -> dict[str, float]:
    pred = np.asarray(pred, dtype=np.float32)
    true = np.asarray(true, dtype=np.float32)
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
        pred_values = pred.reshape(-1)
        true_values = true.reshape(-1)
        top1_hits = []
        top5_hits = []
        regrets = []
        for group in np.unique(group_ids):
            mask = group_ids == group
            group_true = true_values[mask]
            group_pred = pred_values[mask]
            group_actions = action_ids[mask]
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
    selected_group_ids = group_ids[indices] if group_ids is not None else None
    selected_action_ids = action_ids[indices] if action_ids is not None else None
    return _metrics_from_predictions(pred, true, group_ids=selected_group_ids, action_ids=selected_action_ids)


def _make_dataloader(torch, observations, rewards, indices: np.ndarray, spec: dict[str, Any], *, shuffle: bool):
    batch_size = int(spec["batch_size"])
    dataset = torch.utils.data.TensorDataset(
        observations[indices],
        rewards[indices],
        torch.tensor(indices, dtype=torch.long),
    )
    generator = torch.Generator()
    generator.manual_seed(int(spec["seed"]))
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=int(spec.get("num_workers", 0)),
        pin_memory=bool(spec.get("pin_memory", False)),
        generator=generator if shuffle else None,
    )


def _list_ints(value: Any) -> list[int]:
    if not isinstance(value, list):
        return []
    result = []
    for item in value:
        try:
            result.append(int(item))
        except (TypeError, ValueError):
            continue
    return result


def _architecture_hparams(spec: dict[str, Any]) -> dict[str, Any]:
    hidden_sizes = _list_ints(spec.get("hidden_sizes"))
    channels = _list_ints(spec.get("channels"))
    kernels = _list_ints(spec.get("kernel_sizes"))
    param_hidden_sizes = _list_ints(spec.get("param_hidden_sizes"))
    head_hidden_sizes = _list_ints(spec.get("head_hidden_sizes"))
    fusion_hidden_sizes = _list_ints(spec.get("fusion_hidden_sizes"))
    network_type = str(spec.get("type", ""))
    return {
        "architecture_name": str(spec.get("name", "")),
        "architecture_type": network_type,
        "network_type": network_type,
        "learning_rate": float(spec.get("learning_rate", 0.0)),
        "weight_decay": float(spec.get("weight_decay", 0.0)),
        "dropout": float(spec.get("dropout", 0.0)),
        "batch_size": int(spec.get("batch_size", 0)),
        "epochs": int(spec.get("epochs", 0)),
        "seed": int(spec.get("seed", 0)),
        "num_workers": int(spec.get("num_workers", 0)),
        "pin_memory": bool(spec.get("pin_memory", False)),
        "accelerator": str(spec.get("accelerator", "auto")),
        "devices": str(spec.get("devices", "auto")),
        "hidden_layer_count": len(hidden_sizes),
        "hidden_total_units": int(sum(hidden_sizes)),
        "hidden_max_width": int(max(hidden_sizes)) if hidden_sizes else 0,
        "residual_width": int(spec.get("width", 0) or 0),
        "residual_blocks": int(spec.get("blocks", 0) or 0),
        "conv_layer_count": len(channels),
        "conv_channels": ",".join(str(value) for value in channels),
        "conv_total_channels": int(sum(channels)),
        "conv_max_channels": int(max(channels)) if channels else 0,
        "kernel_sizes": ",".join(str(value) for value in kernels),
        "kernel_max": int(max(kernels)) if kernels else 0,
        "embedding_hidden_size": int(spec.get("embedding_hidden_size", 0) or 0),
        "param_hidden_layer_count": len(param_hidden_sizes),
        "param_hidden_total_units": int(sum(param_hidden_sizes)),
        "head_hidden_layer_count": len(head_hidden_sizes),
        "head_hidden_total_units": int(sum(head_hidden_sizes)),
        "fusion_hidden_layer_count": len(fusion_hidden_sizes),
        "fusion_hidden_total_units": int(sum(fusion_hidden_sizes)),
        "recurrent_hidden_size": int(spec.get("hidden_size", 0) or 0),
        "recurrent_layers": int(spec.get("layers", 0) or 0),
        "bidirectional": bool(spec.get("bidirectional", False)),
    }


def _tensorboard_hparams(
    spec: dict[str, Any],
    *,
    cv_folds: int = 1,
    fold: int | None = None,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    hparams: dict[str, Any] = {}
    hparams.update(_architecture_hparams(spec))
    if context:
        for key, value in context.items():
            if isinstance(value, (str, int, float, bool)):
                hparams[key] = value
            elif value is not None:
                hparams[key] = str(value)
    for key, value in spec.items():
        prefixed_key = f"spec_{key}"
        if prefixed_key in hparams:
            continue
        if isinstance(value, (str, int, float, bool)):
            hparams[prefixed_key] = value
        elif isinstance(value, list):
            hparams[prefixed_key] = ",".join(str(item) for item in value)
        elif value is not None:
            hparams[prefixed_key] = str(value)
    hparams["cv_folds"] = int(cv_folds)
    if fold is not None:
        hparams["fold"] = int(fold)
    return hparams


def _tensorboard_hp_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    val_mse = float(metrics["val"]["mse"])
    val_mean_regret = float(metrics["val"]["mean_regret"])
    return {
        "hp_metric": val_mean_regret,
        "hp/val_mse": val_mse,
        "hp/val_mean_regret": val_mean_regret,
        "hp/val_top1_accuracy": float(metrics["val"]["top1_accuracy"]),
        "hp/test_mse": float(metrics["test"]["mse"]),
        "hp/test_mean_regret": float(metrics["test"]["mean_regret"]),
    }


def _log_tensorboard_hparams(
    logger: Any,
    spec: dict[str, Any],
    metrics: dict[str, Any],
    *,
    cv_folds: int = 1,
    fold: int | None = None,
    context: dict[str, Any] | None = None,
) -> None:
    if not logger:
        return
    hparams = _tensorboard_hparams(spec, cv_folds=cv_folds, fold=fold, context=context)
    hp_metrics = _tensorboard_hp_metrics(metrics)
    logger.log_hyperparams(hparams, hp_metrics)
    if hasattr(logger, "experiment"):
        logger.experiment.add_hparams(hparams, hp_metrics, run_name="hparams")
    log_dir = Path(str(logger.log_dir))
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "hparams.yaml").write_text(yaml.safe_dump(hparams, sort_keys=True))
    if hasattr(logger, "save"):
        logger.save()


def _lightning_module_class(lightning, torch):
    class _SupervisedActionValueModule(lightning.LightningModule):
        def __init__(
            self,
            network,
            *,
            learning_rate: float,
            weight_decay: float,
            group_ids: np.ndarray | None,
            action_ids: np.ndarray | None,
        ):
            super().__init__()
            self.network = network
            self.learning_rate = float(learning_rate)
            self.weight_decay = float(weight_decay)
            self.loss_fn = torch.nn.MSELoss()
            self.group_ids = group_ids
            self.action_ids = action_ids
            self.loss_rows: list[dict[str, float | int]] = []
            self._train_losses = []
            self._val_predictions = []
            self._val_targets = []
            self._val_indices = []
            self._test_predictions = []
            self._test_targets = []
            self._test_indices = []
            self.save_hyperparameters(ignore=["network", "group_ids", "action_ids"])

        def forward(self, observation):
            return self.network(observation)

        def on_train_epoch_start(self):
            self._train_losses = []

        def training_step(self, batch, batch_idx):
            observations, rewards, _indices = batch
            predictions = self(observations)
            loss = self.loss_fn(predictions, rewards)
            self._train_losses.append(loss.detach())
            self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
            return loss

        def on_train_epoch_end(self):
            if self._train_losses:
                train_loss = torch.stack(self._train_losses).mean().detach().cpu()
                row = {"epoch": int(self.current_epoch + 1), "train_loss": float(train_loss.item())}
                self.loss_rows.append(row)

        def on_validation_epoch_start(self):
            self._val_predictions = []
            self._val_targets = []
            self._val_indices = []

        def validation_step(self, batch, batch_idx):
            observations, rewards, indices = batch
            predictions = self(observations)
            loss = self.loss_fn(predictions, rewards)
            self._val_predictions.append(predictions.detach().cpu())
            self._val_targets.append(rewards.detach().cpu())
            self._val_indices.append(indices.detach().cpu())
            self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
            return loss

        def on_validation_epoch_end(self):
            if not self._val_predictions:
                return
            pred = torch.cat(self._val_predictions, dim=0).numpy()
            true = torch.cat(self._val_targets, dim=0).numpy()
            indices = torch.cat(self._val_indices, dim=0).numpy()
            selected_group_ids = self.group_ids[indices] if self.group_ids is not None else None
            selected_action_ids = self.action_ids[indices] if self.action_ids is not None else None
            metrics = _metrics_from_predictions(
                pred,
                true,
                group_ids=selected_group_ids,
                action_ids=selected_action_ids,
            )
            for key, value in metrics.items():
                self.log(f"val_{key}", float(value), on_step=False, on_epoch=True, prog_bar=key in {"mse", "mean_regret"})
            if self.loss_rows:
                self.loss_rows[-1].update({f"val_{key}": float(value) for key, value in metrics.items()})

        def on_test_epoch_start(self):
            self._test_predictions = []
            self._test_targets = []
            self._test_indices = []

        def test_step(self, batch, batch_idx):
            observations, rewards, indices = batch
            predictions = self(observations)
            loss = self.loss_fn(predictions, rewards)
            self._test_predictions.append(predictions.detach().cpu())
            self._test_targets.append(rewards.detach().cpu())
            self._test_indices.append(indices.detach().cpu())
            self.log("test_loss", loss, on_step=False, on_epoch=True)
            return loss

        def on_test_epoch_end(self):
            if not self._test_predictions:
                return
            pred = torch.cat(self._test_predictions, dim=0).numpy()
            true = torch.cat(self._test_targets, dim=0).numpy()
            indices = torch.cat(self._test_indices, dim=0).numpy()
            selected_group_ids = self.group_ids[indices] if self.group_ids is not None else None
            selected_action_ids = self.action_ids[indices] if self.action_ids is not None else None
            metrics = _metrics_from_predictions(
                pred,
                true,
                group_ids=selected_group_ids,
                action_ids=selected_action_ids,
            )
            for key, value in metrics.items():
                self.log(f"test_{key}", float(value), on_step=False, on_epoch=True)

        def configure_optimizers(self):
            return torch.optim.Adam(
                self.network.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )

    return _SupervisedActionValueModule


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
    arch_dir: Path | None = None,
    cv_folds: int = 1,
    fold_index: int | None = None,
    tensorboard_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    torch = require_dependency("torch", "ml")
    lightning = require_dependency("lightning.pytorch", "ml")
    TensorBoardLogger = require_dependency("lightning.pytorch.loggers", "ml").TensorBoardLogger
    lightning.seed_everything(int(spec["seed"]), workers=True)
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
    batch_size = int(spec["batch_size"])
    arch_dir = arch_dir or out_dir / _slugify(str(spec["name"]))
    arch_dir.mkdir(parents=True, exist_ok=True)
    train_loader = _make_dataloader(torch, observations, rewards, split_indices["train"], spec, shuffle=True)
    val_loader = _make_dataloader(torch, observations, rewards, split_indices["val"], spec, shuffle=False)
    test_loader = _make_dataloader(torch, observations, rewards, split_indices["test"], spec, shuffle=False)
    module_class = _lightning_module_class(lightning, torch)
    module = module_class(
        network,
        learning_rate=float(spec["learning_rate"]),
        weight_decay=float(spec.get("weight_decay", 0.0)),
        group_ids=group_ids,
        action_ids=action_ids,
    )
    logger = TensorBoardLogger(save_dir=str(arch_dir), name="tensorboard", version="") if tensorboard else False
    trainer = lightning.Trainer(
        max_epochs=int(spec["epochs"]),
        enable_progress_bar=progress,
        logger=logger,
        enable_checkpointing=False,
        accelerator=str(spec.get("accelerator", "auto")),
        devices=spec.get("devices", "auto"),
        deterministic=True,
        num_sanity_val_steps=0,
        enable_model_summary=progress,
        log_every_n_steps=1,
    )
    started = perf_counter()
    trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=val_loader)
    trainer.test(module, dataloaders=test_loader, verbose=False)
    losses = list(module.loss_rows)
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
    _log_tensorboard_hparams(
        trainer.logger,
        spec,
        metrics,
        cv_folds=cv_folds,
        fold=fold_index,
        context=tensorboard_context,
    )
    torch.save({"state_dict": network.state_dict(), "spec": spec, "metadata": metadata}, arch_dir / "checkpoint.pt")
    write_json(arch_dir / "metrics.json", metrics)
    write_json(arch_dir / "resolved_config.json", spec)
    append_csv(arch_dir / "loss.csv", list(losses[0].keys()) if losses else [], losses)
    return metrics


def _aggregate_cross_validation_metrics(
    spec: dict[str, Any],
    fold_metrics: list[dict[str, Any]],
    *,
    arch_dir: Path,
    tensorboard: bool,
    tensorboard_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    assert fold_metrics, "Cannot aggregate cross-validation metrics without folds."
    arch_dir.mkdir(parents=True, exist_ok=True)
    metrics: dict[str, Any] = {
        "name": str(spec["name"]),
        "type": str(spec["type"]),
        "cv_folds": len(fold_metrics),
        "training_seconds": float(sum(float(item.get("training_seconds", 0.0)) for item in fold_metrics)),
        "folds": fold_metrics,
        "checkpoint": str(arch_dir / "checkpoint.pt"),
    }
    for split_name in ("train", "val", "test"):
        metric_names = sorted(
            {
                metric_name
                for fold in fold_metrics
                for metric_name in fold.get(split_name, {}).keys()
            }
        )
        metrics[split_name] = {}
        for metric_name in metric_names:
            values = np.asarray([float(fold[split_name][metric_name]) for fold in fold_metrics], dtype=np.float64)
            metrics[split_name][metric_name] = float(values.mean())
            metrics[split_name][f"{metric_name}_std"] = float(values.std(ddof=0))

    final_fold_checkpoint = Path(str(fold_metrics[-1]["checkpoint"]))
    if final_fold_checkpoint.exists():
        shutil.copyfile(final_fold_checkpoint, arch_dir / "checkpoint.pt")
    write_json(arch_dir / "metrics.json", metrics)
    write_json(arch_dir / "resolved_config.json", spec)

    loss_rows: list[dict[str, Any]] = []
    for fold_index in range(len(fold_metrics)):
        loss_path = arch_dir / f"fold-{fold_index:02d}" / "loss.csv"
        if not loss_path.exists():
            continue
        lines = loss_path.read_text().strip().splitlines()
        if len(lines) < 2:
            continue
        headers = lines[0].split(",")
        for line in lines[1:]:
            values = line.split(",")
            row = {"fold": fold_index}
            row.update({header: values[index] for index, header in enumerate(headers) if index < len(values)})
            loss_rows.append(row)
    if loss_rows:
        append_csv(arch_dir / "loss.csv", list(loss_rows[0].keys()), loss_rows)
    if tensorboard:
        lightning_loggers = require_dependency("lightning.pytorch.loggers", "ml")
        logger = lightning_loggers.TensorBoardLogger(save_dir=str(arch_dir), name="tensorboard", version="cv-summary")
        _log_tensorboard_hparams(logger, spec, metrics, cv_folds=len(fold_metrics), context=tensorboard_context)
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
    tensorboard_context = _tensorboard_context(dataset_path, config, metadata, observations, rewards)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "sweep_config.json", config)
    cv_folds = int(config.get("cv_folds", 1) or 1)
    split_indices = _split_indices(
        observations.shape[0],
        config["split"],
        int(config.get("seed", 7)),
        group_ids=group_ids,
    )
    cv_splits = (
        _cross_validation_splits(
            observations.shape[0],
            cv_folds,
            int(config.get("seed", 7)),
            group_ids=group_ids,
        )
        if cv_folds > 1
        else [split_indices]
    )
    stage_log(
        f"Comparing {len(config['architectures'])} architecture(s). "
        f"Rows={observations.shape[0]} obs={observations.shape[1]} actions={rewards.shape[1]} cv_folds={cv_folds}."
    )
    leaderboard: list[dict[str, Any]] = []
    arch_progress = make_progress_bar(total=len(config["architectures"]), desc="architectures", enabled=progress)
    for spec in config["architectures"]:
        stage_log(f"Training architecture '{spec['name']}' ({spec['type']}).")
        arch_dir = out_dir / _slugify(str(spec["name"]))
        if cv_folds > 1:
            fold_metrics = []
            for fold_index, fold_split_indices in enumerate(cv_splits):
                stage_log(f"Training fold {fold_index + 1}/{cv_folds} for architecture '{spec['name']}'.")
                fold_metrics.append(
                    _train_one(
                        spec,
                        observations_np=observations,
                        rewards_np=rewards,
                        split_indices=fold_split_indices,
                        metadata=metadata,
                        group_ids=group_ids,
                        action_ids=action_ids,
                        out_dir=out_dir,
                        progress=progress,
                        tensorboard=tensorboard,
                        arch_dir=arch_dir / f"fold-{fold_index:02d}",
                        cv_folds=cv_folds,
                        fold_index=fold_index,
                        tensorboard_context=tensorboard_context,
                    )
                )
            metrics = _aggregate_cross_validation_metrics(
                spec,
                fold_metrics,
                arch_dir=arch_dir,
                tensorboard=tensorboard,
                tensorboard_context=tensorboard_context,
            )
        else:
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
                arch_dir=arch_dir,
                cv_folds=cv_folds,
                tensorboard_context=tensorboard_context,
            )
        leaderboard.append(
            {
                "name": metrics["name"],
                "type": metrics["type"],
                "cv_folds": metrics.get("cv_folds", 1),
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
        "cv_folds": cv_folds,
        "leaderboard": leaderboard,
        "best": leaderboard[0] if leaderboard else None,
    }
    write_json(out_dir / "leaderboard.json", payload)
    append_csv(out_dir / "leaderboard.csv", list(leaderboard[0].keys()) if leaderboard else [], leaderboard)
    stage_log(f"Architecture comparison complete. Leaderboard written to {out_dir / 'leaderboard.json'}.")
    return payload
