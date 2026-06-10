from __future__ import annotations

from pathlib import Path
from typing import Any

from .architecture_sweep import compare_architectures
from .manifest import append_csv, write_json


def default_feature_change_search_config(*, epochs: int = 5, seeds: tuple[int, ...] = (7, 11)) -> dict[str, Any]:
    """Return a small CNN/RNN-heavy sweep config for offline action-reward prediction."""
    architectures: list[dict[str, Any]] = []
    for seed in seeds:
        architectures.extend(
            [
                {
                    "name": f"mlp-512-256-s{seed}",
                    "type": "mlp",
                    "hidden_sizes": [512, 256],
                    "learning_rate": 0.001,
                    "batch_size": 64,
                    "epochs": epochs,
                    "seed": seed,
                    "dropout": 0.05,
                },
                {
                    "name": f"cnn-small-s{seed}",
                    "type": "cnn1d",
                    "channels": [32, 64],
                    "kernel_sizes": [7, 5],
                    "embedding_hidden_size": 128,
                    "param_hidden_sizes": [64],
                    "head_hidden_sizes": [128],
                    "learning_rate": 0.001,
                    "batch_size": 64,
                    "epochs": epochs,
                    "seed": seed,
                    "dropout": 0.05,
                },
                {
                    "name": f"cnn-wide-s{seed}",
                    "type": "cnn1d",
                    "channels": [64, 96],
                    "kernel_sizes": [9, 5],
                    "embedding_hidden_size": 192,
                    "param_hidden_sizes": [96],
                    "head_hidden_sizes": [192, 96],
                    "learning_rate": 0.0007,
                    "batch_size": 64,
                    "epochs": epochs,
                    "seed": seed,
                    "dropout": 0.10,
                },
                {
                    "name": f"gru-small-s{seed}",
                    "type": "gru",
                    "hidden_size": 128,
                    "layers": 1,
                    "param_hidden_sizes": [64],
                    "head_hidden_sizes": [128],
                    "learning_rate": 0.001,
                    "batch_size": 64,
                    "epochs": epochs,
                    "seed": seed,
                    "dropout": 0.05,
                },
                {
                    "name": f"gru-bidir-s{seed}",
                    "type": "gru",
                    "hidden_size": 96,
                    "layers": 1,
                    "bidirectional": True,
                    "param_hidden_sizes": [64],
                    "head_hidden_sizes": [128],
                    "learning_rate": 0.0007,
                    "batch_size": 64,
                    "epochs": epochs,
                    "seed": seed,
                    "dropout": 0.05,
                },
                {
                    "name": f"lstm-small-s{seed}",
                    "type": "lstm",
                    "hidden_size": 128,
                    "layers": 1,
                    "param_hidden_sizes": [64],
                    "head_hidden_sizes": [128],
                    "learning_rate": 0.001,
                    "batch_size": 64,
                    "epochs": epochs,
                    "seed": seed,
                    "dropout": 0.05,
                },
            ]
        )
    return {
        "seed": 7,
        "split": {"train": 0.7, "val": 0.15, "test": 0.15},
        "target": "action_reward_as_feature_change_proxy",
        "exclude_failed_rows": True,
        "max_expanded_rows": 20000,
        "architectures": architectures,
    }


def discover_action_datasets(root: Path) -> list[Path]:
    root = Path(root)
    return sorted(path for path in root.glob("*/action_dataset/dataset.npz") if path.exists())


def _dataset_name(dataset_path: Path) -> str:
    parts = Path(dataset_path).parts
    if len(parts) >= 3 and parts[-2] == "action_dataset":
        return parts[-3]
    return Path(dataset_path).stem


def _markdown_table(rows: list[dict[str, Any]]) -> str:
    headers = [
        "dataset",
        "name",
        "type",
        "val_mean_regret",
        "val_mse",
        "val_top1_accuracy",
        "test_mean_regret",
        "test_mse",
        "training_seconds",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        values = []
        for header in headers:
            value = row.get(header, "")
            if isinstance(value, float):
                values.append(f"{value:.6g}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def run_feature_change_search(
    dataset_paths: list[Path],
    out_dir: Path,
    *,
    config: dict[str, Any] | None = None,
    progress: bool = True,
    tensorboard: bool = False,
) -> dict[str, Any]:
    assert dataset_paths, "No action datasets were provided."
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    resolved_config = config or default_feature_change_search_config()
    config_path = out_dir / "search_config.json"
    write_json(config_path, resolved_config)
    rows: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    for dataset_path in dataset_paths:
        dataset_path = Path(dataset_path)
        dataset_label = _dataset_name(dataset_path)
        dataset_out_dir = out_dir / dataset_label
        result = compare_architectures(
            dataset_path,
            config_path,
            dataset_out_dir,
            progress=progress,
            tensorboard=tensorboard,
        )
        results.append(result)
        for rank, row in enumerate(result["leaderboard"], start=1):
            rows.append({"dataset": dataset_label, "rank": rank, **row})
    rows = sorted(rows, key=lambda item: (str(item["dataset"]), float(item["val_mean_regret"]), float(item["val_mse"])))
    payload = {
        "config_path": str(config_path),
        "dataset_paths": [str(path) for path in dataset_paths],
        "rows": rows,
        "best_by_dataset": {
            _dataset_name(Path(result["dataset_path"])): result["best"]
            for result in results
        },
    }
    write_json(out_dir / "combined_leaderboard.json", payload)
    append_csv(out_dir / "combined_leaderboard.csv", list(rows[0].keys()) if rows else [], rows)
    (out_dir / "combined_leaderboard.md").write_text(_markdown_table(rows))
    return payload
