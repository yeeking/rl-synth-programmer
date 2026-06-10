from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np

from .architecture_sweep import compare_architectures
from .config import CurriculumConfig, DQNConfig, ExperimentConfig, RewardConfig, SynthEnvConfig, SynthHostConfig
from .env import make_env
from .hyperparameter_search import (
    default_feature_change_search_config,
    discover_action_datasets,
    run_feature_change_search,
)
from .host import SynthHost
from .offline_dataset import ActionDatasetConfig, estimate_action_dataset, generate_action_dataset
from .smoke import full_smoke_run, generate_target_set, inspect_plugin, smoke_evaluate, smoke_random_env, smoke_train_clap
from .training import evaluate_dqn, run_random_policy, train_dqn, train_dqn_batched

ARTIFACTS_ROOT = Path("artifacts")
TARGETS_DIR_NAME = "targets"
TRAIN_DIR_NAME = "train_dqn"
SMOKE_RANDOM_DIR_NAME = "smoke_random_env"
SMOKE_TRAIN_DIR_NAME = "smoke_train_clap"
SMOKE_EVAL_DIR_NAME = "smoke_evaluate"


def _base_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rl-synth",
        description="VST3 synth inspection, target generation, smoke runs, and DQN training.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect-plugin", help="Load a plugin and print metadata.")
    inspect_parser.add_argument(
        "--plugin",
        required=True,
        help="Path to a .vst3 instrument plugin to inspect. Expected: existing .vst3 file path.",
    )
    inspect_parser.add_argument(
        "--run-folder",
        default="inspect",
        help="Artifact root for this run. Output is always written under artifacts/<name>. The directory is created automatically. Default: inspect.",
    )

    render_parser = subparsers.add_parser("render", help="Render one note from a plugin.")
    render_parser.add_argument(
        "--plugin",
        required=True,
        help="Path to a .vst3 instrument plugin to render. Expected: existing .vst3 file path.",
    )
    render_parser.add_argument(
        "--note",
        type=int,
        default=60,
        help="MIDI note number sent to the synth. Expected range: 0-127. Default: 60.",
    )
    render_parser.add_argument(
        "--duration",
        type=float,
        default=1.0,
        help="Rendered note length in seconds, excluding any configured tail. Expected range: > 0. Default: 1.0.",
    )

    random_parser = subparsers.add_parser("random-agent", help="Run a random agent in the environment.")
    random_parser.add_argument(
        "--plugin",
        required=True,
        help="Path to a .vst3 instrument plugin used by the environment. Expected: existing .vst3 file path.",
    )
    random_parser.add_argument(
        "--episodes",
        type=int,
        default=3,
        help="Number of rollout episodes to run. Expected range: integer >= 1. Default: 3.",
    )
    random_parser.add_argument(
        "--run-folder",
        default=None,
        help="Optional artifact root under artifacts/. If provided, random-agent looks for targets/manifest.json there and uses that preset-derived target set. If omitted, it uses the synthetic target pool.",
    )
    random_parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable live progress output during random rollouts. Use --no-progress for plain final JSON only. Default: enabled.",
    )
    random_parser.add_argument(
        "--episode-log-interval",
        type=int,
        default=1,
        help="Print one episode summary every N completed episodes. Expected range: integer >= 1. Default: 1.",
    )

    train_parser = subparsers.add_parser("train-dqn", help="Train the DQN agent against a generated target set.")
    train_parser.add_argument(
        "--plugin",
        required=True,
        help="Path to the .vst3 synth plugin used for training episodes. Expected: existing .vst3 file path.",
    )
    train_parser.add_argument(
        "--run-folder",
        required=True,
        help="Artifact root under artifacts/ for this experiment, for example artifacts/kr106_real. The command reads targets/manifest.json from there and writes training outputs to train_dqn/ beneath it.",
    )
    train_parser.add_argument(
        "--steps",
        type=int,
        default=2000,
        help="Total environment interaction steps for training. Larger values mean more replay data and more backprop updates. Expected range: integer >= 1. Default: 2000.",
    )
    train_parser.add_argument(
        "--reward-mode",
        choices=("random", "clap"),
        default="random",
        help="Reward source. 'random' is for plumbing checks only. 'clap' uses audio-embedding distance improvement. Expected: random or clap. Default: random.",
    )
    train_parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable live progress bars and stage logs during training. Use --no-progress for plain periodic log lines only. Default: enabled.",
    )
    train_parser.add_argument(
        "--log-interval",
        type=int,
        default=25,
        help="Refresh console training metrics every N steps. Expected range: integer >= 1. Default: 25.",
    )
    train_parser.add_argument(
        "--episode-log-interval",
        type=int,
        default=10,
        help="Print one completed-episode summary every N episodes during training. Expected range: integer >= 1. Default: 10.",
    )
    train_parser.add_argument(
        "--tensorboard",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write TensorBoard event files for training metrics. Use --no-tensorboard to disable. Default: enabled.",
    )
    train_parser.add_argument(
        "--tensorboard-dir",
        default=None,
        help="Optional TensorBoard subdirectory under artifacts/. If omitted, defaults to <run-folder>/train_dqn/tensorboard.",
    )
    train_parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Parallel rollout width. This sets both the number of synth-render worker processes and the number of active episode slots. Expected range: integer >= 1. Default: 1.",
    )
    train_parser.add_argument(
        "--updates-per-tick",
        type=int,
        default=1,
        help="Optimizer updates run after each batched rollout tick. Expected range: integer >= 1. Default: 1.",
    )
    train_parser.add_argument(
        "--clap-batch-size",
        type=int,
        default=None,
        help="Maximum number of audio buffers embedded together in one CLAP batch. If omitted, it defaults to --num-workers. Expected range: integer >= 1.",
    )
    train_parser.add_argument(
        "--epsilon-decay-steps",
        type=int,
        default=None,
        help="Number of action steps over which epsilon decays from epsilon_start to epsilon_end. The current scheduler is step-based, not episode-based. Expected range: integer >= 1. If omitted, the config default is used.",
    )
    train_parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=None,
        help="Maximum actions allowed within one episode before truncation. Expected range: integer >= 1. If omitted, the config default is used.",
    )

    eval_parser = subparsers.add_parser("evaluate", help="Evaluate the latest DQN checkpoint from a run folder.")
    eval_parser.add_argument(
        "--plugin",
        required=True,
        help="Path to the .vst3 synth plugin used during evaluation. Expected: existing .vst3 file path.",
    )
    eval_parser.add_argument(
        "--run-folder",
        required=True,
        help="Artifact root under artifacts/ for this experiment. The command reads targets/manifest.json and train_dqn/dqn_latest.pt from there.",
    )
    eval_parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Number of evaluation episodes to run. Expected range: integer >= 1. Default: 5.",
    )
    eval_parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable live progress output during evaluation. Use --no-progress for plain final JSON only. Default: enabled.",
    )
    eval_parser.add_argument(
        "--episode-log-interval",
        type=int,
        default=1,
        help="Print one evaluation summary every N completed episodes. Expected range: integer >= 1. Default: 1.",
    )
    eval_parser.add_argument(
        "--tensorboard",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write TensorBoard event files for evaluation metrics. Use --no-tensorboard to disable. Default: enabled.",
    )
    eval_parser.add_argument(
        "--tensorboard-dir",
        default=None,
        help="Optional TensorBoard subdirectory under artifacts/. If omitted, defaults to <run-folder>/train_dqn/tensorboard.",
    )

    target_parser = subparsers.add_parser("generate-target-set", help="Generate preset-derived target audio and manifest files.")
    target_parser.add_argument(
        "--plugin",
        required=True,
        help="Path to the .vst3 synth plugin whose built-in presets are turned into target sounds. Expected: existing .vst3 file path.",
    )
    target_parser.add_argument(
        "--run-folder",
        required=True,
        help="Artifact root under artifacts/ for this experiment, for example artifacts/kr106_real. Targets are written to targets/ beneath it.",
    )
    target_parser.add_argument(
        "--subset-limit",
        type=int,
        default=12,
        help="Maximum number of presets to capture into the target set. Use a small value for smoke tests and a larger value for fuller coverage. Expected range: integer >= 1. Default: 12.",
    )
    target_parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable live progress output while rendering target presets. Use --no-progress for quieter output. Default: enabled.",
    )

    dataset_parser = subparsers.add_parser(
        "generate-action-dataset",
        help="Generate a reusable all-actions supervised reward dataset from a target manifest.",
    )
    dataset_parser.add_argument("--plugin", required=True, help="Path to the .vst3 synth plugin.")
    dataset_parser.add_argument(
        "--run-folder",
        required=True,
        help="Artifact root under artifacts/. The command reads targets/manifest.json and writes action_dataset/ beneath it.",
    )
    dataset_parser.add_argument(
        "--reward-mode",
        choices=("clap",),
        default="clap",
        help="Reward source for action labels. Default: clap.",
    )
    dataset_parser.add_argument("--max-states", type=int, default=256, help="Maximum dataset rows to generate. Default: 256.")
    dataset_parser.add_argument(
        "--moves-per-start",
        type=int,
        default=4,
        help="Greedy best-action moves to take from each target/start pair. Default: 4.",
    )
    dataset_parser.add_argument("--num-workers", type=int, default=1, help="Parallel render worker count. Default: 1.")
    dataset_parser.add_argument(
        "--clap-batch-size",
        type=int,
        default=8,
        help="Maximum audio buffers embedded in one CLAP batch. Default: 8.",
    )
    dataset_parser.add_argument(
        "--estimate-only",
        action="store_true",
        help="Run one representative all-action evaluation and print cost estimates without writing dataset.npz.",
    )
    dataset_parser.add_argument(
        "--yes",
        action="store_true",
        help="Confirm generation when estimated render count is large.",
    )
    dataset_parser.add_argument(
        "--render-timeout-seconds",
        type=float,
        default=300.0,
        help="Timeout for one render batch/chunk. Timed-out chunks are skipped by default. Default: 300.",
    )
    dataset_parser.add_argument(
        "--skip-failed-actions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Mark timed-out action chunks with a large negative reward instead of aborting. Default: enabled.",
    )
    dataset_parser.add_argument(
        "--shard-size",
        type=int,
        default=16,
        help="Write a recoverable shard every N generated states. Default: 16.",
    )
    dataset_parser.add_argument(
        "--render-chunk-size",
        type=int,
        default=0,
        help="Actions per render batch. Default 0 means --num-workers, which limits timeout blast radius.",
    )
    dataset_parser.add_argument(
        "--max-state-seconds",
        type=float,
        default=None,
        help="Skip a dataset state after this many seconds and move to the next target/start pair. Default: disabled.",
    )
    dataset_parser.add_argument(
        "--reload-workers-every-renders",
        type=int,
        default=500,
        help="Reload render-worker plugin instances after this many successful action renders. Use 0 to disable. Default: 500.",
    )
    dataset_parser.add_argument(
        "--preset-render-slowdown-threshold",
        type=float,
        default=1.5,
        help="Assert if a preset pair's mean action render time exceeds the prior running mean by this multiplier. Use 0 to disable. Default: 1.5.",
    )
    dataset_parser.add_argument(
        "--reload-workers-on-render-slowdown",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reload render workers immediately when a render chunk crosses the slowdown threshold. Default: enabled.",
    )
    dataset_parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable progress bars and stage logs. Default: enabled.",
    )

    sweep_parser = subparsers.add_parser(
        "compare-architectures",
        help="Train and rank supervised action-value architectures against a generated action dataset.",
    )
    sweep_parser.add_argument("--dataset", required=True, help="Path to action_dataset/dataset.npz.")
    sweep_parser.add_argument("--config", required=True, help="Path to JSON architecture sweep config.")
    sweep_parser.add_argument(
        "--out-dir",
        required=True,
        help="Output directory for architecture sweep artifacts. Relative paths are resolved under artifacts/.",
    )
    sweep_parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable progress bars and stage logs. Default: enabled.",
    )
    sweep_parser.add_argument(
        "--tensorboard",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Write TensorBoard logs for supervised architecture training. Default: disabled.",
    )

    search_parser = subparsers.add_parser(
        "search-feature-change-models",
        help="Run a CNN/RNN-focused architecture search across action datasets.",
    )
    search_parser.add_argument(
        "--dataset",
        action="append",
        default=None,
        help="Path to an action_dataset/dataset.npz file. Repeat to search multiple datasets. If omitted, datasets are discovered under --artifacts-root.",
    )
    search_parser.add_argument(
        "--artifacts-root",
        default=str(ARTIFACTS_ROOT),
        help="Root used to discover */action_dataset/dataset.npz when --dataset is omitted. Default: artifacts.",
    )
    search_parser.add_argument(
        "--config",
        default=None,
        help="Optional JSON sweep config. If omitted, a short CNN/RNN-heavy default config is generated.",
    )
    search_parser.add_argument(
        "--out-dir",
        default="architecture_search/feature_change",
        help="Output directory for combined search artifacts. Relative paths are resolved under artifacts/. Default: architecture_search/feature_change.",
    )
    search_parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Epochs for the generated default config. Ignored when --config is provided. Default: 5.",
    )
    search_parser.add_argument(
        "--cv-folds",
        type=int,
        default=1,
        help="Grouped cross-validation folds for the generated default config. Use 1 to disable. Ignored when --config is provided. Default: 1.",
    )
    search_parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable progress bars and stage logs. Default: enabled.",
    )
    search_parser.add_argument(
        "--tensorboard",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Write TensorBoard logs for supervised training. Default: disabled.",
    )

    smoke_random_parser = subparsers.add_parser("smoke-random-env", help="Run the random-agent smoke baseline.")
    smoke_random_parser.add_argument(
        "--plugin",
        required=True,
        help="Path to the .vst3 synth plugin used for the smoke environment. Expected: existing .vst3 file path.",
    )
    smoke_random_parser.add_argument(
        "--run-folder",
        required=True,
        help="Artifact root under artifacts/ for this experiment. The command reads targets/manifest.json and writes smoke_random_env/ beneath the run folder.",
    )
    smoke_random_parser.add_argument(
        "--episodes",
        type=int,
        default=4,
        help="Number of smoke baseline episodes to run. Expected range: integer >= 1. Default: 4.",
    )
    smoke_random_parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable live smoke progress output. Use --no-progress for quieter output. Default: enabled.",
    )
    smoke_random_parser.add_argument(
        "--episode-log-interval",
        type=int,
        default=1,
        help="Print one smoke-episode summary every N episodes. Expected range: integer >= 1. Default: 1.",
    )

    smoke_train_parser = subparsers.add_parser("smoke-train-clap", help="Run CLAP-backed smoke training.")
    smoke_train_parser.add_argument(
        "--plugin",
        required=True,
        help="Path to the .vst3 synth plugin used for CLAP-backed smoke training. Expected: existing .vst3 file path.",
    )
    smoke_train_parser.add_argument(
        "--run-folder",
        required=True,
        help="Artifact root under artifacts/ for this experiment. The command reads targets/manifest.json and writes smoke_train_clap/ beneath the run folder.",
    )
    smoke_train_parser.add_argument(
        "--steps",
        type=int,
        default=128,
        help="Training steps for the smoke run. This should be large enough to pass replay warmup and exercise backprop. Expected range: integer >= 1. Default: 128.",
    )
    smoke_train_parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable live smoke-training progress output. Use --no-progress for quieter output. Default: enabled.",
    )
    smoke_train_parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        help="Refresh smoke-training console metrics every N steps. Expected range: integer >= 1. Default: 10.",
    )
    smoke_train_parser.add_argument(
        "--episode-log-interval",
        type=int,
        default=1,
        help="Print one smoke-training episode summary every N episodes. Expected range: integer >= 1. Default: 1.",
    )
    smoke_train_parser.add_argument(
        "--tensorboard",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write TensorBoard event files for smoke training. Use --no-tensorboard to disable. Default: enabled.",
    )
    smoke_train_parser.add_argument(
        "--tensorboard-dir",
        default=None,
        help="Optional TensorBoard subdirectory under artifacts/. If omitted, defaults to <run-folder>/smoke_train_clap/tensorboard.",
    )

    smoke_eval_parser = subparsers.add_parser("smoke-evaluate", help="Run held-out evaluation for a smoke checkpoint.")
    smoke_eval_parser.add_argument(
        "--plugin",
        required=True,
        help="Path to the .vst3 synth plugin used for smoke evaluation. Expected: existing .vst3 file path.",
    )
    smoke_eval_parser.add_argument(
        "--run-folder",
        required=True,
        help="Artifact root under artifacts/ for this experiment. The command reads targets/manifest.json and smoke_train_clap/dqn_smoke.pt from there.",
    )
    smoke_eval_parser.add_argument(
        "--episodes",
        type=int,
        default=3,
        help="Number of held-out evaluation episodes to run. Expected range: integer >= 1. Default: 3.",
    )
    smoke_eval_parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable live smoke-evaluation progress output. Use --no-progress for quieter output. Default: enabled.",
    )
    smoke_eval_parser.add_argument(
        "--episode-log-interval",
        type=int,
        default=1,
        help="Print one smoke-evaluation summary every N episodes. Expected range: integer >= 1. Default: 1.",
    )
    smoke_eval_parser.add_argument(
        "--tensorboard",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write TensorBoard event files for smoke evaluation. Use --no-tensorboard to disable. Default: enabled.",
    )
    smoke_eval_parser.add_argument(
        "--tensorboard-dir",
        default=None,
        help="Optional TensorBoard subdirectory under artifacts/. If omitted, defaults to <run-folder>/smoke_train_clap/tensorboard.",
    )

    full_smoke_parser = subparsers.add_parser("full-smoke", help="Run the full end-to-end real-plugin smoke workflow.")
    full_smoke_parser.add_argument(
        "--plugin",
        required=True,
        help="Path to the .vst3 synth plugin used for the end-to-end smoke workflow. Expected: existing .vst3 file path.",
    )
    full_smoke_parser.add_argument(
        "--run-folder",
        default="full_smoke",
        help="Artifact root under artifacts/ for this run. The directory is created automatically. Default: full_smoke.",
    )
    full_smoke_parser.add_argument(
        "--subset-limit",
        type=int,
        default=12,
        help="Maximum number of preset-derived targets to include in the smoke dataset. Expected range: integer >= 1. Default: 12.",
    )
    full_smoke_parser.add_argument(
        "--random-episodes",
        type=int,
        default=6,
        help="Random-baseline episodes run before training for comparison. Expected range: integer >= 1. Default: 6.",
    )
    full_smoke_parser.add_argument(
        "--train-steps",
        type=int,
        default=64,
        help="Training steps in the full smoke run. Choose enough steps to exercise replay warmup and optimizer updates. Expected range: integer >= 1. Default: 64.",
    )
    full_smoke_parser.add_argument(
        "--eval-episodes",
        type=int,
        default=4,
        help="Held-out evaluation episodes run after training. Expected range: integer >= 1. Default: 4.",
    )
    full_smoke_parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable live progress bars and stage logs across the full smoke workflow. Use --no-progress for quieter output. Default: enabled.",
    )
    full_smoke_parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        help="Refresh training metrics every N steps during the full smoke run. Expected range: integer >= 1. Default: 10.",
    )
    full_smoke_parser.add_argument(
        "--episode-log-interval",
        type=int,
        default=1,
        help="Print one episode summary every N episodes during random, train, and eval smoke phases. Expected range: integer >= 1. Default: 1.",
    )
    full_smoke_parser.add_argument(
        "--tensorboard",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write TensorBoard event files for full smoke training and evaluation. Use --no-tensorboard to disable. Default: enabled.",
    )
    full_smoke_parser.add_argument(
        "--tensorboard-dir",
        default=None,
        help="Optional TensorBoard subdirectory under artifacts/. If omitted, defaults to <run-folder>/tensorboard.",
    )
    return parser


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "run"


def _resolve_run_folder(run_folder: str | Path | None, *, default_name: str | None = None, create: bool) -> Path:
    if run_folder is None:
        assert default_name is not None, "default_name is required when run_folder is omitted."
        raw = Path(default_name)
    else:
        raw = Path(run_folder)
    if raw.is_absolute():
        relative = Path(raw.name)
    elif raw.parts and raw.parts[0] == ARTIFACTS_ROOT.name:
        relative = Path(*raw.parts[1:]) if len(raw.parts) > 1 else Path(default_name or "run")
    else:
        relative = raw
    resolved = ARTIFACTS_ROOT / relative
    if create:
        resolved.mkdir(parents=True, exist_ok=True)
    else:
        assert resolved.exists(), f"Expected run folder {resolved} but it was not found."
    return resolved


def _resolve_tensorboard_dir(run_folder: Path, command_name: str, tensorboard_dir: str | None) -> Path:
    if tensorboard_dir is not None:
        return _resolve_run_folder(tensorboard_dir, create=True)
    if command_name == "train-dqn":
        base = run_folder / TRAIN_DIR_NAME
    elif command_name == "smoke-train-clap":
        base = run_folder / SMOKE_TRAIN_DIR_NAME
    elif command_name == "smoke-evaluate":
        base = run_folder / SMOKE_TRAIN_DIR_NAME
    elif command_name == "evaluate":
        base = run_folder / TRAIN_DIR_NAME
    else:
        base = run_folder
    base.mkdir(parents=True, exist_ok=True)
    return base / "tensorboard"


def _find_manifest(run_folder: Path) -> Path:
    manifest_path = run_folder / TARGETS_DIR_NAME / "manifest.json"
    assert manifest_path.exists(), (
        f"Expected {manifest_path} but it was not found. "
        f"Did you run generate-target-set with --run-folder {run_folder}?"
    )
    return manifest_path


def _find_smoke_checkpoint(run_folder: Path) -> Path:
    checkpoint_path = run_folder / SMOKE_TRAIN_DIR_NAME / "dqn_smoke.pt"
    assert checkpoint_path.exists(), (
        f"Expected {checkpoint_path} but it was not found. "
        f"Did you run smoke-train-clap with --run-folder {run_folder}?"
    )
    return checkpoint_path


def _find_train_checkpoint(run_folder: Path) -> Path:
    checkpoint_path = run_folder / TRAIN_DIR_NAME / "dqn_latest.pt"
    assert checkpoint_path.exists(), (
        f"Expected {checkpoint_path} but it was not found. "
        f"Did you run train-dqn with --run-folder {run_folder}?"
    )
    return checkpoint_path


def _is_positive_int(value: object) -> bool:
    return isinstance(value, int) and value >= 1


def _is_nonnegative_int(value: object) -> bool:
    return isinstance(value, int) and value >= 0


def _is_positive_float(value: object) -> bool:
    return isinstance(value, (int, float)) and float(value) > 0.0


def _is_nonnegative_float(value: object) -> bool:
    return isinstance(value, (int, float)) and float(value) >= 0.0


def _argument_error(parser: argparse.ArgumentParser, message: str) -> None:
    parser.error(message)


def _require_plugin_path(parser: argparse.ArgumentParser, plugin_path: str) -> str:
    if not plugin_path:
        _argument_error(parser, "--plugin is required and cannot be empty.")
    path = Path(plugin_path).expanduser()
    if not path.exists():
        _argument_error(parser, f"--plugin must point to an existing VST3 instrument. Not found: {path}")
    if path.suffix.lower() != ".vst3":
        _argument_error(parser, f"--plugin should point to a .vst3 bundle or file. Got: {path}")
    return str(path)


def _require_existing_run_manifest(parser: argparse.ArgumentParser, run_folder: str, command: str) -> None:
    try:
        run_root = _resolve_run_folder(run_folder, create=False)
        _find_manifest(run_root)
    except AssertionError as exc:
        _argument_error(parser, f"{command} requires a generated target set. {exc}")


def _require_existing_checkpoint(parser: argparse.ArgumentParser, run_folder: str, command: str, *, smoke: bool = False) -> None:
    try:
        run_root = _resolve_run_folder(run_folder, create=False)
        if smoke:
            _find_smoke_checkpoint(run_root)
        else:
            _find_train_checkpoint(run_root)
    except AssertionError as exc:
        _argument_error(parser, f"{command} requires an existing checkpoint. {exc}")


def _require_existing_file(parser: argparse.ArgumentParser, option: str, value: str) -> None:
    path = Path(value).expanduser()
    if not path.exists():
        _argument_error(parser, f"{option} must point to an existing file. Not found: {path}")
    if not path.is_file():
        _argument_error(parser, f"{option} must point to a file. Got: {path}")


def _validate_common_logging(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if hasattr(args, "log_interval") and not _is_positive_int(args.log_interval):
        _argument_error(parser, "--log-interval must be an integer >= 1.")
    if hasattr(args, "episode_log_interval") and not _is_positive_int(args.episode_log_interval):
        _argument_error(parser, "--episode-log-interval must be an integer >= 1.")


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    _validate_common_logging(parser, args)
    plugin_commands = {
        "inspect-plugin",
        "render",
        "random-agent",
        "train-dqn",
        "evaluate",
        "generate-target-set",
        "generate-action-dataset",
        "smoke-random-env",
        "smoke-train-clap",
        "smoke-evaluate",
        "full-smoke",
    }
    if args.command in plugin_commands:
        args.plugin = _require_plugin_path(parser, args.plugin)

    if args.command == "render":
        if not (0 <= int(args.note) <= 127):
            _argument_error(parser, "--note must be a MIDI note number in the range 0-127.")
        if not _is_positive_float(args.duration):
            _argument_error(parser, "--duration must be > 0 seconds.")
    elif args.command == "random-agent":
        if not _is_positive_int(args.episodes):
            _argument_error(parser, "--episodes must be an integer >= 1.")
        if args.run_folder is not None:
            _require_existing_run_manifest(parser, args.run_folder, "random-agent")
    elif args.command == "train-dqn":
        if not _is_positive_int(args.steps):
            _argument_error(parser, "--steps must be an integer >= 1.")
        if not _is_positive_int(args.num_workers):
            _argument_error(parser, "--num-workers must be an integer >= 1.")
        if not _is_positive_int(args.updates_per_tick):
            _argument_error(parser, "--updates-per-tick must be an integer >= 1.")
        if args.clap_batch_size is not None and not _is_positive_int(args.clap_batch_size):
            _argument_error(parser, "--clap-batch-size must be an integer >= 1.")
        if args.epsilon_decay_steps is not None and not _is_positive_int(args.epsilon_decay_steps):
            _argument_error(parser, "--epsilon-decay-steps must be an integer >= 1.")
        if args.max_episode_steps is not None and not _is_positive_int(args.max_episode_steps):
            _argument_error(parser, "--max-episode-steps must be an integer >= 1.")
        _require_existing_run_manifest(parser, args.run_folder, "train-dqn")
    elif args.command == "evaluate":
        if not _is_positive_int(args.episodes):
            _argument_error(parser, "--episodes must be an integer >= 1.")
        _require_existing_run_manifest(parser, args.run_folder, "evaluate")
        _require_existing_checkpoint(parser, args.run_folder, "evaluate")
    elif args.command == "generate-target-set":
        if not _is_positive_int(args.subset_limit):
            _argument_error(parser, "--subset-limit must be an integer >= 1.")
    elif args.command == "generate-action-dataset":
        if not _is_positive_int(args.max_states):
            _argument_error(parser, "--max-states must be an integer >= 1.")
        if not _is_positive_int(args.moves_per_start):
            _argument_error(parser, "--moves-per-start must be an integer >= 1.")
        if not _is_positive_int(args.num_workers):
            _argument_error(parser, "--num-workers must be an integer >= 1.")
        if not _is_positive_int(args.clap_batch_size):
            _argument_error(parser, "--clap-batch-size must be an integer >= 1.")
        if not _is_positive_float(args.render_timeout_seconds):
            _argument_error(parser, "--render-timeout-seconds must be > 0.")
        if not _is_positive_int(args.shard_size):
            _argument_error(parser, "--shard-size must be an integer >= 1.")
        if not _is_nonnegative_int(args.render_chunk_size):
            _argument_error(parser, "--render-chunk-size must be an integer >= 0.")
        if args.max_state_seconds is not None and not _is_positive_float(args.max_state_seconds):
            _argument_error(parser, "--max-state-seconds must be > 0 when provided.")
        if not _is_nonnegative_int(args.reload_workers_every_renders):
            _argument_error(parser, "--reload-workers-every-renders must be an integer >= 0.")
        if not _is_nonnegative_float(args.preset_render_slowdown_threshold):
            _argument_error(parser, "--preset-render-slowdown-threshold must be >= 0.")
        _require_existing_run_manifest(parser, args.run_folder, "generate-action-dataset")
    elif args.command == "compare-architectures":
        _require_existing_file(parser, "--dataset", args.dataset)
        _require_existing_file(parser, "--config", args.config)
    elif args.command == "search-feature-change-models":
        if not _is_positive_int(args.epochs):
            _argument_error(parser, "--epochs must be an integer >= 1.")
        if not _is_positive_int(args.cv_folds):
            _argument_error(parser, "--cv-folds must be an integer >= 1.")
        if args.config is not None:
            _require_existing_file(parser, "--config", args.config)
        if args.dataset:
            for dataset_path in args.dataset:
                _require_existing_file(parser, "--dataset", dataset_path)
        else:
            discovered = discover_action_datasets(Path(args.artifacts_root))
            if not discovered:
                _argument_error(
                    parser,
                    f"No action datasets found under {args.artifacts_root}. "
                    "Pass --dataset path/to/action_dataset/dataset.npz or generate one first.",
                )
    elif args.command == "smoke-random-env":
        if not _is_positive_int(args.episodes):
            _argument_error(parser, "--episodes must be an integer >= 1.")
        _require_existing_run_manifest(parser, args.run_folder, "smoke-random-env")
    elif args.command == "smoke-train-clap":
        if not _is_positive_int(args.steps):
            _argument_error(parser, "--steps must be an integer >= 1.")
        _require_existing_run_manifest(parser, args.run_folder, "smoke-train-clap")
    elif args.command == "smoke-evaluate":
        if not _is_positive_int(args.episodes):
            _argument_error(parser, "--episodes must be an integer >= 1.")
        _require_existing_run_manifest(parser, args.run_folder, "smoke-evaluate")
        _require_existing_checkpoint(parser, args.run_folder, "smoke-evaluate", smoke=True)
    elif args.command == "full-smoke":
        if not _is_positive_int(args.subset_limit):
            _argument_error(parser, "--subset-limit must be an integer >= 1.")
        if not _is_positive_int(args.random_episodes):
            _argument_error(parser, "--random-episodes must be an integer >= 1.")
        if not _is_positive_int(args.train_steps):
            _argument_error(parser, "--train-steps must be an integer >= 1.")
        if not _is_positive_int(args.eval_episodes):
            _argument_error(parser, "--eval-episodes must be an integer >= 1.")


def _experiment_config(
    plugin_path: str,
    reward_mode: str = "random",
    manifest_path: Path | None = None,
    artifacts_dir: Path | None = None,
    run_name: str = "default",
    num_workers: int = 1,
    updates_per_tick: int = 1,
    clap_batch_size: int | None = None,
    clap_batch_timeout_ms: int = 10,
    epsilon_decay_steps: int | None = None,
    max_episode_steps: int | None = None,
) -> ExperimentConfig:
    artifact_root = artifacts_dir or ARTIFACTS_ROOT / "default"
    host = SynthHostConfig(plugin_path=Path(plugin_path))
    reward = RewardConfig(mode=reward_mode)
    env = SynthEnvConfig(
        host=host,
        reward=reward,
        target_mode="preset_manifest" if manifest_path else "synthetic_pool",
        artifacts_dir=artifact_root,
    )
    if max_episode_steps is not None:
        env.max_episode_steps = int(max_episode_steps)
    curriculum = CurriculumConfig(manifest_path=manifest_path)
    resolved_clap_batch_size = num_workers if clap_batch_size is None else clap_batch_size
    dqn = DQNConfig()
    if epsilon_decay_steps is not None:
        dqn.epsilon_decay_steps = int(epsilon_decay_steps)
    return ExperimentConfig(
        env=env,
        curriculum=curriculum,
        dqn=dqn,
        output_dir=artifact_root,
        run_name=run_name,
        num_render_workers=num_workers,
        num_parallel_envs=num_workers,
        updates_per_tick=updates_per_tick,
        clap_batch_size=resolved_clap_batch_size,
        clap_batch_timeout_ms=clap_batch_timeout_ms,
    )


def _cmd_inspect(plugin_path: str, run_folder: str) -> None:
    run_root = _resolve_run_folder(run_folder, create=True)
    payload = inspect_plugin(Path(plugin_path), run_root)
    print(json.dumps(payload, indent=2))


def _cmd_render(plugin_path: str, note: int, duration: float) -> None:
    host = SynthHost(SynthHostConfig(plugin_path=Path(plugin_path), note=note, render_duration=duration))
    audio = host.render_note()
    summary = {
        "num_samples": int(audio.shape[0]),
        "dtype": str(audio.dtype),
        "mean_abs": float(np.mean(np.abs(audio))),
        "max_abs": float(np.max(np.abs(audio))),
    }
    print(json.dumps(summary, indent=2))


def _cmd_random_agent(plugin_path: str, episodes: int, run_folder: str | None, progress: bool, episode_log_interval: int) -> None:
    manifest_path = None
    if run_folder is not None:
        run_root = _resolve_run_folder(run_folder, create=False)
        manifest_path = _find_manifest(run_root)
    config = _experiment_config(plugin_path, reward_mode="random", manifest_path=manifest_path)
    env = make_env(config.env, config.curriculum)
    metrics = run_random_policy(env, episodes=episodes, progress=progress, episode_log_interval=episode_log_interval)
    print(json.dumps([asdict(metric) for metric in metrics], indent=2))


def _cmd_train_dqn(
    plugin_path: str,
    run_folder: str,
    steps: int,
    reward_mode: str,
    progress: bool,
    log_interval: int,
    episode_log_interval: int,
    tensorboard: bool,
    tensorboard_dir: str | None,
    num_workers: int,
    updates_per_tick: int,
    clap_batch_size: int | None,
    epsilon_decay_steps: int | None,
    max_episode_steps: int | None,
) -> None:
    run_root = _resolve_run_folder(run_folder, create=True)
    manifest_path = _find_manifest(run_root)
    train_dir = run_root / TRAIN_DIR_NAME
    train_dir.mkdir(parents=True, exist_ok=True)
    config = _experiment_config(
        plugin_path,
        reward_mode=reward_mode,
        manifest_path=manifest_path,
        artifacts_dir=train_dir,
        run_name=train_dir.name,
        num_workers=num_workers,
        updates_per_tick=updates_per_tick,
        clap_batch_size=clap_batch_size,
        epsilon_decay_steps=epsilon_decay_steps,
        max_episode_steps=max_episode_steps,
    )
    resolved_tensorboard_dir = _resolve_tensorboard_dir(run_root, "train-dqn", tensorboard_dir)
    checkpoint_path = train_dir / "dqn_latest.pt"
    batched = num_workers > 1
    train_fn = train_dqn_batched if batched else train_dqn
    agent, logs = train_fn(
        config,
        total_steps=steps,
        progress=progress,
        log_interval=log_interval,
        episode_log_interval=episode_log_interval,
        tensorboard=tensorboard,
        tensorboard_dir=resolved_tensorboard_dir,
    )
    agent.save(checkpoint_path)
    print(
        json.dumps(
            {
                "run_folder": str(run_root),
                "train_dir": str(train_dir),
                "checkpoint": str(checkpoint_path),
                "tensorboard_dir": str(resolved_tensorboard_dir),
                "mode": "batched" if batched else "single_env",
                "last_log": logs[-1] if logs else None,
            },
            indent=2,
        )
    )


def _cmd_evaluate(
    plugin_path: str,
    run_folder: str,
    episodes: int,
    progress: bool,
    episode_log_interval: int,
    tensorboard: bool,
    tensorboard_dir: str | None,
) -> None:
    run_root = _resolve_run_folder(run_folder, create=False)
    manifest_path = _find_manifest(run_root)
    checkpoint_path = _find_train_checkpoint(run_root)
    config = _experiment_config(plugin_path, reward_mode="clap", manifest_path=manifest_path, artifacts_dir=run_root / TRAIN_DIR_NAME)
    metrics = evaluate_dqn(
        config,
        checkpoint=checkpoint_path,
        episodes=episodes,
        progress=progress,
        episode_log_interval=episode_log_interval,
        tensorboard=tensorboard,
        tensorboard_dir=_resolve_tensorboard_dir(run_root, "evaluate", tensorboard_dir),
    )
    print(json.dumps([asdict(metric) for metric in metrics], indent=2))


def _cmd_generate_action_dataset(
    plugin_path: str,
    run_folder: str,
    reward_mode: str,
    max_states: int,
    moves_per_start: int,
    num_workers: int,
    clap_batch_size: int,
    estimate_only: bool,
    yes: bool,
    progress: bool,
    render_timeout_seconds: float,
    skip_failed_actions: bool,
    shard_size: int,
    render_chunk_size: int,
    max_state_seconds: float | None,
    reload_workers_every_renders: int,
    preset_render_slowdown_threshold: float,
    reload_workers_on_render_slowdown: bool,
) -> None:
    run_root = _resolve_run_folder(run_folder, create=True)
    manifest_path = _find_manifest(run_root)
    config = ActionDatasetConfig(
        plugin_path=Path(plugin_path),
        manifest_path=manifest_path,
        output_dir=run_root,
        reward_mode=reward_mode,
        max_states=max_states,
        moves_per_start=moves_per_start,
        num_workers=num_workers,
        clap_batch_size=clap_batch_size,
        render_timeout_seconds=render_timeout_seconds,
        skip_failed_actions=skip_failed_actions,
        shard_size=shard_size,
        render_chunk_size=render_chunk_size,
        max_state_seconds=max_state_seconds,
        reload_workers_every_renders=reload_workers_every_renders,
        preset_render_slowdown_threshold=preset_render_slowdown_threshold,
        reload_workers_on_render_slowdown=reload_workers_on_render_slowdown,
    )
    estimate = estimate_action_dataset(config, progress=progress)
    if estimate_only:
        print(json.dumps({"run_folder": str(run_root), "estimate": estimate}, indent=2))
        return
    result = generate_action_dataset(config, progress=progress, yes=yes, estimate=estimate)
    print(json.dumps(result, indent=2))


def _cmd_compare_architectures(
    dataset_path: str,
    config_path: str,
    out_dir: str,
    progress: bool,
    tensorboard: bool,
) -> None:
    result = compare_architectures(
        Path(dataset_path),
        Path(config_path),
        _resolve_run_folder(out_dir, create=True),
        progress=progress,
        tensorboard=tensorboard,
    )
    print(json.dumps(result, indent=2))


def _cmd_search_feature_change_models(
    dataset_paths: list[str] | None,
    artifacts_root: str,
    config_path: str | None,
    out_dir: str,
    epochs: int,
    cv_folds: int,
    progress: bool,
    tensorboard: bool,
) -> None:
    if dataset_paths:
        datasets = [Path(path) for path in dataset_paths]
    else:
        datasets = discover_action_datasets(Path(artifacts_root))
    if config_path is not None:
        config = json.loads(Path(config_path).read_text())
    else:
        config = default_feature_change_search_config(epochs=epochs, cv_folds=cv_folds)
    result = run_feature_change_search(
        datasets,
        _resolve_run_folder(out_dir, create=True),
        config=config,
        progress=progress,
        tensorboard=tensorboard,
    )
    print(json.dumps(result, indent=2))


def main() -> None:
    parser = _base_parser()
    args = parser.parse_args()
    _validate_args(parser, args)
    try:
        if args.command == "inspect-plugin":
            _cmd_inspect(args.plugin, args.run_folder)
        elif args.command == "render":
            _cmd_render(args.plugin, args.note, args.duration)
        elif args.command == "random-agent":
            _cmd_random_agent(args.plugin, args.episodes, args.run_folder, args.progress, args.episode_log_interval)
        elif args.command == "train-dqn":
            _cmd_train_dqn(
                args.plugin,
                args.run_folder,
                args.steps,
                args.reward_mode,
                args.progress,
                args.log_interval,
                args.episode_log_interval,
                args.tensorboard,
                args.tensorboard_dir,
                args.num_workers,
                args.updates_per_tick,
                args.clap_batch_size,
                args.epsilon_decay_steps,
                args.max_episode_steps,
            )
        elif args.command == "evaluate":
            _cmd_evaluate(
                args.plugin,
                args.run_folder,
                args.episodes,
                args.progress,
                args.episode_log_interval,
                args.tensorboard,
                args.tensorboard_dir,
            )
        elif args.command == "generate-target-set":
            run_root = _resolve_run_folder(args.run_folder, create=True)
            print(
                json.dumps(
                    generate_target_set(
                        Path(args.plugin),
                        run_root,
                        subset_limit=args.subset_limit,
                        progress=args.progress,
                    ),
                    indent=2,
                )
            )
        elif args.command == "generate-action-dataset":
            _cmd_generate_action_dataset(
                args.plugin,
                args.run_folder,
                args.reward_mode,
                args.max_states,
                args.moves_per_start,
                args.num_workers,
                args.clap_batch_size,
                args.estimate_only,
                args.yes,
                args.progress,
                args.render_timeout_seconds,
                args.skip_failed_actions,
                args.shard_size,
                args.render_chunk_size,
                args.max_state_seconds,
                args.reload_workers_every_renders,
                args.preset_render_slowdown_threshold,
                args.reload_workers_on_render_slowdown,
            )
        elif args.command == "compare-architectures":
            _cmd_compare_architectures(
                args.dataset,
                args.config,
                args.out_dir,
                args.progress,
                args.tensorboard,
            )
        elif args.command == "search-feature-change-models":
            _cmd_search_feature_change_models(
                args.dataset,
                args.artifacts_root,
                args.config,
                args.out_dir,
                args.epochs,
                args.cv_folds,
                args.progress,
                args.tensorboard,
            )
        elif args.command == "smoke-random-env":
            run_root = _resolve_run_folder(args.run_folder, create=False)
            print(
                json.dumps(
                    smoke_random_env(
                        Path(args.plugin),
                        run_root,
                        _find_manifest(run_root),
                        episodes=args.episodes,
                        progress=args.progress,
                        episode_log_interval=args.episode_log_interval,
                    ),
                    indent=2,
                )
            )
        elif args.command == "smoke-train-clap":
            run_root = _resolve_run_folder(args.run_folder, create=True)
            print(
                json.dumps(
                    smoke_train_clap(
                        Path(args.plugin),
                        run_root,
                        _find_manifest(run_root),
                        steps=args.steps,
                        progress=args.progress,
                        log_interval=args.log_interval,
                        episode_log_interval=args.episode_log_interval,
                        tensorboard=args.tensorboard,
                        tensorboard_dir=_resolve_tensorboard_dir(run_root, "smoke-train-clap", args.tensorboard_dir),
                    ),
                    indent=2,
                )
            )
        elif args.command == "smoke-evaluate":
            run_root = _resolve_run_folder(args.run_folder, create=False)
            print(
                json.dumps(
                    smoke_evaluate(
                        Path(args.plugin),
                        run_root,
                        _find_manifest(run_root),
                        _find_smoke_checkpoint(run_root),
                        episodes=args.episodes,
                        progress=args.progress,
                        episode_log_interval=args.episode_log_interval,
                        tensorboard=args.tensorboard,
                        tensorboard_dir=_resolve_tensorboard_dir(run_root, "smoke-evaluate", args.tensorboard_dir),
                    ),
                    indent=2,
                )
            )
        elif args.command == "full-smoke":
            run_root = _resolve_run_folder(args.run_folder, create=True)
            print(
                json.dumps(
                    full_smoke_run(
                        Path(args.plugin),
                        run_root,
                        subset_limit=args.subset_limit,
                        random_episodes=args.random_episodes,
                        train_steps=args.train_steps,
                        eval_episodes=args.eval_episodes,
                        progress=args.progress,
                        log_interval=args.log_interval,
                        episode_log_interval=args.episode_log_interval,
                        tensorboard=args.tensorboard,
                        tensorboard_dir=_resolve_tensorboard_dir(run_root, "full-smoke", args.tensorboard_dir),
                    ),
                    indent=2,
                )
            )
        else:
            raise AssertionError(f"Unhandled command: {args.command}")
    except (AssertionError, RuntimeError, ValueError) as exc:
        print(f"rl-synth: error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
