from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np

from .config import CurriculumConfig, DQNConfig, ExperimentConfig, RewardConfig, SynthEnvConfig, SynthHostConfig
from .env import make_env
from .host import SynthHost
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

    full_smoke_parser = subparsers.add_parser("full-smoke", help="Run the full corrected KR-106-style smoke workflow.")
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


def main() -> None:
    parser = _base_parser()
    args = parser.parse_args()
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


if __name__ == "__main__":
    main()
