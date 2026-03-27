from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class SynthHostConfig:
    plugin_path: Path
    sample_rate: int = 44_100
    note: int = 60
    velocity: int = 100
    render_duration: float = 1.0
    tail_duration: float = 0.25
    warmup_duration: float = 0.0
    block_size: int = 512


@dataclass(slots=True)
class RewardConfig:
    mode: str = "random"
    clap_version: str = "2023"
    clap_checkpoint: Path | None = None
    clap_text_model_path: Path | None = None
    distance_metric: str = "cosine"


@dataclass(slots=True)
class CurriculumConfig:
    pool_size: int = 32
    train_size: int = 24
    val_size: int = 4
    test_size: int = 4
    dwell_episodes: int = 4
    seed: int = 7
    switching_mode: str = "uniform_rotation"
    manifest_path: Path | None = None
    subset_limit: int | None = None


@dataclass(slots=True)
class SynthEnvConfig:
    host: SynthHostConfig
    reward: RewardConfig = field(default_factory=RewardConfig)
    parameter_allowlist: list[str] = field(default_factory=list)
    parameter_denylist: list[str] = field(default_factory=list)
    action_step: float = 0.05
    max_episode_steps: int = 24
    success_threshold: float = 0.05
    target_mode: str = "synthetic_pool"
    seed: int = 7
    artifacts_dir: Path = Path("artifacts")


@dataclass(slots=True)
class DQNConfig:
    learning_rate: float = 1e-3
    gamma: float = 0.99
    batch_size: int = 64
    replay_capacity: int = 50_000
    warmup_steps: int = 512
    target_sync_interval: int = 250
    epsilon_start: float = 1.0
    epsilon_end: float = 0.3
    epsilon_decay_steps: int = 20_000
    hidden_sizes: tuple[int, ...] = (512, 256)


@dataclass(slots=True)
class ExperimentConfig:
    env: SynthEnvConfig
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    dqn: DQNConfig = field(default_factory=DQNConfig)
    output_dir: Path = Path("artifacts")
    run_name: str = "default"
    num_render_workers: int = 1
    num_parallel_envs: int = 1
    updates_per_tick: int = 1
    clap_batch_size: int = 8
    clap_batch_timeout_ms: int = 10
    render_queue_size: int = 0
    embed_queue_size: int = 0
