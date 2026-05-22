"""RL synth programmer package."""

from .agent import DQNAgent, RandomAgent
from .architecture_sweep import compare_architectures
from .config import (
    CurriculumConfig,
    DQNConfig,
    ExperimentConfig,
    RewardConfig,
    SynthEnvConfig,
    SynthHostConfig,
)
from .curriculum import TargetPool, TargetSpec
from .env import SynthProgrammingEnv
from .host import ParameterSpec, SynthHost
from .offline_dataset import ActionDatasetConfig, estimate_action_dataset, generate_action_dataset as generate_action_dataset_file
from .parallel_rollout import BatchedRolloutCoordinator, ParallelRenderPool
from .reward import CLAPEmbedder, RandomRewardModel, SimilarityRewardModel
from .smoke import generate_target_set, inspect_plugin, smoke_evaluate, smoke_random_env, smoke_train_clap
from .training import train_dqn_batched

__all__ = [
    "BatchedRolloutCoordinator",
    "CLAPEmbedder",
    "CurriculumConfig",
    "DQNAgent",
    "DQNConfig",
    "ExperimentConfig",
    "ParallelRenderPool",
    "ParameterSpec",
    "RandomAgent",
    "RandomRewardModel",
    "RewardConfig",
    "SimilarityRewardModel",
    "SynthEnvConfig",
    "SynthHost",
    "SynthHostConfig",
    "SynthProgrammingEnv",
    "TargetPool",
    "TargetSpec",
    "ActionDatasetConfig",
    "compare_architectures",
    "estimate_action_dataset",
    "generate_action_dataset_file",
    "train_dqn_batched",
    "generate_target_set",
    "inspect_plugin",
    "smoke_evaluate",
    "smoke_random_env",
    "smoke_train_clap",
]
