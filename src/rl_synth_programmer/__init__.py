"""RL synth programmer package."""

from .agent import DQNAgent, RandomAgent
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
from .reward import CLAPEmbedder, RandomRewardModel, SimilarityRewardModel
from .smoke import generate_target_set, inspect_plugin, smoke_evaluate, smoke_random_env, smoke_train_clap

__all__ = [
    "CLAPEmbedder",
    "CurriculumConfig",
    "DQNAgent",
    "DQNConfig",
    "ExperimentConfig",
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
    "generate_target_set",
    "inspect_plugin",
    "smoke_evaluate",
    "smoke_random_env",
    "smoke_train_clap",
]
