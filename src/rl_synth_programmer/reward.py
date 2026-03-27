from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np
import yaml

from .config import RewardConfig
from .optional_deps import require_dependency


class AudioEmbedder(Protocol):
    def embed_audio(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        ...


@dataclass(slots=True)
class RandomRewardModel:
    seed: int = 7
    _rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)

    def reward(self, previous_distance: float | None = None, new_distance: float | None = None) -> float:
        _ = previous_distance, new_distance
        return float(self._rng.uniform(-1.0, 1.0))


class CLAPEmbedder:
    """Wrap msclap so the rest of the code works with in-memory numpy audio."""

    def __init__(self, config: RewardConfig):
        self.config = config
        msclap = require_dependency("msclap", "ml")
        model_fp = None if config.clap_checkpoint is None else str(config.clap_checkpoint)
        text_model_path = None if config.clap_text_model_path is None else str(config.clap_text_model_path)
        self._model = self._build_model(msclap, model_fp, config.clap_version, text_model_path)

    @staticmethod
    def _build_model(msclap, model_fp: str | None, version: str, text_model_path: str | None):
        if text_model_path is None:
            return msclap.CLAP(version=version, model_fp=model_fp, use_cuda=False)

        wrapper_mod = require_dependency("msclap.CLAPWrapper", "ml")
        wrapper = wrapper_mod.CLAPWrapper.__new__(wrapper_mod.CLAPWrapper)
        wrapper.supported_versions = wrapper_mod.CLAPWrapper.model_name.keys()
        if version not in wrapper.supported_versions:
            raise ValueError(f"Unsupported CLAP version: {version}")
        import argparse
        import os
        import re
        import sys

        wrapper.np_str_obj_array_pattern = re.compile(r"[SaUO]")
        wrapper.file_path = os.path.realpath(wrapper_mod.__file__)
        wrapper.default_collate_err_msg_format = (
            "default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found {}"
        )
        config_path = Path(wrapper_mod.__file__).parent / f"configs/config_{version}.yml"
        config_data = yaml.safe_load(config_path.read_text())
        config_data["text_model"] = text_model_path
        wrapper.config_as_str = yaml.safe_dump(config_data)
        wrapper.model_fp = model_fp
        wrapper.use_cuda = False
        if "clapcap" in version:
            wrapper.clapcap, wrapper.tokenizer, wrapper.args = wrapper.load_clapcap()
        else:
            wrapper.clap, wrapper.tokenizer, wrapper.args = wrapper.load_clap()
        return wrapper

    def embed_audio(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        assert audio.ndim == 1, f"Expected mono audio, got shape {audio.shape}."
        torch = require_dependency("torch", "ml")
        target_rate = int(self._model.args.sampling_rate)
        target_duration = int(self._model.args.duration)
        processed = self._resample_audio(audio, sample_rate, target_rate)
        target_length = target_rate * target_duration
        if processed.shape[0] < target_length:
            repeats = int(np.ceil(target_length / max(processed.shape[0], 1)))
            processed = np.tile(processed, repeats)[:target_length]
        else:
            processed = processed[:target_length]
        tensor = torch.tensor(processed, dtype=torch.float32).reshape(1, 1, -1)
        with torch.no_grad():
            embedding = self._model._get_audio_embeddings(tensor)
        return np.asarray(embedding[0].detach().cpu().numpy(), dtype=np.float32)

    @staticmethod
    def _resample_audio(audio: np.ndarray, sample_rate: int, target_rate: int) -> np.ndarray:
        if sample_rate == target_rate:
            return np.asarray(audio, dtype=np.float32)
        duration = (len(audio) - 1) / float(sample_rate)
        old_times = np.linspace(0.0, duration, num=len(audio), dtype=np.float32)
        new_length = max(1, int(round(len(audio) * target_rate / sample_rate)))
        new_times = np.linspace(0.0, duration, num=new_length, dtype=np.float32)
        return np.interp(new_times, old_times, audio).astype(np.float32)


@dataclass(slots=True)
class SimilarityRewardModel:
    metric: str = "cosine"

    def distance(self, lhs: np.ndarray, rhs: np.ndarray) -> float:
        lhs = np.asarray(lhs, dtype=np.float32)
        rhs = np.asarray(rhs, dtype=np.float32)
        assert lhs.shape == rhs.shape, f"Embedding shapes must match, got {lhs.shape} and {rhs.shape}."
        if self.metric == "cosine":
            lhs_scale = max(float(np.linalg.norm(lhs)), 1e-8)
            rhs_scale = max(float(np.linalg.norm(rhs)), 1e-8)
            lhs_norm = lhs / lhs_scale
            rhs_norm = rhs / rhs_scale
            return float(1.0 - np.dot(lhs_norm, rhs_norm))
        if self.metric == "l2":
            return float(np.linalg.norm(lhs - rhs))
        raise ValueError(f"Unsupported distance metric: {self.metric}")

    def reward(self, previous_distance: float, new_distance: float) -> float:
        return float(previous_distance - new_distance)


def build_embedder(config: RewardConfig) -> AudioEmbedder | None:
    if config.mode != "clap":
        return None
    if config.clap_checkpoint is not None:
        checkpoint = Path(config.clap_checkpoint)
        assert checkpoint.exists(), f"CLAP checkpoint path does not exist: {checkpoint}"
    return CLAPEmbedder(config)
