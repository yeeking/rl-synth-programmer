from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .config import CurriculumConfig, RewardConfig, SynthEnvConfig
from .curriculum import TargetPool, TargetSpec
from .host import ParameterSpec, SynthHost
from .reward import AudioEmbedder, RandomRewardModel, SimilarityRewardModel, build_embedder

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    gym = None

    class _Discrete:
        def __init__(self, n: int):
            self.n = n

        def sample(self) -> int:
            return int(np.random.randint(0, self.n))

    class _Box:
        def __init__(self, low, high, shape, dtype=np.float32):
            self.low = low
            self.high = high
            self.shape = shape
            self.dtype = dtype

    class _Spaces:
        Discrete = _Discrete
        Box = _Box

    spaces = _Spaces()

    class _Env:
        pass

    class _Gym:
        Env = _Env

    gym = _Gym()


@dataclass(slots=True)
class StepResult:
    observation: np.ndarray
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any]


class SynthProgrammingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        config: SynthEnvConfig,
        curriculum_config: CurriculumConfig,
        host: SynthHost,
        embedder: AudioEmbedder | None = None,
    ):
        self.config = config
        self.host = host
        self.parameter_specs = host.filter_parameters(
            allowlist=config.parameter_allowlist,
            denylist=config.parameter_denylist,
        )
        self.curriculum = TargetPool(curriculum_config, self.parameter_specs)
        self.embedder = embedder if embedder is not None else build_embedder(config.reward)
        self.distance_model = SimilarityRewardModel(metric=config.reward.distance_metric)
        self.random_reward = RandomRewardModel(seed=config.seed)
        self._rng = np.random.default_rng(config.seed)
        self._current_target: TargetSpec | None = None
        self._current_params: dict[str, float] = host.get_normalized_defaults(self.parameter_specs)
        self._current_embedding: np.ndarray | None = None
        self._current_distance: float | None = None
        self._step_count = 0
        self.action_space = spaces.Discrete(2 * len(self.parameter_specs))
        obs_size = len(self.parameter_specs)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_size,),
            dtype=np.float32,
        )

    def _decode_action(self, action: int) -> tuple[str, float]:
        assert 0 <= action < self.action_space.n, f"Invalid action {action}."
        parameter_index = action // 2
        direction = 1.0 if action % 2 == 0 else -1.0
        parameter_id = self.parameter_specs[parameter_index].stable_id
        return parameter_id, direction * self.config.action_step

    def _render_current_audio(self) -> np.ndarray:
        return self.host.render_note(self._current_params)

    def _embed_audio(self, audio: np.ndarray) -> np.ndarray:
        if self.embedder is None:
            return np.asarray(audio[: len(self.parameter_specs)], dtype=np.float32)
        return self.embedder.embed_audio(audio, self.host.config.sample_rate)

    def _flatten_observation(self, current_embedding: np.ndarray, target_embedding: np.ndarray) -> np.ndarray:
        delta = target_embedding - current_embedding
        params = np.array([self._current_params[spec.stable_id] for spec in self.parameter_specs], dtype=np.float32)
        return np.concatenate([target_embedding, current_embedding, delta, params]).astype(np.float32)

    def _target_embedding(self, target: TargetSpec) -> np.ndarray:
        if target.embedding is None:
            previous_state = None
            if target.preset_state_path:
                try:
                    previous_state = self.host.capture_preset_state()
                    self.host.restore_preset_state(Path(target.preset_state_path).read_bytes())
                except Exception:
                    previous_state = None
            if target.preset_state_path:
                target.audio = self.host.render_note(None)
            else:
                self.host.set_parameters(target.parameters, self.parameter_specs)
                target.audio = self.host.render_note(target.parameters)
            target.embedding = self._embed_audio(target.audio)
            if previous_state is not None:
                self.host.restore_preset_state(previous_state)
        return target.embedding

    def _sample_initial_params(self) -> dict[str, float]:
        params = dict(self.host.get_normalized_defaults(self.parameter_specs))
        for index, spec in enumerate(self.parameter_specs[: min(6, len(self.parameter_specs))]):
            # Deterministically move away from defaults without relying on random full-range starts.
            direction = 1.0 if index % 2 == 0 else -1.0
            scale = (index + 1) * self.config.action_step
            params[spec.stable_id] = float(np.clip(params[spec.stable_id] + direction * scale, 0.0, 1.0))
        return params

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        _ = options
        self._current_target = self.curriculum.maybe_advance()
        self._current_params = self._sample_initial_params()
        target_embedding = self._target_embedding(self._current_target)
        self.host.set_parameters(self._current_params, self.parameter_specs)
        audio = self._render_current_audio()
        self._current_embedding = self._embed_audio(audio)
        self._current_distance = self.distance_model.distance(self._current_embedding, target_embedding)
        if self._current_distance <= 1e-8 and self.parameter_specs:
            for index, spec in enumerate(self.parameter_specs[: min(8, len(self.parameter_specs))]):
                self._current_params[spec.stable_id] = float(
                    np.clip(self._current_params[spec.stable_id] + (index + 1) * self.config.action_step, 0.0, 1.0)
                )
            self.host.set_parameters(self._current_params, self.parameter_specs)
            audio = self._render_current_audio()
            self._current_embedding = self._embed_audio(audio)
            self._current_distance = self.distance_model.distance(self._current_embedding, target_embedding)
        assert np.isfinite(self._current_distance), "Initial distance must be finite."
        self._step_count = 0
        observation = self._flatten_observation(self._current_embedding, target_embedding)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=observation.shape,
            dtype=np.float32,
        )
        return observation, self._info(action=None)

    def _info(self, action: int | None) -> dict[str, Any]:
        parameter_snapshot = {spec.stable_id: self._current_params[spec.stable_id] for spec in self.parameter_specs}
        action_decoded = None
        parameter_changed = None
        parameter_delta = None
        if action is not None:
            action_decoded = self._decode_action(action)
            parameter_changed, parameter_delta = action_decoded
        assert self._current_target is not None
        return {
            "current_distance": self._current_distance,
            "target_id": self._current_target.target_id,
            "target_label": self._current_target.label,
            "target_split": self._current_target.split,
            "parameter_snapshot": parameter_snapshot,
            "action_decoded": action_decoded,
            "parameter_changed": parameter_changed,
            "parameter_delta": parameter_delta,
            "step_count": self._step_count,
        }

    def step(self, action: int):
        assert self._current_target is not None, "Call reset before step."
        assert self._current_embedding is not None
        assert self._current_distance is not None
        parameter_id, delta = self._decode_action(action)
        self._current_params[parameter_id] = float(np.clip(self._current_params[parameter_id] + delta, 0.0, 1.0))
        self.host.set_parameters(self._current_params, self.parameter_specs)
        audio = self._render_current_audio()
        new_embedding = self._embed_audio(audio)
        target_embedding = self._target_embedding(self._current_target)
        new_distance = self.distance_model.distance(new_embedding, target_embedding)
        if self.config.reward.mode == "random":
            reward = self.random_reward.reward(self._current_distance, new_distance)
        else:
            reward = self.distance_model.reward(self._current_distance, new_distance)
        self._current_embedding = new_embedding
        self._current_distance = new_distance
        self._step_count += 1
        terminated = bool(new_distance <= self.config.success_threshold)
        truncated = bool(self._step_count >= self.config.max_episode_steps)
        observation = self._flatten_observation(new_embedding, target_embedding)
        return observation, reward, terminated, truncated, self._info(action=action)


def make_env(config: SynthEnvConfig, curriculum_config: CurriculumConfig) -> SynthProgrammingEnv:
    host = SynthHost(config.host)
    return SynthProgrammingEnv(config, curriculum_config, host)
