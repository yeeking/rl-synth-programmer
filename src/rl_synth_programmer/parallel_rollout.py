from __future__ import annotations

from dataclasses import dataclass
import multiprocessing as mp
import os
from pathlib import Path
from time import perf_counter
from typing import Iterable

import numpy as np

from .config import CurriculumConfig, RewardConfig, SynthEnvConfig, SynthHostConfig
from .curriculum import TargetPool, TargetSpec
from .host import ParameterSpec, SynthHost
from .reward import AudioEmbedder, RandomRewardModel, SimilarityRewardModel

_WORKER_HOST: SynthHost | None = None


@dataclass(slots=True)
class RenderRequest:
    slot_id: int
    render_mode: str
    parameters: dict[str, float] | None = None
    preset_state: bytes | None = None


@dataclass(slots=True)
class RenderResult:
    slot_id: int
    worker_id: int
    audio: np.ndarray
    sample_rate: int
    render_seconds: float


@dataclass(slots=True)
class EpisodeSlotState:
    slot_id: int
    episode_id: int
    target: TargetSpec
    current_params: dict[str, float]
    current_embedding: np.ndarray
    current_distance: float
    initial_distance: float
    observation: np.ndarray
    step_count: int = 0
    total_reward: float = 0.0
    last_worker_id: int | None = None


@dataclass(slots=True)
class BatchedStepResult:
    slot_id: int
    episode_id: int
    target_id: str
    target_label: str | None
    reward: float
    previous_distance: float
    current_distance: float
    distance_delta: float
    parameter_changed: str
    parameter_delta: float
    step_in_episode: int
    initial_distance: float
    terminated: bool
    truncated: bool
    worker_id: int


def _render_worker_init(host_config: SynthHostConfig) -> None:
    global _WORKER_HOST
    _WORKER_HOST = SynthHost(host_config)
    _WORKER_HOST.load()


def _render_worker(request: RenderRequest) -> RenderResult:
    global _WORKER_HOST
    assert _WORKER_HOST is not None, "Render worker host is not initialized."
    started = perf_counter()
    if request.render_mode == "target_state":
        assert request.preset_state is not None, "target_state render requires preset_state."
        _WORKER_HOST.restore_preset_state(request.preset_state)
        audio = _WORKER_HOST.render_note(None)
    elif request.render_mode == "parameter_state":
        assert request.parameters is not None, "parameter_state render requires parameters."
        _WORKER_HOST.set_parameters(request.parameters)
        audio = _WORKER_HOST.render_note(None)
    else:
        raise ValueError(f"Unsupported render_mode: {request.render_mode}")
    return RenderResult(
        slot_id=request.slot_id,
        worker_id=os.getpid(),
        audio=np.asarray(audio, dtype=np.float32),
        sample_rate=int(_WORKER_HOST.config.sample_rate),
        render_seconds=float(perf_counter() - started),
    )


class ParallelRenderPool:
    def __init__(self, host_config: SynthHostConfig, num_workers: int):
        assert num_workers >= 1, f"num_workers must be >= 1, got {num_workers}"
        self.host_config = host_config
        self.num_workers = int(num_workers)
        self._ctx = mp.get_context("spawn")
        self._pool = self._ctx.Pool(
            processes=self.num_workers,
            initializer=_render_worker_init,
            initargs=(self.host_config,),
        )

    def render_batch(self, requests: Iterable[RenderRequest]) -> list[RenderResult]:
        request_list = list(requests)
        if not request_list:
            return []
        return list(self._pool.map(_render_worker, request_list, chunksize=1))

    def close(self) -> None:
        self._pool.close()
        self._pool.join()

    def terminate(self) -> None:
        self._pool.terminate()
        self._pool.join()

    def __enter__(self) -> ParallelRenderPool:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is None:
            self.close()
        else:
            self.terminate()


def embed_audio_batch(
    embedder: AudioEmbedder | None,
    audios: list[np.ndarray],
    sample_rates: list[int],
    *,
    fallback_size: int,
    batch_size: int,
) -> np.ndarray:
    assert len(audios) == len(sample_rates), "audios and sample_rates must have the same length."
    if not audios:
        return np.empty((0, fallback_size), dtype=np.float32)
    if embedder is None:
        return np.stack([np.asarray(audio[:fallback_size], dtype=np.float32) for audio in audios]).astype(np.float32)
    batches: list[np.ndarray] = []
    for start in range(0, len(audios), max(1, batch_size)):
        stop = start + max(1, batch_size)
        audio_chunk = audios[start:stop]
        rate_chunk = sample_rates[start:stop]
        if hasattr(embedder, "embed_audio_batch"):
            chunk = embedder.embed_audio_batch(audio_chunk, rate_chunk)
        else:
            chunk = np.stack([embedder.embed_audio(audio, sr) for audio, sr in zip(audio_chunk, rate_chunk)]).astype(np.float32)
        batches.append(np.asarray(chunk, dtype=np.float32))
    return np.concatenate(batches, axis=0)


class BatchedRolloutCoordinator:
    def __init__(self, config: SynthEnvConfig, curriculum_config: CurriculumConfig, parameter_specs: list[ParameterSpec]):
        self.config = config
        self.parameter_specs = list(parameter_specs)
        self.curriculum = TargetPool(curriculum_config, self.parameter_specs)
        self.distance_model = SimilarityRewardModel(metric=config.reward.distance_metric)
        self.random_reward = RandomRewardModel(seed=config.seed)
        self._rng = np.random.default_rng(config.seed)
        self._next_episode_id = 1

    @property
    def action_size(self) -> int:
        return 2 * len(self.parameter_specs)

    def decode_action(self, action: int) -> tuple[str, float]:
        assert 0 <= action < self.action_size, f"Invalid action {action}."
        parameter_index = action // 2
        direction = 1.0 if action % 2 == 0 else -1.0
        parameter_id = self.parameter_specs[parameter_index].stable_id
        return parameter_id, direction * self.config.action_step

    def preset_start_candidates(self, target: TargetSpec) -> list[TargetSpec]:
        same_split = [
            candidate
            for candidate in self.curriculum.all_targets()
            if candidate.target_id != target.target_id
            and candidate.preset_state_path
            and candidate.split == target.split
        ]
        if same_split:
            return same_split
        return [
            candidate
            for candidate in self.curriculum.all_targets()
            if candidate.target_id != target.target_id and candidate.preset_state_path
        ]

    def sample_initial_state(self, target: TargetSpec, default_params: dict[str, float]) -> tuple[dict[str, float], bytes | None]:
        _ = default_params
        preset_candidates = self.preset_start_candidates(target)
        if preset_candidates:
            start_target = preset_candidates[int(self._rng.integers(0, len(preset_candidates)))]
            return dict(start_target.parameters), Path(start_target.preset_state_path).read_bytes()
        return self.sample_initial_params(default_params), None

    def sample_initial_params(self, default_params: dict[str, float]) -> dict[str, float]:
        _ = default_params
        return {
            spec.stable_id: float(self._rng.uniform(0.0, 1.0))
            for spec in self.parameter_specs
        }

    def flatten_observation(
        self,
        current_embedding: np.ndarray,
        target_embedding: np.ndarray,
        current_params: dict[str, float],
    ) -> np.ndarray:
        delta = target_embedding - current_embedding
        params = np.array([current_params[spec.stable_id] for spec in self.parameter_specs], dtype=np.float32)
        return np.concatenate([target_embedding, current_embedding, delta, params]).astype(np.float32)

    def build_target_render_requests(self) -> list[RenderRequest]:
        requests: list[RenderRequest] = []
        for index, target in enumerate(self.curriculum.all_targets()):
            if target.embedding is not None:
                continue
            if target.preset_state_path:
                requests.append(
                    RenderRequest(
                        slot_id=index,
                        render_mode="target_state",
                        preset_state=Path(target.preset_state_path).read_bytes(),
                    )
                )
            else:
                requests.append(
                    RenderRequest(
                        slot_id=index,
                        render_mode="parameter_state",
                        parameters=dict(target.parameters),
                    )
                )
        return requests

    def apply_target_embeddings(self, requests: list[RenderRequest], render_results: list[RenderResult], embeddings: np.ndarray) -> None:
        for request, result, embedding in zip(requests, render_results, embeddings):
            target = self.curriculum.all_targets()[request.slot_id]
            target.audio = result.audio
            target.embedding = np.asarray(embedding, dtype=np.float32)

    def _fresh_episode_id(self) -> int:
        episode_id = self._next_episode_id
        self._next_episode_id += 1
        return episode_id

    def reset_slot_requests(self, slot_ids: list[int], default_params: dict[str, float]) -> tuple[list[EpisodeSlotState], list[RenderRequest]]:
        states: list[EpisodeSlotState] = []
        requests: list[RenderRequest] = []
        for slot_id in slot_ids:
            target = self.curriculum.maybe_advance()
            assert target.embedding is not None, f"Target {target.target_id} has no cached embedding."
            params, initial_preset_state = self.sample_initial_state(target, default_params)
            states.append(
                EpisodeSlotState(
                    slot_id=slot_id,
                    episode_id=self._fresh_episode_id(),
                    target=target,
                    current_params=params,
                    current_embedding=np.empty((0,), dtype=np.float32),
                    current_distance=float("nan"),
                    initial_distance=float("nan"),
                    observation=np.empty((0,), dtype=np.float32),
                )
            )
            if initial_preset_state is not None:
                requests.append(
                    RenderRequest(
                        slot_id=slot_id,
                        render_mode="target_state",
                        preset_state=initial_preset_state,
                    )
                )
            else:
                requests.append(RenderRequest(slot_id=slot_id, render_mode="parameter_state", parameters=dict(params)))
        return states, requests

    def finalize_reset_states(
        self,
        states: list[EpisodeSlotState],
        render_results: list[RenderResult],
        embeddings: np.ndarray,
    ) -> list[EpisodeSlotState]:
        result_by_slot = {result.slot_id: result for result in render_results}
        embedding_by_slot = {state.slot_id: np.asarray(embedding, dtype=np.float32) for state, embedding in zip(states, embeddings)}
        finalized: list[EpisodeSlotState] = []
        for state in states:
            result = result_by_slot[state.slot_id]
            embedding = embedding_by_slot[state.slot_id]
            distance = self.distance_model.distance(embedding, state.target.embedding)
            if distance <= 1e-8 and self.parameter_specs:
                for index, spec in enumerate(self.parameter_specs[: min(8, len(self.parameter_specs))]):
                    state.current_params[spec.stable_id] = float(
                        np.clip(state.current_params[spec.stable_id] + (index + 1) * self.config.action_step, 0.0, 1.0)
                    )
                raise RuntimeError("Zero-distance reset requires rerender and should be handled upstream.")
            state.current_embedding = embedding
            state.current_distance = float(distance)
            state.initial_distance = float(distance)
            state.observation = self.flatten_observation(embedding, state.target.embedding, state.current_params)
            state.last_worker_id = result.worker_id
            finalized.append(state)
        return finalized

    def rerender_zero_distance_requests(self, states: list[EpisodeSlotState]) -> list[RenderRequest]:
        requests: list[RenderRequest] = []
        for state in states:
            requests.append(
                RenderRequest(
                    slot_id=state.slot_id,
                    render_mode="parameter_state",
                    parameters=dict(state.current_params),
                )
            )
        return requests

    def build_step_requests(
        self,
        slot_states: list[EpisodeSlotState],
        actions: list[int],
    ) -> tuple[list[RenderRequest], list[dict[str, float | int | str | bool]]]:
        requests: list[RenderRequest] = []
        pending: list[dict[str, float | int | str | bool]] = []
        for slot_state, action in zip(slot_states, actions):
            parameter_id, delta = self.decode_action(action)
            next_params = dict(slot_state.current_params)
            next_params[parameter_id] = float(np.clip(next_params[parameter_id] + delta, 0.0, 1.0))
            requests.append(RenderRequest(slot_id=slot_state.slot_id, render_mode="parameter_state", parameters=next_params))
            pending.append(
                {
                    "slot_id": slot_state.slot_id,
                    "episode_id": slot_state.episode_id,
                    "action": int(action),
                    "parameter_changed": parameter_id,
                    "parameter_delta": float(delta),
                }
            )
        return requests, pending

    def apply_step_results(
        self,
        slot_state: EpisodeSlotState,
        next_params: dict[str, float],
        next_embedding: np.ndarray,
        action: int,
        worker_id: int,
    ) -> tuple[EpisodeSlotState, BatchedStepResult]:
        _ = action
        parameter_changed, parameter_delta = self.decode_action(action)
        previous_distance = float(slot_state.current_distance)
        new_distance = self.distance_model.distance(next_embedding, slot_state.target.embedding)
        if self.config.reward.mode == "random":
            reward = self.random_reward.reward(previous_distance, new_distance)
        else:
            reward = self.distance_model.reward(previous_distance, new_distance)
        slot_state.current_params = next_params
        slot_state.current_embedding = np.asarray(next_embedding, dtype=np.float32)
        slot_state.current_distance = float(new_distance)
        slot_state.step_count += 1
        slot_state.total_reward += float(reward)
        slot_state.observation = self.flatten_observation(slot_state.current_embedding, slot_state.target.embedding, slot_state.current_params)
        slot_state.last_worker_id = worker_id
        terminated = bool(new_distance <= self.config.success_threshold)
        truncated = bool(slot_state.step_count >= self.config.max_episode_steps)
        result = BatchedStepResult(
            slot_id=slot_state.slot_id,
            episode_id=slot_state.episode_id,
            target_id=slot_state.target.target_id,
            target_label=slot_state.target.label,
            reward=float(reward),
            previous_distance=previous_distance,
            current_distance=float(new_distance),
            distance_delta=float(previous_distance - new_distance),
            parameter_changed=parameter_changed,
            parameter_delta=float(parameter_delta),
            step_in_episode=int(slot_state.step_count),
            initial_distance=float(slot_state.initial_distance),
            terminated=terminated,
            truncated=truncated,
            worker_id=int(worker_id),
        )
        return slot_state, result
