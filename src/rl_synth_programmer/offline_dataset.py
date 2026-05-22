from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import resource
from time import perf_counter
from typing import Any

import numpy as np

from .config import CurriculumConfig, RewardConfig, SynthEnvConfig, SynthHostConfig
from .curriculum import TargetSpec
from .host import SynthHost
from .logging_utils import make_progress_bar, stage_log
from .manifest import write_json
from .parallel_rollout import BatchedRolloutCoordinator, ParallelRenderPool, RenderRequest, embed_audio_batch
from .reward import build_embedder

LARGE_RENDER_WARNING_THRESHOLD = 10_000


@dataclass(slots=True)
class ActionDatasetConfig:
    """Configuration for offline all-actions dataset generation.

    plugin_path: VST3 instrument plugin used for rendering candidate synth states.
    manifest_path: target manifest produced by generate-target-set.
    output_dir: run folder where action_dataset/ artifacts are written.
    reward_mode: reward backend; v1 supports CLAP-backed distance rewards.
    max_states: maximum dataset rows to generate. one row is values for all parameter tweaks from a given position in terms of how much closer you get from that position to a target position when you make a given parameter tweak
    moves_per_start: greedy best-action steps to collect from each target/start preset pair.
    num_workers: number of parallel render worker processes.
    clap_batch_size: number of rendered audio buffers embedded per CLAP batch.
    action_step: normalized parameter delta used by each +/- action.
    seed: deterministic seed for curriculum/coordinator setup.
    render_timeout_seconds: timeout for one render batch/chunk; None disables it.
    skip_failed_actions: if true, timed-out chunks receive failed_action_reward and generation continues.
    failed_action_reward: label value assigned to actions skipped after timeout/slow-state limits.
    shard_size: write a recoverable shard after this many generated rows.
    render_chunk_size: actions per render batch; 0 means use num_workers.
    max_state_seconds: skip the rest of a state after this many seconds; None disables it.
    reload_workers_every_pair: reload plugin worker processes after each target/start pair.
    """

    plugin_path: Path
    manifest_path: Path
    output_dir: Path
    reward_mode: str = "clap"
    max_states: int = 256
    moves_per_start: int = 4
    num_workers: int = 1
    clap_batch_size: int = 8
    action_step: float = 0.05
    seed: int = 7
    render_timeout_seconds: float | None = 300.0
    skip_failed_actions: bool = True
    failed_action_reward: float = -1_000_000.0
    shard_size: int = 16
    render_chunk_size: int = 0
    max_state_seconds: float | None = None
    reload_workers_every_pair: bool = True


def _memory_snapshot() -> dict[str, float | str]:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    # macOS reports bytes, Linux reports KiB. Keep both explicit and conservative.
    raw = float(usage.ru_maxrss)
    if raw > 10_000_000:
        mib = raw / (1024.0 * 1024.0)
        unit = "bytes"
    else:
        mib = raw / 1024.0
        unit = "kib"
    return {"max_rss_raw": raw, "max_rss_unit": unit, "max_rss_mib_estimate": mib}


def _target_start_pairs(targets: list[TargetSpec]) -> list[tuple[int, int]]:
    preset_indices = [index for index, target in enumerate(targets) if target.preset_state_path]
    pairs: list[tuple[int, int]] = []
    for target_index in preset_indices:
        target = targets[target_index]
        same_split = [
            start_index
            for start_index in preset_indices
            if start_index != target_index and targets[start_index].split == target.split
        ]
        candidates = same_split or [start_index for start_index in preset_indices if start_index != target_index]
        for start_index in candidates:
            pairs.append((target_index, start_index))
    return pairs


def _sample_state_count(config: ActionDatasetConfig, pairs: list[tuple[int, int]]) -> int:
    return min(int(config.max_states), len(pairs) * int(config.moves_per_start))


def _build_config(config: ActionDatasetConfig) -> tuple[SynthEnvConfig, CurriculumConfig]:
    reward = RewardConfig(mode=config.reward_mode)
    env = SynthEnvConfig(
        host=SynthHostConfig(plugin_path=config.plugin_path),
        reward=reward,
        action_step=float(config.action_step),
        target_mode="preset_manifest",
        seed=int(config.seed),
        artifacts_dir=config.output_dir,
    )
    curriculum = CurriculumConfig(manifest_path=config.manifest_path, seed=int(config.seed))
    return env, curriculum


def _prime_target_embeddings(
    coordinator: BatchedRolloutCoordinator,
    render_pool: ParallelRenderPool,
    embedder,
    *,
    batch_size: int,
    progress: bool,
    timeout_seconds: float | None = None,
) -> None:
    requests = coordinator.build_target_render_requests()
    if not requests:
        return
    stage_log(f"Precomputing target embeddings for {len(requests)} target(s).")
    progress_bar = make_progress_bar(total=len(requests), desc="target embeddings", enabled=progress)
    for start in range(0, len(requests), max(1, batch_size)):
        batch = requests[start : start + max(1, batch_size)]
        results = _render_batch(render_pool, batch, timeout_seconds=timeout_seconds)
        embeddings = embed_audio_batch(
            embedder,
            [result.audio for result in results],
            [result.sample_rate for result in results],
            fallback_size=len(coordinator.parameter_specs),
            batch_size=max(1, batch_size),
        )
        coordinator.apply_target_embeddings(batch, results, embeddings)
        progress_bar.update(len(batch))
    progress_bar.close()


def _render_batch(
    render_pool: ParallelRenderPool,
    requests: list[RenderRequest],
    *,
    timeout_seconds: float | None,
):
    try:
        return render_pool.render_batch(requests, timeout_seconds=timeout_seconds)
    except TypeError:
        # Test fakes and older render pools may not accept the timeout keyword.
        return render_pool.render_batch(requests)


def _render_start_state(
    coordinator: BatchedRolloutCoordinator,
    render_pool: ParallelRenderPool,
    embedder,
    target: TargetSpec,
    start: TargetSpec,
    *,
    batch_size: int,
    timeout_seconds: float | None = None,
) -> tuple[dict[str, float], np.ndarray, float, np.ndarray]:
    assert start.preset_state_path is not None, f"Start target {start.target_id} has no preset state."
    assert target.embedding is not None, f"Target {target.target_id} has no embedding."
    params = dict(start.parameters)
    request = RenderRequest(slot_id=0, render_mode="target_state", preset_state=Path(start.preset_state_path).read_bytes())
    result = _render_batch(render_pool, [request], timeout_seconds=timeout_seconds)[0]
    embedding = embed_audio_batch(
        embedder,
        [result.audio],
        [result.sample_rate],
        fallback_size=len(coordinator.parameter_specs),
        batch_size=max(1, batch_size),
    )[0]
    distance = coordinator.distance_model.distance(np.asarray(embedding, dtype=np.float32), target.embedding)
    observation = coordinator.flatten_observation(np.asarray(embedding, dtype=np.float32), target.embedding, params)
    return params, np.asarray(embedding, dtype=np.float32), float(distance), observation


def _evaluate_all_actions(
    coordinator: BatchedRolloutCoordinator,
    render_pool: ParallelRenderPool,
    embedder,
    target: TargetSpec,
    current_params: dict[str, float],
    current_distance: float,
    *,
    batch_size: int,
    timeout_seconds: float | None,
    skip_failed_actions: bool,
    failed_action_reward: float,
    render_chunk_size: int,
    state_number: int | None = None,
    deadline: float | None = None,
    render_progress=None,
) -> tuple[np.ndarray, list[dict[str, float]], list[np.ndarray], list[float], list[int]]:
    assert target.embedding is not None, f"Target {target.target_id} has no embedding."
    requests: list[RenderRequest] = []
    next_params_list: list[dict[str, float]] = []
    for action in range(coordinator.action_size):
        parameter_id, delta = coordinator.decode_action(action)
        next_params = dict(current_params)
        next_params[parameter_id] = float(np.clip(next_params[parameter_id] + delta, 0.0, 1.0))
        next_params_list.append(next_params)
        requests.append(RenderRequest(slot_id=action, render_mode="parameter_state", parameters=next_params))
    chunk_size = int(render_chunk_size) if int(render_chunk_size) > 0 else len(requests)
    rewards = np.full((len(requests),), float(failed_action_reward), dtype=np.float32)
    distances = [float("nan")] * len(requests)
    next_embeddings = [np.zeros_like(target.embedding, dtype=np.float32) for _ in requests]
    failed_actions: list[int] = []
    for start in range(0, len(requests), max(1, chunk_size)):
        if deadline is not None and perf_counter() >= deadline:
            action_ids = [request.slot_id for request in requests[start:]]
            if not skip_failed_actions:
                raise TimeoutError(f"State {state_number} exceeded max_state_seconds before all actions rendered.")
            stage_log(
                f"State {state_number} exceeded max_state_seconds; "
                f"marking remaining actions {action_ids[0]}-{action_ids[-1]} as failed."
            )
            failed_actions.extend(action_ids)
            if render_progress is not None:
                render_progress.update(len(action_ids))
            break
        request_chunk = requests[start : start + max(1, chunk_size)]
        render_started = perf_counter()
        try:
            results = _render_batch(render_pool, request_chunk, timeout_seconds=timeout_seconds)
        except TimeoutError:
            action_ids = [request.slot_id for request in request_chunk]
            decoded = [coordinator.decode_action(action_id) for action_id in action_ids[:8]]
            message = (
                f"Render timeout for actions {action_ids[0]}-{action_ids[-1]} "
                f"target={target.target_id}; first decoded actions={decoded}."
            )
            if not skip_failed_actions:
                raise TimeoutError(message)
            stage_log(message + f" Marking {len(action_ids)} action(s) as failed.")
            failed_actions.extend(action_ids)
            if render_progress is not None:
                render_progress.update(len(action_ids))
            continue
        embeddings = embed_audio_batch(
            embedder,
            [result.audio for result in results],
            [result.sample_rate for result in results],
            fallback_size=len(coordinator.parameter_specs),
            batch_size=max(1, batch_size),
        )
        for result, embedding in zip(results, embeddings):
            action_id = int(result.slot_id)
            next_embedding = np.asarray(embedding, dtype=np.float32)
            next_distance = coordinator.distance_model.distance(next_embedding, target.embedding)
            rewards[action_id] = float(current_distance - next_distance)
            distances[action_id] = float(next_distance)
            next_embeddings[action_id] = next_embedding
        if render_progress is not None:
            render_progress.update(len(results))
    if not np.any(rewards > float(failed_action_reward)):
        raise RuntimeError(f"All action renders failed for target={target.target_id} state={state_number}.")
    return rewards, next_params_list, next_embeddings, distances, failed_actions


def estimate_action_dataset(config: ActionDatasetConfig, *, progress: bool = True) -> dict[str, Any]:
    assert config.max_states >= 1, f"max_states must be >= 1, got {config.max_states}"
    assert config.moves_per_start >= 1, f"moves_per_start must be >= 1, got {config.moves_per_start}"
    env_config, curriculum_config = _build_config(config)
    stage_log("Loading host and manifest for dataset estimate.")
    probe_host = SynthHost(env_config.host)
    parameter_specs = probe_host.filter_parameters()
    coordinator = BatchedRolloutCoordinator(env_config, curriculum_config, parameter_specs)
    targets = coordinator.curriculum.all_targets()
    pairs = _target_start_pairs(targets)
    assert pairs, "Manifest must contain at least two preset-state targets for preset-pair dataset generation."
    embedder = build_embedder(env_config.reward)
    with _open_render_pool(env_config.host, config.num_workers) as render_pool:
        _prime_target_embeddings(
            coordinator,
            render_pool,
            embedder,
            batch_size=config.clap_batch_size,
            progress=progress,
            timeout_seconds=config.render_timeout_seconds,
        )
        target_index, start_index = pairs[0]
        params, _embedding, current_distance, _observation = _render_start_state(
            coordinator,
            render_pool,
            embedder,
            targets[target_index],
            targets[start_index],
            batch_size=config.clap_batch_size,
            timeout_seconds=config.render_timeout_seconds,
        )
        started = perf_counter()
        rewards, _params_list, _embeddings, _distances, _failed_actions = _evaluate_all_actions(
            coordinator,
            render_pool,
            embedder,
            targets[target_index],
            params,
            current_distance,
            batch_size=config.clap_batch_size,
            timeout_seconds=config.render_timeout_seconds,
            skip_failed_actions=config.skip_failed_actions,
            failed_action_reward=config.failed_action_reward,
            render_chunk_size=config.render_chunk_size or config.num_workers,
            state_number=1,
        )
        seconds_per_state = perf_counter() - started
    sample_states = _sample_state_count(config, pairs)
    observation_size = 3 * int(targets[target_index].embedding.shape[0]) + len(parameter_specs)
    action_count = int(rewards.shape[0])
    total_renders = int(sample_states * action_count + len(targets) + sample_states)
    estimated_npz_bytes = int(sample_states * (observation_size + action_count + 6) * np.dtype(np.float32).itemsize)
    estimate = {
        "target_count": len(targets),
        "preset_pair_count": len(pairs),
        "sample_states": sample_states,
        "action_count": action_count,
        "observation_size": observation_size,
        "seconds_per_state": float(seconds_per_state),
        "estimated_seconds": float(seconds_per_state * sample_states),
        "estimated_total_renders": total_renders,
        "estimated_npz_bytes": estimated_npz_bytes,
        "sample_best_reward": float(np.max(rewards)),
        "num_workers": int(config.num_workers),
        "clap_batch_size": int(config.clap_batch_size),
        "render_timeout_seconds": config.render_timeout_seconds,
        "max_state_seconds": config.max_state_seconds,
        "memory": _memory_snapshot(),
    }
    stage_log(
        "Dataset estimate: "
        f"states={sample_states} actions={action_count} renders~{total_renders} "
        f"seconds~{estimate['estimated_seconds']:.1f} npz~{estimated_npz_bytes / 1_000_000:.2f}MB"
    )
    return estimate


def _summary(rows: dict[str, list[Any]], target_ids: list[str], action_count: int) -> dict[str, Any]:
    rewards = np.asarray(rows["action_rewards"], dtype=np.float32)
    best_rewards = np.asarray(rows["best_rewards"], dtype=np.float32)
    target_indices = np.asarray(rows["target_indices"], dtype=np.int32)
    per_target_counts = {
        target_id: int(np.sum(target_indices == index))
        for index, target_id in enumerate(target_ids)
        if int(np.sum(target_indices == index)) > 0
    }
    return {
        "row_count": int(rewards.shape[0]),
        "action_count": int(action_count),
        "reward_mean": float(np.mean(rewards)),
        "reward_std": float(np.std(rewards)),
        "reward_min": float(np.min(rewards)),
        "reward_max": float(np.max(rewards)),
        "best_reward_mean": float(np.mean(best_rewards)),
        "best_reward_max": float(np.max(best_rewards)),
        "positive_action_fraction": float(np.mean(rewards > 0.0)),
        "per_target_counts": per_target_counts,
    }


def _summary_from_arrays(arrays: dict[str, np.ndarray], target_ids: list[str], action_count: int) -> dict[str, Any]:
    rewards = np.asarray(arrays["action_rewards"], dtype=np.float32)
    best_rewards = np.asarray(arrays["best_rewards"], dtype=np.float32)
    target_indices = np.asarray(arrays["target_indices"], dtype=np.int32)
    failed_counts = np.asarray(arrays.get("failed_action_counts", np.zeros((rewards.shape[0],), dtype=np.int32)), dtype=np.int32)
    skipped_states = np.asarray(arrays.get("state_skipped", np.zeros((rewards.shape[0],), dtype=np.int32)), dtype=np.int32)
    per_target_counts = {
        target_id: int(np.sum(target_indices == index))
        for index, target_id in enumerate(target_ids)
        if int(np.sum(target_indices == index)) > 0
    }
    return {
        "row_count": int(rewards.shape[0]),
        "action_count": int(action_count),
        "reward_mean": float(np.mean(rewards)),
        "reward_std": float(np.std(rewards)),
        "reward_min": float(np.min(rewards)),
        "reward_max": float(np.max(rewards)),
        "best_reward_mean": float(np.mean(best_rewards)),
        "best_reward_max": float(np.max(best_rewards)),
        "positive_action_fraction": float(np.mean(rewards > 0.0)),
        "failed_action_count": int(np.sum(failed_counts)),
        "rows_with_failed_actions": int(np.sum(failed_counts > 0)),
        "skipped_state_count": int(np.sum(skipped_states > 0)),
        "per_target_counts": per_target_counts,
    }


def _empty_rows() -> dict[str, list[Any]]:
    return {
        "observations": [],
        "action_rewards": [],
        "current_distances": [],
        "target_indices": [],
        "start_indices": [],
        "move_indices": [],
        "best_actions": [],
        "best_rewards": [],
        "failed_action_counts": [],
        "state_skipped": [],
    }


def _arrays_from_rows(rows: dict[str, list[Any]]) -> dict[str, np.ndarray]:
    return {
        "observations": np.stack(rows["observations"]).astype(np.float32),
        "action_rewards": np.stack(rows["action_rewards"]).astype(np.float32),
        "current_distances": np.asarray(rows["current_distances"], dtype=np.float32),
        "target_indices": np.asarray(rows["target_indices"], dtype=np.int32),
        "start_indices": np.asarray(rows["start_indices"], dtype=np.int32),
        "move_indices": np.asarray(rows["move_indices"], dtype=np.int32),
        "best_actions": np.asarray(rows["best_actions"], dtype=np.int32),
        "best_rewards": np.asarray(rows["best_rewards"], dtype=np.float32),
        "failed_action_counts": np.asarray(rows["failed_action_counts"], dtype=np.int32),
        "state_skipped": np.asarray(rows["state_skipped"], dtype=np.int32),
    }


def _write_shard(shards_dir: Path, rows: dict[str, list[Any]], shard_index: int) -> Path | None:
    if not rows["observations"]:
        return None
    shards_dir.mkdir(parents=True, exist_ok=True)
    path = shards_dir / f"shard-{shard_index:05d}.npz"
    np.savez_compressed(path, **_arrays_from_rows(rows))
    return path


def _merge_shards(shard_paths: list[Path]) -> dict[str, np.ndarray]:
    assert shard_paths, "No shard paths to merge."
    loaded = [np.load(path) for path in shard_paths]
    keys = list(loaded[0].files)
    arrays = {key: np.concatenate([item[key] for item in loaded], axis=0) for key in keys}
    for item in loaded:
        item.close()
    return arrays


def _open_render_pool(host_config: SynthHostConfig, num_workers: int):
    return ParallelRenderPool(host_config, num_workers)


def _close_render_pool(render_pool) -> None:
    if hasattr(render_pool, "close"):
        render_pool.close()
    elif hasattr(render_pool, "__exit__"):
        render_pool.__exit__(None, None, None)


def generate_action_dataset(
    config: ActionDatasetConfig,
    *,
    progress: bool = True,
    yes: bool = False,
    estimate: dict[str, Any] | None = None,
) -> dict[str, Any]:
    assert config.max_states >= 1, f"max_states must be >= 1, got {config.max_states}"
    assert config.moves_per_start >= 1, f"moves_per_start must be >= 1, got {config.moves_per_start}"
    if estimate is None:
        estimate = estimate_action_dataset(config, progress=progress)
    if int(estimate["estimated_total_renders"]) > LARGE_RENDER_WARNING_THRESHOLD and not yes:
        raise RuntimeError(
            "Dataset generation is estimated to require "
            f"{estimate['estimated_total_renders']} renders. Re-run with --yes to confirm."
        )

    env_config, curriculum_config = _build_config(config)
    stage_log("Loading host, manifest, and embedder for action dataset generation.")
    probe_host = SynthHost(env_config.host)
    parameter_specs = probe_host.filter_parameters()
    coordinator = BatchedRolloutCoordinator(env_config, curriculum_config, parameter_specs)
    embedder = build_embedder(env_config.reward)
    output_dir = config.output_dir / "action_dataset"
    output_dir.mkdir(parents=True, exist_ok=True)
    shards_dir = output_dir / "shards"
    rows = _empty_rows()
    shard_paths: list[Path] = []
    shard_index = 0
    total_written_rows = 0
    started = perf_counter()
    render_pool = _open_render_pool(env_config.host, config.num_workers)
    try:
        _prime_target_embeddings(
            coordinator,
            render_pool,
            embedder,
            batch_size=config.clap_batch_size,
            progress=progress,
            timeout_seconds=config.render_timeout_seconds,
        )
        if config.reload_workers_every_pair:
            _close_render_pool(render_pool)
            render_pool = _open_render_pool(env_config.host, config.num_workers)
        targets = coordinator.curriculum.all_targets()
        pairs = _target_start_pairs(targets)
        assert pairs, "Manifest must contain at least two preset-state targets for preset-pair dataset generation."
        target_ids = [target.target_id for target in targets]
        total_states = _sample_state_count(config, pairs)
        total_action_renders = total_states * int(coordinator.action_size)
        progress_bar = make_progress_bar(total=total_action_renders, desc="action renders", enabled=progress)
        for target_index, start_index in pairs:
            if total_written_rows + len(rows["observations"]) >= total_states:
                break
            target = targets[target_index]
            start = targets[start_index]
            current_params, current_embedding, current_distance, observation = _render_start_state(
                coordinator,
                render_pool,
                embedder,
                target,
                start,
                batch_size=config.clap_batch_size,
                timeout_seconds=config.render_timeout_seconds,
            )
            for move_index in range(config.moves_per_start):
                current_row_number = total_written_rows + len(rows["observations"]) + 1
                if current_row_number > total_states:
                    break
                state_started = perf_counter()
                deadline = None
                if config.max_state_seconds is not None and float(config.max_state_seconds) > 0.0:
                    deadline = state_started + float(config.max_state_seconds)
                rewards, next_params_list, next_embeddings, next_distances, failed_actions = _evaluate_all_actions(
                    coordinator,
                    render_pool,
                    embedder,
                    target,
                    current_params,
                    current_distance,
                    batch_size=config.clap_batch_size,
                    timeout_seconds=config.render_timeout_seconds,
                    skip_failed_actions=config.skip_failed_actions,
                    failed_action_reward=config.failed_action_reward,
                    render_chunk_size=config.render_chunk_size or config.num_workers,
                    state_number=current_row_number,
                    deadline=deadline,
                    render_progress=progress_bar,
                )
                best_action = int(np.argmax(rewards))
                state_skipped = int(
                    config.max_state_seconds is not None
                    and float(config.max_state_seconds) > 0.0
                    and perf_counter() >= state_started + float(config.max_state_seconds)
                    and len(failed_actions) > 0
                )
                rows["observations"].append(np.asarray(observation, dtype=np.float32))
                rows["action_rewards"].append(rewards)
                rows["current_distances"].append(float(current_distance))
                rows["target_indices"].append(int(target_index))
                rows["start_indices"].append(int(start_index))
                rows["move_indices"].append(int(move_index))
                rows["best_actions"].append(best_action)
                rows["best_rewards"].append(float(rewards[best_action]))
                rows["failed_action_counts"].append(len(failed_actions))
                rows["state_skipped"].append(state_skipped)
                if state_skipped:
                    stage_log(
                        f"Skipping slow state {current_row_number}/{total_states}: "
                        f"target={target.target_id} start={start.target_id} "
                        f"elapsed={perf_counter() - state_started:.2f}s failed_actions={len(failed_actions)}."
                    )
                else:
                    current_params = next_params_list[best_action]
                    current_embedding = next_embeddings[best_action]
                    current_distance = next_distances[best_action]
                    observation = coordinator.flatten_observation(current_embedding, target.embedding, current_params)
                if failed_actions:
                    stage_log(
                        f"State {current_row_number}/{total_states} completed with "
                        f"{len(failed_actions)} failed action(s); best_action={best_action}."
                    )
                progress_bar.set_postfix(
                    {
                        "state": f"{current_row_number}/{total_states}",
                        "target": target.target_id,
                        "best": f"{float(rewards[best_action]):.4f}",
                    }
                )
                if len(rows["observations"]) >= max(1, config.shard_size):
                    path = _write_shard(shards_dir, rows, shard_index)
                    if path is not None:
                        shard_paths.append(path)
                        total_written_rows += len(rows["observations"])
                        shard_index += 1
                        rows = _empty_rows()
                if state_skipped:
                    break
            if config.reload_workers_every_pair and total_written_rows + len(rows["observations"]) < total_states:
                _close_render_pool(render_pool)
                render_pool = _open_render_pool(env_config.host, config.num_workers)
        progress_bar.close()
    finally:
        _close_render_pool(render_pool)

    if rows["observations"]:
        path = _write_shard(shards_dir, rows, shard_index)
        if path is not None:
            shard_paths.append(path)
            total_written_rows += len(rows["observations"])
    assert shard_paths, "No dataset rows were generated."
    dataset_path = output_dir / "dataset.npz"
    arrays = _merge_shards(shard_paths)
    np.savez_compressed(dataset_path, **arrays)
    targets = coordinator.curriculum.all_targets()
    target_ids = [target.target_id for target in targets]
    embedding_size = int(targets[0].embedding.shape[0])
    metadata = {
        "plugin_path": str(config.plugin_path),
        "manifest_path": str(config.manifest_path),
        "reward_mode": config.reward_mode,
        "action_step": float(config.action_step),
        "parameter_ids": [spec.stable_id for spec in parameter_specs],
        "target_ids": target_ids,
        "observation_layout": {
            "target_embedding": [0, embedding_size],
            "current_embedding": [embedding_size, 2 * embedding_size],
            "delta_embedding": [2 * embedding_size, 3 * embedding_size],
            "params": [3 * embedding_size, 3 * embedding_size + len(parameter_specs)],
        },
        "embedding_size": embedding_size,
        "param_count": len(parameter_specs),
        "action_count": int(coordinator.action_size),
        "shapes": {key: list(value.shape) for key, value in arrays.items()},
        "dtypes": {key: str(value.dtype) for key, value in arrays.items()},
        "generation_seconds": float(perf_counter() - started),
        "memory": _memory_snapshot(),
        "estimate": estimate,
        "args": {
            "max_states": int(config.max_states),
            "moves_per_start": int(config.moves_per_start),
            "num_workers": int(config.num_workers),
            "clap_batch_size": int(config.clap_batch_size),
            "seed": int(config.seed),
            "render_timeout_seconds": config.render_timeout_seconds,
            "skip_failed_actions": bool(config.skip_failed_actions),
            "failed_action_reward": float(config.failed_action_reward),
            "shard_size": int(config.shard_size),
            "render_chunk_size": int(config.render_chunk_size or config.num_workers),
            "max_state_seconds": config.max_state_seconds,
            "reload_workers_every_pair": bool(config.reload_workers_every_pair),
        },
        "shards": [str(path) for path in shard_paths],
    }
    summary = _summary_from_arrays(arrays, target_ids, coordinator.action_size)
    write_json(output_dir / "metadata.json", metadata)
    write_json(output_dir / "summary.json", summary)
    stage_log(f"Action dataset written to {dataset_path}. Rows={summary['row_count']}.")
    return {
        "dataset_path": str(dataset_path),
        "metadata_path": str(output_dir / "metadata.json"),
        "summary_path": str(output_dir / "summary.json"),
        "summary": summary,
    }
