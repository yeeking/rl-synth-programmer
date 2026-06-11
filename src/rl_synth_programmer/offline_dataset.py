from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import resource
from time import perf_counter
from typing import Any, Callable

import numpy as np

from .config import CurriculumConfig, RewardConfig, SynthEnvConfig, SynthHostConfig
from .curriculum import TargetSpec
from .host import RenderKingSynthHost, SynthHost
from .logging_utils import make_progress_bar, stage_log
from .manifest import write_json
from .parallel_rollout import BatchedRolloutCoordinator, ParallelRenderPool, RenderRequest, embed_audio_batch
from .reward import build_embedder

LARGE_RENDER_WARNING_THRESHOLD = 10_000


def make_synth_host(config: SynthHostConfig):
    if str(getattr(config, "host_backend", "pedalboard")) == "renderking":
        return RenderKingSynthHost(config)
    return SynthHost(config)


@dataclass(slots=True)
class ActionDatasetConfig:
    """Configuration for offline all-actions dataset generation.

    plugin_path: VST3 instrument plugin used for rendering candidate synth states.
    manifest_path: target manifest produced by generate-target-set.
    output_dir: run folder where action_dataset/ artifacts are written.
    host_backend: synth render backend, either pedalboard or renderking.
    reward_mode: reward backend; v1 supports CLAP-backed distance rewards.
    rows_to_generate: target number of dataset rows to write. Each row is one current synth
        state plus dense labels for every available +/- parameter action. Generation keeps
        cycling across target/start preset pairs until this many rows are written, unless every
        pair becomes inactive because of skipped slow states.
    moves_per_cycle: number of greedy best-action moves to take from one target/start pair
        before moving to the next pair in the cycle. When the generator returns to that pair
        on a later cycle, it continues from the last state reached rather than resetting.
    num_workers: number of parallel render worker processes.
    clap_batch_size: number of rendered audio buffers embedded per CLAP batch.
    clap_device: CLAP embedding device, either auto, cpu, or cuda.
    action_step: normalized parameter delta used by each +/- action.
    seed: deterministic seed for curriculum/coordinator setup.
    render_timeout_seconds: timeout for one render batch/chunk; None disables it.
    skip_failed_actions: if true, timed-out chunks receive failed_action_reward and generation continues.
    failed_action_reward: label value assigned to actions skipped after timeout/slow-state limits.
    shard_size: write a recoverable shard after this many generated rows.
    render_chunk_size: actions per render batch; 0 means use num_workers.
    max_state_seconds: skip the rest of a state after this many seconds; None disables it.
    reload_workers_every_renders: reload plugin worker processes after this many successful action renders.
    preset_render_slowdown_threshold: assert if a row's mean action render time exceeds the
        prior running mean by this multiplier. Use 0 to disable.
    reload_workers_on_render_slowdown: reload workers immediately when one render chunk crosses
        preset_render_slowdown_threshold. If false, assert at the first slow chunk.
    action_step_calibration: probe start states and replace the global action_step with
        per-parameter steps normalized by embedding-distance sensitivity.
    """

    plugin_path: Path
    manifest_path: Path
    output_dir: Path
    host_backend: str = "pedalboard"
    reward_mode: str = "clap"
    rows_to_generate: int = 256
    moves_per_cycle: int = 4
    num_workers: int = 1
    clap_batch_size: int = 8
    clap_device: str = "auto"
    action_step: float = 0.05
    seed: int = 7
    render_timeout_seconds: float | None = 300.0
    skip_failed_actions: bool = True
    failed_action_reward: float = -1_000_000.0
    shard_size: int = 16
    render_chunk_size: int = 0
    max_state_seconds: float | None = None
    reload_workers_every_renders: int = 500
    preset_render_slowdown_threshold: float = 1.5
    reload_workers_on_render_slowdown: bool = True
    action_step_calibration: bool = True
    calibration_probe_states: int = 4
    calibration_probe_deltas: tuple[float, ...] = (0.01, 0.1, 0.25, 0.5)
    calibration_reference_delta: float = 0.25
    calibration_min_step: float = 0.01
    calibration_max_step: float = 0.5
    calibration_epsilon: float = 1e-8


@dataclass(slots=True)
class _PairRolloutState:
    params: dict[str, float]
    embedding: np.ndarray
    distance: float
    observation: np.ndarray
    moves_taken: int = 0
    active: bool = True


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
    _ = pairs
    return int(config.rows_to_generate)


def _build_config(config: ActionDatasetConfig) -> tuple[SynthEnvConfig, CurriculumConfig]:
    reward = RewardConfig(mode=config.reward_mode, clap_device=config.clap_device)
    env = SynthEnvConfig(
        host=SynthHostConfig(plugin_path=config.plugin_path, host_backend=config.host_backend),
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


def _action_deltas(coordinator: BatchedRolloutCoordinator) -> list[float]:
    return [float(coordinator.decode_action(action)[1]) for action in range(coordinator.action_size)]


def _calibrate_action_steps(
    coordinator: BatchedRolloutCoordinator,
    render_pool: ParallelRenderPool,
    embedder,
    targets: list[TargetSpec],
    pairs: list[tuple[int, int]],
    config: ActionDatasetConfig,
    *,
    progress: bool,
    reload_after_rendered: Callable[[int, Any], Any] | None = None,
) -> tuple[dict[str, float], dict[str, Any]]:
    parameter_ids = [spec.stable_id for spec in coordinator.parameter_specs]
    fixed_steps = {parameter_id: float(config.action_step) for parameter_id in parameter_ids}
    probe_deltas = [float(delta) for delta in config.calibration_probe_deltas]
    reference_delta = float(config.calibration_reference_delta)
    assert probe_deltas, "calibration_probe_deltas must contain at least one delta."
    assert reference_delta in probe_deltas, "calibration_reference_delta must be present in calibration_probe_deltas."
    assert config.calibration_probe_states >= 1, "calibration_probe_states must be >= 1."
    assert config.calibration_min_step > 0.0, "calibration_min_step must be > 0."
    assert config.calibration_max_step >= config.calibration_min_step, "calibration_max_step must be >= calibration_min_step."

    if not config.action_step_calibration:
        return fixed_steps, {
            "enabled": False,
            "probe_deltas": probe_deltas,
            "reference_delta": reference_delta,
            "probe_state_count": 0,
            "per_parameter_effects": {parameter_id: {} for parameter_id in parameter_ids},
            "target_effect": None,
            "clipped_min_count": 0,
            "clipped_max_count": 0,
            "failed_probe_count": 0,
        }

    selected_pairs = pairs[: min(int(config.calibration_probe_states), len(pairs))]
    stage_log(
        "Calibrating per-parameter action steps: "
        f"states={len(selected_pairs)} params={len(parameter_ids)} deltas={probe_deltas}."
    )
    effects: dict[str, dict[str, list[float]]] = {
        parameter_id: {str(delta): [] for delta in probe_deltas}
        for parameter_id in parameter_ids
    }
    actual_deltas: dict[str, dict[str, list[float]]] = {
        parameter_id: {str(delta): [] for delta in probe_deltas}
        for parameter_id in parameter_ids
    }
    failed_probe_count = 0
    rendered_probe_count = 0
    total_probes = len(selected_pairs) * len(parameter_ids) * len(probe_deltas) * 2
    progress_bar = make_progress_bar(total=total_probes, desc="calibration renders", enabled=progress)
    chunk_size = int(config.render_chunk_size) if int(config.render_chunk_size) > 0 else int(config.num_workers)
    chunk_size = max(1, chunk_size)
    for target_index, start_index in selected_pairs:
        target = targets[target_index]
        start = targets[start_index]
        baseline_params, baseline_embedding, _distance, _observation = _render_start_state(
            coordinator,
            render_pool,
            embedder,
            target,
            start,
            batch_size=config.clap_batch_size,
            timeout_seconds=config.render_timeout_seconds,
        )
        requests: list[RenderRequest] = []
        probe_records: list[tuple[str, float, float]] = []
        for parameter_id in parameter_ids:
            baseline_value = float(baseline_params[parameter_id])
            for delta in probe_deltas:
                for direction in (1.0, -1.0):
                    next_params = dict(baseline_params)
                    next_value = float(np.clip(baseline_value + direction * delta, 0.0, 1.0))
                    next_params[parameter_id] = next_value
                    probe_records.append((parameter_id, float(delta), float(next_value - baseline_value)))
                    requests.append(
                        RenderRequest(
                            slot_id=len(probe_records) - 1,
                            render_mode="parameter_state",
                            parameters=next_params,
                        )
                    )
        for start_offset in range(0, len(requests), chunk_size):
            request_chunk = requests[start_offset : start_offset + chunk_size]
            try:
                results = _render_batch(render_pool, request_chunk, timeout_seconds=config.render_timeout_seconds)
            except TimeoutError:
                failed_probe_count += len(request_chunk)
                progress_bar.update(len(request_chunk))
                if not config.skip_failed_actions:
                    raise
                continue
            embeddings = embed_audio_batch(
                embedder,
                [result.audio for result in results],
                [result.sample_rate for result in results],
                fallback_size=len(coordinator.parameter_specs),
                batch_size=max(1, config.clap_batch_size),
            )
            for result, embedding in zip(results, embeddings):
                parameter_id, requested_delta, actual_delta = probe_records[int(result.slot_id)]
                if abs(actual_delta) <= 1e-12:
                    continue
                effect = coordinator.distance_model.distance(
                    np.asarray(embedding, dtype=np.float32),
                    np.asarray(baseline_embedding, dtype=np.float32),
                )
                effects[parameter_id][str(requested_delta)].append(float(effect))
                actual_deltas[parameter_id][str(requested_delta)].append(float(abs(actual_delta)))
            rendered_probe_count += len(results)
            progress_bar.update(len(results))
            if reload_after_rendered is not None:
                render_pool = reload_after_rendered(len(results), render_pool)
    progress_bar.close()

    reference_key = str(reference_delta)
    reference_effects = {
        parameter_id: float(np.mean(values[reference_key]))
        for parameter_id, values in effects.items()
        if values[reference_key]
    }
    valid_effects = [value for value in reference_effects.values() if np.isfinite(value) and value > 0.0]
    target_effect = float(np.mean(valid_effects)) if valid_effects else float(reference_delta)
    clipped_min_count = 0
    clipped_max_count = 0
    steps: dict[str, float] = {}
    for parameter_id in parameter_ids:
        effect = reference_effects.get(parameter_id, 0.0)
        raw_step = reference_delta * target_effect / max(float(effect), float(config.calibration_epsilon))
        step = float(np.clip(raw_step, float(config.calibration_min_step), float(config.calibration_max_step)))
        clipped_min_count += int(step <= float(config.calibration_min_step) and raw_step < float(config.calibration_min_step))
        clipped_max_count += int(step >= float(config.calibration_max_step) and raw_step > float(config.calibration_max_step))
        steps[parameter_id] = step

    per_parameter_effects = {
        parameter_id: {
            "step": float(steps[parameter_id]),
            "reference_effect": float(reference_effects.get(parameter_id, 0.0)),
            "effects": {
                delta_key: float(np.mean(delta_values)) if delta_values else None
                for delta_key, delta_values in values.items()
            },
            "probe_counts": {delta_key: int(len(delta_values)) for delta_key, delta_values in values.items()},
            "mean_actual_deltas": {
                delta_key: float(np.mean(delta_values)) if delta_values else None
                for delta_key, delta_values in actual_deltas[parameter_id].items()
            },
        }
        for parameter_id, values in effects.items()
    }
    return steps, {
        "enabled": True,
        "probe_deltas": probe_deltas,
        "reference_delta": reference_delta,
        "probe_state_count": len(selected_pairs),
        "per_parameter_effects": per_parameter_effects,
        "target_effect": target_effect,
        "clipped_min_count": int(clipped_min_count),
        "clipped_max_count": int(clipped_max_count),
        "failed_probe_count": int(failed_probe_count),
        "rendered_probe_count": int(rendered_probe_count),
        "min_step": float(config.calibration_min_step),
        "max_step": float(config.calibration_max_step),
        "epsilon": float(config.calibration_epsilon),
    }


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
    reload_after_rendered: Callable[[int, Any], Any] | None = None,
    render_slowdown_baseline: float | None = None,
    render_slowdown_threshold: float = 0.0,
    render_slowdown_context: str = "",
    reload_on_render_slowdown: Callable[[Any], Any] | None = None,
) -> tuple[np.ndarray, list[dict[str, float]], list[np.ndarray], list[float], list[int], float, int, int]:
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
    render_seconds = 0.0
    rendered_action_count = 0
    slow_render_chunk_count = 0
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
        chunk_render_seconds = perf_counter() - render_started
        render_seconds += chunk_render_seconds
        rendered_action_count += len(results)
        reloaded_after_slowdown = False
        chunk_mean_seconds = float(chunk_render_seconds / max(1, len(results)))
        if (
            render_slowdown_baseline is not None
            and float(render_slowdown_baseline) > 0.0
            and float(render_slowdown_threshold) > 0.0
            and chunk_mean_seconds > float(render_slowdown_baseline) * float(render_slowdown_threshold)
        ):
            action_ids = [request.slot_id for request in request_chunk]
            message = (
                "Preset render slowdown detected: "
                f"{render_slowdown_context} "
                f"actions={action_ids[0]}-{action_ids[-1]} "
                f"chunk_mean_seconds_per_action={chunk_mean_seconds:.6f} "
                f"baseline_mean_seconds_per_action={float(render_slowdown_baseline):.6f} "
                f"threshold_multiplier={float(render_slowdown_threshold):.3f}"
            )
            slow_render_chunk_count += 1
            if reload_on_render_slowdown is None:
                raise AssertionError(message)
            stage_log(message + ". Reloading render workers.")
            render_pool = reload_on_render_slowdown(render_pool)
            reloaded_after_slowdown = True
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
        if reload_after_rendered is not None and not reloaded_after_slowdown:
            render_pool = reload_after_rendered(len(results), render_pool)
    if not np.any(rewards > float(failed_action_reward)):
        raise RuntimeError(f"All action renders failed for target={target.target_id} state={state_number}.")
    return (
        rewards,
        next_params_list,
        next_embeddings,
        distances,
        failed_actions,
        render_seconds,
        rendered_action_count,
        slow_render_chunk_count,
    )


def estimate_action_dataset(config: ActionDatasetConfig, *, progress: bool = True) -> dict[str, Any]:
    assert config.rows_to_generate >= 1, f"rows_to_generate must be >= 1, got {config.rows_to_generate}"
    assert config.moves_per_cycle >= 1, f"moves_per_cycle must be >= 1, got {config.moves_per_cycle}"
    env_config, curriculum_config = _build_config(config)
    stage_log("Loading host and manifest for dataset estimate.")
    probe_host = make_synth_host(env_config.host)
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
        (
            rewards,
            _params_list,
            _embeddings,
            _distances,
            _failed_actions,
            _render_seconds,
            _rendered_count,
            _slow_render_chunk_count,
        ) = _evaluate_all_actions(
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
    calibration_probe_states = min(int(config.calibration_probe_states), len(pairs)) if config.action_step_calibration else 0
    calibration_renders = int(
        calibration_probe_states * (1 + len(parameter_specs) * len(tuple(config.calibration_probe_deltas)) * 2)
    )
    total_renders = int(sample_states * action_count + len(targets) + sample_states + calibration_renders)
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
        "estimated_calibration_renders": calibration_renders,
        "estimated_npz_bytes": estimated_npz_bytes,
        "sample_best_reward": float(np.max(rewards)),
        "num_workers": int(config.num_workers),
        "clap_batch_size": int(config.clap_batch_size),
        "clap_device": str(config.clap_device),
        "host_backend": str(config.host_backend),
        "resolved_clap_device": str(getattr(embedder, "device", config.clap_device)),
        "render_timeout_seconds": config.render_timeout_seconds,
        "max_state_seconds": config.max_state_seconds,
        "reload_workers_every_renders": int(config.reload_workers_every_renders),
        "preset_render_slowdown_threshold": float(config.preset_render_slowdown_threshold),
        "reload_workers_on_render_slowdown": bool(config.reload_workers_on_render_slowdown),
        "action_step_calibration": bool(config.action_step_calibration),
        "calibration_probe_states": int(config.calibration_probe_states),
        "sampling_scheme": "cyclic_greedy",
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
    slow_chunk_counts = np.asarray(
        arrays.get("slow_render_chunk_counts", np.zeros((rewards.shape[0],), dtype=np.int32)),
        dtype=np.int32,
    )
    mean_render_seconds = np.asarray(
        arrays.get("mean_render_seconds_per_action", np.zeros((rewards.shape[0],), dtype=np.float32)),
        dtype=np.float32,
    )
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
        "mean_render_seconds_per_action": float(np.mean(mean_render_seconds)),
        "max_render_seconds_per_action": float(np.max(mean_render_seconds)),
        "slow_render_chunk_count": int(np.sum(slow_chunk_counts)),
        "rows_with_slow_render_chunks": int(np.sum(slow_chunk_counts > 0)),
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
        "render_seconds": [],
        "mean_render_seconds_per_action": [],
        "slow_render_chunk_counts": [],
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
        "render_seconds": np.asarray(rows["render_seconds"], dtype=np.float32),
        "mean_render_seconds_per_action": np.asarray(rows["mean_render_seconds_per_action"], dtype=np.float32),
        "slow_render_chunk_counts": np.asarray(rows["slow_render_chunk_counts"], dtype=np.int32),
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
    confirm_large_run: bool = False,
    estimate: dict[str, Any] | None = None,
) -> dict[str, Any]:
    assert config.rows_to_generate >= 1, f"rows_to_generate must be >= 1, got {config.rows_to_generate}"
    assert config.moves_per_cycle >= 1, f"moves_per_cycle must be >= 1, got {config.moves_per_cycle}"
    if estimate is None:
        estimate = estimate_action_dataset(config, progress=progress)
    if int(estimate["estimated_total_renders"]) > LARGE_RENDER_WARNING_THRESHOLD and not confirm_large_run:
        raise RuntimeError(
            "Dataset generation is estimated to require "
            f"{estimate['estimated_total_renders']} renders. Re-run with --confirm-large-run to confirm."
        )

    env_config, curriculum_config = _build_config(config)
    stage_log("Loading host, manifest, and embedder for action dataset generation.")
    probe_host = make_synth_host(env_config.host)
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
    successful_renders_since_reload = 0
    render_mean_history: list[float] = []
    started = perf_counter()
    render_pool = _open_render_pool(env_config.host, config.num_workers)

    def _reload_after_rendered(rendered_count: int, current_pool):
        nonlocal successful_renders_since_reload, render_pool
        if int(config.reload_workers_every_renders) <= 0:
            return current_pool
        successful_renders_since_reload += int(rendered_count)
        if successful_renders_since_reload < int(config.reload_workers_every_renders):
            return current_pool
        _close_render_pool(current_pool)
        successful_renders_since_reload = 0
        render_pool = _open_render_pool(env_config.host, config.num_workers)
        return render_pool

    def _reload_render_pool_now(current_pool):
        nonlocal successful_renders_since_reload, render_pool
        _close_render_pool(current_pool)
        successful_renders_since_reload = 0
        render_pool = _open_render_pool(env_config.host, config.num_workers)
        return render_pool

    try:
        _prime_target_embeddings(
            coordinator,
            render_pool,
            embedder,
            batch_size=config.clap_batch_size,
            progress=progress,
            timeout_seconds=config.render_timeout_seconds,
        )
        targets = coordinator.curriculum.all_targets()
        pairs = _target_start_pairs(targets)
        assert pairs, "Manifest must contain at least two preset-state targets for preset-pair dataset generation."
        target_ids = [target.target_id for target in targets]
        total_states = _sample_state_count(config, pairs)
        action_steps_by_parameter, calibration_metadata = _calibrate_action_steps(
            coordinator,
            render_pool,
            embedder,
            targets,
            pairs,
            config,
            progress=progress,
            reload_after_rendered=_reload_after_rendered,
        )
        coordinator.config.action_steps_by_parameter = dict(action_steps_by_parameter)
        total_action_renders = total_states * int(coordinator.action_size)
        progress_bar = make_progress_bar(total=total_action_renders, desc="action renders", enabled=progress)
        pair_states: dict[int, _PairRolloutState] = {}
        cycle_index = 0
        while total_written_rows + len(rows["observations"]) < total_states:
            rows_before_cycle = total_written_rows + len(rows["observations"])
            for pair_index, (target_index, start_index) in enumerate(pairs):
                if total_written_rows + len(rows["observations"]) >= total_states:
                    break
                target = targets[target_index]
                start = targets[start_index]
                pair_state = pair_states.get(pair_index)
                if pair_state is None:
                    current_params, current_embedding, current_distance, observation = _render_start_state(
                        coordinator,
                        render_pool,
                        embedder,
                        target,
                        start,
                        batch_size=config.clap_batch_size,
                        timeout_seconds=config.render_timeout_seconds,
                    )
                    pair_state = _PairRolloutState(
                        params=current_params,
                        embedding=current_embedding,
                        distance=current_distance,
                        observation=observation,
                    )
                    pair_states[pair_index] = pair_state
                if not pair_state.active:
                    continue
                for _cycle_move in range(config.moves_per_cycle):
                    if total_written_rows + len(rows["observations"]) >= total_states:
                        break
                    move_index = int(pair_state.moves_taken)
                    # pair_state is deliberately updated after each successful row. When this
                    # pair is revisited in a later cycle, generation resumes from that latest
                    # greedy best-action state rather than from the original preset start.
                    assert pair_state.active, f"Pair {pair_index} became inactive during cycle {cycle_index}."
                    current_row_number = total_written_rows + len(rows["observations"]) + 1
                    state_started = perf_counter()
                    deadline = None
                    if config.max_state_seconds is not None and float(config.max_state_seconds) > 0.0:
                        deadline = state_started + float(config.max_state_seconds)
                    (
                        rewards,
                        next_params_list,
                        next_embeddings,
                        next_distances,
                        failed_actions,
                        render_seconds,
                        rendered_action_count,
                        slow_render_chunk_count,
                    ) = _evaluate_all_actions(
                        coordinator,
                        render_pool,
                        embedder,
                        target,
                        pair_state.params,
                        pair_state.distance,
                        batch_size=config.clap_batch_size,
                        timeout_seconds=config.render_timeout_seconds,
                        skip_failed_actions=config.skip_failed_actions,
                        failed_action_reward=config.failed_action_reward,
                        render_chunk_size=config.render_chunk_size or config.num_workers,
                        state_number=current_row_number,
                        deadline=deadline,
                        render_progress=progress_bar,
                        reload_after_rendered=_reload_after_rendered,
                        render_slowdown_baseline=float(np.mean(render_mean_history)) if render_mean_history else None,
                        render_slowdown_threshold=float(config.preset_render_slowdown_threshold),
                        render_slowdown_context=(
                            f"target={target.target_id} start={start.target_id} "
                            f"target_index={target_index} start_index={start_index} "
                            f"cycle={cycle_index} move={move_index} state={current_row_number}/{total_states}"
                        ),
                        reload_on_render_slowdown=_reload_render_pool_now
                        if bool(config.reload_workers_on_render_slowdown)
                        else None,
                    )
                    best_action = int(np.argmax(rewards))
                    mean_render_seconds_per_action = float(render_seconds / max(1, rendered_action_count))
                    if (
                        float(config.preset_render_slowdown_threshold) > 0.0
                        and render_mean_history
                        and rendered_action_count > 0
                        and slow_render_chunk_count == 0
                    ):
                        baseline = float(np.mean(render_mean_history))
                        allowed = baseline * float(config.preset_render_slowdown_threshold)
                        assert mean_render_seconds_per_action <= allowed, (
                            "Preset render slowdown detected: "
                            f"target={target.target_id} start={start.target_id} "
                            f"target_index={target_index} start_index={start_index} "
                            f"cycle={cycle_index} move={move_index} state={current_row_number}/{total_states} "
                            f"mean_render_seconds_per_action={mean_render_seconds_per_action:.6f} "
                            f"baseline_mean_seconds_per_action={baseline:.6f} "
                            f"threshold_multiplier={float(config.preset_render_slowdown_threshold):.3f} "
                            f"rendered_actions={rendered_action_count} failed_actions={len(failed_actions)}"
                        )
                    state_skipped = int(
                        config.max_state_seconds is not None
                        and float(config.max_state_seconds) > 0.0
                        and perf_counter() >= state_started + float(config.max_state_seconds)
                        and len(failed_actions) > 0
                    )
                    rows["observations"].append(np.asarray(pair_state.observation, dtype=np.float32))
                    rows["action_rewards"].append(rewards)
                    rows["current_distances"].append(float(pair_state.distance))
                    rows["target_indices"].append(int(target_index))
                    rows["start_indices"].append(int(start_index))
                    rows["move_indices"].append(int(move_index))
                    rows["best_actions"].append(best_action)
                    rows["best_rewards"].append(float(rewards[best_action]))
                    rows["failed_action_counts"].append(len(failed_actions))
                    rows["state_skipped"].append(state_skipped)
                    rows["render_seconds"].append(float(render_seconds))
                    rows["mean_render_seconds_per_action"].append(mean_render_seconds_per_action)
                    rows["slow_render_chunk_counts"].append(int(slow_render_chunk_count))
                    if rendered_action_count > 0 and not state_skipped:
                        render_mean_history.append(mean_render_seconds_per_action)
                    if state_skipped:
                        stage_log(
                            f"Skipping slow state {current_row_number}/{total_states}: "
                            f"target={target.target_id} start={start.target_id} "
                            f"elapsed={perf_counter() - state_started:.2f}s failed_actions={len(failed_actions)}."
                        )
                        pair_state.active = False
                        break
                    pair_state.params = next_params_list[best_action]
                    pair_state.embedding = next_embeddings[best_action]
                    pair_state.distance = next_distances[best_action]
                    pair_state.observation = coordinator.flatten_observation(
                        pair_state.embedding,
                        target.embedding,
                        pair_state.params,
                    )
                    pair_state.moves_taken += 1
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
            rows_after_cycle = total_written_rows + len(rows["observations"])
            assert rows_after_cycle > rows_before_cycle, (
                "No dataset rows were generated during a complete source-target cycle. "
                "All pairs may have become inactive due to slow-state skips before rows_to_generate was reached."
            )
            cycle_index += 1
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
        "host_backend": str(config.host_backend),
        "clap_device": str(config.clap_device),
        "resolved_clap_device": str(getattr(embedder, "device", config.clap_device)),
        "action_step": float(config.action_step),
        "action_step_mode": "calibrated" if bool(config.action_step_calibration) else "fixed",
        "action_step_by_parameter": {spec.stable_id: float(coordinator.config.action_steps_by_parameter[spec.stable_id]) for spec in parameter_specs},
        "action_deltas": _action_deltas(coordinator),
        "action_step_calibration": calibration_metadata,
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
        "sampling_scheme": "cyclic_greedy",
        "memory": _memory_snapshot(),
        "estimate": estimate,
        "args": {
            "rows_to_generate": int(config.rows_to_generate),
            "moves_per_cycle": int(config.moves_per_cycle),
            "num_workers": int(config.num_workers),
            "clap_batch_size": int(config.clap_batch_size),
            "clap_device": str(config.clap_device),
            "host_backend": str(config.host_backend),
            "seed": int(config.seed),
            "render_timeout_seconds": config.render_timeout_seconds,
            "skip_failed_actions": bool(config.skip_failed_actions),
            "failed_action_reward": float(config.failed_action_reward),
            "shard_size": int(config.shard_size),
            "render_chunk_size": int(config.render_chunk_size or config.num_workers),
            "max_state_seconds": config.max_state_seconds,
            "reload_workers_every_renders": int(config.reload_workers_every_renders),
            "preset_render_slowdown_threshold": float(config.preset_render_slowdown_threshold),
            "reload_workers_on_render_slowdown": bool(config.reload_workers_on_render_slowdown),
            "action_step_calibration": bool(config.action_step_calibration),
            "calibration_probe_states": int(config.calibration_probe_states),
            "calibration_probe_deltas": [float(delta) for delta in config.calibration_probe_deltas],
            "calibration_reference_delta": float(config.calibration_reference_delta),
            "calibration_min_step": float(config.calibration_min_step),
            "calibration_max_step": float(config.calibration_max_step),
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
