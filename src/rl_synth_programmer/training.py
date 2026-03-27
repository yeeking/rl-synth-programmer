from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np

from .agent import DQNAgent, RandomAgent, ReplayTransition
from .config import ExperimentConfig
from .env import SynthProgrammingEnv, make_env
from .host import SynthHost
from .logging_utils import create_summary_writer, log_run_metadata, make_progress_bar, stage_log
from .optional_deps import require_dependency
from .parallel_rollout import (
    BatchedRolloutCoordinator,
    EpisodeSlotState,
    ParallelRenderPool,
    RenderRequest,
    embed_audio_batch,
)
from .reward import build_embedder


@dataclass(slots=True)
class EpisodeMetrics:
    total_reward: float
    steps: int
    initial_distance: float
    final_distance: float
    target_id: str
    target_label: str | None = None
    initial_parameter_snapshot: dict[str, float] | None = None


def _run_metadata(config: ExperimentConfig, total_steps: int | None = None) -> dict[str, str | int | float]:
    metadata: dict[str, str | int | float] = {
        "plugin_path": str(config.env.host.plugin_path),
        "reward_mode": str(config.env.reward.mode),
        "target_mode": str(config.env.target_mode),
        "hidden_sizes": "x".join(str(size) for size in config.dqn.hidden_sizes),
        "learning_rate": float(config.dqn.learning_rate),
        "gamma": float(config.dqn.gamma),
        "batch_size": int(config.dqn.batch_size),
        "warmup_steps": int(config.dqn.warmup_steps),
        "epsilon_start": float(config.dqn.epsilon_start),
        "epsilon_end": float(config.dqn.epsilon_end),
        "epsilon_decay_steps": int(config.dqn.epsilon_decay_steps),
        "action_step": float(config.env.action_step),
        "max_episode_steps": int(config.env.max_episode_steps),
        "run_name": str(config.run_name),
        "num_render_workers": int(config.num_render_workers),
        "num_parallel_envs": int(config.num_parallel_envs),
        "updates_per_tick": int(config.updates_per_tick),
        "clap_batch_size": int(config.clap_batch_size),
        "clap_batch_timeout_ms": int(config.clap_batch_timeout_ms),
    }
    if config.curriculum.manifest_path is not None:
        metadata["manifest_path"] = str(config.curriculum.manifest_path)
    if total_steps is not None:
        metadata["total_steps"] = int(total_steps)
    return metadata


def _episode_summary_line(prefix: str, episode_index: int, metric: EpisodeMetrics) -> str:
    delta = float(metric.initial_distance - metric.final_distance)
    label = f" ({metric.target_label})" if metric.target_label else ""
    return (
        f"{prefix} episode {episode_index}: target={metric.target_id}{label} "
        f"steps={metric.steps} reward={metric.total_reward:.4f} "
        f"distance={metric.initial_distance:.4f}->{metric.final_distance:.4f} delta={delta:.4f}"
    )


def _maybe_text_log(writer, tag: str, text: str, step: int, *, enabled: bool) -> None:
    if enabled:
        writer.add_text(tag, text, step)


def run_random_policy(
    env: SynthProgrammingEnv,
    episodes: int,
    seed: int = 7,
    *,
    progress: bool = True,
    episode_log_interval: int = 1,
) -> list[EpisodeMetrics]:
    assert episodes >= 1, f"episodes must be >= 1, got {episodes}"
    assert episode_log_interval >= 1, f"episode_log_interval must be >= 1, got {episode_log_interval}"
    agent = RandomAgent(env.action_space.n, seed=seed)
    metrics: list[EpisodeMetrics] = []
    stage_log(f"Running random policy for {episodes} episode(s).")
    progress_bar = make_progress_bar(total=episodes, desc="random episodes", enabled=progress)
    for episode_index in range(1, episodes + 1):
        observation, info = env.reset()
        total_reward = 0.0
        initial_distance = float(info["current_distance"])
        initial_parameter_snapshot = dict(info.get("parameter_snapshot", {}))
        done = False
        truncated = False
        steps = 0
        while not done and not truncated:
            action = agent.act(observation)
            observation, reward, done, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
        metrics.append(
            EpisodeMetrics(
                total_reward=total_reward,
                steps=steps,
                initial_distance=initial_distance,
                final_distance=float(info["current_distance"]),
                target_id=str(info["target_id"]),
                target_label=info.get("target_label"),
                initial_parameter_snapshot=initial_parameter_snapshot,
            )
        )
        if episode_index == 1 or episode_index % episode_log_interval == 0 or episode_index == episodes:
            stage_log(_episode_summary_line("random", episode_index, metrics[-1]))
        progress_bar.update(1)
    progress_bar.close()
    return metrics


def train_dqn(
    config: ExperimentConfig,
    total_steps: int,
    *,
    progress: bool = True,
    log_interval: int = 25,
    episode_log_interval: int = 10,
    tensorboard: bool = True,
    tensorboard_dir: Path | None = None,
) -> tuple[DQNAgent, list[dict[str, float | str]]]:
    assert total_steps >= 1, f"total_steps must be >= 1, got {total_steps}"
    assert log_interval >= 1, f"log_interval must be >= 1, got {log_interval}"
    assert episode_log_interval >= 1, f"episode_log_interval must be >= 1, got {episode_log_interval}"
    stage_log("Creating training environment.")
    env = make_env(config.env, config.curriculum)
    stage_log("Resetting environment.")
    observation, info = env.reset(seed=config.env.seed)
    stage_log("Building DQN agent.")
    agent = DQNAgent(observation_size=observation.shape[0], action_size=env.action_space.n, config=config.dqn)
    writer = create_summary_writer(tensorboard, tensorboard_dir or config.output_dir / "tensorboard")
    log_run_metadata(writer, _run_metadata(config, total_steps=total_steps))
    logs: list[dict[str, float | str]] = []
    current_observation = observation
    current_initial_distance = float(info["current_distance"])
    episode_index = 1
    episode_reward = 0.0
    episode_steps = 0
    episode_target_id = str(info["target_id"])
    episode_target_label = info.get("target_label")
    warmup_size = max(config.dqn.batch_size, config.dqn.warmup_steps)
    stage_log(f"Starting training for {total_steps} step(s). Replay warmup threshold: {warmup_size}.")
    progress_bar = make_progress_bar(total=total_steps, desc="train steps", enabled=progress)
    warmup_announced = False
    for step_index in range(1, total_steps + 1):
        action = agent.act(current_observation, explore=True)
        previous_distance = float(info["current_distance"])
        next_observation, reward, terminated, truncated, info = env.step(action)
        transition = ReplayTransition(
            observation=current_observation,
            action=action,
            reward=reward,
            next_observation=next_observation,
            done=bool(terminated or truncated),
            target_id=str(info["target_id"]),
        )
        agent.observe(transition)
        loss = agent.train_step()
        epsilon = float(agent.epsilon())
        replay_size = len(agent.replay)
        current_distance = float(info["current_distance"])
        distance_delta = float(previous_distance - current_distance)
        step_in_episode = int(info["step_count"])
        parameter_changed = str(info.get("parameter_changed") or "")
        parameter_delta = float(info.get("parameter_delta") or 0.0)
        logs.append(
            {
                "step": float(agent.total_steps),
                "episode_id": float(episode_index),
                "step_in_episode": float(step_in_episode),
                "reward": float(reward),
                "initial_distance": float(current_initial_distance),
                "previous_distance": previous_distance,
                "distance": current_distance,
                "distance_delta": distance_delta,
                "loss": float(loss) if loss is not None else float("nan"),
                "epsilon": epsilon,
                "replay_size": float(replay_size),
                "parameter_changed": parameter_changed,
                "parameter_delta": parameter_delta,
                "target_id": str(info["target_id"]),
                "target_label": str(info.get("target_label") or ""),
            }
        )
        writer.add_scalar("train/reward", float(reward), agent.total_steps)
        writer.add_scalar("train/previous_distance", previous_distance, agent.total_steps)
        writer.add_scalar("train/distance", current_distance, agent.total_steps)
        writer.add_scalar("train/distance_delta", distance_delta, agent.total_steps)
        writer.add_scalar("train/initial_distance", float(current_initial_distance), agent.total_steps)
        writer.add_scalar("train/epsilon", epsilon, agent.total_steps)
        writer.add_scalar("train/replay_size", float(replay_size), agent.total_steps)
        writer.add_scalar("train/step_in_episode", float(step_in_episode), agent.total_steps)
        writer.add_scalar("train/episode_id", float(episode_index), agent.total_steps)
        if loss is not None and np.isfinite(float(loss)):
            writer.add_scalar("train/loss", float(loss), agent.total_steps)
        if agent.total_steps % config.dqn.target_sync_interval == 0 and loss is not None:
            writer.add_scalar("train/target_sync", 1.0, agent.total_steps)
        if step_index == 1 or step_index % log_interval == 0 or step_index == total_steps:
            _maybe_text_log(
                writer,
                "train/parameter_changed",
                parameter_changed or "<none>",
                agent.total_steps,
                enabled=bool(parameter_changed),
            )
        episode_reward += float(reward)
        episode_steps += 1
        if replay_size >= warmup_size and not warmup_announced:
            stage_log(f"Replay warmup complete at step {agent.total_steps}. Optimizer updates are active.")
            warmup_announced = True
        if step_index == 1 or step_index % log_interval == 0 or step_index == total_steps:
            loss_text = "warmup" if loss is None or not np.isfinite(float(loss)) else f"{float(loss):.4f}"
            progress_bar.set_postfix(
                {
                    "step": agent.total_steps,
                    "eps": f"{epsilon:.3f}",
                    "reward": f"{float(reward):.3f}",
                    "dist": f"{current_distance:.3f}",
                    "delta": f"{distance_delta:.3f}",
                    "loss": loss_text,
                    "replay": replay_size,
                    "param": parameter_changed or "-",
                }
            )
            if not progress:
                stage_log(
                    f"train step {agent.total_steps}/{total_steps}: reward={float(reward):.4f} "
                    f"distance={previous_distance:.4f}->{current_distance:.4f} delta={distance_delta:.4f} "
                    f"loss={loss_text} epsilon={epsilon:.4f} replay={replay_size} "
                    f"param={parameter_changed or '-'} target={info['target_id']}"
                )
        if terminated or truncated:
            metric = EpisodeMetrics(
                total_reward=float(episode_reward),
                steps=int(episode_steps),
                initial_distance=float(current_initial_distance),
                final_distance=float(info["current_distance"]),
                target_id=episode_target_id,
                target_label=episode_target_label,
            )
            writer.add_scalar("episode/total_reward", metric.total_reward, episode_index)
            writer.add_scalar("episode/initial_distance", metric.initial_distance, episode_index)
            writer.add_scalar("episode/final_distance", metric.final_distance, episode_index)
            writer.add_scalar("episode/distance_reduction", metric.initial_distance - metric.final_distance, episode_index)
            writer.add_scalar("episode/length", float(metric.steps), episode_index)
            if episode_index == 1 or episode_index % episode_log_interval == 0:
                stage_log(_episode_summary_line("train", episode_index, metric))
            current_observation, reset_info = env.reset()
            info = reset_info
            current_initial_distance = float(reset_info["current_distance"])
            episode_reward = 0.0
            episode_steps = 0
            episode_index += 1
            episode_target_id = str(reset_info["target_id"])
            episode_target_label = reset_info.get("target_label")
        else:
            current_observation = next_observation
        progress_bar.update(1)
    progress_bar.close()
    writer.flush()
    writer.close()
    return agent, logs


def _prime_target_embeddings(
    coordinator: BatchedRolloutCoordinator,
    render_pool: ParallelRenderPool,
    embedder,
    *,
    progress: bool,
    batch_size: int,
) -> None:
    requests = coordinator.build_target_render_requests()
    if not requests:
        return
    stage_log(f"Precomputing target embeddings for {len(requests)} target(s).")
    progress_bar = make_progress_bar(total=len(requests), desc="target embeddings", enabled=progress)
    for start in range(0, len(requests), max(1, batch_size)):
        request_batch = requests[start : start + max(1, batch_size)]
        render_results = render_pool.render_batch(request_batch)
        embeddings = embed_audio_batch(
            embedder,
            [result.audio for result in render_results],
            [result.sample_rate for result in render_results],
            fallback_size=len(coordinator.parameter_specs),
            batch_size=max(1, batch_size),
        )
        coordinator.apply_target_embeddings(request_batch, render_results, embeddings)
        progress_bar.update(len(request_batch))
    progress_bar.close()


def _reset_slot_batch(
    coordinator: BatchedRolloutCoordinator,
    slot_ids: list[int],
    default_params: dict[str, float],
    render_pool: ParallelRenderPool,
    embedder,
    *,
    batch_size: int,
) -> list[EpisodeSlotState]:
    pending_states, requests = coordinator.reset_slot_requests(slot_ids, default_params)
    render_results = render_pool.render_batch(requests)
    embeddings = embed_audio_batch(
        embedder,
        [result.audio for result in render_results],
        [result.sample_rate for result in render_results],
        fallback_size=len(coordinator.parameter_specs),
        batch_size=max(1, batch_size),
    )
    finalized: list[EpisodeSlotState] = []
    rerender_states: list[EpisodeSlotState] = []
    rerender_requests: list[RenderRequest] = []
    for state, result, embedding in zip(pending_states, render_results, embeddings):
        distance = coordinator.distance_model.distance(np.asarray(embedding, dtype=np.float32), state.target.embedding)
        min_reset_distance = max(1e-8, float(coordinator.config.success_threshold))
        if distance <= min_reset_distance and coordinator.parameter_specs:
            for index, spec in enumerate(coordinator.parameter_specs[: min(8, len(coordinator.parameter_specs))]):
                state.current_params[spec.stable_id] = float(
                    np.clip(state.current_params[spec.stable_id] + (index + 1) * coordinator.config.action_step, 0.0, 1.0)
                )
            rerender_states.append(state)
            rerender_requests.append(
                RenderRequest(slot_id=state.slot_id, render_mode="parameter_state", parameters=dict(state.current_params))
            )
            continue
        state.current_embedding = np.asarray(embedding, dtype=np.float32)
        state.current_distance = float(distance)
        state.initial_distance = float(distance)
        state.observation = coordinator.flatten_observation(state.current_embedding, state.target.embedding, state.current_params)
        state.last_worker_id = result.worker_id
        finalized.append(state)
    if rerender_requests:
        rerender_results = render_pool.render_batch(rerender_requests)
        rerender_embeddings = embed_audio_batch(
            embedder,
            [result.audio for result in rerender_results],
            [result.sample_rate for result in rerender_results],
            fallback_size=len(coordinator.parameter_specs),
            batch_size=max(1, batch_size),
        )
        for state, result, embedding in zip(rerender_states, rerender_results, rerender_embeddings):
            distance = coordinator.distance_model.distance(np.asarray(embedding, dtype=np.float32), state.target.embedding)
            assert np.isfinite(distance), "Initial distance must be finite after rerender."
            state.current_embedding = np.asarray(embedding, dtype=np.float32)
            state.current_distance = float(distance)
            state.initial_distance = float(distance)
            state.observation = coordinator.flatten_observation(state.current_embedding, state.target.embedding, state.current_params)
            state.last_worker_id = result.worker_id
            finalized.append(state)
    return sorted(finalized, key=lambda item: item.slot_id)


def train_dqn_batched(
    config: ExperimentConfig,
    total_steps: int,
    *,
    progress: bool = True,
    log_interval: int = 25,
    episode_log_interval: int = 10,
    tensorboard: bool = True,
    tensorboard_dir: Path | None = None,
) -> tuple[DQNAgent, list[dict[str, float | str]]]:
    assert total_steps >= 1, f"total_steps must be >= 1, got {total_steps}"
    assert log_interval >= 1, f"log_interval must be >= 1, got {log_interval}"
    assert episode_log_interval >= 1, f"episode_log_interval must be >= 1, got {episode_log_interval}"
    assert config.num_render_workers >= 1, f"num_render_workers must be >= 1, got {config.num_render_workers}"
    assert config.num_parallel_envs >= 1, f"num_parallel_envs must be >= 1, got {config.num_parallel_envs}"
    assert config.updates_per_tick >= 1, f"updates_per_tick must be >= 1, got {config.updates_per_tick}"
    assert config.clap_batch_size >= 1, f"clap_batch_size must be >= 1, got {config.clap_batch_size}"
    stage_log("Loading coordinator host for parameter discovery.")
    probe_host = SynthHost(config.env.host)
    parameter_specs = probe_host.filter_parameters(
        allowlist=config.env.parameter_allowlist,
        denylist=config.env.parameter_denylist,
    )
    default_params = probe_host.get_normalized_defaults(parameter_specs)
    coordinator = BatchedRolloutCoordinator(config.env, config.curriculum, parameter_specs)
    embedder = build_embedder(config.env.reward)
    writer = create_summary_writer(tensorboard, tensorboard_dir or config.output_dir / "tensorboard")
    log_run_metadata(writer, _run_metadata(config, total_steps=total_steps))
    logs: list[dict[str, float | str]] = []
    stage_log(
        f"Starting batched training with {config.num_parallel_envs} parallel env(s) and {config.num_render_workers} render worker(s)."
    )
    with ParallelRenderPool(config.env.host, config.num_render_workers) as render_pool:
        _prime_target_embeddings(
            coordinator,
            render_pool,
            embedder,
            progress=progress,
            batch_size=config.clap_batch_size,
        )
        slot_states = _reset_slot_batch(
            coordinator,
            list(range(config.num_parallel_envs)),
            default_params,
            render_pool,
            embedder,
            batch_size=config.clap_batch_size,
        )
        observation_size = slot_states[0].observation.shape[0]
        agent = DQNAgent(observation_size=observation_size, action_size=coordinator.action_size, config=config.dqn)
        warmup_size = max(config.dqn.batch_size, config.dqn.warmup_steps)
        stage_log(f"Batched replay warmup threshold: {warmup_size}.")
        progress_bar = make_progress_bar(total=total_steps, desc="batched train steps", enabled=progress)
        warmup_announced = False
        completed_episodes = 0
        while agent.total_steps < total_steps:
            remaining = total_steps - agent.total_steps
            active_slots = slot_states[: min(len(slot_states), remaining)]
            actions = [agent.act(slot_state.observation, explore=True) for slot_state in active_slots]
            requests, _ = coordinator.build_step_requests(active_slots, actions)
            next_params_by_slot = {
                request.slot_id: dict(request.parameters or {})
                for request in requests
            }
            render_started = perf_counter()
            render_results = render_pool.render_batch(requests)
            render_elapsed = perf_counter() - render_started
            embeddings_started = perf_counter()
            next_embeddings = embed_audio_batch(
                embedder,
                [result.audio for result in render_results],
                [result.sample_rate for result in render_results],
                fallback_size=len(coordinator.parameter_specs),
                batch_size=config.clap_batch_size,
            )
            embed_elapsed = perf_counter() - embeddings_started
            result_by_slot = {result.slot_id: result for result in render_results}
            embedding_by_slot = {
                slot_state.slot_id: np.asarray(embedding, dtype=np.float32)
                for slot_state, embedding in zip(active_slots, next_embeddings)
            }
            pending_resets: list[int] = []
            for slot_state, action in zip(active_slots, actions):
                previous_distance = float(slot_state.current_distance)
                previous_observation = np.asarray(slot_state.observation, dtype=np.float32).copy()
                updated_slot, step_result = coordinator.apply_step_results(
                    slot_state,
                    next_params_by_slot[slot_state.slot_id],
                    embedding_by_slot[slot_state.slot_id],
                    action,
                    result_by_slot[slot_state.slot_id].worker_id,
                )
                transition = ReplayTransition(
                    observation=previous_observation,
                    action=action,
                    reward=step_result.reward,
                    next_observation=updated_slot.observation,
                    done=bool(step_result.terminated or step_result.truncated),
                    target_id=updated_slot.target.target_id,
                )
                agent.observe(transition)
                epsilon = float(agent.epsilon())
                replay_size = len(agent.replay)
                row = {
                    "step": float(agent.total_steps),
                    "episode_id": float(updated_slot.episode_id),
                    "slot_id": float(updated_slot.slot_id),
                    "worker_id": float(step_result.worker_id),
                    "step_in_episode": float(step_result.step_in_episode),
                    "reward": float(step_result.reward),
                    "initial_distance": float(step_result.initial_distance),
                    "previous_distance": previous_distance,
                    "distance": float(step_result.current_distance),
                    "distance_delta": float(step_result.distance_delta),
                    "loss": float("nan"),
                    "epsilon": epsilon,
                    "replay_size": float(replay_size),
                    "parameter_changed": step_result.parameter_changed,
                    "parameter_delta": float(step_result.parameter_delta),
                    "target_id": step_result.target_id,
                    "target_label": str(step_result.target_label or ""),
                }
                loss_value = None
                for _ in range(config.updates_per_tick):
                    maybe_loss = agent.train_step()
                    if maybe_loss is not None and np.isfinite(float(maybe_loss)):
                        loss_value = float(maybe_loss)
                if loss_value is not None:
                    row["loss"] = loss_value
                logs.append(row)
                writer.add_scalar("train/reward", float(step_result.reward), agent.total_steps)
                writer.add_scalar("train/previous_distance", previous_distance, agent.total_steps)
                writer.add_scalar("train/distance", float(step_result.current_distance), agent.total_steps)
                writer.add_scalar("train/distance_delta", float(step_result.distance_delta), agent.total_steps)
                writer.add_scalar("train/initial_distance", float(step_result.initial_distance), agent.total_steps)
                writer.add_scalar("train/epsilon", epsilon, agent.total_steps)
                writer.add_scalar("train/replay_size", float(replay_size), agent.total_steps)
                writer.add_scalar("train/step_in_episode", float(step_result.step_in_episode), agent.total_steps)
                writer.add_scalar("train/episode_id", float(updated_slot.episode_id), agent.total_steps)
                writer.add_scalar("train/slot_id", float(updated_slot.slot_id), agent.total_steps)
                writer.add_scalar("train/worker_id", float(step_result.worker_id), agent.total_steps)
                writer.add_scalar("batched/render_seconds", float(render_elapsed), agent.total_steps)
                writer.add_scalar("batched/embed_seconds", float(embed_elapsed), agent.total_steps)
                writer.add_scalar("batched/clap_batch_size", float(len(render_results)), agent.total_steps)
                writer.add_scalar(
                    "batched/render_items_per_second",
                    float(len(render_results) / max(render_elapsed, 1e-8)),
                    agent.total_steps,
                )
                writer.add_scalar(
                    "batched/embed_items_per_second",
                    float(len(render_results) / max(embed_elapsed, 1e-8)),
                    agent.total_steps,
                )
                if loss_value is not None:
                    writer.add_scalar("train/loss", loss_value, agent.total_steps)
                if agent.total_steps % config.dqn.target_sync_interval == 0 and loss_value is not None:
                    writer.add_scalar("train/target_sync", 1.0, agent.total_steps)
                if agent.total_steps == 1 or agent.total_steps % log_interval == 0 or agent.total_steps == total_steps:
                    _maybe_text_log(
                        writer,
                        "train/parameter_changed",
                        step_result.parameter_changed or "<none>",
                        agent.total_steps,
                        enabled=bool(step_result.parameter_changed),
                    )
                if replay_size >= warmup_size and not warmup_announced:
                    stage_log(f"Replay warmup complete at step {agent.total_steps}. Optimizer updates are active.")
                    warmup_announced = True
                if agent.total_steps == 1 or agent.total_steps % log_interval == 0 or agent.total_steps == total_steps:
                    loss_text = "warmup" if loss_value is None else f"{loss_value:.4f}"
                    progress_bar.set_postfix(
                        {
                            "step": agent.total_steps,
                            "eps": f"{epsilon:.3f}",
                            "reward": f"{float(step_result.reward):.3f}",
                            "dist": f"{float(step_result.current_distance):.3f}",
                            "delta": f"{float(step_result.distance_delta):.3f}",
                            "loss": loss_text,
                            "replay": replay_size,
                            "slot": updated_slot.slot_id,
                            "param": step_result.parameter_changed or "-",
                        }
                    )
                    if not progress:
                        stage_log(
                            f"batched train step {agent.total_steps}/{total_steps}: "
                            f"slot={updated_slot.slot_id} reward={float(step_result.reward):.4f} "
                            f"distance={previous_distance:.4f}->{float(step_result.current_distance):.4f} "
                            f"delta={float(step_result.distance_delta):.4f} loss={loss_text} "
                            f"epsilon={epsilon:.4f} replay={replay_size} "
                            f"param={step_result.parameter_changed or '-'} target={step_result.target_id}"
                        )
                if step_result.terminated or step_result.truncated:
                    metric = EpisodeMetrics(
                        total_reward=float(updated_slot.total_reward),
                        steps=int(updated_slot.step_count),
                        initial_distance=float(updated_slot.initial_distance),
                        final_distance=float(updated_slot.current_distance),
                        target_id=updated_slot.target.target_id,
                        target_label=updated_slot.target.label,
                    )
                    completed_episodes += 1
                    writer.add_scalar("episode/total_reward", metric.total_reward, completed_episodes)
                    writer.add_scalar("episode/initial_distance", metric.initial_distance, completed_episodes)
                    writer.add_scalar("episode/final_distance", metric.final_distance, completed_episodes)
                    writer.add_scalar(
                        "episode/distance_reduction",
                        metric.initial_distance - metric.final_distance,
                        completed_episodes,
                    )
                    writer.add_scalar("episode/length", float(metric.steps), completed_episodes)
                    writer.add_scalar("episode/slot_id", float(updated_slot.slot_id), completed_episodes)
                    if completed_episodes == 1 or completed_episodes % episode_log_interval == 0:
                        stage_log(_episode_summary_line("batched-train", completed_episodes, metric))
                    pending_resets.append(updated_slot.slot_id)
                progress_bar.update(1)
            if pending_resets:
                reset_states = _reset_slot_batch(
                    coordinator,
                    pending_resets,
                    default_params,
                    render_pool,
                    embedder,
                    batch_size=config.clap_batch_size,
                )
                slot_by_id = {slot_state.slot_id: slot_state for slot_state in slot_states}
                for state in reset_states:
                    slot_by_id[state.slot_id] = state
                slot_states = [slot_by_id[index] for index in sorted(slot_by_id)]
        progress_bar.close()
    writer.flush()
    writer.close()
    return agent, logs


def evaluate_dqn(
    config: ExperimentConfig,
    checkpoint: Path,
    episodes: int,
    *,
    progress: bool = True,
    episode_log_interval: int = 1,
    tensorboard: bool = False,
    tensorboard_dir: Path | None = None,
    tensorboard_writer=None,
    global_step: int | None = None,
) -> list[EpisodeMetrics]:
    assert episodes >= 1, f"episodes must be >= 1, got {episodes}"
    assert episode_log_interval >= 1, f"episode_log_interval must be >= 1, got {episode_log_interval}"
    stage_log(f"Creating evaluation environment for {episodes} episode(s).")
    env = make_env(config.env, config.curriculum)
    observation, _ = env.reset(seed=config.env.seed)
    agent = DQNAgent(observation_size=observation.shape[0], action_size=env.action_space.n, config=config.dqn)
    stage_log(f"Loading checkpoint: {checkpoint}")
    agent.load(checkpoint)
    writer = tensorboard_writer if tensorboard_writer is not None else create_summary_writer(tensorboard, tensorboard_dir)
    created_writer = tensorboard_writer is None
    metrics: list[EpisodeMetrics] = []
    progress_bar = make_progress_bar(total=episodes, desc="eval episodes", enabled=progress)
    for episode_index in range(1, episodes + 1):
        observation, info = env.reset()
        total_reward = 0.0
        initial_distance = float(info["current_distance"])
        initial_parameter_snapshot = dict(info.get("parameter_snapshot", {}))
        done = False
        truncated = False
        steps = 0
        while not done and not truncated:
            action = agent.act(observation, explore=False)
            observation, reward, done, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
        metrics.append(
            EpisodeMetrics(
                total_reward=total_reward,
                steps=steps,
                initial_distance=initial_distance,
                final_distance=float(info["current_distance"]),
                target_id=str(info["target_id"]),
                target_label=info.get("target_label"),
                initial_parameter_snapshot=initial_parameter_snapshot,
            )
        )
        metric = metrics[-1]
        event_step = int(global_step if global_step is not None else episode_index)
        writer.add_scalar("eval/episode_total_reward", metric.total_reward, event_step)
        writer.add_scalar("eval/episode_initial_distance", metric.initial_distance, event_step)
        writer.add_scalar("eval/episode_final_distance", metric.final_distance, event_step)
        writer.add_scalar("eval/episode_distance_reduction", metric.initial_distance - metric.final_distance, event_step)
        writer.add_scalar("eval/episode_length", float(metric.steps), event_step)
        if episode_index == 1 or episode_index % episode_log_interval == 0 or episode_index == episodes:
            stage_log(_episode_summary_line("eval", episode_index, metric))
        progress_bar.update(1)
    progress_bar.close()
    initial_values = np.asarray([metric.initial_distance for metric in metrics], dtype=np.float64)
    final_values = np.asarray([metric.final_distance for metric in metrics], dtype=np.float64)
    summary_step = int(global_step if global_step is not None else episodes)
    writer.add_scalar("eval/mean_initial_distance", float(np.mean(initial_values)), summary_step)
    writer.add_scalar("eval/mean_final_distance", float(np.mean(final_values)), summary_step)
    writer.add_scalar("eval/mean_distance_reduction", float(np.mean(initial_values - final_values)), summary_step)
    writer.flush()
    if created_writer:
        writer.close()
    return metrics


class LightningDQNModule:
    """Optional Lightning wrapper for offline supervised diagnostics on replay batches."""

    def __init__(self, agent: DQNAgent):
        lightning = require_dependency("lightning", "ml")
        torch = require_dependency("torch", "ml")

        class _Module(lightning.LightningModule):
            def __init__(self):
                super().__init__()
                self.network = agent.online_network
                self.loss_fn = torch.nn.MSELoss()

            def training_step(self, batch, batch_idx):
                obs, targets = batch
                predictions = self.network(obs)
                loss = self.loss_fn(predictions, targets)
                self.log("train_loss", loss)
                return loss

            def configure_optimizers(self):
                return torch.optim.Adam(self.network.parameters(), lr=agent.config.learning_rate)

        self.module = _Module()
