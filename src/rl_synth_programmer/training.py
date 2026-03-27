from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .agent import DQNAgent, RandomAgent, ReplayTransition
from .config import ExperimentConfig
from .env import SynthProgrammingEnv, make_env
from .logging_utils import create_summary_writer, log_run_metadata, make_progress_bar, stage_log
from .optional_deps import require_dependency


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
