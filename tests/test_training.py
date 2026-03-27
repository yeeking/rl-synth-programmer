from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rl_synth_programmer.config import ExperimentConfig, SynthEnvConfig, SynthHostConfig
from rl_synth_programmer.host import ParameterSpec
from rl_synth_programmer.training import train_dqn, train_dqn_batched


class FakeWriter:
    def __init__(self):
        self.scalars: list[tuple[str, float, int]] = []
        self.texts: list[tuple[str, str, int]] = []

    def add_scalar(self, tag: str, scalar_value: float, global_step: int) -> None:
        self.scalars.append((tag, float(scalar_value), int(global_step)))

    def add_text(self, tag: str, text_string: str, global_step: int = 0) -> None:
        self.texts.append((tag, str(text_string), int(global_step)))

    def flush(self) -> None:
        return None

    def close(self) -> None:
        return None


class FakeAgent:
    def __init__(self, observation_size: int, action_size: int, config):
        _ = observation_size, action_size
        self.config = config
        self.total_steps = 0
        self.replay = []

    def act(self, observation, explore: bool = True) -> int:
        _ = observation, explore
        return 0

    def observe(self, transition) -> None:
        self.replay.append(transition)
        self.total_steps += 1

    def train_step(self):
        return None

    def epsilon(self) -> float:
        return 0.5


class FakeActionSpace:
    n = 2


class FakeBatchedEmbedder:
    def embed_audio_batch(self, audios, sample_rates):
        _ = sample_rates
        return np.stack([np.asarray(audio, dtype=np.float32) for audio in audios]).astype(np.float32)


class FakeCoordinatorHost:
    def __init__(self, config):
        self.config = config

    def filter_parameters(self, allowlist=None, denylist=None):
        _ = allowlist, denylist
        return [
            ParameterSpec("cutoff", "Cutoff", 0, 0.25),
            ParameterSpec("resonance", "Resonance", 1, 0.5),
        ]

    def get_normalized_defaults(self, parameter_specs):
        _ = parameter_specs
        return {"cutoff": 0.25, "resonance": 0.5}


class FakeRenderPool:
    def __init__(self, host_config, num_workers):
        _ = host_config
        self.num_workers = num_workers

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        _ = exc_type, exc, tb
        return None

    def render_batch(self, requests):
        from rl_synth_programmer.parallel_rollout import RenderResult

        results = []
        for request in requests:
            if request.parameters is not None:
                cutoff = float(request.parameters.get("cutoff", 0.0))
                resonance = float(request.parameters.get("resonance", 0.0))
                audio = np.array([cutoff + resonance, cutoff - resonance], dtype=np.float32)
            else:
                audio = np.array([0.1, -0.2], dtype=np.float32)
            results.append(
                RenderResult(
                    slot_id=request.slot_id,
                    worker_id=100 + int(request.slot_id),
                    audio=audio,
                    sample_rate=44_100,
                    render_seconds=0.001,
                )
            )
        return results


class FakeEnv:
    def __init__(self):
        self.action_space = FakeActionSpace()
        self._reset_calls = 0

    def reset(self, seed=None):
        _ = seed
        self._reset_calls += 1
        obs = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        info = {
            "current_distance": 1.0 if self._reset_calls == 1 else 0.75,
            "target_id": "target-001",
            "target_label": "Test Target",
            "parameter_snapshot": {"cutoff": 0.5},
            "step_count": 0,
        }
        return obs, info

    def step(self, action):
        _ = action
        obs = np.array([0.4, 0.5, 0.6], dtype=np.float32)
        info = {
            "current_distance": 0.75,
            "target_id": "target-001",
            "target_label": "Test Target",
            "parameter_snapshot": {"cutoff": 0.6},
            "parameter_changed": "cutoff",
            "parameter_delta": 0.05,
            "step_count": 1,
        }
        return obs, 0.25, False, False, info


class TrainingAuditLoggingTests(unittest.TestCase):
    def test_train_log_includes_audit_fields(self) -> None:
        config = ExperimentConfig(env=SynthEnvConfig(host=SynthHostConfig(plugin_path=Path("dummy.vst3"))))
        writer = FakeWriter()
        with patch("rl_synth_programmer.training.make_env", return_value=FakeEnv()):
            with patch("rl_synth_programmer.training.DQNAgent", FakeAgent):
                with patch("rl_synth_programmer.training.create_summary_writer", return_value=writer):
                    agent, logs = train_dqn(
                        config,
                        total_steps=1,
                        progress=False,
                        log_interval=1,
                        episode_log_interval=1,
                        tensorboard=True,
                        tensorboard_dir=Path("/tmp/tb"),
                    )
        _ = agent
        self.assertEqual(len(logs), 1)
        row = logs[0]
        for key in (
            "episode_id",
            "step_in_episode",
            "previous_distance",
            "distance",
            "distance_delta",
            "parameter_changed",
            "parameter_delta",
        ):
            self.assertIn(key, row)
        self.assertAlmostEqual(float(row["previous_distance"]), 1.0)
        self.assertAlmostEqual(float(row["distance"]), 0.75)
        self.assertAlmostEqual(float(row["distance_delta"]), 0.25)
        self.assertAlmostEqual(float(row["reward"]), 0.25)
        self.assertEqual(row["parameter_changed"], "cutoff")
        scalar_tags = {tag for tag, _, _ in writer.scalars}
        self.assertIn("train/previous_distance", scalar_tags)
        self.assertIn("train/distance_delta", scalar_tags)
        self.assertIn("train/step_in_episode", scalar_tags)
        self.assertIn("train/episode_id", scalar_tags)
        text_tags = {tag for tag, _, _ in writer.texts}
        self.assertIn("train/parameter_changed", text_tags)

    def test_batched_train_log_includes_slot_and_worker_audit_fields(self) -> None:
        config = ExperimentConfig(env=SynthEnvConfig(host=SynthHostConfig(plugin_path=Path("dummy.vst3"))))
        config.env.reward.mode = "clap"
        config.num_render_workers = 2
        config.num_parallel_envs = 2
        config.clap_batch_size = 2
        writer = FakeWriter()
        with patch("rl_synth_programmer.training.SynthHost", FakeCoordinatorHost):
            with patch("rl_synth_programmer.training.ParallelRenderPool", FakeRenderPool):
                with patch("rl_synth_programmer.training.build_embedder", return_value=FakeBatchedEmbedder()):
                    with patch("rl_synth_programmer.training.DQNAgent", FakeAgent):
                        with patch("rl_synth_programmer.training.create_summary_writer", return_value=writer):
                            agent, logs = train_dqn_batched(
                                config,
                                total_steps=2,
                                progress=False,
                                log_interval=1,
                                episode_log_interval=1,
                                tensorboard=True,
                                tensorboard_dir=Path("/tmp/tb"),
                            )
        _ = agent
        self.assertEqual(len(logs), 2)
        row = logs[0]
        for key in (
            "slot_id",
            "worker_id",
            "episode_id",
            "step_in_episode",
            "previous_distance",
            "distance",
            "distance_delta",
            "parameter_changed",
            "parameter_delta",
        ):
            self.assertIn(key, row)
        self.assertAlmostEqual(float(row["reward"]), float(row["distance_delta"]))
        scalar_tags = {tag for tag, _, _ in writer.scalars}
        self.assertIn("batched/render_seconds", scalar_tags)
        self.assertIn("batched/embed_seconds", scalar_tags)
        self.assertIn("batched/clap_batch_size", scalar_tags)
        self.assertIn("train/slot_id", scalar_tags)
        self.assertIn("train/worker_id", scalar_tags)


if __name__ == "__main__":
    unittest.main()
