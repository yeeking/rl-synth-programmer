from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rl_synth_programmer.config import CurriculumConfig, RewardConfig, SynthEnvConfig, SynthHostConfig
from rl_synth_programmer.env import SynthProgrammingEnv
from rl_synth_programmer.host import ParameterSpec


class FakeHost:
    def __init__(self):
        self.config = SynthHostConfig(plugin_path=Path("dummy.vst3"))
        self.params = {"cutoff": 0.5, "resonance": 0.25}
        self._preset_states = {
            b"preset-a": {"cutoff": 1.0, "resonance": 0.0},
            b"preset-b": {"cutoff": 0.0, "resonance": 1.0},
        }

    def filter_parameters(self, allowlist=None, denylist=None):
        _ = allowlist, denylist
        return [
            ParameterSpec("cutoff", "Cutoff", 0, 0.5),
            ParameterSpec("resonance", "Resonance", 1, 0.25),
        ]

    def get_normalized_defaults(self, parameter_specs):
        return {spec.stable_id: self.params[spec.stable_id] for spec in parameter_specs}

    def set_parameters(self, normalized_values, parameter_specs=None):
        _ = parameter_specs
        self.params.update(normalized_values)

    def capture_preset_state(self):
        return b"preset-a"

    def restore_preset_state(self, state):
        self.params = dict(self._preset_states[state])

    def render_note(self, parameter_values=None, note=None, duration=None, velocity=None):
        _ = note, duration, velocity
        if parameter_values:
            self.params.update(parameter_values)
        return np.array([self.params["cutoff"], self.params["resonance"]], dtype=np.float32)


class FakeEmbedder:
    def embed_audio(self, audio, sample_rate):
        _ = sample_rate
        return np.asarray(audio, dtype=np.float32)


class EnvTests(unittest.TestCase):
    def test_manifest_reset_prefers_other_preset_start(self):
        config = SynthEnvConfig(
            host=SynthHostConfig(plugin_path=Path("dummy.vst3")),
            reward=RewardConfig(mode="clap"),
            action_step=0.1,
            max_episode_steps=2,
        )
        manifest_path = Path("/tmp/test_manifest_reset_prefers_other_preset_start.json")
        manifest_path.write_text(
            """
{
  "targets": [
    {
      "target_id": "train-a",
      "split": "train",
      "label": "A",
      "parameter_snapshot": {"cutoff": 1.0, "resonance": 0.0},
      "preset_state_path": "/tmp/test-preset-a.bin"
    },
    {
      "target_id": "train-b",
      "split": "train",
      "label": "B",
      "parameter_snapshot": {"cutoff": 0.0, "resonance": 1.0},
      "preset_state_path": "/tmp/test-preset-b.bin"
    }
  ]
}
            """.strip()
        )
        Path("/tmp/test-preset-a.bin").write_bytes(b"preset-a")
        Path("/tmp/test-preset-b.bin").write_bytes(b"preset-b")
        env = SynthProgrammingEnv(
            config,
            CurriculumConfig(pool_size=2, train_size=2, val_size=0, test_size=0, manifest_path=manifest_path),
            FakeHost(),
            FakeEmbedder(),
        )
        _, info = env.reset(seed=0)
        snapshot = info["parameter_snapshot"]
        self.assertEqual(snapshot, {"cutoff": 0.0, "resonance": 1.0})
        self.assertEqual(info["target_id"], "train-a")

    def test_env_step_changes_one_parameter(self):
        config = SynthEnvConfig(
            host=SynthHostConfig(plugin_path=Path("dummy.vst3")),
            reward=RewardConfig(mode="clap"),
            action_step=0.1,
            max_episode_steps=2,
        )
        env = SynthProgrammingEnv(config, CurriculumConfig(pool_size=4, train_size=2, val_size=1, test_size=1), FakeHost(), FakeEmbedder())
        observation, info = env.reset(seed=0)
        self.assertGreater(observation.shape[0], 2)
        next_observation, reward, terminated, truncated, info = env.step(0)
        self.assertIsInstance(reward, float)
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        snapshot = info["parameter_snapshot"]
        self.assertNotEqual(snapshot["cutoff"], 0.5)
        self.assertEqual(next_observation.shape, observation.shape)
        self.assertEqual(info["parameter_changed"], "cutoff")
        self.assertAlmostEqual(info["parameter_delta"], 0.1)
        self.assertEqual(info["step_count"], 1)


if __name__ == "__main__":
    unittest.main()
