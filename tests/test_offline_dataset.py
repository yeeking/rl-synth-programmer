from __future__ import annotations

import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rl_synth_programmer.host import ParameterSpec
from rl_synth_programmer.offline_dataset import ActionDatasetConfig, estimate_action_dataset, generate_action_dataset


class FakeHost:
    def __init__(self, config):
        self.config = config

    def filter_parameters(self):
        return [
            ParameterSpec("cutoff", "Cutoff", 0, 0.5),
            ParameterSpec("resonance", "Resonance", 1, 0.5),
        ]


class FakeEmbedder:
    def embed_audio_batch(self, audios, sample_rates):
        _ = sample_rates
        return np.stack([np.asarray(audio, dtype=np.float32) for audio in audios]).astype(np.float32)


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
            if request.preset_state == b"a":
                audio = np.array([1.0, 0.0], dtype=np.float32)
            elif request.preset_state == b"b":
                audio = np.array([0.0, 1.0], dtype=np.float32)
            elif request.preset_state == b"c":
                audio = np.array([0.5, 0.5], dtype=np.float32)
            else:
                params = request.parameters or {}
                audio = np.array(
                    [
                        float(params.get("cutoff", 0.0)),
                        float(params.get("resonance", 0.0)),
                    ],
                    dtype=np.float32,
                )
            results.append(RenderResult(request.slot_id, 123, audio, 44_100, 0.001))
        return results


class TimeoutRenderPool(FakeRenderPool):
    def render_batch(self, requests, timeout_seconds=None):
        _ = timeout_seconds
        if any(request.slot_id == 0 and request.render_mode == "parameter_state" for request in requests):
            raise TimeoutError("simulated timeout")
        return super().render_batch(requests)


class SlowRenderPool(FakeRenderPool):
    def render_batch(self, requests, timeout_seconds=None):
        _ = timeout_seconds
        if any(request.render_mode == "parameter_state" for request in requests):
            time.sleep(0.02)
        return super().render_batch(requests)


class SlowHighParamRenderPool(FakeRenderPool):
    def render_batch(self, requests, timeout_seconds=None):
        _ = timeout_seconds
        if any(
            request.render_mode == "parameter_state"
            and request.parameters is not None
            and float(request.parameters.get("cutoff", 0.0)) > 0.8
            and float(request.parameters.get("resonance", 0.0)) > 0.8
            for request in requests
        ):
            time.sleep(0.01)
        return super().render_batch(requests)


class SensitivityRenderPool(FakeRenderPool):
    def render_batch(self, requests, timeout_seconds=None):
        from rl_synth_programmer.parallel_rollout import RenderResult

        _ = timeout_seconds
        preset_params = {
            b"a": {"cutoff": 1.0, "resonance": 0.0},
            b"b": {"cutoff": 0.0, "resonance": 1.0},
            b"c": {"cutoff": 0.9, "resonance": 0.9},
        }
        results = []
        for request in requests:
            params = preset_params.get(request.preset_state, request.parameters or {})
            audio = np.array(
                [
                    1.0 + float(params.get("cutoff", 0.0)),
                    1.0 + 0.1 * float(params.get("resonance", 0.0)),
                ],
                dtype=np.float32,
            )
            results.append(RenderResult(request.slot_id, 123, audio, 44_100, 0.001))
        return results


class CountingRenderPool(FakeRenderPool):
    opened = 0
    closed = 0

    def __init__(self, host_config, num_workers):
        type(self).opened += 1
        super().__init__(host_config, num_workers)

    def close(self):
        type(self).closed += 1

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return None


def _write_manifest(root: Path) -> Path:
    (root / "states").mkdir(parents=True)
    state_a = root / "states/a.bin"
    state_b = root / "states/b.bin"
    state_a.write_bytes(b"a")
    state_b.write_bytes(b"b")
    manifest = {
        "targets": [
            {
                "target_id": "a",
                "split": "train",
                "label": "A",
                "parameter_snapshot": {"cutoff": 1.0, "resonance": 0.0},
                "preset_state_path": str(state_a),
            },
            {
                "target_id": "b",
                "split": "train",
                "label": "B",
                "parameter_snapshot": {"cutoff": 0.0, "resonance": 1.0},
                "preset_state_path": str(state_b),
            },
        ]
    }
    path = root / "manifest.json"
    path.write_text(json.dumps(manifest))
    return path


def _write_three_manifest(root: Path) -> Path:
    (root / "states").mkdir(parents=True)
    state_a = root / "states/a.bin"
    state_b = root / "states/b.bin"
    state_c = root / "states/c.bin"
    state_a.write_bytes(b"a")
    state_b.write_bytes(b"b")
    state_c.write_bytes(b"c")
    manifest = {
        "targets": [
            {
                "target_id": "a",
                "split": "train",
                "label": "A",
                "parameter_snapshot": {"cutoff": 1.0, "resonance": 0.0},
                "preset_state_path": str(state_a),
            },
            {
                "target_id": "b",
                "split": "train",
                "label": "B",
                "parameter_snapshot": {"cutoff": 0.0, "resonance": 1.0},
                "preset_state_path": str(state_b),
            },
            {
                "target_id": "c",
                "split": "train",
                "label": "C",
                "parameter_snapshot": {"cutoff": 0.9, "resonance": 0.9},
                "preset_state_path": str(state_c),
            },
        ]
    }
    path = root / "manifest.json"
    path.write_text(json.dumps(manifest))
    return path


class OfflineDatasetTests(unittest.TestCase):
    def test_estimate_action_dataset_does_not_write_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = _write_manifest(root)
            config = ActionDatasetConfig(
                plugin_path=Path("dummy.vst3"),
                manifest_path=manifest_path,
                output_dir=root,
                max_states=2,
                moves_per_start=1,
                preset_render_slowdown_threshold=0.0,
            )
            with patch("rl_synth_programmer.offline_dataset.SynthHost", FakeHost):
                with patch("rl_synth_programmer.offline_dataset.ParallelRenderPool", FakeRenderPool):
                    with patch("rl_synth_programmer.offline_dataset.build_embedder", return_value=FakeEmbedder()):
                        estimate = estimate_action_dataset(config, progress=False)
            self.assertEqual(estimate["sample_states"], 2)
            self.assertEqual(estimate["action_count"], 4)
            self.assertFalse((root / "action_dataset" / "dataset.npz").exists())

    def test_generate_action_dataset_writes_dense_action_rewards(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = _write_manifest(root)
            config = ActionDatasetConfig(
                plugin_path=Path("dummy.vst3"),
                manifest_path=manifest_path,
                output_dir=root,
                max_states=2,
                moves_per_start=1,
            )
            estimate = {
                "estimated_total_renders": 10,
                "sample_states": 2,
                "action_count": 4,
                "observation_size": 8,
                "seconds_per_state": 0.01,
                "estimated_seconds": 0.02,
                "estimated_npz_bytes": 100,
            }
            with patch("rl_synth_programmer.offline_dataset.SynthHost", FakeHost):
                with patch("rl_synth_programmer.offline_dataset.ParallelRenderPool", FakeRenderPool):
                    with patch("rl_synth_programmer.offline_dataset.build_embedder", return_value=FakeEmbedder()):
                        result = generate_action_dataset(config, progress=False, yes=True, estimate=estimate)
            dataset = np.load(result["dataset_path"])
            self.assertEqual(dataset["observations"].shape, (2, 8))
            self.assertEqual(dataset["action_rewards"].shape, (2, 4))
            self.assertTrue(np.all(dataset["best_actions"] == np.argmax(dataset["action_rewards"], axis=1)))
            metadata = json.loads(Path(result["metadata_path"]).read_text())
            self.assertEqual(metadata["param_count"], 2)
            self.assertEqual(metadata["embedding_size"], 2)
            self.assertEqual(metadata["shapes"]["action_rewards"], [2, 4])
            self.assertEqual(metadata["action_step_mode"], "calibrated")
            self.assertEqual(len(metadata["action_deltas"]), 4)
            self.assertIn("cutoff", metadata["action_step_by_parameter"])
            self.assertIn("action_step_calibration", metadata)
            self.assertTrue((root / "action_dataset" / "shards" / "shard-00000.npz").exists())

    def test_generate_action_dataset_calibrates_larger_steps_for_less_sensitive_parameters(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = _write_manifest(root)
            config = ActionDatasetConfig(
                plugin_path=Path("dummy.vst3"),
                manifest_path=manifest_path,
                output_dir=root,
                max_states=1,
                moves_per_start=1,
                calibration_probe_states=1,
                preset_render_slowdown_threshold=0.0,
            )
            estimate = {
                "estimated_total_renders": 100,
                "sample_states": 1,
                "action_count": 4,
                "observation_size": 8,
                "seconds_per_state": 0.01,
                "estimated_seconds": 0.02,
                "estimated_npz_bytes": 100,
            }
            with patch("rl_synth_programmer.offline_dataset.SynthHost", FakeHost):
                with patch("rl_synth_programmer.offline_dataset.ParallelRenderPool", SensitivityRenderPool):
                    with patch("rl_synth_programmer.offline_dataset.build_embedder", return_value=FakeEmbedder()):
                        result = generate_action_dataset(config, progress=False, yes=True, estimate=estimate)
            metadata = json.loads(Path(result["metadata_path"]).read_text())
            cutoff_step = float(metadata["action_step_by_parameter"]["cutoff"])
            resonance_step = float(metadata["action_step_by_parameter"]["resonance"])
            self.assertGreater(resonance_step, cutoff_step)
            self.assertEqual(metadata["action_deltas"], [cutoff_step, -cutoff_step, resonance_step, -resonance_step])

    def test_generate_action_dataset_can_disable_action_step_calibration(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = _write_manifest(root)
            config = ActionDatasetConfig(
                plugin_path=Path("dummy.vst3"),
                manifest_path=manifest_path,
                output_dir=root,
                max_states=1,
                moves_per_start=1,
                action_step=0.05,
                action_step_calibration=False,
                preset_render_slowdown_threshold=0.0,
            )
            estimate = {
                "estimated_total_renders": 10,
                "sample_states": 1,
                "action_count": 4,
                "observation_size": 8,
                "seconds_per_state": 0.01,
                "estimated_seconds": 0.02,
                "estimated_npz_bytes": 100,
            }
            with patch("rl_synth_programmer.offline_dataset.SynthHost", FakeHost):
                with patch("rl_synth_programmer.offline_dataset.ParallelRenderPool", FakeRenderPool):
                    with patch("rl_synth_programmer.offline_dataset.build_embedder", return_value=FakeEmbedder()):
                        result = generate_action_dataset(config, progress=False, yes=True, estimate=estimate)
            metadata = json.loads(Path(result["metadata_path"]).read_text())
            self.assertEqual(metadata["action_step_mode"], "fixed")
            self.assertEqual(metadata["action_deltas"], [0.05, -0.05, 0.05, -0.05])

    def test_generate_action_dataset_samples_pairs_round_robin(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = _write_manifest(root)
            config = ActionDatasetConfig(
                plugin_path=Path("dummy.vst3"),
                manifest_path=manifest_path,
                output_dir=root,
                max_states=3,
                moves_per_start=2,
                preset_render_slowdown_threshold=0.0,
            )
            estimate = {
                "estimated_total_renders": 15,
                "sample_states": 3,
                "action_count": 4,
                "observation_size": 8,
                "seconds_per_state": 0.01,
                "estimated_seconds": 0.03,
                "estimated_npz_bytes": 100,
            }
            with patch("rl_synth_programmer.offline_dataset.SynthHost", FakeHost):
                with patch("rl_synth_programmer.offline_dataset.ParallelRenderPool", FakeRenderPool):
                    with patch("rl_synth_programmer.offline_dataset.build_embedder", return_value=FakeEmbedder()):
                        result = generate_action_dataset(config, progress=False, yes=True, estimate=estimate)
            dataset = np.load(result["dataset_path"])
            self.assertEqual(dataset["target_indices"].tolist(), [0, 1, 0])
            self.assertEqual(dataset["start_indices"].tolist(), [1, 0, 1])
            self.assertEqual(dataset["move_indices"].tolist(), [0, 0, 1])
            metadata = json.loads(Path(result["metadata_path"]).read_text())
            self.assertEqual(metadata["sampling_scheme"], "round_robin_greedy")

    def test_generate_action_dataset_marks_timed_out_actions_failed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = _write_manifest(root)
            config = ActionDatasetConfig(
                plugin_path=Path("dummy.vst3"),
                manifest_path=manifest_path,
                output_dir=root,
                max_states=1,
                moves_per_start=1,
                render_timeout_seconds=0.01,
                render_chunk_size=1,
                failed_action_reward=-123.0,
                preset_render_slowdown_threshold=0.0,
            )
            estimate = {
                "estimated_total_renders": 5,
                "sample_states": 1,
                "action_count": 4,
                "observation_size": 8,
                "seconds_per_state": 0.01,
                "estimated_seconds": 0.01,
                "estimated_npz_bytes": 100,
            }
            with patch("rl_synth_programmer.offline_dataset.SynthHost", FakeHost):
                with patch("rl_synth_programmer.offline_dataset.ParallelRenderPool", TimeoutRenderPool):
                    with patch("rl_synth_programmer.offline_dataset.build_embedder", return_value=FakeEmbedder()):
                        result = generate_action_dataset(config, progress=False, yes=True, estimate=estimate)
            dataset = np.load(result["dataset_path"])
            self.assertEqual(float(dataset["action_rewards"][0, 0]), -123.0)
            self.assertEqual(int(dataset["failed_action_counts"][0]), 1)
            summary = json.loads(Path(result["summary_path"]).read_text())
            self.assertEqual(summary["failed_action_count"], 1)

    def test_generate_action_dataset_skips_slow_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = _write_manifest(root)
            config = ActionDatasetConfig(
                plugin_path=Path("dummy.vst3"),
                manifest_path=manifest_path,
                output_dir=root,
                max_states=1,
                moves_per_start=1,
                render_chunk_size=1,
                max_state_seconds=0.01,
                failed_action_reward=-123.0,
                preset_render_slowdown_threshold=0.0,
            )
            estimate = {
                "estimated_total_renders": 5,
                "sample_states": 1,
                "action_count": 4,
                "observation_size": 8,
                "seconds_per_state": 0.01,
                "estimated_seconds": 0.01,
                "estimated_npz_bytes": 100,
            }
            with patch("rl_synth_programmer.offline_dataset.SynthHost", FakeHost):
                with patch("rl_synth_programmer.offline_dataset.ParallelRenderPool", SlowRenderPool):
                    with patch("rl_synth_programmer.offline_dataset.build_embedder", return_value=FakeEmbedder()):
                        result = generate_action_dataset(config, progress=False, yes=True, estimate=estimate)
            dataset = np.load(result["dataset_path"])
            self.assertEqual(int(dataset["state_skipped"][0]), 1)
            self.assertGreater(int(dataset["failed_action_counts"][0]), 0)
            summary = json.loads(Path(result["summary_path"]).read_text())
            self.assertEqual(summary["skipped_state_count"], 1)

    def test_generate_action_dataset_reloads_workers_after_render_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = _write_manifest(root)
            config = ActionDatasetConfig(
                plugin_path=Path("dummy.vst3"),
                manifest_path=manifest_path,
                output_dir=root,
                max_states=2,
                moves_per_start=1,
                shard_size=1,
                reload_workers_every_renders=3,
                preset_render_slowdown_threshold=0.0,
            )
            estimate = {
                "estimated_total_renders": 10,
                "sample_states": 2,
                "action_count": 4,
                "observation_size": 8,
                "seconds_per_state": 0.01,
                "estimated_seconds": 0.02,
                "estimated_npz_bytes": 100,
            }
            CountingRenderPool.opened = 0
            CountingRenderPool.closed = 0
            with patch("rl_synth_programmer.offline_dataset.SynthHost", FakeHost):
                with patch("rl_synth_programmer.offline_dataset.ParallelRenderPool", CountingRenderPool):
                    with patch("rl_synth_programmer.offline_dataset.build_embedder", return_value=FakeEmbedder()):
                        result = generate_action_dataset(config, progress=False, yes=True, estimate=estimate)
            self.assertGreaterEqual(CountingRenderPool.opened, 2)
            self.assertEqual(CountingRenderPool.opened, CountingRenderPool.closed)
            metadata = json.loads(Path(result["metadata_path"]).read_text())
            self.assertEqual(metadata["args"]["reload_workers_every_renders"], 3)

    def test_generate_action_dataset_asserts_on_preset_render_slowdown(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = _write_three_manifest(root)
            config = ActionDatasetConfig(
                plugin_path=Path("dummy.vst3"),
                manifest_path=manifest_path,
                output_dir=root,
                max_states=2,
                moves_per_start=1,
                render_chunk_size=1,
                preset_render_slowdown_threshold=1.1,
                reload_workers_on_render_slowdown=False,
            )
            estimate = {
                "estimated_total_renders": 20,
                "sample_states": 2,
                "action_count": 4,
                "observation_size": 8,
                "seconds_per_state": 0.01,
                "estimated_seconds": 0.02,
                "estimated_npz_bytes": 100,
            }
            with patch("rl_synth_programmer.offline_dataset.SynthHost", FakeHost):
                with patch("rl_synth_programmer.offline_dataset.ParallelRenderPool", SlowHighParamRenderPool):
                    with patch("rl_synth_programmer.offline_dataset.build_embedder", return_value=FakeEmbedder()):
                        with self.assertRaises(AssertionError) as ctx:
                            generate_action_dataset(config, progress=False, yes=True, estimate=estimate)
            message = str(ctx.exception)
            self.assertIn("Preset render slowdown detected", message)
            self.assertIn("target=a", message)
            self.assertIn("start=c", message)

    def test_generate_action_dataset_requires_confirmation_for_large_estimate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = _write_manifest(root)
            config = ActionDatasetConfig(
                plugin_path=Path("dummy.vst3"),
                manifest_path=manifest_path,
                output_dir=root,
                max_states=2,
                moves_per_start=1,
            )
            with self.assertRaises(RuntimeError) as ctx:
                generate_action_dataset(
                    config,
                    progress=False,
                    yes=False,
                    estimate={"estimated_total_renders": 10001},
                )
            self.assertIn("--yes", str(ctx.exception))
            self.assertFalse((root / "action_dataset" / "dataset.npz").exists())


if __name__ == "__main__":
    unittest.main()
