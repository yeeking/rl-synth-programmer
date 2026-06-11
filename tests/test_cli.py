from __future__ import annotations

import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rl_synth_programmer.cli import _base_parser, _find_manifest, _resolve_run_folder, _validate_args


class CliLoggingOptionsTest(unittest.TestCase):
    def test_train_parser_exposes_logging_options(self) -> None:
        parser = _base_parser()
        args = parser.parse_args(
            [
                "train-dqn",
                "--plugin",
                "/tmp/test.vst3",
                "--run-folder",
                "artifacts/test_run",
                "--reward-mode",
                "clap",
            ]
        )
        self.assertTrue(args.progress)
        self.assertEqual(args.log_interval, 25)
        self.assertEqual(args.episode_log_interval, 10)
        self.assertTrue(args.tensorboard)
        self.assertIsNone(args.tensorboard_dir)
        self.assertEqual(args.run_folder, "artifacts/test_run")
        self.assertEqual(args.num_workers, 1)
        self.assertEqual(args.updates_per_tick, 1)
        self.assertIsNone(args.clap_batch_size)
        self.assertEqual(args.clap_device, "auto")
        self.assertIsNone(args.epsilon_decay_steps)
        self.assertIsNone(args.max_episode_steps)

    def test_train_parser_accepts_epsilon_decay_steps(self) -> None:
        parser = _base_parser()
        args = parser.parse_args(
            [
                "train-dqn",
                "--plugin",
                "/tmp/test.vst3",
                "--run-folder",
                "artifacts/test_run",
                "--epsilon-decay-steps",
                "50000",
            ]
        )
        self.assertEqual(args.epsilon_decay_steps, 50000)

    def test_train_parser_accepts_max_episode_steps(self) -> None:
        parser = _base_parser()
        args = parser.parse_args(
            [
                "train-dqn",
                "--plugin",
                "/tmp/test.vst3",
                "--run-folder",
                "artifacts/test_run",
                "--max-episode-steps",
                "48",
            ]
        )
        self.assertEqual(args.max_episode_steps, 48)

    def test_generate_action_dataset_parser_defaults(self) -> None:
        parser = _base_parser()
        args = parser.parse_args(
            [
                "generate-action-dataset",
                "--plugin",
                "/tmp/test.vst3",
                "--run-folder",
                "artifacts/test_run",
            ]
        )
        self.assertEqual(args.reward_mode, "clap")
        self.assertEqual(args.rows_to_generate, 256)
        self.assertEqual(args.moves_per_cycle, 4)
        self.assertEqual(args.num_workers, 1)
        self.assertEqual(args.clap_batch_size, 8)
        self.assertEqual(args.clap_device, "auto")
        self.assertEqual(args.host_backend, "pedalboard")
        self.assertFalse(args.estimate_only)
        self.assertFalse(args.confirm_large_run)
        self.assertEqual(args.render_timeout_seconds, 300.0)
        self.assertTrue(args.skip_failed_actions)
        self.assertEqual(args.shard_size, 16)
        self.assertEqual(args.render_chunk_size, 0)
        self.assertIsNone(args.max_state_seconds)
        self.assertEqual(args.reload_workers_every_renders, 500)
        self.assertEqual(args.preset_render_slowdown_threshold, 1.5)
        self.assertTrue(args.reload_workers_on_render_slowdown)
        self.assertTrue(args.action_step_calibration)
        self.assertEqual(args.calibration_probe_states, 4)
        self.assertEqual(args.calibration_probe_deltas, "0.01,0.1,0.25,0.5")
        self.assertEqual(args.calibration_reference_delta, 0.25)
        self.assertEqual(args.calibration_min_step, 0.01)
        self.assertEqual(args.calibration_max_step, 0.5)
        self.assertTrue(args.progress)

    def test_generate_action_dataset_parser_accepts_legacy_aliases(self) -> None:
        parser = _base_parser()
        args = parser.parse_args(
            [
                "generate-action-dataset",
                "--plugin",
                "/tmp/test.vst3",
                "--run-folder",
                "artifacts/test_run",
                "--max-states",
                "12",
                "--moves-per-start",
                "3",
                "--yes",
            ]
        )
        self.assertEqual(args.rows_to_generate, 12)
        self.assertEqual(args.moves_per_cycle, 3)
        self.assertTrue(args.confirm_large_run)

    def test_generate_action_dataset_parser_accepts_renderking_backend(self) -> None:
        parser = _base_parser()
        args = parser.parse_args(
            [
                "generate-action-dataset",
                "--plugin",
                "/tmp/test.vst3",
                "--run-folder",
                "artifacts/test_run",
                "--host-backend",
                "renderking",
            ]
        )
        self.assertEqual(args.host_backend, "renderking")

    def test_compare_architectures_parser_defaults(self) -> None:
        parser = _base_parser()
        args = parser.parse_args(
            [
                "compare-architectures",
                "--dataset",
                "/tmp/dataset.npz",
                "--config",
                "/tmp/sweep.json",
                "--out-dir",
                "artifacts/sweep",
            ]
        )
        self.assertEqual(args.dataset, "/tmp/dataset.npz")
        self.assertEqual(args.config, "/tmp/sweep.json")
        self.assertEqual(args.out_dir, "artifacts/sweep")
        self.assertTrue(args.progress)
        self.assertFalse(args.tensorboard)

    def test_search_feature_change_models_parser_defaults(self) -> None:
        parser = _base_parser()
        args = parser.parse_args(["search-feature-change-models"])
        self.assertIsNone(args.dataset)
        self.assertEqual(args.artifacts_root, "artifacts")
        self.assertIsNone(args.config)
        self.assertEqual(args.out_dir, "architecture_search/feature_change")
        self.assertEqual(args.epochs, 5)
        self.assertEqual(args.cv_folds, 1)
        self.assertEqual(args.dataloader_num_workers, 2)
        self.assertTrue(args.progress)
        self.assertFalse(args.tensorboard)

    def test_smoke_train_parser_accepts_disable_flags(self) -> None:
        parser = _base_parser()
        args = parser.parse_args(
            [
                "smoke-train-clap",
                "--plugin",
                "/tmp/test.vst3",
                "--run-folder",
                "artifacts/test_run",
                "--no-progress",
                "--no-tensorboard",
            ]
        )
        self.assertFalse(args.progress)
        self.assertFalse(args.tensorboard)

    def test_removed_manifest_flag_is_rejected(self) -> None:
        parser = _base_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(
                [
                    "smoke-train-clap",
                    "--plugin",
                    "/tmp/test.vst3",
                    "--run-folder",
                    "artifacts/test_run",
                    "--manifest",
                    "/tmp/manifest.json",
                ]
            )

    def test_removed_parallel_flags_are_rejected(self) -> None:
        parser = _base_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(
                [
                    "train-dqn",
                    "--plugin",
                    "/tmp/test.vst3",
                    "--run-folder",
                    "artifacts/test_run",
                    "--num-render-workers",
                    "4",
                ]
            )

    def test_run_folder_resolves_under_artifacts(self) -> None:
        path = _resolve_run_folder("my_run", create=True)
        self.assertEqual(path, Path("artifacts/my_run"))

    def test_missing_manifest_has_helpful_error(self) -> None:
        run_root = _resolve_run_folder("missing_manifest_test", create=True)
        with self.assertRaises(AssertionError) as ctx:
            _find_manifest(run_root)
        self.assertIn("Expected artifacts/missing_manifest_test/targets/manifest.json", str(ctx.exception))

    def test_generate_action_dataset_missing_manifest_suggests_target_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            plugin_path = Path(tmp) / "test.vst3"
            plugin_path.mkdir()
            parser = _base_parser()
            args = parser.parse_args(
                [
                    "generate-action-dataset",
                    "--plugin",
                    str(plugin_path),
                    "--run-folder",
                    "missing_dataset_manifest_test",
                ]
            )
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                with self.assertRaises(SystemExit) as ctx:
                    _validate_args(parser, args)
            self.assertNotEqual(ctx.exception.code, 0)
            message = stderr.getvalue()
            self.assertIn("requires generated preset targets first", message)
            self.assertIn(f"rl-synth generate-target-set --plugin {plugin_path}", message)
            self.assertIn("--run-folder artifacts/missing_dataset_manifest_test", message)

    def test_cli_validation_rejects_missing_plugin_path(self) -> None:
        parser = _base_parser()
        args = parser.parse_args(
            [
                "inspect-plugin",
                "--plugin",
                "/tmp/does-not-exist.vst3",
            ]
        )
        with self.assertRaises(SystemExit):
            _validate_args(parser, args)

    def test_cli_validation_rejects_bad_training_steps_before_manifest_lookup(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            plugin_path = Path(tmp) / "test.vst3"
            plugin_path.mkdir()
            parser = _base_parser()
            args = parser.parse_args(
                [
                    "train-dqn",
                    "--plugin",
                    str(plugin_path),
                    "--run-folder",
                    "missing_manifest_test",
                    "--steps",
                    "0",
                ]
            )
            with self.assertRaises(SystemExit):
                _validate_args(parser, args)


if __name__ == "__main__":
    unittest.main()
