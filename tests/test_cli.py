from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rl_synth_programmer.cli import _base_parser, _find_manifest, _resolve_run_folder


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

    def test_run_folder_resolves_under_artifacts(self) -> None:
        path = _resolve_run_folder("my_run", create=True)
        self.assertEqual(path, Path("artifacts/my_run"))

    def test_missing_manifest_has_helpful_error(self) -> None:
        run_root = _resolve_run_folder("missing_manifest_test", create=True)
        with self.assertRaises(AssertionError) as ctx:
            _find_manifest(run_root)
        self.assertIn("Did you run generate-target-set", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
