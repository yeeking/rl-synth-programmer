from __future__ import annotations

import json
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rl_synth_programmer.architecture_sweep import compare_architectures, load_sweep_config


class ArchitectureSweepTests(unittest.TestCase):
    def test_load_sweep_config_rejects_unknown_architecture_type(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            path.write_text(
                json.dumps(
                    {
                        "architectures": [
                            {
                                "name": "bad",
                                "type": "transformer",
                                "learning_rate": 0.001,
                                "batch_size": 4,
                                "epochs": 1,
                                "seed": 1,
                            }
                        ]
                    }
                )
            )
            with self.assertRaises(ValueError):
                load_sweep_config(path)

    def test_compare_architectures_writes_ranked_leaderboard(self) -> None:
        if importlib.util.find_spec("torch") is None:
            self.skipTest("torch is not installed; install the ml extra to run architecture training tests.")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rng = np.random.default_rng(1)
            observations = rng.normal(size=(12, 7)).astype(np.float32)
            action_rewards = np.stack(
                [
                    observations[:, 0],
                    -observations[:, 1],
                    observations[:, 2] * 0.5,
                    observations[:, -1],
                ],
                axis=1,
            ).astype(np.float32)
            dataset_path = root / "dataset.npz"
            np.savez_compressed(dataset_path, observations=observations, action_rewards=action_rewards)
            (root / "metadata.json").write_text(
                json.dumps(
                    {
                        "embedding_size": 2,
                        "param_count": 1,
                        "observation_layout": {
                            "target_embedding": [0, 2],
                            "current_embedding": [2, 4],
                            "delta_embedding": [4, 6],
                            "params": [6, 7],
                        },
                    }
                )
            )
            config_path = root / "sweep.json"
            config_path.write_text(
                json.dumps(
                    {
                        "split": {"train": 0.6, "val": 0.2, "test": 0.2},
                        "architectures": [
                            {
                                "name": "tiny-mlp",
                                "type": "mlp",
                                "hidden_sizes": [8],
                                "learning_rate": 0.01,
                                "batch_size": 4,
                                "epochs": 1,
                                "seed": 1,
                            },
                            {
                                "name": "tiny-residual",
                                "type": "residual_mlp",
                                "width": 8,
                                "blocks": 1,
                                "learning_rate": 0.01,
                                "batch_size": 4,
                                "epochs": 1,
                                "seed": 2,
                            },
                            {
                                "name": "tiny-cnn",
                                "type": "cnn1d",
                                "channels": [4],
                                "kernel_sizes": [3],
                                "embedding_hidden_size": 4,
                                "param_hidden_sizes": [4],
                                "head_hidden_sizes": [8],
                                "learning_rate": 0.01,
                                "batch_size": 4,
                                "epochs": 1,
                                "seed": 3,
                            },
                        ],
                    }
                )
            )
            out_dir = root / "sweep"
            result = compare_architectures(dataset_path, config_path, out_dir, progress=False, tensorboard=False)
            self.assertEqual(len(result["leaderboard"]), 3)
            self.assertTrue((out_dir / "leaderboard.json").exists())
            self.assertTrue((out_dir / "leaderboard.csv").exists())
            for row in result["leaderboard"]:
                self.assertIn("val_mse", row)
                self.assertIn("val_top1_accuracy", row)
                self.assertIn("val_mean_regret", row)
                self.assertTrue(Path(row["checkpoint"]).exists())

    def test_compare_architectures_supports_action_conditioned_target(self) -> None:
        if importlib.util.find_spec("torch") is None:
            self.skipTest("torch is not installed; install the ml extra to run architecture training tests.")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rng = np.random.default_rng(2)
            observations = rng.normal(size=(6, 8)).astype(np.float32)
            action_rewards = rng.normal(size=(6, 4)).astype(np.float32)
            dataset_path = root / "dataset.npz"
            np.savez_compressed(
                dataset_path,
                observations=observations,
                action_rewards=action_rewards,
                failed_action_counts=np.zeros((6,), dtype=np.int32),
                state_skipped=np.zeros((6,), dtype=np.int32),
            )
            (root / "metadata.json").write_text(
                json.dumps(
                    {
                        "embedding_size": 2,
                        "param_count": 2,
                        "action_step": 0.05,
                    }
                )
            )
            config_path = root / "action_conditioned.json"
            config_path.write_text(
                json.dumps(
                    {
                        "seed": 3,
                        "split": {"train": 0.5, "val": 0.25, "test": 0.25},
                        "target": "action_reward_as_feature_change_proxy",
                        "max_expanded_rows": 0,
                        "architectures": [
                            {
                                "name": "tiny-gru",
                                "type": "gru",
                                "hidden_size": 4,
                                "param_hidden_sizes": [],
                                "head_hidden_sizes": [4],
                                "learning_rate": 0.01,
                                "batch_size": 4,
                                "epochs": 1,
                                "seed": 3,
                            }
                        ],
                    }
                )
            )
            result = compare_architectures(dataset_path, config_path, root / "out", progress=False, tensorboard=False)
            self.assertEqual(result["action_count"], 1)
            self.assertEqual(result["row_count"], 24)
            self.assertEqual(result["best"]["type"], "gru")


if __name__ == "__main__":
    unittest.main()
