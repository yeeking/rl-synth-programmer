from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rl_synth_programmer.config import RewardConfig
from rl_synth_programmer.reward import CLAPEmbedder, SimilarityRewardModel


class FakeCLAPWrapper:
    model_repo = "fake/repo"
    model_name = {"2023": "CLAP_weights_2023.pth"}


class FakeWrapperModule:
    CLAPWrapper = FakeCLAPWrapper

    def __init__(self, root: Path):
        self.__file__ = str(root / "CLAPWrapper.py")


class RewardTests(unittest.TestCase):
    def test_cosine_distance_and_reward(self):
        model = SimilarityRewardModel(metric="cosine")
        lhs = np.array([1.0, 0.0], dtype=np.float32)
        rhs = np.array([1.0, 0.0], dtype=np.float32)
        other = np.array([0.0, 1.0], dtype=np.float32)
        same_distance = model.distance(lhs, rhs)
        diff_distance = model.distance(lhs, other)
        self.assertAlmostEqual(same_distance, 0.0, places=5)
        self.assertGreater(diff_distance, same_distance)
        self.assertGreater(model.reward(diff_distance, same_distance), 0.0)

    def test_clap_asset_resolution_prefers_local_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            weights_dir = root / "clap-weights"
            weights_dir.mkdir()
            checkpoint = weights_dir / "CLAP_weights_2023.pth"
            checkpoint.write_bytes(b"checkpoint")
            gpt2_dir = weights_dir / "gpt2"
            gpt2_dir.mkdir()
            (gpt2_dir / "config.json").write_text("{}")
            (gpt2_dir / "model.safetensors").write_bytes(b"model")
            (gpt2_dir / "tokenizer_config.json").write_text("{}")
            configs = root / "configs"
            configs.mkdir()
            (configs / "config_2023.yml").write_text("text_model: gpt2\n")
            (root / "CLAPWrapper.py").write_text("")

            with patch("rl_synth_programmer.reward.CLAP_WEIGHTS_DIR", weights_dir):
                with patch("rl_synth_programmer.reward.require_dependency") as require:
                    model_path, text_path = CLAPEmbedder._resolve_clap_assets(
                        RewardConfig(mode="clap"),
                        FakeWrapperModule(root),
                    )

            self.assertEqual(Path(model_path), checkpoint)
            self.assertEqual(Path(text_path), gpt2_dir)
            require.assert_not_called()

    def test_clap_checkpoint_downloads_to_local_cache_when_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            weights_dir = root / "clap-weights"

            class FakeDownloader:
                @staticmethod
                def hf_hub_download(repo_id, filename, local_dir):
                    self.assertEqual(repo_id, "fake/repo")
                    path = Path(local_dir) / filename
                    path.write_bytes(b"downloaded")
                    return str(path)

            with patch("rl_synth_programmer.reward.CLAP_WEIGHTS_DIR", weights_dir):
                with patch("rl_synth_programmer.reward.require_dependency", return_value=FakeDownloader):
                    checkpoint = CLAPEmbedder._resolve_clap_checkpoint(
                        RewardConfig(mode="clap"),
                        FakeWrapperModule(root),
                    )

            self.assertEqual(checkpoint, weights_dir / "CLAP_weights_2023.pth")
            self.assertTrue(checkpoint.exists())


if __name__ == "__main__":
    unittest.main()
