from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rl_synth_programmer.reward import SimilarityRewardModel


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


if __name__ == "__main__":
    unittest.main()
