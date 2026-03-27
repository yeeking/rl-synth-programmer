from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rl_synth_programmer.config import CurriculumConfig
from rl_synth_programmer.curriculum import TargetPool
from rl_synth_programmer.host import ParameterSpec


class CurriculumTests(unittest.TestCase):
    def test_target_pool_rotation_uses_dwell_episodes(self):
        specs = [ParameterSpec("cutoff", "Cutoff", 0, 0.5)]
        pool = TargetPool(
            CurriculumConfig(pool_size=4, train_size=2, val_size=1, test_size=1, dwell_episodes=2),
            specs,
        )
        first = pool.maybe_advance()
        second = pool.maybe_advance()
        third = pool.maybe_advance()
        self.assertEqual(first.target_id, second.target_id)
        self.assertNotEqual(second.target_id, third.target_id)


if __name__ == "__main__":
    unittest.main()
