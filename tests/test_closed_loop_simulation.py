from __future__ import annotations

import unittest
import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from run_closed_loop_simulation import (
    Circle,
    METHODS,
    build_fields,
    generate_episode,
    perception_tape,
    segment_metrics,
    simulate_episode,
)


class ClosedLoopSimulationTests(unittest.TestCase):
    def test_episode_generation_is_deterministic(self) -> None:
        first = generate_episode(123, 4)
        second = generate_episode(123, 4)
        self.assertEqual(first, second)

    def test_perception_fields_have_frozen_membership(self) -> None:
        obstacles, _, _ = generate_episode(123, 1)
        fields = build_fields(perception_tape(obstacles, 123, 1, 0))
        self.assertEqual(set(fields), {"fixed_seed0", "fixed_seed1", "fixed_seed2", "consensus", "union", "cwrf"})
        self.assertTrue(all(field.shape == (36, 48) for field in fields.values()))
        self.assertTrue(all(np.isfinite(field).all() for field in fields.values()))

    def test_segment_detects_footprint_collision(self) -> None:
        obstacle = Circle(5.0, 5.0, 2.0)
        truth = np.zeros((36, 48), dtype=float)
        collision, violation, _, _ = segment_metrics(
            np.asarray([1.0, 5.0]), np.asarray([8.0, 5.0]), [obstacle], truth
        )
        self.assertTrue(collision)
        self.assertTrue(violation)

    def test_small_simulation_returns_every_method(self) -> None:
        rows = simulate_episode(123, 0)
        self.assertEqual({row["method"] for row in rows}, set(METHODS))
        self.assertEqual(len(rows), len(METHODS))


if __name__ == "__main__":
    unittest.main()
