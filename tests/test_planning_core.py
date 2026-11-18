from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from planning_core import (  # noqa: E402
    Detection,
    astar,
    build_risk_field,
    path_metrics,
)


class RiskFieldTests(unittest.TestCase):
    def test_low_confidence_expands_uncertainty_aware_field(self) -> None:
        high = build_risk_field(
            [Detection(20, 20, 30, 30, 0.95)], 64, 64, mode="carf"
        )
        low = build_risk_field(
            [Detection(20, 20, 30, 30, 0.20)], 64, 64, mode="carf"
        )
        self.assertGreater(float(low[25, 36]), float(high[25, 36]))
        self.assertGreater(float(high[25, 25]), float(low[25, 25]))

    def test_risk_weight_changes_a_star_route(self) -> None:
        field = np.zeros((41, 41), dtype=np.float32)
        field[12:30, 18:23] = 1.0
        start, goal = (20, 39), (20, 1)
        geometric, _ = astar(field, start, goal, risk_weight=0.0)
        aware, _ = astar(field, start, goal, risk_weight=20.0)
        self.assertIsNotNone(geometric)
        self.assertIsNotNone(aware)
        geometric_points = np.asarray(geometric)
        aware_points = np.asarray(aware)
        self.assertTrue(np.any(field[geometric_points[:, 1], geometric_points[:, 0]] > 0.5))
        self.assertFalse(np.any(field[aware_points[:, 1], aware_points[:, 0]] > 0.5))

    def test_exposure_is_weighted_by_segment_length(self) -> None:
        evaluation = np.ones((8, 8), dtype=np.float32) * 0.5
        safety = np.zeros((8, 8), dtype=bool)
        clearance = np.ones((8, 8), dtype=np.float32)
        path = [(0, 0), (1, 1), (2, 2)]
        metrics = path_metrics(path, evaluation, safety, clearance, (0, 0), (2, 2))
        expected = 0.5 * (2.0 * math.sqrt(2.0))
        self.assertAlmostEqual(float(metrics["risk_exposure"]), expected, places=6)


if __name__ == "__main__":
    unittest.main()
