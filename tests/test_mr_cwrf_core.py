from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from mr_cwrf_core import (  # noqa: E402
    CANDIDATE_ORDER,
    integrated_path_cost,
    select_minimax_regret,
)


def repeated_paths(path: list[tuple[int, int]]) -> dict[str, list[tuple[int, int]]]:
    return {candidate: list(path) for candidate in CANDIDATE_ORDER}


class MrCwrfCoreTests(unittest.TestCase):
    def test_integrated_cost_matches_length_plus_weighted_exposure(self) -> None:
        field = np.ones((4, 4), dtype=np.float32) * 0.5
        path = [(0, 0), (1, 1), (2, 2)]
        cost, length = integrated_path_cost(path, field, risk_weight=2.0)
        expected_length = 2.0 * math.sqrt(2.0)
        self.assertAlmostEqual(length, expected_length)
        self.assertAlmostEqual(cost, expected_length * 2.0)

    def test_identical_paths_select_frozen_first_candidate(self) -> None:
        fields = [np.zeros((4, 4), dtype=np.float32) for _ in range(3)]
        decision = select_minimax_regret(
            repeated_paths([(0, 0), (1, 1)]), fields, 15.0, 1.0
        )
        self.assertEqual(decision.selected_candidate, "fixed_seed0")

    def test_selector_prefers_balanced_route_over_scenario_extremes(self) -> None:
        fields = []
        for x in (0, 2, 4):
            field = np.zeros((5, 5), dtype=np.float32)
            field[:, x] = 1.0
            fields.append(field)
        paths = {
            "fixed_seed0": [(0, y) for y in range(5)],
            "fixed_seed1": [(2, y) for y in range(5)],
            "fixed_seed2": [(4, y) for y in range(5)],
            "consensus": [(1, y) for y in range(5)],
            "union": [(3, y) for y in range(5)],
            "cwrf": [(1, y) for y in range(5)],
        }
        decision = select_minimax_regret(paths, fields, 15.0, 1.0)
        self.assertIn(decision.selected_candidate, {"consensus", "union", "cwrf"})

    def test_candidate_membership_is_strict(self) -> None:
        paths = repeated_paths([(0, 0), (1, 1)])
        del paths["union"]
        with self.assertRaises(ValueError):
            select_minimax_regret(
                paths, [np.zeros((3, 3), dtype=np.float32)], 15.0, 1.0
            )


if __name__ == "__main__":
    unittest.main()
