from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from run_planning_benchmark import paired_statistics  # noqa: E402
from compile_route_seed_robustness import holm_adjust  # noqa: E402


class PairedStatisticsTests(unittest.TestCase):
    def test_inference_uses_frames_as_independent_units(self) -> None:
        rows = []
        for frame_id, differences in ((1, [1.0, 1.0, 1.0]), (2, [3.0])):
            for route_index, difference in enumerate(differences):
                scenario_id = f"{frame_id}_challenge_{route_index}"
                for method, value in (("fixed_w15", 0.0), ("carf_w15", difference)):
                    rows.append(
                        {
                            "frame_id": frame_id,
                            "scenario_id": scenario_id,
                            "scenario_type": "challenge",
                            "method": method,
                            "risk_exposure": value,
                            "length_ratio": value,
                            "minimum_clearance": value,
                            "violation": value,
                        }
                    )

        statistics = paired_statistics(pd.DataFrame(rows), samples=100, seed=7)
        row = statistics[
            (statistics["metric"] == "risk_exposure")
            & (statistics["baseline"] == "fixed_w15")
            & (statistics["candidate"] == "carf_w15")
        ].iloc[0]

        self.assertEqual(int(row["paired_routes"]), 4)
        self.assertEqual(int(row["paired_frames"]), 2)
        self.assertAlmostEqual(float(row["candidate_minus_baseline_mean"]), 2.0)

    def test_holm_adjustment_controls_two_primary_tests(self) -> None:
        adjusted = holm_adjust([0.0319, 0.0372])
        self.assertAlmostEqual(adjusted[0], 0.0638)
        self.assertAlmostEqual(adjusted[1], 0.0638)


if __name__ == "__main__":
    unittest.main()
