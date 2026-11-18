from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from edrf_core import (
    build_edrf_field,
    cluster_ensemble_detections,
    scale_ensemble_detections,
)
from mr_cwrf_core import CANDIDATE_ORDER, select_minimax_regret
from planning_core import (
    Detection,
    astar,
    build_risk_field,
    build_safety_mask,
    clearance_map,
    normalized_boxes_from_yolo,
    path_metrics,
    scale_boxes,
)
from run_planning_benchmark import bootstrap_interval, select_scenarios, summarize


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the frozen hypothesis-aware MR-CWRF benchmark."
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, nargs=3, required=True)
    parser.add_argument("--frame-partition", type=Path, default=None)
    parser.add_argument("--frozen-selection", type=Path, default=None)
    parser.add_argument("--evaluation-part", choices=("selection", "all"), required=True)
    parser.add_argument("--exclude-prediction-file", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--lambdas", default="0,0.25,0.5,1,2")
    parser.add_argument("--grid-width", type=int, default=96)
    parser.add_argument("--cluster-iou", type=float, default=0.50)
    parser.add_argument("--risk-weight", type=float, default=15.0)
    parser.add_argument("--max-challenge-routes", type=int, default=3)
    parser.add_argument("--route-seed", type=int, default=20260726)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def lambda_label(value: float) -> str:
    return f"mr_l{format(value, 'g').replace('.', 'p')}"


def paired_against_cwrf(
    raw: pd.DataFrame, lambdas: list[float], samples: int, seed: int
) -> pd.DataFrame:
    challenge = raw[raw["scenario_type"] == "challenge"]
    rows = []
    for metric in ("violation", "risk_exposure", "length_ratio", "minimum_clearance"):
        pivot = challenge.pivot(
            index=["frame_id", "scenario_id"], columns="method", values=metric
        )
        for lambda_value in lambdas:
            candidate = lambda_label(lambda_value)
            paired = pivot[["cwrf", candidate]].dropna()
            route_difference = paired[candidate] - paired["cwrf"]
            frame_difference = route_difference.groupby(level="frame_id").mean().to_numpy(float)
            low, high = bootstrap_interval(frame_difference, samples, seed + len(rows))
            nonzero = frame_difference[np.abs(frame_difference) > 1e-12]
            if len(nonzero):
                test = wilcoxon(
                    frame_difference, zero_method="wilcox", alternative="two-sided"
                )
                statistic, p_value = float(test.statistic), float(test.pvalue)
            else:
                statistic, p_value = 0.0, 1.0
            rows.append(
                {
                    "metric": metric,
                    "baseline": "cwrf",
                    "candidate": candidate,
                    "lambda": lambda_value,
                    "paired_routes": len(route_difference),
                    "paired_frames": len(frame_difference),
                    "candidate_minus_cwrf_mean": float(np.mean(frame_difference)),
                    "bootstrap_95_low": low,
                    "bootstrap_95_high": high,
                    "wilcoxon_statistic": statistic,
                    "wilcoxon_p": p_value,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    data_root = args.data_root.resolve()
    output_dir = args.output_dir.resolve()
    lambdas = [float(value) for value in args.lambdas.split(",")]
    frozen_selection = None
    if args.evaluation_part == "selection":
        if lambdas != [0.0, 0.25, 0.5, 1.0, 2.0]:
            raise ValueError("MR-CWRF development requires the frozen lambda grid")
        if args.frozen_selection is not None:
            raise ValueError("development must not consume a post-selection artifact")
    else:
        if args.frozen_selection is None:
            raise ValueError("post-freeze evaluation requires --frozen-selection")
        frozen_selection = json.loads(
            args.frozen_selection.read_text(encoding="utf-8")
        )
        if frozen_selection.get("status") != "SELECTED":
            raise ValueError("frozen selection does not authorize test evaluation")
        selected_lambda = float(frozen_selection["selected"]["lambda"])
        if lambdas != [selected_lambda]:
            raise ValueError("post-freeze evaluation must run only the selected lambda")

    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in args.predictions]
    if not all(payload.get("metadata", {}).get("paper_eligible", False) for payload in payloads):
        raise ValueError("All prediction caches must be paper-eligible")
    predictions = [
        {int(row["frame_id"]): row for row in payload["predictions"]}
        for payload in payloads
    ]
    frame_sets = [set(rows) for rows in predictions]
    if not all(frame_set == frame_sets[0] for frame_set in frame_sets[1:]):
        raise ValueError("Prediction caches do not cover identical frame IDs")

    if args.evaluation_part == "selection":
        if args.frame_partition is None:
            raise ValueError("selection evaluation requires --frame-partition")
        partition = json.loads(args.frame_partition.read_text(encoding="utf-8"))
        frame_ids = [int(value) for value in partition["selection_frame_ids"]]
    else:
        frame_ids = sorted(frame_sets[0])
    excluded_frame_ids: list[int] = []
    if args.exclude_prediction_file is not None:
        exclusion_payload = json.loads(
            args.exclude_prediction_file.read_text(encoding="utf-8")
        )
        excluded = {int(row["frame_id"]) for row in exclusion_payload["predictions"]}
        excluded_frame_ids = sorted(set(frame_ids) & excluded)
        frame_ids = [frame_id for frame_id in frame_ids if frame_id not in excluded]
    if not frame_ids or not set(frame_ids) <= frame_sets[0]:
        raise ValueError("Evaluation frame selection is empty or unavailable")

    with (data_root / "split_manifest.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        manifest = {int(row["frame_id"]): row for row in csv.DictReader(handle)}

    planning_rows: list[dict[str, object]] = []
    decision_rows: list[dict[str, object]] = []
    selection_counts: dict[str, Counter[str]] = {
        lambda_label(value): Counter() for value in lambdas
    }
    scenario_counts: defaultdict[str, int] = defaultdict(int)

    for frame_id in frame_ids:
        model_rows = [rows[frame_id] for rows in predictions]
        shapes = {
            (int(row["image_width"]), int(row["image_height"])) for row in model_rows
        }
        if len(shapes) != 1:
            raise ValueError(f"Prediction image-size mismatch for frame {frame_id}")
        image_w, image_h = shapes.pop()
        grid_w = args.grid_width
        grid_h = max(40, int(round(grid_w * image_h / image_w)))
        normalized = [
            [Detection(**item) for item in row["detections"]] for row in model_rows
        ]
        detector_boxes = [scale_boxes(items, grid_w, grid_h) for items in normalized]
        clusters = cluster_ensemble_detections(normalized, args.cluster_iou)
        scaled_clusters = scale_ensemble_detections(clusters, grid_w, grid_h)

        gt_normalized = normalized_boxes_from_yolo(
            data_root / manifest[frame_id]["label"]
        )
        gt_boxes = scale_boxes(gt_normalized, grid_w, grid_h)
        evaluation_field = build_risk_field(
            gt_boxes, grid_w, grid_h, mode="evaluation"
        )
        safety_mask = build_safety_mask(gt_boxes, grid_w, grid_h)
        clearances = clearance_map(gt_boxes, grid_w, grid_h)

        fields = {
            f"fixed_seed{index}": build_risk_field(
                boxes, grid_w, grid_h, mode="fixed"
            )
            for index, boxes in enumerate(detector_boxes)
        }
        fields["consensus"] = build_edrf_field(
            scaled_clusters, grid_w, grid_h, 0.0, 0.0, minimum_support=2
        )
        fields["union"] = build_edrf_field(
            scaled_clusters, grid_w, grid_h, 1.0, 0.0, minimum_support=1
        )
        fields["cwrf"] = build_edrf_field(
            scaled_clusters, grid_w, grid_h, 0.5, 0.0, minimum_support=1
        )
        scenario_fields = [fields[f"fixed_seed{index}"] for index in range(3)]

        scenarios = select_scenarios(
            frame_id,
            safety_mask,
            args.max_challenge_routes,
            args.route_seed,
        )
        for local_index, (scenario_type, start, goal) in enumerate(scenarios):
            scenario_counts[scenario_type] += 1
            scenario_id = f"{frame_id}_{scenario_type}_{local_index}"
            paths = {}
            runtimes = {}
            base = {
                "scenario_id": scenario_id,
                "frame_id": frame_id,
                "scenario_type": scenario_type,
                "start_x": start[0],
                "start_y": start[1],
                "goal_x": goal[0],
                "goal_y": goal[1],
            }
            for method in CANDIDATE_ORDER:
                path, runtime = astar(
                    fields[method], start, goal, risk_weight=args.risk_weight
                )
                paths[method] = path
                runtimes[method] = runtime * 1000.0
                planning_rows.append(
                    {
                        **base,
                        "method": method,
                        "risk_weight": args.risk_weight,
                        "selected_candidate": method,
                        **path_metrics(
                            path,
                            evaluation_field,
                            safety_mask,
                            clearances,
                            start,
                            goal,
                        ),
                        "runtime_ms": runtimes[method],
                    }
                )

            candidate_runtime = float(sum(runtimes.values()))
            for lambda_value in lambdas:
                method = lambda_label(lambda_value)
                started = time.perf_counter()
                decision = select_minimax_regret(
                    paths,
                    scenario_fields,
                    args.risk_weight,
                    lambda_value,
                    CANDIDATE_ORDER,
                )
                selector_runtime = (time.perf_counter() - started) * 1000.0
                selection_counts[method][decision.selected_candidate] += 1
                selected_path = paths[decision.selected_candidate]
                planning_rows.append(
                    {
                        **base,
                        "method": method,
                        "risk_weight": args.risk_weight,
                        "selected_candidate": decision.selected_candidate,
                        **path_metrics(
                            selected_path,
                            evaluation_field,
                            safety_mask,
                            clearances,
                            start,
                            goal,
                        ),
                        "runtime_ms": candidate_runtime + selector_runtime,
                    }
                )
                for candidate in decision.candidates:
                    decision_rows.append(
                        {
                            **base,
                            "method": method,
                            "lambda": lambda_value,
                            "candidate": candidate.candidate,
                            "is_selected": candidate.candidate
                            == decision.selected_candidate,
                            "scenario_cost_seed0": candidate.scenario_costs[0],
                            "scenario_cost_seed1": candidate.scenario_costs[1],
                            "scenario_cost_seed2": candidate.scenario_costs[2],
                            "regret_seed0": candidate.regrets[0],
                            "regret_seed1": candidate.regrets[1],
                            "regret_seed2": candidate.regrets[2],
                            "mean_regret": candidate.mean_regret,
                            "max_regret": candidate.max_regret,
                            "mean_scenario_cost": candidate.mean_scenario_cost,
                            "path_length": candidate.path_length,
                            "score": candidate.score,
                            "selector_runtime_ms": selector_runtime,
                        }
                    )

    planning = pd.DataFrame(planning_rows)
    decisions = pd.DataFrame(decision_rows)
    summary = summarize(planning)
    paired = paired_against_cwrf(
        planning, lambdas, args.bootstrap_samples, args.route_seed
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    planning.to_csv(output_dir / "planning_trials.csv", index=False)
    decisions.to_csv(output_dir / "decision_scores.csv", index=False)
    summary.to_csv(output_dir / "planning_summary.csv", index=False)
    paired.to_csv(output_dir / "paired_statistics.csv", index=False)
    metadata = {
        "artifact_type": "Frozen MR-CWRF benchmark",
        "data_role": (
            "validation-selection development"
            if args.evaluation_part == "selection"
            else "post-freeze evaluation"
        ),
        "prediction_files": [str(path.resolve()) for path in args.predictions],
        "prediction_sha256": [sha256(path) for path in args.predictions],
        "frame_partition": (
            str(args.frame_partition.resolve()) if args.frame_partition else None
        ),
        "frame_partition_sha256": (
            sha256(args.frame_partition) if args.frame_partition else None
        ),
        "frozen_selection": (
            str(args.frozen_selection.resolve()) if args.frozen_selection else None
        ),
        "frozen_selection_sha256": (
            sha256(args.frozen_selection) if args.frozen_selection else None
        ),
        "evaluation_part": args.evaluation_part,
        "excluded_frame_ids": excluded_frame_ids,
        "evaluated_frames": len(frame_ids),
        "candidate_order": list(CANDIDATE_ORDER),
        "lambdas": lambdas,
        "selection_counts": {
            method: dict(counts) for method, counts in selection_counts.items()
        },
        "scenario_counts": dict(scenario_counts),
        "risk_weight": args.risk_weight,
        "grid_width": args.grid_width,
        "cluster_iou": args.cluster_iou,
        "route_seed": args.route_seed,
        "bootstrap_samples": args.bootstrap_samples,
        "statistical_unit": "frame; route differences averaged within frame",
        "ground_truth_blinding": (
            "Ground truth constructs evaluation metrics and challenge-route sampling only; "
            "MR candidate costs, regrets, and selection use detector fields only."
        ),
        "runtime_scope": (
            "MR runtime is the sum of six candidate A* searches and selector time; "
            "detector inference and field construction are excluded."
        ),
    }
    (output_dir / "benchmark_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))
    print(summary[summary["scenario_type"] == "challenge"].to_string(index=False))
    print(paired.to_string(index=False))


if __name__ == "__main__":
    main()
