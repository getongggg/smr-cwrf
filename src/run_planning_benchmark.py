from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run paired lunar image-plane planning trials.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--grid-width", type=int, default=96)
    parser.add_argument("--fixed-weight", type=float, default=15.0)
    parser.add_argument("--fixed-weights", default="5,10,15,20")
    parser.add_argument("--carf-weights", default="5,10,15,20")
    parser.add_argument("--primary-carf-weight", type=float, default=15.0)
    parser.add_argument("--max-challenge-routes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument(
        "--exclude-first-frames",
        type=int,
        default=0,
        help="Exclude the first N sorted prediction frame IDs (e.g. pilot-tuning frames).",
    )
    parser.add_argument(
        "--exclude-prediction-file",
        type=Path,
        default=None,
        help="Exclude frame IDs present in another prediction JSON, such as a pilot run.",
    )
    return parser.parse_args()


def candidate_pairs(width: int, height: int) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    starts = [(int(round(frac * (width - 1))), height - 2) for frac in (0.15, 0.32, 0.50, 0.68, 0.85)]
    goals = [(int(round(frac * (width - 1))), 1) for frac in (0.18, 0.38, 0.62, 0.82)]
    return [(start, goal) for start in starts for goal in goals]


def select_scenarios(
    frame_id: int,
    safety_mask: np.ndarray,
    max_challenge: int,
    seed: int,
) -> list[tuple[str, tuple[int, int], tuple[int, int]]]:
    height, width = safety_mask.shape
    zero = np.zeros_like(safety_mask, dtype=np.float32)
    valid = []
    challenge = []
    for start, goal in candidate_pairs(width, height):
        if safety_mask[start[1], start[0]] or safety_mask[goal[1], goal[0]]:
            continue
        path, _ = astar(zero, start, goal, risk_weight=0.0)
        if path is None:
            continue
        valid.append((start, goal))
        points = np.asarray(path, dtype=np.int32)
        if np.any(safety_mask[points[:, 1], points[:, 0]]):
            challenge.append((start, goal))
    rng = np.random.default_rng(seed + frame_id)
    selected = []
    if challenge:
        indices = rng.choice(len(challenge), size=min(max_challenge, len(challenge)), replace=False)
        selected.extend(("challenge", *challenge[int(index)]) for index in np.atleast_1d(indices))
    if valid:
        random_pair = valid[int(rng.integers(0, len(valid)))]
        selected.append(("random", *random_pair))
    return selected


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    return (
        raw.groupby(["scenario_type", "method"], as_index=False)
        .agg(
            trials=("scenario_id", "count"),
            success_rate=("success", "mean"),
            violation_rate=("violation", "mean"),
            exposure_mean=("risk_exposure", "mean"),
            exposure_median=("risk_exposure", "median"),
            length_ratio_mean=("length_ratio", "mean"),
            minimum_clearance_mean=("minimum_clearance", "mean"),
            runtime_ms_mean=("runtime_ms", "mean"),
        )
    )


def bootstrap_interval(values: np.ndarray, samples: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=float)
    for index in range(samples):
        means[index] = np.mean(rng.choice(values, size=len(values), replace=True))
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def paired_statistics(
    raw: pd.DataFrame,
    samples: int,
    seed: int,
    fixed_weight: float = 15.0,
    carf_weight: float = 15.0,
) -> pd.DataFrame:
    fixed_method = f"fixed_w{fixed_weight:g}"
    carf_method = f"carf_w{carf_weight:g}"
    comparisons = [
        ("geometric", fixed_method),
        ("geometric", carf_method),
        (fixed_method, carf_method),
    ]
    rows = []
    challenge = raw[raw["scenario_type"] == "challenge"]
    for metric in ("risk_exposure", "length_ratio", "minimum_clearance", "violation"):
        pivot = challenge.pivot(
            index=["frame_id", "scenario_id"], columns="method", values=metric
        )
        for baseline, candidate in comparisons:
            if baseline not in pivot or candidate not in pivot:
                continue
            paired = pivot[[baseline, candidate]].dropna()
            route_difference = paired[candidate] - paired[baseline]
            frame_difference = route_difference.groupby(level="frame_id").mean().to_numpy(float)
            if len(frame_difference) == 0:
                continue
            low, high = bootstrap_interval(
                frame_difference, samples=samples, seed=seed + len(rows)
            )
            nonzero = frame_difference[np.abs(frame_difference) > 1e-12]
            if len(nonzero) == 0:
                statistic, p_value = 0.0, 1.0
            else:
                test = wilcoxon(frame_difference, zero_method="wilcox", alternative="two-sided")
                statistic, p_value = float(test.statistic), float(test.pvalue)
            rows.append(
                {
                    "metric": metric,
                    "baseline": baseline,
                    "candidate": candidate,
                    "paired_routes": len(route_difference),
                    "paired_frames": len(frame_difference),
                    "candidate_minus_baseline_mean": float(np.mean(frame_difference)),
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
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = json.loads(args.predictions.read_text(encoding="utf-8"))
    prediction_metadata = payload["metadata"]
    if not prediction_metadata.get("paper_eligible", False):
        print("WARNING: predictions are marked pilot_invalid_for_paper")

    with (data_root / "split_manifest.csv").open("r", encoding="utf-8", newline="") as handle:
        manifest = {int(row["frame_id"]): row for row in csv.DictReader(handle)}
    prediction_by_frame = {int(row["frame_id"]): row for row in payload["predictions"]}
    sorted_frame_ids = sorted(prediction_by_frame)
    if args.exclude_first_frames < 0:
        raise ValueError("--exclude-first-frames must be non-negative")
    if args.exclude_first_frames >= len(sorted_frame_ids):
        raise ValueError("--exclude-first-frames must leave at least one frame")
    excluded_frame_id_set = set(sorted_frame_ids[: args.exclude_first_frames])
    exclusion_source = None
    if args.exclude_prediction_file is not None:
        exclusion_source = args.exclude_prediction_file.resolve()
        exclusion_payload = json.loads(exclusion_source.read_text(encoding="utf-8"))
        pilot_ids = {int(row["frame_id"]) for row in exclusion_payload["predictions"]}
        excluded_frame_id_set.update(set(sorted_frame_ids) & pilot_ids)
    excluded_frame_ids = sorted(excluded_frame_id_set)
    evaluation_frame_ids = [
        frame_id for frame_id in sorted_frame_ids if frame_id not in excluded_frame_id_set
    ]
    if not evaluation_frame_ids:
        raise ValueError("Frame exclusions must leave at least one prediction frame")
    fixed_weights = [float(value) for value in args.fixed_weights.split(",")]
    if args.fixed_weight not in fixed_weights:
        fixed_weights.append(args.fixed_weight)
    fixed_weights = sorted(set(fixed_weights))
    carf_weights = sorted(set(float(value) for value in args.carf_weights.split(",")))
    if args.primary_carf_weight not in carf_weights:
        carf_weights.append(args.primary_carf_weight)
        carf_weights.sort()
    raw_rows = []
    scenario_count_by_type = defaultdict(int)

    for frame_id in evaluation_frame_ids:
        row = manifest[frame_id]
        prediction = prediction_by_frame[frame_id]
        image_w = int(prediction["image_width"])
        image_h = int(prediction["image_height"])
        grid_w = args.grid_width
        grid_h = max(40, int(round(grid_w * image_h / image_w)))

        gt_normalized = normalized_boxes_from_yolo(data_root / row["label"])
        gt_boxes = scale_boxes(gt_normalized, grid_w, grid_h)
        predicted_normalized = [Detection(**item) for item in prediction["detections"]]
        predicted_boxes = scale_boxes(predicted_normalized, grid_w, grid_h)
        evaluation_field = build_risk_field(gt_boxes, grid_w, grid_h, mode="evaluation")
        safety_mask = build_safety_mask(gt_boxes, grid_w, grid_h)
        clearances = clearance_map(gt_boxes, grid_w, grid_h)
        fixed_field = build_risk_field(predicted_boxes, grid_w, grid_h, mode="fixed")
        carf_field = build_risk_field(predicted_boxes, grid_w, grid_h, mode="carf")
        zero_field = np.zeros((grid_h, grid_w), dtype=np.float32)

        scenarios = select_scenarios(
            frame_id, safety_mask, args.max_challenge_routes, args.seed
        )
        for local_index, (scenario_type, start, goal) in enumerate(scenarios):
            scenario_count_by_type[scenario_type] += 1
            scenario_id = f"{frame_id}_{scenario_type}_{local_index}"
            method_specs = [("geometric", zero_field, 0.0)]
            method_specs.extend(
                (f"fixed_w{weight:g}", fixed_field, weight) for weight in fixed_weights
            )
            method_specs.extend(
                (f"carf_w{weight:g}", carf_field, weight) for weight in carf_weights
            )
            for method, field, weight in method_specs:
                path, runtime = astar(field, start, goal, risk_weight=weight)
                metrics = path_metrics(
                    path, evaluation_field, safety_mask, clearances, start, goal
                )
                raw_rows.append(
                    {
                        "scenario_id": scenario_id,
                        "frame_id": frame_id,
                        "scenario_type": scenario_type,
                        "method": method,
                        "risk_weight": weight,
                        "prediction_count": len(predicted_boxes),
                        "ground_truth_count": len(gt_boxes),
                        "start_x": start[0],
                        "start_y": start[1],
                        "goal_x": goal[0],
                        "goal_y": goal[1],
                        **metrics,
                        "runtime_ms": runtime * 1000.0,
                    }
                )

    raw = pd.DataFrame(raw_rows)
    summary = summarize(raw)
    paired = paired_statistics(
        raw,
        args.bootstrap_samples,
        args.seed,
        fixed_weight=args.fixed_weight,
        carf_weight=args.primary_carf_weight,
    )
    raw.to_csv(output_dir / "planning_trials.csv", index=False)
    summary.to_csv(output_dir / "planning_summary.csv", index=False)
    paired.to_csv(output_dir / "paired_statistics.csv", index=False)
    metadata = {
        "prediction_metadata": prediction_metadata,
        "paper_eligible": bool(prediction_metadata.get("paper_eligible", False)),
        "grid_width": args.grid_width,
        "fixed_weight": args.fixed_weight,
        "primary_carf_weight": args.primary_carf_weight,
        "fixed_weights": fixed_weights,
        "carf_weights": carf_weights,
        "scenario_counts": dict(scenario_count_by_type),
        "raw_rows": len(raw),
        "bootstrap_samples": args.bootstrap_samples,
        "statistical_unit": "frame; route differences are averaged within frame before inference",
        "seed": args.seed,
        "available_prediction_frames": len(sorted_frame_ids),
        "evaluated_prediction_frames": len(evaluation_frame_ids),
        "excluded_first_frames": args.exclude_first_frames,
        "exclude_prediction_file": str(exclusion_source) if exclusion_source else None,
        "excluded_frame_ids": excluded_frame_ids,
        "primary_comparison": (
            f"fixed_w{args.fixed_weight:g} versus "
            f"carf_w{args.primary_carf_weight:g} on challenge scenarios"
        ),
        "risk_exposure_definition": "path-length-weighted integral of ground-truth proximity risk",
        "violation_definition": "path intersects a ground-truth box inflated by 10% of its maximum side",
    }
    (output_dir / "benchmark_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))
    print(summary.to_string(index=False))
    print(paired.to_string(index=False))


if __name__ == "__main__":
    main()
