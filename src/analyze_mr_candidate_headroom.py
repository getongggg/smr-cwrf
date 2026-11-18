from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from edrf_core import (
    build_edrf_field,
    cluster_ensemble_detections,
    scale_ensemble_detections,
)
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


CANDIDATE_ORDER = (
    "fixed_seed0",
    "fixed_seed1",
    "fixed_seed2",
    "consensus",
    "union",
    "cwrf",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validation-only oracle headroom for the frozen MR-CWRF candidate set."
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, nargs=3, required=True)
    parser.add_argument("--frame-partition", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
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


def choose_oracle(rows: list[dict[str, object]]) -> dict[str, object]:
    successful = [row for row in rows if int(row["success"]) == 1]
    if not successful:
        return min(rows, key=lambda row: CANDIDATE_ORDER.index(str(row["method"])))
    return min(
        successful,
        key=lambda row: (
            int(row["violation"]),
            float(row["risk_exposure"]),
            float(row["length_ratio"]),
            CANDIDATE_ORDER.index(str(row["method"])),
        ),
    )


def paired_headroom(
    raw: pd.DataFrame, samples: int, seed: int
) -> pd.DataFrame:
    challenge = raw[raw["scenario_type"] == "challenge"]
    rows = []
    for metric in ("violation", "risk_exposure", "length_ratio", "minimum_clearance"):
        pivot = challenge.pivot(
            index=["frame_id", "scenario_id"], columns="method", values=metric
        )
        paired = pivot[["cwrf", "oracle"]].dropna()
        route_difference = paired["oracle"] - paired["cwrf"]
        frame_difference = route_difference.groupby(level="frame_id").mean().to_numpy(float)
        low, high = bootstrap_interval(frame_difference, samples, seed + len(rows))
        rows.append(
            {
                "metric": metric,
                "baseline": "cwrf",
                "candidate": "oracle",
                "paired_routes": len(route_difference),
                "paired_frames": len(frame_difference),
                "oracle_minus_cwrf_mean": float(np.mean(frame_difference)),
                "bootstrap_95_low": low,
                "bootstrap_95_high": high,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    data_root = args.data_root.resolve()
    output_dir = args.output_dir.resolve()
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

    partition = json.loads(args.frame_partition.read_text(encoding="utf-8"))
    frame_ids = [int(value) for value in partition["selection_frame_ids"]]
    if not set(frame_ids) <= frame_sets[0]:
        raise ValueError("Validation-selection IDs are missing from prediction caches")
    with (data_root / "split_manifest.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        manifest = {int(row["frame_id"]): row for row in csv.DictReader(handle)}

    candidate_rows: list[dict[str, object]] = []
    oracle_rows: list[dict[str, object]] = []
    selection_counts: Counter[str] = Counter()
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

        scenarios = select_scenarios(
            frame_id,
            safety_mask,
            args.max_challenge_routes,
            args.route_seed,
        )
        for local_index, (scenario_type, start, goal) in enumerate(scenarios):
            scenario_counts[scenario_type] += 1
            scenario_id = f"{frame_id}_{scenario_type}_{local_index}"
            current = []
            for method in CANDIDATE_ORDER:
                path, runtime = astar(
                    fields[method], start, goal, risk_weight=args.risk_weight
                )
                row = {
                    "scenario_id": scenario_id,
                    "frame_id": frame_id,
                    "scenario_type": scenario_type,
                    "method": method,
                    "start_x": start[0],
                    "start_y": start[1],
                    "goal_x": goal[0],
                    "goal_y": goal[1],
                    **path_metrics(
                        path,
                        evaluation_field,
                        safety_mask,
                        clearances,
                        start,
                        goal,
                    ),
                    "runtime_ms": runtime * 1000.0,
                }
                current.append(row)
                candidate_rows.append(row)
            selected = choose_oracle(current)
            selection_counts[str(selected["method"])] += 1
            oracle_rows.append(
                {
                    **selected,
                    "method": "oracle",
                    "selected_candidate": selected["method"],
                }
            )

    candidates = pd.DataFrame(candidate_rows)
    oracle = pd.DataFrame(oracle_rows)
    raw = pd.concat([candidates, oracle], ignore_index=True)
    summary = summarize(raw)
    paired = paired_headroom(raw, args.bootstrap_samples, args.route_seed)
    challenge = summary[summary["scenario_type"] == "challenge"].set_index("method")
    cwrf = challenge.loc["cwrf"]
    oracle_row = challenge.loc["oracle"]
    violation_reduction = float(cwrf["violation_rate"] - oracle_row["violation_rate"])
    exposure_reduction_relative = float(
        (cwrf["exposure_mean"] - oracle_row["exposure_mean"])
        / max(float(cwrf["exposure_mean"]), 1e-9)
    )
    length_delta = float(oracle_row["length_ratio_mean"] - cwrf["length_ratio_mean"])
    headroom_gate = bool(
        oracle_row["violation_rate"] < cwrf["violation_rate"]
        and oracle_row["exposure_mean"] < cwrf["exposure_mean"]
        and length_delta <= 0.01
        and (violation_reduction >= 0.005 or exposure_reduction_relative >= 0.05)
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(output_dir / "candidate_trials.csv", index=False)
    oracle.to_csv(output_dir / "oracle_trials.csv", index=False)
    summary.to_csv(output_dir / "planning_summary.csv", index=False)
    paired.to_csv(output_dir / "paired_headroom.csv", index=False)
    metadata = {
        "artifact_type": "MR-CWRF validation-only oracle headroom",
        "status": "PROCEED" if headroom_gate else "STOP",
        "data_role": "held-out validation-selection only",
        "prediction_files": [str(path.resolve()) for path in args.predictions],
        "prediction_sha256": [sha256(path) for path in args.predictions],
        "frame_partition": str(args.frame_partition.resolve()),
        "frame_partition_sha256": sha256(args.frame_partition),
        "evaluated_frames": len(frame_ids),
        "candidate_order": list(CANDIDATE_ORDER),
        "oracle_order": ["violation", "risk_exposure", "length_ratio", "candidate_id"],
        "scenario_counts": dict(scenario_counts),
        "oracle_selection_counts": dict(selection_counts),
        "risk_weight": args.risk_weight,
        "grid_width": args.grid_width,
        "cluster_iou": args.cluster_iou,
        "route_seed": args.route_seed,
        "bootstrap_samples": args.bootstrap_samples,
        "statistical_unit": "frame; route differences averaged within frame",
        "headroom_rule": {
            "violation_reduction_absolute": violation_reduction,
            "exposure_reduction_relative": exposure_reduction_relative,
            "length_ratio_delta": length_delta,
            "gate_pass": headroom_gate,
        },
        "claim_boundary": "Unattainable validation diagnostic; not a deployable method or test result.",
    }
    (output_dir / "headroom_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))
    print(summary[summary["scenario_type"] == "challenge"].to_string(index=False))
    print(paired.to_string(index=False))


if __name__ == "__main__":
    main()
