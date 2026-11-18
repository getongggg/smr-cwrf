from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_planning_benchmark import bootstrap_interval


ROOT = Path(__file__).resolve().parents[1]
FIXED = "fixed_w15"
CWRF = "edrf_a50_g0_w15"
METRICS = ("violation", "risk_exposure", "length_ratio")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compile frozen route-seed, sequence-stratum, and statistical sensitivity evidence."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=ROOT / "results" / "cwrf" / "route_seed_robustness",
    )
    parser.add_argument(
        "--primary-trials",
        type=Path,
        default=ROOT / "results" / "edrf" / "frozen_test" / "planning_trials.csv",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "results" / "paper_tables"
    )
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260728)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def holm_adjust(p_values: list[float]) -> list[float]:
    order = np.argsort(np.asarray(p_values, dtype=float))
    adjusted_sorted: list[float] = []
    running = 0.0
    count = len(p_values)
    for rank, index in enumerate(order):
        value = min(1.0, (count - rank) * float(p_values[int(index)]))
        running = max(running, value)
        adjusted_sorted.append(running)
    adjusted = [0.0] * count
    for index, value in zip(order, adjusted_sorted, strict=True):
        adjusted[int(index)] = value
    return adjusted


def frame_differences(raw: pd.DataFrame) -> pd.DataFrame:
    challenge = raw[raw.scenario_type == "challenge"]
    rows: dict[str, pd.Series] = {}
    for metric in METRICS:
        pivot = challenge.pivot(
            index=["frame_id", "scenario_id"], columns="method", values=metric
        )
        route_difference = pivot[CWRF] - pivot[FIXED]
        rows[metric] = route_difference.groupby(level="frame_id").mean()
    return pd.DataFrame(rows).dropna()


def summarize_seed(directory: Path) -> dict[str, float | int | bool]:
    metadata = json.loads(
        (directory / "benchmark_metadata.json").read_text(encoding="utf-8")
    )
    summary = pd.read_csv(directory / "planning_summary.csv")
    summary = summary[summary.scenario_type == "challenge"].set_index("method")
    paired = pd.read_csv(directory / "paired_statistics.csv")
    paired = paired[
        (paired.baseline == FIXED) & (paired.candidate == CWRF)
    ].set_index("metric")
    fixed = summary.loc[FIXED]
    cwrf = summary.loc[CWRF]
    row: dict[str, float | int | bool] = {
        "route_seed": int(metadata["seed"]),
        "challenge_routes": int(cwrf.trials),
        "challenge_frames": int(paired.loc["violation", "paired_frames"]),
        "fixed_violation": float(fixed.violation_rate),
        "cwrf_violation": float(cwrf.violation_rate),
        "violation_difference": float(
            paired.loc["violation", "candidate_minus_baseline_mean"]
        ),
        "violation_ci_low": float(paired.loc["violation", "bootstrap_95_low"]),
        "violation_ci_high": float(paired.loc["violation", "bootstrap_95_high"]),
        "violation_wilcoxon_p": float(paired.loc["violation", "wilcoxon_p"]),
        "fixed_exposure": float(fixed.exposure_mean),
        "cwrf_exposure": float(cwrf.exposure_mean),
        "exposure_difference": float(
            paired.loc["risk_exposure", "candidate_minus_baseline_mean"]
        ),
        "exposure_ci_low": float(
            paired.loc["risk_exposure", "bootstrap_95_low"]
        ),
        "exposure_ci_high": float(
            paired.loc["risk_exposure", "bootstrap_95_high"]
        ),
        "exposure_wilcoxon_p": float(
            paired.loc["risk_exposure", "wilcoxon_p"]
        ),
        "fixed_length_ratio": float(fixed.length_ratio_mean),
        "cwrf_length_ratio": float(cwrf.length_ratio_mean),
        "length_difference": float(
            paired.loc["length_ratio", "candidate_minus_baseline_mean"]
        ),
    }
    row["directional_gate_pass"] = bool(
        row["violation_difference"] < 0
        and row["exposure_difference"] < 0
        and row["length_difference"] <= 0.02
    )
    return row


def summarize_strata(
    raw: pd.DataFrame, samples: int, seed: int
) -> pd.DataFrame:
    challenge = raw[raw.scenario_type == "challenge"]
    frame_ids = np.asarray(sorted(challenge.frame_id.unique()), dtype=int)
    rows: list[dict[str, float | int | bool]] = []
    for stratum_index, frame_group in enumerate(np.array_split(frame_ids, 3), start=1):
        subset = challenge[challenge.frame_id.isin(frame_group)]
        summary = (
            subset[subset.method.isin([FIXED, CWRF])]
            .groupby("method")
            .agg(
                routes=("scenario_id", "count"),
                violation=("violation", "mean"),
                exposure=("risk_exposure", "mean"),
                length_ratio=("length_ratio", "mean"),
            )
        )
        differences = frame_differences(subset)
        row: dict[str, float | int | bool] = {
            "stratum": stratum_index,
            "frame_id_min": int(frame_group.min()),
            "frame_id_max": int(frame_group.max()),
            "frames": int(len(frame_group)),
            "routes": int(summary.loc[CWRF, "routes"]),
            "fixed_violation": float(summary.loc[FIXED, "violation"]),
            "cwrf_violation": float(summary.loc[CWRF, "violation"]),
            "fixed_exposure": float(summary.loc[FIXED, "exposure"]),
            "cwrf_exposure": float(summary.loc[CWRF, "exposure"]),
            "fixed_length_ratio": float(summary.loc[FIXED, "length_ratio"]),
            "cwrf_length_ratio": float(summary.loc[CWRF, "length_ratio"]),
        }
        for metric in METRICS:
            values = differences[metric].to_numpy(float)
            low, high = bootstrap_interval(
                values, samples=samples, seed=seed + 10 * stratum_index + len(row)
            )
            prefix = "exposure" if metric == "risk_exposure" else metric
            row[f"{prefix}_difference"] = float(values.mean())
            row[f"{prefix}_ci_low"] = low
            row[f"{prefix}_ci_high"] = high
        row["directional_gate_pass"] = bool(
            row["violation_difference"] < 0
            and row["exposure_difference"] < 0
            and row["length_ratio_difference"] <= 0.02
        )
        rows.append(row)
    return pd.DataFrame(rows)


def statistical_sensitivity(
    raw: pd.DataFrame, samples: int, seed: int
) -> dict[str, object]:
    differences = frame_differences(raw)
    primary = pd.read_csv(
        ROOT / "results" / "paper_tables" / "cwrf_primary_paired_statistics.csv"
    )
    primary = primary[
        (primary.baseline == "fixed_w15") & (primary.candidate == "cwrf_w15")
    ].set_index("metric")
    metrics = ["violation", "risk_exposure"]
    p_values = [float(primary.loc[metric, "wilcoxon_p"]) for metric in metrics]
    adjusted = holm_adjust(p_values)
    rng = np.random.default_rng(seed)
    values = differences[["violation", "risk_exposure"]].to_numpy(float)
    joint = 0
    for _ in range(samples):
        indices = rng.integers(0, len(values), size=len(values))
        joint += int(bool(np.all(values[indices].mean(axis=0) < 0)))
    return {
        "statistical_unit": "frame; route differences averaged within frame",
        "bootstrap_samples": samples,
        "bootstrap_seed": seed,
        "holm_tests": [
            {
                "metric": metric,
                "unadjusted_wilcoxon_p": p_value,
                "holm_adjusted_p": adjusted_value,
            }
            for metric, p_value, adjusted_value in zip(
                metrics, p_values, adjusted, strict=True
            )
        ],
        "joint_bootstrap_probability_both_mean_differences_below_zero": joint
        / samples,
        "interpretation_boundary": "Post-primary sensitivity analysis; it does not replace the prespecified unadjusted intervals.",
    }


def main() -> None:
    args = parse_args()
    directories = sorted(
        path for path in args.input_root.glob("seed_*") if path.is_dir()
    )
    if len(directories) != 4:
        raise ValueError(
            f"Expected four frozen route-seed directories, found {len(directories)}"
        )
    seed_summary = pd.DataFrame(
        summarize_seed(path) for path in directories
    ).sort_values("route_seed")
    primary_raw = pd.read_csv(args.primary_trials)
    strata = summarize_strata(primary_raw, args.bootstrap_samples, args.seed)
    sensitivity = statistical_sensitivity(
        primary_raw, args.bootstrap_samples, args.seed
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_summary.to_csv(
        args.output_dir / "cwrf_route_seed_robustness.csv", index=False
    )
    strata.to_csv(
        args.output_dir / "cwrf_sequence_strata_robustness.csv", index=False
    )
    (args.output_dir / "cwrf_statistical_sensitivity.json").write_text(
        json.dumps(sensitivity, indent=2), encoding="utf-8"
    )

    provenance = {
        "artifact_type": "Frozen CWRF route-seed and sequence-stratum robustness",
        "contract": str(
            (ROOT / "RESEARCH_EXTENSION_CONTRACT_20260728.md").resolve()
        ),
        "contract_sha256": sha256(
            ROOT / "RESEARCH_EXTENSION_CONTRACT_20260728.md"
        ),
        "route_seed_directories": [str(path.resolve()) for path in directories],
        "route_seed_metadata_sha256": {
            path.name: sha256(path / "benchmark_metadata.json")
            for path in directories
        },
        "primary_trials": str(args.primary_trials.resolve()),
        "primary_trials_sha256": sha256(args.primary_trials),
        "route_seed_gate_pass": bool(seed_summary.directional_gate_pass.all()),
        "sequence_strata_gate_pass": bool(strata.directional_gate_pass.all()),
        "reporting_boundary": "Parameters were frozen before extension runs; results are robustness evidence, not new confirmatory tests.",
    }
    (args.output_dir / "cwrf_route_seed_robustness_provenance.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )
    print(seed_summary.to_string(index=False))
    print(strata.to_string(index=False))
    print(json.dumps(sensitivity, indent=2))
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
