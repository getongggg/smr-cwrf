from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from mr_cwrf_core import CANDIDATE_ORDER
from run_planning_benchmark import bootstrap_interval, summarize


METRICS = ("violation", "risk_exposure", "length_ratio", "minimum_clearance")
WEIGHT_STEP = 0.05
LAMBDAS = (0.0, 0.1, 0.25, 0.5, 1.0, 2.0)
METHOD = "rwsr"
SMR_METHOD = "mr_l0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Develop or evaluate frozen RWSR-CWRF from cached candidates."
    )
    parser.add_argument("--planning-trials", type=Path, required=True)
    parser.add_argument(
        "--smr-trials",
        type=Path,
        help="Optional SMR-only trials when the candidate cache was archived separately.",
    )
    parser.add_argument("--decision-scores", type=Path, required=True)
    parser.add_argument("--mode", choices=("development", "evaluation"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--contract", type=Path)
    parser.add_argument("--frozen-selection", type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260730)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def frozen_grid() -> list[tuple[tuple[float, float, float], float]]:
    configs = []
    for first_index in range(21):
        for second_index in range(21 - first_index):
            first = first_index * WEIGHT_STEP
            second = second_index * WEIGHT_STEP
            third = 1.0 - first - second
            for lambda_value in LAMBDAS:
                configs.append(((first, second, third), lambda_value))
    return configs


class CachedCandidates:
    def __init__(self, planning: pd.DataFrame, decisions: pd.DataFrame) -> None:
        source_method = str(decisions["method"].iloc[0])
        decisions = decisions[decisions["method"] == source_method].drop_duplicates(
            ["scenario_id", "candidate"]
        )
        self.scenario_ids = decisions["scenario_id"].drop_duplicates().tolist()
        self.scenario_index = {
            scenario_id: index for index, scenario_id in enumerate(self.scenario_ids)
        }
        order_index = {name: index for index, name in enumerate(CANDIDATE_ORDER)}
        row_count = len(self.scenario_ids)
        self.costs = np.full((row_count, len(CANDIDATE_ORDER), 3), np.nan)
        self.path_lengths = np.full((row_count, len(CANDIDATE_ORDER)), np.nan)
        for row in decisions.itertuples(index=False):
            i = self.scenario_index[row.scenario_id]
            j = order_index[row.candidate]
            self.costs[i, j] = (
                row.scenario_cost_seed0,
                row.scenario_cost_seed1,
                row.scenario_cost_seed2,
            )
            self.path_lengths[i, j] = row.path_length
        if not np.isfinite(self.costs).all() or not np.isfinite(self.path_lengths).all():
            raise ValueError("cached decision matrix is incomplete or non-finite")

        minima = self.costs.min(axis=1, keepdims=True)
        self.regrets = (self.costs - minima) / (minima + 1e-9)
        self.max_regret = self.regrets.max(axis=2)
        self.mean_cost = self.costs.mean(axis=2)
        candidates = planning[planning["method"].isin(CANDIDATE_ORDER)].copy()
        duplicate = candidates.duplicated(["scenario_id", "method"]).any()
        if duplicate:
            raise ValueError("candidate planning trials are not unique")
        candidates = candidates.set_index(["scenario_id", "method"], drop=False)
        self.candidate_rows = candidates
        self.metrics = np.full((row_count, len(CANDIDATE_ORDER), len(METRICS)), np.nan)
        self.frames = np.empty(row_count, dtype=object)
        self.scenario_types = np.empty(row_count, dtype=object)
        for scenario_id, i in self.scenario_index.items():
            rows = candidates.loc[scenario_id].loc[list(CANDIDATE_ORDER)]
            self.metrics[i] = rows.loc[:, METRICS].to_numpy(float)
            self.frames[i] = str(rows.iloc[0]["frame_id"])
            self.scenario_types[i] = str(rows.iloc[0]["scenario_type"])
        if not np.isfinite(self.metrics).all():
            raise ValueError("candidate evaluation metrics are incomplete or non-finite")

    def choose(self, weights: tuple[float, float, float], lambda_value: float) -> np.ndarray:
        weighted = self.regrets @ np.asarray(weights, dtype=float)
        score = weighted + lambda_value * self.max_regret
        candidate_id = np.broadcast_to(
            np.arange(len(CANDIDATE_ORDER)), score.shape
        )
        ranked = np.lexsort(
            (
                candidate_id,
                self.path_lengths,
                self.mean_cost,
                weighted,
                self.max_regret,
                score,
            ),
            axis=1,
        )
        return ranked[:, 0]

    def selected_metrics(self, selected: np.ndarray) -> np.ndarray:
        return self.metrics[np.arange(len(selected)), selected]

    def selected_trials(self, selected: np.ndarray, method: str = METHOD) -> pd.DataFrame:
        rows = []
        for i, candidate_index in enumerate(selected):
            scenario_id = self.scenario_ids[i]
            candidate = CANDIDATE_ORDER[int(candidate_index)]
            row = self.candidate_rows.loc[(scenario_id, candidate)].copy()
            row["method"] = method
            row["selected_candidate"] = candidate
            row["runtime_ms"] = np.nan
            rows.append(row)
        return pd.DataFrame(rows).reset_index(drop=True)


def aggregate(metrics: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if not np.any(mask):
        return np.full(len(METRICS), np.nan)
    return metrics[mask].mean(axis=0)


def frame_interval(
    delta: np.ndarray,
    frames: np.ndarray,
    challenge: np.ndarray,
    metric_index: int,
    samples: int,
    seed: int,
) -> tuple[float, float, float]:
    unique_frames = sorted(set(frames[challenge]))
    frame_delta = np.asarray(
        [delta[(frames == frame) & challenge, metric_index].mean() for frame in unique_frames]
    )
    low, high = bootstrap_interval(frame_delta, samples, seed)
    return float(frame_delta.mean()), low, high


def paired_statistics(
    selected_metrics: np.ndarray,
    cache: CachedCandidates,
    planning: pd.DataFrame,
    samples: int,
    seed: int,
) -> pd.DataFrame:
    baseline_rows = planning[planning["method"].isin(("cwrf", SMR_METHOD))].copy()
    baseline_rows = baseline_rows.drop_duplicates(["scenario_id", "method"])
    baseline_rows = baseline_rows.set_index(["scenario_id", "method"])
    challenge = cache.scenario_types == "challenge"
    rows = []
    for baseline_index, baseline in enumerate(("cwrf", SMR_METHOD)):
        baseline_metrics = np.asarray(
            [
                baseline_rows.loc[(scenario_id, baseline), list(METRICS)].to_numpy(float)
                for scenario_id in cache.scenario_ids
            ]
        )
        delta = selected_metrics - baseline_metrics
        for metric_index, metric in enumerate(METRICS):
            mean, low, high = frame_interval(
                delta,
                cache.frames,
                challenge,
                metric_index,
                samples,
                seed + 100 * baseline_index + metric_index,
            )
            rows.append(
                {
                    "metric": metric,
                    "baseline": baseline,
                    "candidate": METHOD,
                    "candidate_minus_baseline_mean": mean,
                    "bootstrap_95_low": low,
                    "bootstrap_95_high": high,
                    "paired_frames": len(set(cache.frames[challenge])),
                }
            )
    return pd.DataFrame(rows)


def point_dominates(candidate: np.ndarray, baseline: np.ndarray) -> bool:
    return bool(np.all(candidate[:3] < baseline[:3]))


def cross_validate_family(
    cache: CachedCandidates,
    configs: list[tuple[tuple[float, float, float], float]],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    challenge = cache.scenario_types == "challenge"
    equal_selection = cache.choose((1 / 3, 1 / 3, 1 / 3), 0.0)
    smr_metrics = cache.selected_metrics(equal_selection)
    unique_frames = np.asarray(sorted(set(cache.frames)), dtype=object)
    rng = np.random.default_rng(20260730)
    permuted = rng.permutation(unique_frames)
    frame_fold = {frame: index % 5 for index, frame in enumerate(permuted)}
    folds = np.asarray([frame_fold[frame] for frame in cache.frames])
    held_selection = equal_selection.copy()
    fold_rows = []
    for fold in range(5):
        train = challenge & (folds != fold)
        test = challenge & (folds == fold)
        smr_train = aggregate(smr_metrics, train)
        eligible = []
        for weights, lambda_value in configs:
            selected = cache.choose(weights, lambda_value)
            values = aggregate(cache.selected_metrics(selected), train)
            if point_dominates(values, smr_train):
                eligible.append((*values[:3], lambda_value, *weights, selected))
        if eligible:
            winner = min(eligible, key=lambda item: item[:7])
            chosen = winner[-1]
            status = "SELECTED"
            weights = tuple(float(value) for value in winner[4:7])
            lambda_value = float(winner[3])
        else:
            chosen = equal_selection
            status = "SMR_FALLBACK"
            weights = (1 / 3, 1 / 3, 1 / 3)
            lambda_value = 0.0
        held_selection[folds == fold] = chosen[folds == fold]
        held_values = aggregate(cache.selected_metrics(chosen), test)
        fold_rows.append(
            {
                "fold": fold,
                "status": status,
                "train_frames": len(set(cache.frames[train])),
                "heldout_frames": len(set(cache.frames[test])),
                "weight_seed0": weights[0],
                "weight_seed1": weights[1],
                "weight_seed2": weights[2],
                "lambda": lambda_value,
                **{f"heldout_{metric}": held_values[i] for i, metric in enumerate(METRICS)},
            }
        )

    held_metrics = cache.selected_metrics(held_selection)
    pooled = aggregate(held_metrics, challenge)
    cwrf = aggregate(cache.metrics[:, CANDIDATE_ORDER.index("cwrf")], challenge)
    gate = bool(
        pooled[0] < cwrf[0]
        and pooled[1] < cwrf[1]
        and pooled[2] <= cwrf[2] + 0.005
    )
    trials = cache.selected_trials(held_selection, method="rwsr_cross_validated")
    summary = {
        "gate_pass": gate,
        "pooled_heldout": {metric: float(pooled[i]) for i, metric in enumerate(METRICS)},
        "cwrf": {metric: float(cwrf[i]) for i, metric in enumerate(METRICS)},
        "split_rule": (
            "default_rng(20260730) permutation of sorted frame IDs, round-robin five folds"
        ),
    }
    return pd.DataFrame(fold_rows), trials, summary


def development(
    args: argparse.Namespace,
    planning: pd.DataFrame,
    cache: CachedCandidates,
    output_dir: Path,
) -> None:
    if args.contract is None or args.frozen_selection is not None or args.smr_trials:
        raise ValueError(
            "development requires --contract and forbids --frozen-selection/--smr-trials"
        )
    configs = frozen_grid()
    challenge = cache.scenario_types == "challenge"
    smr_selection = cache.choose((1 / 3, 1 / 3, 1 / 3), 0.0)
    smr_metrics = cache.selected_metrics(smr_selection)
    smr_values = aggregate(smr_metrics, challenge)
    config_rows = []
    eligible_records = []
    for weights, lambda_value in configs:
        selected = cache.choose(weights, lambda_value)
        selected_metrics = cache.selected_metrics(selected)
        values = aggregate(selected_metrics, challenge)
        record = {
            "weight_seed0": weights[0],
            "weight_seed1": weights[1],
            "weight_seed2": weights[2],
            "lambda": lambda_value,
            **{metric: values[i] for i, metric in enumerate(METRICS)},
            "dominates_smr_primary_resource": point_dominates(values, smr_values),
            "violation_ci_low_vs_smr": np.nan,
            "violation_ci_high_vs_smr": np.nan,
            "exposure_ci_low_vs_smr": np.nan,
            "exposure_ci_high_vs_smr": np.nan,
            "interval_gate": False,
            "eligible": False,
        }
        if record["dominates_smr_primary_resource"]:
            delta = selected_metrics - smr_metrics
            _, violation_low, violation_high = frame_interval(
                delta, cache.frames, challenge, 0, args.bootstrap_samples, args.bootstrap_seed
            )
            _, exposure_low, exposure_high = frame_interval(
                delta,
                cache.frames,
                challenge,
                1,
                args.bootstrap_samples,
                args.bootstrap_seed + 1,
            )
            record.update(
                {
                    "violation_ci_low_vs_smr": violation_low,
                    "violation_ci_high_vs_smr": violation_high,
                    "exposure_ci_low_vs_smr": exposure_low,
                    "exposure_ci_high_vs_smr": exposure_high,
                    "interval_gate": violation_high <= 0.0 or exposure_high <= 0.0,
                }
            )
            record["eligible"] = bool(record["interval_gate"])
        config_rows.append(record)
        if record["eligible"]:
            eligible_records.append((record, selected, selected_metrics))

    fold_table, cross_trials, cross_summary = cross_validate_family(cache, configs)
    winner = min(
        eligible_records,
        key=lambda item: (
            item[0]["violation"],
            item[0]["risk_exposure"],
            item[0]["length_ratio"],
            item[0]["lambda"],
            item[0]["weight_seed0"],
            item[0]["weight_seed1"],
            item[0]["weight_seed2"],
        ),
        default=None,
    )
    status = "SELECTED" if winner is not None and cross_summary["gate_pass"] else "ABANDONED"
    selected_record = winner[0] if status == "SELECTED" else None
    selection = {
        "artifact_type": "Frozen RWSR-CWRF validation selection",
        "status": status,
        "selection_rule": (
            "Strictly lower validation violation, exposure, and length ratio than SMR; "
            "at least one primary frame-bootstrap upper bound <=0; lexicographic safety "
            "selection; five-fold family gate against CWRF."
        ),
        "grid_size": len(configs),
        "point_dominating_configs": sum(
            bool(row["dominates_smr_primary_resource"]) for row in config_rows
        ),
        "eligible_configs": len(eligible_records),
        "selected": selected_record,
        "cross_validation": cross_summary,
        "test_authorization": status == "SELECTED",
        "known_evidence_boundary": (
            "SMR clean/external aggregates were known before this branch; neither was used "
            "for RWSR design or selection."
        ),
        "source_files": {
            str(args.planning_trials.resolve()): sha256(args.planning_trials),
            str(args.decision_scores.resolve()): sha256(args.decision_scores),
            str(args.contract.resolve()): sha256(args.contract),
        },
    }
    pd.DataFrame(config_rows).to_csv(output_dir / "configuration_grid.csv", index=False)
    fold_table.to_csv(output_dir / "cross_validation_folds.csv", index=False)
    cross_trials.to_csv(output_dir / "cross_validation_trials.csv", index=False)
    (output_dir / "selection.json").write_text(
        json.dumps(selection, indent=2), encoding="utf-8"
    )
    if status == "SELECTED":
        selected_trials = cache.selected_trials(winner[1])
        combined = pd.concat([planning, selected_trials], ignore_index=True)
        summary = summarize(combined)
        paired = paired_statistics(
            winner[2], cache, planning, args.bootstrap_samples, args.bootstrap_seed
        )
        selected_trials.to_csv(output_dir / "planning_trials.csv", index=False)
        summary.to_csv(output_dir / "planning_summary.csv", index=False)
        paired.to_csv(output_dir / "paired_statistics.csv", index=False)
    print(json.dumps(selection, indent=2))


def evaluation(
    args: argparse.Namespace,
    planning: pd.DataFrame,
    cache: CachedCandidates,
    output_dir: Path,
) -> None:
    if args.frozen_selection is None or args.contract is not None:
        raise ValueError("evaluation requires --frozen-selection and forbids --contract")
    selection = json.loads(args.frozen_selection.read_text(encoding="utf-8"))
    if selection.get("status") != "SELECTED" or not selection.get("test_authorization"):
        raise ValueError("frozen selection does not authorize evaluation")
    selected = selection["selected"]
    weights = (
        float(selected["weight_seed0"]),
        float(selected["weight_seed1"]),
        float(selected["weight_seed2"]),
    )
    lambda_value = float(selected["lambda"])
    chosen = cache.choose(weights, lambda_value)
    selected_metrics = cache.selected_metrics(chosen)
    selected_trials = cache.selected_trials(chosen)
    combined = pd.concat([planning, selected_trials], ignore_index=True)
    summary = summarize(combined)
    paired = paired_statistics(
        selected_metrics, cache, planning, args.bootstrap_samples, args.bootstrap_seed
    )
    counts = Counter(CANDIDATE_ORDER[int(index)] for index in chosen)
    metadata = {
        "artifact_type": "Post-freeze RWSR-CWRF cached-candidate evaluation",
        "weights": weights,
        "lambda": lambda_value,
        "selection_counts": dict(counts),
        "candidate_metrics_reused_without_new_astar": True,
        "runtime_eligible": False,
        "frozen_selection": str(args.frozen_selection.resolve()),
        "frozen_selection_sha256": sha256(args.frozen_selection),
        "planning_trials": str(args.planning_trials.resolve()),
        "planning_trials_sha256": sha256(args.planning_trials),
        "decision_scores": str(args.decision_scores.resolve()),
        "decision_scores_sha256": sha256(args.decision_scores),
        "separate_smr_trials": (
            str(args.smr_trials.resolve()) if args.smr_trials else None
        ),
        "separate_smr_trials_sha256": (
            sha256(args.smr_trials) if args.smr_trials else None
        ),
    }
    selected_trials.to_csv(output_dir / "planning_trials.csv", index=False)
    summary.to_csv(output_dir / "planning_summary.csv", index=False)
    paired.to_csv(output_dir / "paired_statistics.csv", index=False)
    (output_dir / "benchmark_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))
    print(summary[summary["scenario_type"] == "challenge"].to_string(index=False))
    print(paired.to_string(index=False))


def main() -> None:
    args = parse_args()
    planning = pd.read_csv(args.planning_trials)
    if args.smr_trials is not None:
        smr = pd.read_csv(args.smr_trials)
        smr = smr[smr["method"] == SMR_METHOD].copy()
        if smr.empty:
            raise ValueError("--smr-trials does not contain mr_l0 rows")
        planning = pd.concat(
            [planning[planning["method"] != SMR_METHOD], smr], ignore_index=True
        )
    decisions = pd.read_csv(args.decision_scores)
    cache = CachedCandidates(planning, decisions)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == "development":
        development(args, planning, cache, output_dir)
    else:
        evaluation(args, planning, cache, output_dir)


if __name__ == "__main__":
    main()
