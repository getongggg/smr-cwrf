from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from mr_cwrf_core import CANDIDATE_ORDER
from run_planning_benchmark import bootstrap_interval, summarize
from run_rwsr_cwrf_selector import CachedCandidates


RULES = (
    "mean_relative_regret",
    "mean_scenario_cost",
    "worst_relative_regret",
    "worst_scenario_cost",
)
METRICS = ("violation", "risk_exposure", "length_ratio", "minimum_clearance")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Post-freeze selector-rule ablation on cached candidate paths."
    )
    parser.add_argument("--planning-trials", type=Path, required=True)
    parser.add_argument("--decision-scores", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260730)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def choose(cache: CachedCandidates, rule: str) -> np.ndarray:
    mean_regret = cache.regrets.mean(axis=2)
    if rule == "mean_relative_regret":
        primary = mean_regret
    elif rule == "mean_scenario_cost":
        primary = cache.mean_cost
    elif rule == "worst_relative_regret":
        primary = cache.max_regret
    elif rule == "worst_scenario_cost":
        primary = cache.costs.max(axis=2)
    else:
        raise ValueError(f"unknown selector rule: {rule}")
    candidate_id = np.broadcast_to(np.arange(len(CANDIDATE_ORDER)), primary.shape)
    ranked = np.lexsort(
        (
            candidate_id,
            cache.path_lengths,
            cache.mean_cost,
            mean_regret,
            cache.max_regret,
            primary,
        ),
        axis=1,
    )
    return ranked[:, 0]


def paired(
    cache: CachedCandidates,
    selections: dict[str, np.ndarray],
    samples: int,
    seed: int,
) -> pd.DataFrame:
    challenge = cache.scenario_types == "challenge"
    frames = cache.frames[challenge]
    unique_frames = sorted(set(frames))
    baseline = selections["mean_relative_regret"]
    baseline_metrics = cache.selected_metrics(baseline)[challenge]
    rows = []
    for rule_index, rule in enumerate(RULES):
        if rule == "mean_relative_regret":
            continue
        delta = cache.selected_metrics(selections[rule])[challenge] - baseline_metrics
        for metric_index, metric in enumerate(METRICS):
            frame_delta = np.asarray(
                [delta[frames == frame, metric_index].mean() for frame in unique_frames]
            )
            low, high = bootstrap_interval(
                frame_delta,
                samples,
                seed + 100 * rule_index + metric_index,
            )
            rows.append(
                {
                    "baseline": "mean_relative_regret",
                    "candidate": rule,
                    "metric": metric,
                    "candidate_minus_baseline_mean": float(frame_delta.mean()),
                    "bootstrap_95_low": low,
                    "bootstrap_95_high": high,
                    "paired_frames": len(unique_frames),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    planning = pd.read_csv(args.planning_trials)
    decisions = pd.read_csv(args.decision_scores)
    cache = CachedCandidates(planning, decisions)
    selections = {rule: choose(cache, rule) for rule in RULES}
    trials = pd.concat(
        [cache.selected_trials(selection, method=rule) for rule, selection in selections.items()],
        ignore_index=True,
    )
    summary = summarize(trials)
    statistics = paired(
        cache, selections, args.bootstrap_samples, args.bootstrap_seed
    )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    trials.to_csv(output_dir / "planning_trials.csv", index=False)
    summary.to_csv(output_dir / "planning_summary.csv", index=False)
    statistics.to_csv(output_dir / "paired_statistics.csv", index=False)
    metadata = {
        "artifact_type": "Post-freeze selector-rule mechanism ablation",
        "rules": list(RULES),
        "selection_counts": {
            rule: {
                str(name): int(count)
                for name, count in zip(
                    *np.unique(np.asarray(CANDIDATE_ORDER)[selection], return_counts=True)
                )
            }
            for rule, selection in selections.items()
        },
        "claim_role": (
            "Post-test descriptive mechanism analysis; not method or parameter selection."
        ),
        "planning_trials": str(args.planning_trials.resolve()),
        "planning_trials_sha256": sha256(args.planning_trials),
        "decision_scores": str(args.decision_scores.resolve()),
        "decision_scores_sha256": sha256(args.decision_scores),
    }
    (output_dir / "benchmark_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(summary[summary["scenario_type"] == "challenge"].to_string(index=False))
    print(statistics.to_string(index=False))


if __name__ == "__main__":
    main()
