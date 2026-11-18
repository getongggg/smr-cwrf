from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_mr_candidate_headroom import choose_oracle
from mr_cwrf_core import CANDIDATE_ORDER
from rcmr_cwrf_core import selector_components
from run_planning_benchmark import bootstrap_interval
from run_rwsr_cwrf_selector import CachedCandidates, METRICS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Post-freeze SMR mechanism analysis.")
    parser.add_argument("--validation-planning", type=Path, required=True)
    parser.add_argument("--validation-decisions", type=Path, required=True)
    parser.add_argument("--test-planning", type=Path, required=True)
    parser.add_argument("--test-decisions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260803)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def selected_metrics(cache: CachedCandidates, method: str) -> tuple[np.ndarray, np.ndarray]:
    if method == "cwrf":
        selected = np.full(len(cache.scenario_ids), CANDIDATE_ORDER.index("cwrf"), dtype=int)
    elif method == "smr":
        selected = cache.choose((1 / 3, 1 / 3, 1 / 3), 0.0)
    else:
        raise ValueError(method)
    return selected, cache.selected_metrics(selected)


def oracle_selection(cache: CachedCandidates) -> np.ndarray:
    selected = []
    for route in range(len(cache.scenario_ids)):
        rows = [
            {
                "method": candidate,
                "success": 1,
                "violation": cache.metrics[route, index, 0],
                "risk_exposure": cache.metrics[route, index, 1],
                "length_ratio": cache.metrics[route, index, 2],
            }
            for index, candidate in enumerate(CANDIDATE_ORDER)
        ]
        selected.append(CANDIDATE_ORDER.index(str(choose_oracle(rows)["method"])))
    return np.asarray(selected, dtype=int)


def frame_interval(
    delta: np.ndarray,
    frames: np.ndarray,
    mask: np.ndarray,
    samples: int,
    seed: int,
) -> tuple[float, float, float, int]:
    unique = sorted(set(frames[mask]))
    values = np.asarray([delta[(frames == frame) & mask].mean() for frame in unique])
    low, high = bootstrap_interval(values, samples, seed)
    return float(values.mean()), low, high, len(values)


def main() -> None:
    args = parse_args()
    validation = CachedCandidates(
        pd.read_csv(args.validation_planning), pd.read_csv(args.validation_decisions)
    )
    test = CachedCandidates(pd.read_csv(args.test_planning), pd.read_csv(args.test_decisions))
    validation_d = selector_components(validation.costs, validation.path_lengths)[-1]
    validation_challenge = validation.scenario_types == "challenge"
    thresholds = np.quantile(validation_d[validation_challenge], [0.25, 0.50, 0.75])

    test_d = selector_components(test.costs, test.path_lengths)[-1]
    challenge = test.scenario_types == "challenge"
    bins = np.digitize(test_d, thresholds, right=True)
    _, cwrf = selected_metrics(test, "cwrf")
    smr_selection, smr = selected_metrics(test, "smr")
    oracle = oracle_selection(test)
    oracle_metrics = test.selected_metrics(oracle)

    disagreement_rows = []
    for bin_index, label in enumerate(("Q1", "Q2", "Q3", "Q4")):
        mask = challenge & (bins == bin_index)
        for metric_index, metric in enumerate(METRICS):
            delta = smr[:, metric_index] - cwrf[:, metric_index]
            mean, low, high, frames = frame_interval(
                delta,
                test.frames,
                mask,
                args.bootstrap_samples,
                args.bootstrap_seed + 100 * bin_index + metric_index,
            )
            disagreement_rows.append({
                "disagreement_bin": label,
                "validation_lower": 0.0 if bin_index == 0 else thresholds[bin_index - 1],
                "validation_upper": thresholds[bin_index] if bin_index < 3 else np.inf,
                "metric": metric,
                "smr_minus_cwrf_frame_mean": mean,
                "bootstrap_95_low": low,
                "bootstrap_95_high": high,
                "routes": int(mask.sum()),
                "frames": frames,
            })

    gap_rows = []
    for metric_index, metric in enumerate(METRICS):
        direction = -1.0 if metric == "minimum_clearance" else 1.0
        components = {
            "candidate_headroom_cwrf_minus_oracle": direction * (cwrf[:, metric_index] - oracle_metrics[:, metric_index]),
            "captured_by_smr_cwrf_minus_smr": direction * (cwrf[:, metric_index] - smr[:, metric_index]),
            "residual_selector_gap_smr_minus_oracle": direction * (smr[:, metric_index] - oracle_metrics[:, metric_index]),
        }
        for component_index, (component, values) in enumerate(components.items()):
            mean, low, high, frames = frame_interval(
                values,
                test.frames,
                challenge,
                args.bootstrap_samples,
                args.bootstrap_seed + 1000 + 100 * metric_index + component_index,
            )
            gap_rows.append({
                "metric": metric,
                "component": component,
                "improvement_positive_mean": mean,
                "bootstrap_95_low": low,
                "bootstrap_95_high": high,
                "routes": int(challenge.sum()),
                "frames": frames,
            })

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    disagreement = pd.DataFrame(disagreement_rows)
    gap = pd.DataFrame(gap_rows)
    disagreement.to_csv(output / "disagreement_mechanism.csv", index=False)
    gap.to_csv(output / "oracle_gap_decomposition.csv", index=False)
    route_table = pd.DataFrame({
        "scenario_id": test.scenario_ids,
        "frame_id": test.frames,
        "scenario_type": test.scenario_types,
        "disagreement": test_d,
        "disagreement_bin": [f"Q{value + 1}" for value in bins],
        "smr_candidate": [CANDIDATE_ORDER[value] for value in smr_selection],
        "oracle_candidate": [CANDIDATE_ORDER[value] for value in oracle],
        "smr_matches_oracle": (smr_selection == oracle).astype(int),
    })
    route_table.to_csv(output / "route_diagnostics.csv", index=False)
    metadata = {
        "artifact_type": "Post-freeze SMR disagreement and oracle-gap mechanism analysis",
        "claim_role": "Descriptive mechanism analysis; not algorithm or threshold selection.",
        "validation_disagreement_quartiles": thresholds.tolist(),
        "test_smr_oracle_match_rate_challenge": float(
            (smr_selection[challenge] == oracle[challenge]).mean()
        ),
        "oracle_definition": "Lexicographic violation, exposure, length ratio, frozen candidate order; ground truth evaluation only.",
        "source_files": {
            str(path.resolve()): sha256(path)
            for path in (
                args.validation_planning,
                args.validation_decisions,
                args.test_planning,
                args.test_decisions,
            )
        },
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    print(disagreement.to_string(index=False))
    print(gap.to_string(index=False))


if __name__ == "__main__":
    main()

