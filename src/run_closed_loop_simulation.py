from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from mr_cwrf_core import CANDIDATE_ORDER, integrated_path_cost
from planning_core import astar
from run_planning_benchmark import bootstrap_interval


GRID_W = 48
GRID_H = 36
ROBOT_RADIUS = 1.0
SAFETY_RADIUS = 1.5
RISK_WEIGHT = 15.0
MAX_REPLANS = 30
EXECUTION_STEPS = 3
METHODS = (
    "detector_1",
    "consensus",
    "union",
    "cwrf",
    "mean_scenario_cost",
    "worst_relative_regret",
    "smr_cwrf",
)


@dataclass(frozen=True)
class Circle:
    x: float
    y: float
    radius: float
    detector: int = -1


@dataclass
class RobotState:
    position: np.ndarray
    collision: bool = False
    violation: bool = False
    success: bool = False
    exposure: float = 0.0
    path_length: float = 0.0
    replans: int = 0
    selected: Counter[str] | None = None

    def __post_init__(self) -> None:
        if self.selected is None:
            self.selected = Counter()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Frozen procedural closed-loop study.")
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def point_clear(point: tuple[int, int], obstacles: list[Circle], margin: float) -> bool:
    x, y = point
    return all(math.hypot(x - item.x, y - item.y) > item.radius + margin for item in obstacles)


def connected(start: tuple[int, int], goal: tuple[int, int], obstacles: list[Circle]) -> bool:
    free = np.ones((GRID_H, GRID_W), dtype=bool)
    yy, xx = np.mgrid[0:GRID_H, 0:GRID_W]
    for item in obstacles:
        free &= np.hypot(xx - item.x, yy - item.y) > item.radius + ROBOT_RADIUS
    queue = deque([start])
    seen = {start}
    while queue:
        x, y = queue.popleft()
        if (x, y) == goal:
            return True
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nx, ny = x + dx, y + dy
            if 0 <= nx < GRID_W and 0 <= ny < GRID_H and free[ny, nx] and (nx, ny) not in seen:
                seen.add((nx, ny))
                queue.append((nx, ny))
    return False


def generate_episode(seed: int, episode: int) -> tuple[list[Circle], tuple[int, int], tuple[int, int]]:
    rng = np.random.default_rng(np.random.SeedSequence([seed, episode, 11]))
    for _ in range(1000):
        obstacles = [
            Circle(
                float(rng.uniform(5.0, GRID_W - 5.0)),
                float(rng.uniform(3.5, GRID_H - 3.5)),
                float(rng.uniform(1.5, 3.5)),
            )
            for _ in range(int(rng.integers(8, 15)))
        ]
        start = (int(rng.integers(1, 5)), int(rng.integers(3, GRID_H - 3)))
        goal = (int(rng.integers(GRID_W - 5, GRID_W - 1)), int(rng.integers(3, GRID_H - 3)))
        if point_clear(start, obstacles, SAFETY_RADIUS) and point_clear(goal, obstacles, SAFETY_RADIUS) and connected(start, goal, obstacles):
            return obstacles, start, goal
    raise RuntimeError(f"could not generate feasible episode {episode}")


def perception_tape(obstacles: list[Circle], seed: int, episode: int, replan: int) -> list[list[Circle]]:
    rng = np.random.default_rng(np.random.SeedSequence([seed, episode, replan, 23]))
    probabilities = (0.82, 0.78, 0.74)
    visibility = rng.beta(6.0, 2.0, len(obstacles))
    outputs: list[list[Circle]] = []
    for detector, probability in enumerate(probabilities):
        detections = []
        for index, item in enumerate(obstacles):
            effective = np.clip(0.35 + 0.75 * visibility[index], 0.0, 1.0) * probability
            if rng.random() <= effective:
                detections.append(
                    Circle(
                        float(np.clip(item.x + rng.normal(0.0, 0.55), 0.0, GRID_W - 1)),
                        float(np.clip(item.y + rng.normal(0.0, 0.55), 0.0, GRID_H - 1)),
                        float(np.clip(item.radius * rng.lognormal(0.0, 0.12), 0.8, 5.0)),
                        detector,
                    )
                )
        for _ in range(int(rng.poisson(0.8))):
            detections.append(
                Circle(
                    float(rng.uniform(1.0, GRID_W - 2.0)),
                    float(rng.uniform(1.0, GRID_H - 2.0)),
                    float(rng.uniform(0.8, 2.5)),
                    detector,
                )
            )
        outputs.append(detections)
    return outputs


def cluster_detections(detections: list[list[Circle]]) -> list[list[Circle]]:
    clusters: list[list[Circle]] = []
    for detector_rows in detections:
        for item in detector_rows:
            matches = []
            for index, cluster in enumerate(clusters):
                if any(member.detector == item.detector for member in cluster):
                    continue
                cx = float(np.mean([member.x for member in cluster]))
                cy = float(np.mean([member.y for member in cluster]))
                radius = float(np.mean([member.radius for member in cluster]))
                threshold = max(2.0, 0.6 * (radius + item.radius))
                distance = math.hypot(item.x - cx, item.y - cy)
                if distance <= threshold:
                    matches.append((distance, index))
            if matches:
                clusters[min(matches)[1]].append(item)
            else:
                clusters.append([item])
    return clusters


def field_from_circles(circles: list[tuple[Circle, float]]) -> np.ndarray:
    yy, xx = np.mgrid[0:GRID_H, 0:GRID_W]
    field = np.zeros((GRID_H, GRID_W), dtype=np.float32)
    for item, amplitude in circles:
        distance = np.maximum(np.hypot(xx - item.x, yy - item.y) - item.radius, 0.0)
        sigma = 0.24 * item.radius + 0.8
        field = np.maximum(field, amplitude * np.exp(-0.5 * np.square(distance / sigma)))
    return np.clip(field, 0.0, 1.0).astype(np.float32)


def build_fields(detections: list[list[Circle]]) -> dict[str, np.ndarray]:
    fields = {
        f"fixed_seed{index}": field_from_circles([(item, 1.0) for item in rows])
        for index, rows in enumerate(detections)
    }
    clusters = cluster_detections(detections)
    fused: list[tuple[Circle, int]] = []
    for cluster in clusters:
        weights = np.asarray([1.0 for _ in cluster])
        fused.append(
            (
                Circle(
                    float(np.average([item.x for item in cluster], weights=weights)),
                    float(np.average([item.y for item in cluster], weights=weights)),
                    float(np.average([item.radius for item in cluster], weights=weights)),
                ),
                len({item.detector for item in cluster}),
            )
        )
    fields["consensus"] = field_from_circles([(item, 1.0) for item, support in fused if support >= 2])
    fields["union"] = field_from_circles([(item, 1.0) for item, _ in fused])
    fields["cwrf"] = field_from_circles([(item, 1.0 if support >= 2 else 0.5) for item, support in fused])
    return fields


def choose_candidate(
    paths: dict[str, list[tuple[int, int]] | None],
    fields: dict[str, np.ndarray],
    rule: str,
) -> str:
    costs = np.asarray([
        [integrated_path_cost(paths[name], fields[f"fixed_seed{scenario}"], RISK_WEIGHT)[0] for scenario in range(3)]
        for name in CANDIDATE_ORDER
    ])
    best = costs.min(axis=0)
    regrets = (costs - best[None, :]) / best[None, :]
    lengths = np.asarray([
        integrated_path_cost(paths[name], fields["fixed_seed0"], 0.0)[1]
        for name in CANDIDATE_ORDER
    ])
    if rule == "mean_scenario_cost":
        primary = costs.mean(axis=1)
    elif rule == "worst_relative_regret":
        primary = regrets.max(axis=1)
    elif rule == "smr_cwrf":
        primary = regrets.mean(axis=1)
    else:
        raise ValueError(rule)
    return min(
        CANDIDATE_ORDER,
        key=lambda name: (
            float(primary[CANDIDATE_ORDER.index(name)]),
            float(regrets[CANDIDATE_ORDER.index(name)].max()),
            float(regrets[CANDIDATE_ORDER.index(name)].mean()),
            float(costs[CANDIDATE_ORDER.index(name)].mean()),
            float(lengths[CANDIDATE_ORDER.index(name)]),
            CANDIDATE_ORDER.index(name),
        ),
    )


def ground_truth_field(obstacles: list[Circle]) -> np.ndarray:
    return field_from_circles([(item, 1.0) for item in obstacles])


def segment_metrics(first: np.ndarray, second: np.ndarray, obstacles: list[Circle], truth: np.ndarray) -> tuple[bool, bool, float, float]:
    distance = float(np.linalg.norm(second - first))
    samples = max(2, int(math.ceil(distance / 0.1)) + 1)
    points = np.linspace(first, second, samples)
    collision = False
    violation = False
    for point in points:
        for item in obstacles:
            radius = math.hypot(float(point[0]) - item.x, float(point[1]) - item.y)
            collision |= radius <= item.radius + ROBOT_RADIUS
            violation |= radius <= item.radius + SAFETY_RADIUS
    x = np.clip(np.rint(points[:, 0]).astype(int), 0, GRID_W - 1)
    y = np.clip(np.rint(points[:, 1]).astype(int), 0, GRID_H - 1)
    exposure = float(np.trapezoid(truth[y, x], dx=distance / max(samples - 1, 1)))
    return collision, violation, exposure, distance


def simulate_episode(seed: int, episode: int) -> list[dict[str, object]]:
    obstacles, start, goal = generate_episode(seed, episode)
    direct = max(math.hypot(goal[0] - start[0], goal[1] - start[1]), 1e-9)
    truth = ground_truth_field(obstacles)
    states = {method: RobotState(np.asarray(start, dtype=float)) for method in METHODS}

    for replan in range(MAX_REPLANS):
        active = [method for method, state in states.items() if not (state.collision or state.success)]
        if not active:
            break
        fields = build_fields(perception_tape(obstacles, seed, episode, replan))
        grouped: dict[tuple[int, int], list[str]] = {}
        for method in active:
            point = tuple(np.clip(np.rint(states[method].position).astype(int), [0, 0], [GRID_W - 1, GRID_H - 1]))
            grouped.setdefault(point, []).append(method)
        for point, group in grouped.items():
            paths = {name: astar(fields[name], point, goal, RISK_WEIGHT)[0] for name in CANDIDATE_ORDER}
            selected_by_rule = {
                rule: choose_candidate(paths, fields, rule)
                for rule in ("mean_scenario_cost", "worst_relative_regret", "smr_cwrf")
            }
            mapping = {
                "detector_1": "fixed_seed0",
                "consensus": "consensus",
                "union": "union",
                "cwrf": "cwrf",
                **selected_by_rule,
            }
            for method in group:
                state = states[method]
                state.replans += 1
                selected = mapping[method]
                state.selected[selected] += 1
                path = paths[selected]
                if not path or len(path) < 2:
                    continue
                for step_index, command in enumerate(path[1 : 1 + EXECUTION_STEPS]):
                    noise_rng = np.random.default_rng(
                        np.random.SeedSequence([seed, episode, replan, step_index, 47])
                    )
                    target = np.asarray(command, dtype=float) + noise_rng.normal(0.0, 0.20, 2)
                    target = np.clip(target, [0.0, 0.0], [GRID_W - 1.0, GRID_H - 1.0])
                    collision, violation, exposure, length = segment_metrics(
                        state.position, target, obstacles, truth
                    )
                    state.exposure += exposure
                    state.path_length += length
                    state.violation |= violation
                    state.position = target
                    if collision:
                        state.collision = True
                        break
                    if np.linalg.norm(state.position - np.asarray(goal, dtype=float)) <= 1.0:
                        state.success = True
                        break

    rows = []
    for method, state in states.items():
        rows.append({
            "episode": episode,
            "method": method,
            "collision": int(state.collision),
            "violation": int(state.violation),
            "success": int(state.success),
            "cumulative_exposure": state.exposure,
            "path_length_ratio": state.path_length / direct,
            "replans": state.replans,
            "selected_candidates": json.dumps(dict(state.selected), sort_keys=True),
        })
    return rows


def summarize(trials: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, group in trials.groupby("method", sort=False):
        rows.append({
            "method": method,
            "episodes": len(group),
            "collision_rate": group["collision"].mean(),
            "violation_rate": group["violation"].mean(),
            "success_rate": group["success"].mean(),
            "exposure_mean": group["cumulative_exposure"].mean(),
            "length_ratio_mean": group["path_length_ratio"].mean(),
            "replans_mean": group["replans"].mean(),
        })
    return pd.DataFrame(rows)


def paired_statistics(trials: pd.DataFrame, samples: int, seed: int) -> pd.DataFrame:
    metrics = (
        "collision",
        "violation",
        "success",
        "cumulative_exposure",
        "path_length_ratio",
        "replans",
    )
    baseline = trials[trials["method"] == "cwrf"].set_index("episode")
    rows = []
    for method_index, method in enumerate(METHODS):
        if method == "cwrf":
            continue
        candidate = trials[trials["method"] == method].set_index("episode")
        common = baseline.index.intersection(candidate.index)
        for metric_index, metric in enumerate(metrics):
            delta = candidate.loc[common, metric].to_numpy(float) - baseline.loc[common, metric].to_numpy(float)
            low, high = bootstrap_interval(delta, samples, seed + 100 * method_index + metric_index)
            rows.append({
                "baseline": "cwrf",
                "candidate": method,
                "metric": metric,
                "candidate_minus_baseline_mean": float(delta.mean()),
                "bootstrap_95_low": low,
                "bootstrap_95_high": high,
                "paired_episodes": len(delta),
            })
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    rows = []
    for episode in range(args.episodes):
        rows.extend(simulate_episode(args.seed, episode))
        if (episode + 1) % 50 == 0:
            print(f"completed {episode + 1}/{args.episodes}", flush=True)
    trials = pd.DataFrame(rows)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    summary = summarize(trials)
    paired = paired_statistics(trials, args.bootstrap_samples, 20260802)
    trials.to_csv(output / "episode_trials.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    paired.to_csv(output / "paired_statistics.csv", index=False)
    metadata = {
        "artifact_type": "Frozen procedural closed-loop simulation",
        "split": args.split,
        "episodes": args.episodes,
        "seed": args.seed,
        "methods": list(METHODS),
        "rcmr_status": "OMITTED_VALIDATION_GATE_FAILED",
        "grid": [GRID_W, GRID_H],
        "robot_radius": ROBOT_RADIUS,
        "safety_radius": SAFETY_RADIUS,
        "execution_noise_sigma": 0.20,
        "execution_steps_per_replan": EXECUTION_STEPS,
        "max_replans": MAX_REPLANS,
        "paired_random_tape": True,
        "contract": str(args.contract.resolve()),
        "contract_sha256": sha256(args.contract),
        "source_sha256": sha256(Path(__file__).resolve()),
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    print(paired.to_string(index=False))


if __name__ == "__main__":
    main()
