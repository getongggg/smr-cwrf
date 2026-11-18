from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


CANDIDATE_ORDER = (
    "fixed_seed0",
    "fixed_seed1",
    "fixed_seed2",
    "consensus",
    "union",
    "cwrf",
)


@dataclass(frozen=True)
class CandidateDecision:
    candidate: str
    scenario_costs: tuple[float, ...]
    regrets: tuple[float, ...]
    mean_regret: float
    max_regret: float
    mean_scenario_cost: float
    path_length: float
    score: float


@dataclass(frozen=True)
class MrDecision:
    selected_candidate: str
    candidates: tuple[CandidateDecision, ...]


def integrated_path_cost(
    path: Sequence[tuple[int, int]] | None,
    cost_field: np.ndarray,
    risk_weight: float,
) -> tuple[float, float]:
    """Return the exact geometric-plus-risk cost used by planning_core.astar."""
    if not path or len(path) < 2:
        return math.inf, math.inf
    points = np.asarray(path, dtype=np.int32)
    x = points[:, 0]
    y = points[:, 1]
    if (
        np.any(x < 0)
        or np.any(y < 0)
        or np.any(x >= cost_field.shape[1])
        or np.any(y >= cost_field.shape[0])
    ):
        raise ValueError("path contains a point outside the cost field")
    step_lengths = np.linalg.norm(np.diff(points.astype(float), axis=0), axis=1)
    risks = cost_field[y, x].astype(float)
    exposure = float(np.sum(0.5 * (risks[:-1] + risks[1:]) * step_lengths))
    length = float(np.sum(step_lengths))
    return length + risk_weight * exposure, length


def select_minimax_regret(
    paths: Mapping[str, Sequence[tuple[int, int]] | None],
    scenario_fields: Sequence[np.ndarray],
    risk_weight: float,
    lambda_value: float,
    candidate_order: Sequence[str] = CANDIDATE_ORDER,
) -> MrDecision:
    if lambda_value < 0.0:
        raise ValueError("lambda_value must be non-negative")
    if not scenario_fields:
        raise ValueError("at least one scenario field is required")
    if set(paths) != set(candidate_order):
        raise ValueError("paths must exactly match the frozen candidate set")
    shapes = {field.shape for field in scenario_fields}
    if len(shapes) != 1:
        raise ValueError("scenario fields must have identical shapes")

    costs: dict[str, tuple[float, ...]] = {}
    lengths: dict[str, float] = {}
    for candidate in candidate_order:
        candidate_costs = []
        candidate_length = math.inf
        for field in scenario_fields:
            cost, length = integrated_path_cost(paths[candidate], field, risk_weight)
            candidate_costs.append(cost)
            candidate_length = length
        costs[candidate] = tuple(candidate_costs)
        lengths[candidate] = candidate_length

    matrix = np.asarray([costs[candidate] for candidate in candidate_order], dtype=float)
    best = np.min(matrix, axis=0)
    if np.any(~np.isfinite(best)):
        raise ValueError("every perception scenario must have a successful candidate")
    regrets = (matrix - best[None, :]) / (best[None, :] + 1e-9)

    decisions = []
    for index, candidate in enumerate(candidate_order):
        candidate_regrets = regrets[index]
        finite = bool(np.all(np.isfinite(candidate_regrets)))
        mean_regret = float(np.mean(candidate_regrets)) if finite else math.inf
        max_regret = float(np.max(candidate_regrets)) if finite else math.inf
        mean_cost = float(np.mean(matrix[index])) if finite else math.inf
        score = mean_regret + lambda_value * max_regret if finite else math.inf
        decisions.append(
            CandidateDecision(
                candidate=candidate,
                scenario_costs=costs[candidate],
                regrets=tuple(float(value) for value in candidate_regrets),
                mean_regret=mean_regret,
                max_regret=max_regret,
                mean_scenario_cost=mean_cost,
                path_length=lengths[candidate],
                score=score,
            )
        )

    order_index = {candidate: index for index, candidate in enumerate(candidate_order)}
    selected = min(
        decisions,
        key=lambda item: (
            item.score,
            item.max_regret,
            item.mean_regret,
            item.mean_scenario_cost,
            item.path_length,
            order_index[item.candidate],
        ),
    )
    if not math.isfinite(selected.score):
        raise RuntimeError("MR-CWRF has no successful candidate path")
    return MrDecision(selected_candidate=selected.candidate, candidates=tuple(decisions))
