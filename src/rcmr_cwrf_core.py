from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mr_cwrf_core import CANDIDATE_ORDER


@dataclass(frozen=True)
class RcmrConfig:
    epsilon_u: float
    epsilon_v: float
    beta: float
    gamma: float
    fallback: str


@dataclass(frozen=True)
class RcmrDiagnostics:
    selected: np.ndarray
    disagreement: np.ndarray
    used_fallback: np.ndarray
    admissible_count: np.ndarray


def selector_components(
    costs: np.ndarray, path_lengths: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return mean regret, max regret, normalized worst cost, length excess, and D."""
    costs = np.asarray(costs, dtype=float)
    path_lengths = np.asarray(path_lengths, dtype=float)
    if costs.ndim != 3 or costs.shape[1:] != (len(CANDIDATE_ORDER), 3):
        raise ValueError("costs must have shape (routes, 6, 3)")
    if path_lengths.shape != costs.shape[:2]:
        raise ValueError("path_lengths must have shape (routes, 6)")
    if not np.isfinite(costs).all() or np.any(costs <= 0.0):
        raise ValueError("costs must be finite and strictly positive")
    if not np.isfinite(path_lengths).all() or np.any(path_lengths <= 0.0):
        raise ValueError("path lengths must be finite and strictly positive")

    scenario_min = costs.min(axis=1, keepdims=True)
    regrets = (costs - scenario_min) / scenario_min
    mean_regret = regrets.mean(axis=2)
    max_regret = regrets.max(axis=2)

    worst_cost = costs.max(axis=2)
    normalized_worst = worst_cost / worst_cost.min(axis=1, keepdims=True) - 1.0
    length_excess = path_lengths / path_lengths.min(axis=1, keepdims=True) - 1.0

    profiles = np.swapaxes(costs / scenario_min, 1, 2)
    pairwise = []
    for first, second in ((0, 1), (0, 2), (1, 2)):
        pairwise.append(np.linalg.norm(profiles[:, first] - profiles[:, second], axis=1))
    disagreement = np.mean(np.stack(pairwise, axis=1), axis=1)
    return mean_regret, max_regret, normalized_worst, length_excess, disagreement


def select_rcmr(
    costs: np.ndarray,
    path_lengths: np.ndarray,
    config: RcmrConfig,
) -> RcmrDiagnostics:
    if config.epsilon_u < 0.0 or config.epsilon_v < 0.0:
        raise ValueError("risk tolerances must be non-negative")
    if config.beta < 0.0 or config.gamma < 0.0:
        raise ValueError("beta and gamma must be non-negative")
    if config.fallback not in {"cwrf", "union"}:
        raise ValueError("fallback must be cwrf or union")

    mean_regret, max_regret, normalized_worst, length_excess, disagreement = (
        selector_components(costs, path_lengths)
    )
    fallback_index = CANDIDATE_ORDER.index(config.fallback)
    selected = np.empty(costs.shape[0], dtype=np.int64)
    used_fallback = disagreement > config.gamma
    admissible_count = np.zeros(costs.shape[0], dtype=np.int64)
    mean_cost = costs.mean(axis=2)

    for route in range(costs.shape[0]):
        if used_fallback[route]:
            selected[route] = fallback_index
            admissible_count[route] = 1
            continue
        admissible = (
            (normalized_worst[route] <= config.epsilon_u + 1e-12)
            & (
                max_regret[route]
                <= max_regret[route].min() + config.epsilon_v + 1e-12
            )
        )
        indices = np.flatnonzero(admissible)
        if not len(indices):
            indices = np.arange(len(CANDIDATE_ORDER))
            primary = normalized_worst[route]
        else:
            primary = mean_regret[route] + config.beta * length_excess[route]
        admissible_count[route] = len(indices)
        selected[route] = min(
            indices,
            key=lambda index: (
                float(primary[index]),
                float(normalized_worst[route, index]),
                float(max_regret[route, index]),
                float(mean_cost[route, index]),
                float(path_lengths[route, index]),
                int(index),
            ),
        )
    return RcmrDiagnostics(
        selected=selected,
        disagreement=disagreement,
        used_fallback=used_fallback,
        admissible_count=admissible_count,
    )

