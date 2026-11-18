from __future__ import annotations

import heapq
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


@dataclass(frozen=True)
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float


def normalized_boxes_from_yolo(label_path: Path) -> list[Detection]:
    boxes: list[Detection] = []
    text = label_path.read_text(encoding="utf-8").strip()
    if not text:
        return boxes
    for line in text.splitlines():
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"Unexpected YOLO label in {label_path}: {line}")
        _, xc, yc, width, height = map(float, parts)
        boxes.append(
            Detection(
                x1=max(0.0, xc - width / 2.0),
                y1=max(0.0, yc - height / 2.0),
                x2=min(1.0, xc + width / 2.0),
                y2=min(1.0, yc + height / 2.0),
                confidence=1.0,
            )
        )
    return boxes


def scale_boxes(boxes: Iterable[Detection], grid_w: int, grid_h: int) -> list[Detection]:
    return [
        Detection(
            x1=box.x1 * (grid_w - 1),
            y1=box.y1 * (grid_h - 1),
            x2=box.x2 * (grid_w - 1),
            y2=box.y2 * (grid_h - 1),
            confidence=box.confidence,
        )
        for box in boxes
    ]


def _rectangle_distance(
    xx: np.ndarray, yy: np.ndarray, box: Detection
) -> np.ndarray:
    dx = np.maximum(np.maximum(box.x1 - xx, 0.0), xx - box.x2)
    dy = np.maximum(np.maximum(box.y1 - yy, 0.0), yy - box.y2)
    return np.hypot(dx, dy)


def build_risk_field(
    boxes: Iterable[Detection],
    grid_w: int,
    grid_h: int,
    mode: str,
) -> np.ndarray:
    """Build fixed or confidence-adaptive image-plane risk fields.

    The confidence-adaptive field preserves a conservative amplitude floor and
    increases spatial inflation for low-confidence and near-field detections.
    """
    yy, xx = np.mgrid[0:grid_h, 0:grid_w]
    field = np.zeros((grid_h, grid_w), dtype=np.float32)
    for box in boxes:
        width = max(1.0, box.x2 - box.x1)
        height = max(1.0, box.y2 - box.y1)
        scale = max(width, height)
        if mode == "fixed":
            amplitude = 1.0
            sigma = 0.12 * scale + 0.8
        elif mode in {"carf", "urf"}:
            confidence = float(np.clip(box.confidence, 0.0, 1.0))
            amplitude = 0.55 + 0.45 * confidence
            uncertainty = 1.0 + 1.5 * (1.0 - confidence)
            near_field = 0.85 + 0.45 * (box.y2 / max(1.0, grid_h - 1))
            sigma = (0.12 * scale + 0.8) * uncertainty * near_field
        elif mode == "evaluation":
            amplitude = 1.0
            sigma = 0.10 * scale + 0.8
        else:
            raise ValueError(f"Unknown risk-field mode: {mode}")
        distance = _rectangle_distance(xx, yy, box)
        local = amplitude * np.exp(-0.5 * np.square(distance / sigma))
        field = np.maximum(field, local.astype(np.float32))
    return np.clip(field, 0.0, 1.0)


def build_safety_mask(
    boxes: Iterable[Detection], grid_w: int, grid_h: int
) -> np.ndarray:
    mask = np.zeros((grid_h, grid_w), dtype=np.uint8)
    for box in boxes:
        width = max(1.0, box.x2 - box.x1)
        height = max(1.0, box.y2 - box.y1)
        margin = max(1, int(round(0.10 * max(width, height))))
        x1 = max(0, int(math.floor(box.x1)) - margin)
        y1 = max(0, int(math.floor(box.y1)) - margin)
        x2 = min(grid_w - 1, int(math.ceil(box.x2)) + margin)
        y2 = min(grid_h - 1, int(math.ceil(box.y2)) + margin)
        mask[y1 : y2 + 1, x1 : x2 + 1] = 1
    return mask.astype(bool)


def clearance_map(boxes: Iterable[Detection], grid_w: int, grid_h: int) -> np.ndarray:
    core = np.zeros((grid_h, grid_w), dtype=np.uint8)
    for box in boxes:
        x1 = max(0, int(math.floor(box.x1)))
        y1 = max(0, int(math.floor(box.y1)))
        x2 = min(grid_w - 1, int(math.ceil(box.x2)))
        y2 = min(grid_h - 1, int(math.ceil(box.y2)))
        core[y1 : y2 + 1, x1 : x2 + 1] = 1
    if not core.any():
        return np.full((grid_h, grid_w), math.hypot(grid_w, grid_h), dtype=np.float32)
    return cv2.distanceTransform((1 - core).astype(np.uint8), cv2.DIST_L2, 5)


def astar(
    cost_field: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
    risk_weight: float,
) -> tuple[list[tuple[int, int]] | None, float]:
    started = time.perf_counter()
    height, width = cost_field.shape
    sx, sy = start
    gx, gy = goal
    neighbors = [
        (-1, 0, 1.0),
        (1, 0, 1.0),
        (0, -1, 1.0),
        (0, 1, 1.0),
        (-1, -1, math.sqrt(2.0)),
        (1, -1, math.sqrt(2.0)),
        (-1, 1, math.sqrt(2.0)),
        (1, 1, math.sqrt(2.0)),
    ]

    def heuristic(x: int, y: int) -> float:
        return math.hypot(x - gx, y - gy)

    queue = [(heuristic(sx, sy), 0.0, sx, sy)]
    best = {(sx, sy): 0.0}
    parent: dict[tuple[int, int], tuple[int, int]] = {}
    while queue:
        _, current_cost, x, y = heapq.heappop(queue)
        if current_cost > best.get((x, y), math.inf):
            continue
        if (x, y) == (gx, gy):
            path = [(x, y)]
            while (x, y) in parent:
                x, y = parent[(x, y)]
                path.append((x, y))
            path.reverse()
            return path, time.perf_counter() - started
        for dx, dy, step in neighbors:
            nx, ny = x + dx, y + dy
            if not (0 <= nx < width and 0 <= ny < height):
                continue
            local = 0.5 * (float(cost_field[y, x]) + float(cost_field[ny, nx]))
            candidate = current_cost + step * (1.0 + risk_weight * local)
            if candidate < best.get((nx, ny), math.inf):
                best[(nx, ny)] = candidate
                parent[(nx, ny)] = (x, y)
                heapq.heappush(queue, (candidate + heuristic(nx, ny), candidate, nx, ny))
    return None, time.perf_counter() - started


def path_metrics(
    path: list[tuple[int, int]] | None,
    evaluation_field: np.ndarray,
    safety_mask: np.ndarray,
    clearances: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
) -> dict[str, float | int]:
    if not path or len(path) < 2:
        return {
            "success": 0,
            "violation": 1,
            "risk_exposure": math.nan,
            "max_risk": math.nan,
            "path_length": math.nan,
            "length_ratio": math.nan,
            "minimum_clearance": math.nan,
        }
    points = np.asarray(path, dtype=np.int32)
    x = points[:, 0]
    y = points[:, 1]
    step_lengths = np.linalg.norm(np.diff(points.astype(float), axis=0), axis=1)
    risks = evaluation_field[y, x]
    exposure = float(np.sum(0.5 * (risks[:-1] + risks[1:]) * step_lengths))
    length = float(np.sum(step_lengths))
    straight = max(1e-9, math.hypot(goal[0] - start[0], goal[1] - start[1]))
    return {
        "success": 1,
        "violation": int(np.any(safety_mask[y, x])),
        "risk_exposure": exposure,
        "max_risk": float(np.max(risks)),
        "path_length": length,
        "length_ratio": length / straight,
        "minimum_clearance": float(np.min(clearances[y, x])),
    }
