from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from planning_core import Detection, _rectangle_distance
from qcrf_core import box_iou


@dataclass(frozen=True)
class EnsembleDetection:
    box: Detection
    support: int
    edge_disagreement_ratio: float
    member_confidences: tuple[float, ...]


def _fused_box(members: list[tuple[int, Detection]]) -> Detection:
    weights = np.asarray([max(item.confidence, 1e-3) for _, item in members], dtype=float)
    coordinates = np.asarray(
        [[item.x1, item.y1, item.x2, item.y2] for _, item in members], dtype=float
    )
    fused = np.average(coordinates, axis=0, weights=weights)
    return Detection(
        x1=float(fused[0]),
        y1=float(fused[1]),
        x2=float(fused[2]),
        y2=float(fused[3]),
        confidence=float(np.mean([item.confidence for _, item in members])),
    )


def cluster_ensemble_detections(
    predictions_by_model: Iterable[Iterable[Detection]],
    iou_threshold: float = 0.50,
) -> list[EnsembleDetection]:
    indexed = [
        (model_index, detection)
        for model_index, detections in enumerate(predictions_by_model)
        for detection in detections
    ]
    indexed.sort(key=lambda item: item[1].confidence, reverse=True)
    clusters: list[list[tuple[int, Detection]]] = []
    for model_index, detection in indexed:
        best_index = None
        best_iou = iou_threshold
        for cluster_index, cluster in enumerate(clusters):
            if any(member_model == model_index for member_model, _ in cluster):
                continue
            overlap = box_iou(detection, _fused_box(cluster))
            if overlap >= best_iou:
                best_iou = overlap
                best_index = cluster_index
        if best_index is None:
            clusters.append([(model_index, detection)])
        else:
            clusters[best_index].append((model_index, detection))

    output = []
    for cluster in clusters:
        fused = _fused_box(cluster)
        coordinates = np.asarray(
            [[item.x1, item.y1, item.x2, item.y2] for _, item in cluster], dtype=float
        )
        scale = max(fused.x2 - fused.x1, fused.y2 - fused.y1, 1e-6)
        edge_disagreement = (
            float(np.max(np.std(coordinates, axis=0))) / scale
            if len(cluster) > 1
            else 0.0
        )
        output.append(
            EnsembleDetection(
                box=fused,
                support=len({model_index for model_index, _ in cluster}),
                edge_disagreement_ratio=edge_disagreement,
                member_confidences=tuple(item.confidence for _, item in cluster),
            )
        )
    return output


def scale_ensemble_detections(
    detections: Iterable[EnsembleDetection], grid_w: int, grid_h: int
) -> list[EnsembleDetection]:
    output = []
    for detection in detections:
        box = detection.box
        output.append(
            EnsembleDetection(
                box=Detection(
                    x1=box.x1 * (grid_w - 1),
                    y1=box.y1 * (grid_h - 1),
                    x2=box.x2 * (grid_w - 1),
                    y2=box.y2 * (grid_h - 1),
                    confidence=box.confidence,
                ),
                support=detection.support,
                edge_disagreement_ratio=detection.edge_disagreement_ratio,
                member_confidences=detection.member_confidences,
            )
        )
    return output


def build_edrf_field(
    detections: Iterable[EnsembleDetection],
    grid_w: int,
    grid_h: int,
    singleton_amplitude: float,
    disagreement_gain: float,
    minimum_support: int = 1,
) -> np.ndarray:
    if not 0.0 <= singleton_amplitude <= 1.0:
        raise ValueError("singleton_amplitude must be in [0, 1]")
    if disagreement_gain < 0.0:
        raise ValueError("disagreement_gain must be non-negative")
    yy, xx = np.mgrid[0:grid_h, 0:grid_w]
    field = np.zeros((grid_h, grid_w), dtype=np.float32)
    for detection in detections:
        if detection.support < minimum_support:
            continue
        box = detection.box
        scale = max(1.0, box.x2 - box.x1, box.y2 - box.y1)
        amplitude = 1.0 if detection.support >= 2 else singleton_amplitude
        sigma = (0.12 * scale + 0.8) * (
            1.0 + disagreement_gain * detection.edge_disagreement_ratio
        )
        distance = _rectangle_distance(xx, yy, box)
        local = amplitude * np.exp(-0.5 * np.square(distance / sigma))
        field = np.maximum(field, local.astype(np.float32))
    return np.clip(field, 0.0, 1.0)
