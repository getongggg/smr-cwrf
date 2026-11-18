from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np

from planning_core import Detection


DEFAULT_CONFIDENCE_EDGES = (0.10, 0.25, 0.50, 0.75, 1.01)


def box_iou(left: Detection, right: Detection) -> float:
    x1 = max(left.x1, right.x1)
    y1 = max(left.y1, right.y1)
    x2 = min(left.x2, right.x2)
    y2 = min(left.y2, right.y2)
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left.x2 - left.x1) * max(0.0, left.y2 - left.y1)
    right_area = max(0.0, right.x2 - right.x1) * max(0.0, right.y2 - right.y1)
    union = left_area + right_area - intersection
    return intersection / union if union > 0.0 else 0.0


def greedy_matches(
    predictions: Iterable[Detection],
    targets: Iterable[Detection],
    minimum_iou: float,
) -> list[tuple[Detection, Detection, float]]:
    predicted = list(predictions)
    truth = list(targets)
    candidates = sorted(
        (
            (box_iou(prediction, target), prediction_index, target_index)
            for prediction_index, prediction in enumerate(predicted)
            for target_index, target in enumerate(truth)
        ),
        reverse=True,
    )
    used_predictions: set[int] = set()
    used_targets: set[int] = set()
    matches = []
    for iou, prediction_index, target_index in candidates:
        if iou < minimum_iou:
            break
        if prediction_index in used_predictions or target_index in used_targets:
            continue
        used_predictions.add(prediction_index)
        used_targets.add(target_index)
        matches.append((predicted[prediction_index], truth[target_index], iou))
    return matches


def required_expansion_ratio(
    prediction: Detection,
    target: Detection,
    safety_fraction: float = 0.10,
) -> float:
    target_scale = max(target.x2 - target.x1, target.y2 - target.y1, 1e-6)
    safety_margin = safety_fraction * target_scale
    target_x1 = target.x1 - safety_margin
    target_y1 = target.y1 - safety_margin
    target_x2 = target.x2 + safety_margin
    target_y2 = target.y2 + safety_margin
    required_margin = max(
        prediction.x1 - target_x1,
        prediction.y1 - target_y1,
        target_x2 - prediction.x2,
        target_y2 - prediction.y2,
        0.0,
    )
    prediction_scale = max(
        prediction.x2 - prediction.x1,
        prediction.y2 - prediction.y1,
        1e-6,
    )
    return required_margin / prediction_scale


def confidence_bin_index(confidence: float, edges: Iterable[float]) -> int:
    boundaries = np.asarray(tuple(edges), dtype=float)
    index = int(np.searchsorted(boundaries, confidence, side="right") - 1)
    return int(np.clip(index, 0, len(boundaries) - 2))


def fit_qcrf_calibration(
    predictions_by_frame: Mapping[int, Iterable[Detection]],
    targets_by_frame: Mapping[int, Iterable[Detection]],
    frame_ids: Iterable[int],
    quantiles: Iterable[float],
    minimum_iou: float = 0.10,
    confidence_edges: Iterable[float] = DEFAULT_CONFIDENCE_EDGES,
    minimum_bin_samples: int = 30,
    maximum_expansion_ratio: float = 1.50,
) -> dict:
    edges = tuple(float(value) for value in confidence_edges)
    scores_by_bin: list[list[float]] = [[] for _ in range(len(edges) - 1)]
    match_ious = []
    frame_id_list = [int(value) for value in frame_ids]
    for frame_id in frame_id_list:
        matches = greedy_matches(
            predictions_by_frame.get(frame_id, []),
            targets_by_frame.get(frame_id, []),
            minimum_iou=minimum_iou,
        )
        for prediction, target, iou in matches:
            bin_index = confidence_bin_index(prediction.confidence, edges)
            scores_by_bin[bin_index].append(
                required_expansion_ratio(prediction, target)
            )
            match_ious.append(iou)

    all_scores = np.asarray(
        [score for values in scores_by_bin for score in values], dtype=float
    )
    if len(all_scores) == 0:
        raise ValueError("No prediction/target matches available for QCRF calibration")

    quantile_rows = {}
    for quantile in quantiles:
        quantile = float(quantile)
        if not 0.0 < quantile < 1.0:
            raise ValueError("Calibration quantiles must be between zero and one")
        global_value = float(np.quantile(all_scores, quantile, method="higher"))
        raw_values = []
        for values in scores_by_bin:
            source = np.asarray(values, dtype=float)
            raw_values.append(
                float(np.quantile(source, quantile, method="higher"))
                if len(source) >= minimum_bin_samples
                else global_value
            )
        monotone_values = raw_values.copy()
        for index in range(len(monotone_values) - 2, -1, -1):
            monotone_values[index] = max(
                monotone_values[index], monotone_values[index + 1]
            )
        monotone_values = [
            float(np.clip(value, 0.0, maximum_expansion_ratio))
            for value in monotone_values
        ]
        key = f"q{int(round(quantile * 100)):02d}"
        quantile_rows[key] = {
            "quantile": quantile,
            "global_expansion_ratio": min(global_value, maximum_expansion_ratio),
            "raw_bin_expansion_ratios": raw_values,
            "monotone_bin_expansion_ratios": monotone_values,
        }

    return {
        "method": "QCRF",
        "score_definition": "smallest symmetric prediction-box expansion, normalized by prediction maximum side, that covers the ground-truth box plus a 10% safety margin",
        "matching": "greedy one-to-one IoU matching",
        "minimum_iou": float(minimum_iou),
        "confidence_edges": list(edges),
        "minimum_bin_samples": int(minimum_bin_samples),
        "maximum_expansion_ratio": float(maximum_expansion_ratio),
        "calibration_frames": len(frame_id_list),
        "matched_boxes": int(len(all_scores)),
        "bin_sample_counts": [len(values) for values in scores_by_bin],
        "match_iou_mean": float(np.mean(match_ious)),
        "score_mean": float(np.mean(all_scores)),
        "score_median": float(np.median(all_scores)),
        "quantiles": quantile_rows,
    }


def inflate_qcrf_boxes(
    boxes: Iterable[Detection], calibration: Mapping, quantile_key: str
) -> list[Detection]:
    edges = calibration["confidence_edges"]
    ratios = calibration["quantiles"][quantile_key][
        "monotone_bin_expansion_ratios"
    ]
    inflated = []
    for box in boxes:
        scale = max(box.x2 - box.x1, box.y2 - box.y1, 1e-6)
        ratio = float(ratios[confidence_bin_index(box.confidence, edges)])
        margin = ratio * scale
        inflated.append(
            Detection(
                x1=max(0.0, box.x1 - margin),
                y1=max(0.0, box.y1 - margin),
                x2=min(1.0, box.x2 + margin),
                y2=min(1.0, box.y2 + margin),
                confidence=box.confidence,
            )
        )
    return inflated
