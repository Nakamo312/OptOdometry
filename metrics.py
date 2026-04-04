"""
metrics.py
Метрики для оценки corridor-based boundary pipeline.

Модуль не привязан к конкретной реализации сегментации или трекинга:
он задаёт единый контракт оценки качества для research-ветки.
"""

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

import numpy as np


def _safe_mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values]
    return float(np.mean(vals)) if vals else 0.0


def boundary_position_error_px(pred_u: float, true_u: float) -> float:
    """
    Абсолютная ошибка положения границы в пикселях.
    """
    return float(abs(pred_u - true_u))


def boundary_angle_error_deg(pred_phi: float, true_phi: float) -> float:
    """
    Ошибка угла границы в градусах с учётом периодичности.
    """
    delta = float(pred_phi - true_phi)
    delta = float(np.arctan2(np.sin(delta), np.cos(delta)))
    return float(abs(np.degrees(delta)))


def crossing_timing_error_frames(pred_frame: int, true_frame: int) -> int:
    """
    Ошибка момента пересечения границы в кадрах.
    """
    return int(abs(pred_frame - true_frame))


def precision_recall_f1(tp: int, fp: int, fn: int) -> Dict[str, float]:
    """
    Классические метрики детекции событий/границ.
    """
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2.0 * precision * recall / (precision + recall)
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def relative_pose_error(
    pred_delta: np.ndarray,
    true_delta: np.ndarray,
) -> float:
    """
    Норма ошибки относительного смещения между двумя кадрами.
    """
    pred = np.asarray(pred_delta, dtype=np.float32).reshape(-1)
    true = np.asarray(true_delta, dtype=np.float32).reshape(-1)
    if pred.shape != true.shape:
        raise ValueError(f"Shape mismatch: {pred.shape} vs {true.shape}")
    return float(np.linalg.norm(pred - true))


def absolute_trajectory_error(
    pred_positions: np.ndarray,
    true_positions: np.ndarray,
) -> float:
    """
    Среднеквадратичная ошибка траектории.
    """
    pred = np.asarray(pred_positions, dtype=np.float32)
    true = np.asarray(true_positions, dtype=np.float32)
    if pred.shape != true.shape:
        raise ValueError(f"Shape mismatch: {pred.shape} vs {true.shape}")
    if pred.ndim != 2:
        raise ValueError("Expected shape (N, D)")
    sq = np.sum((pred - true) ** 2, axis=1)
    return float(np.sqrt(np.mean(sq))) if len(sq) > 0 else 0.0


@dataclass
class BoundaryFrameEval:
    frame_number: int
    position_error_px: Optional[float] = None
    angle_error_deg: Optional[float] = None
    timing_error_frames: Optional[int] = None
    matched: bool = False
    metadata: Dict[str, float] = field(default_factory=dict)


@dataclass
class TrackingEval:
    mean_track_lifetime: float = 0.0
    mean_track_hits: float = 0.0
    confirmed_tracks: int = 0
    total_tracks: int = 0


@dataclass
class PipelineMetrics:
    frame_evals: List[BoundaryFrameEval] = field(default_factory=list)
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tracking: TrackingEval = field(default_factory=TrackingEval)
    rpe_values: List[float] = field(default_factory=list)
    ate_value: Optional[float] = None

    def add_frame_eval(self, item: BoundaryFrameEval):
        self.frame_evals.append(item)

    def add_rpe(self, value: float):
        self.rpe_values.append(float(value))

    def summary(self) -> Dict[str, float]:
        pr = precision_recall_f1(self.tp, self.fp, self.fn)
        pos_errors = [
            item.position_error_px for item in self.frame_evals
            if item.position_error_px is not None
        ]
        angle_errors = [
            item.angle_error_deg for item in self.frame_evals
            if item.angle_error_deg is not None
        ]
        timing_errors = [
            float(item.timing_error_frames) for item in self.frame_evals
            if item.timing_error_frames is not None
        ]
        return {
            "boundary_precision": pr["precision"],
            "boundary_recall": pr["recall"],
            "boundary_f1": pr["f1"],
            "boundary_position_error_px": _safe_mean(pos_errors),
            "boundary_angle_error_deg": _safe_mean(angle_errors),
            "crossing_timing_error_frames": _safe_mean(timing_errors),
            "track_lifetime": float(self.tracking.mean_track_lifetime),
            "track_hits": float(self.tracking.mean_track_hits),
            "confirmed_tracks": float(self.tracking.confirmed_tracks),
            "total_tracks": float(self.tracking.total_tracks),
            "rpe": _safe_mean(self.rpe_values),
            "ate": float(self.ate_value) if self.ate_value is not None else 0.0,
        }
