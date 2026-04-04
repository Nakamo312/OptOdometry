"""
measurement.py
Универсальные структуры данных для boundary-aware пайплайна.

Текущий detector.py работает на уровне события перехода между зонами.
Этот модуль вводит следующий уровень абстракции: описание самой границы,
её локальной геометрии и итогового измерения для навигации.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

import numpy as np


BoundaryModelType = Literal["line", "poly2", "poly3", "spline", "polyline", "unknown"]


@dataclass
class BoundaryCrossingGeometry:
    """
    Локальная геометрия границы в точке пересечения с контрольной осью ROI.

    Параметры
    ----------
    u_cross :
        Поперечная координата пересечения в системе ROI.
    phi_cross :
        Угол касательной к границе в точке пересечения, рад.
    curvature :
        Локальная кривизна. Для прямой обычно равна 0.
    tangent: np.ndarray
        Единичный касательный вектор в точке пересечения.
    point: np.ndarray
        2D-точка пересечения в координатах ROI.
    """

    u_cross: float
    phi_cross: float
    curvature: float = 0.0
    tangent: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0], dtype=np.float32))
    point: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))

    def __post_init__(self):
        self.tangent = np.asarray(self.tangent, dtype=np.float32).reshape(2)
        self.point = np.asarray(self.point, dtype=np.float32).reshape(2)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "u_cross": float(self.u_cross),
            "phi_cross": float(self.phi_cross),
            "curvature": float(self.curvature),
            "tangent": self.tangent.tolist(),
            "point": self.point.tolist(),
        }


@dataclass
class BoundaryCandidate:
    """
    Сырая гипотеза границы из одного кадра.

    curve_points и model_params описывают найденную кривую максимально
    универсально, чтобы не ограничиваться прямой первого порядка.
    """

    frame_number: int
    model_type: BoundaryModelType = "unknown"
    curve_points: np.ndarray = field(default_factory=lambda: np.empty((0, 2), dtype=np.float32))
    model_params: np.ndarray = field(default_factory=lambda: np.empty((0,), dtype=np.float32))
    score: float = 0.0
    support_mask_area: int = 0
    curve_length: float = 0.0
    class_old: str = "unknown"
    class_new: str = "unknown"
    roi_id: str = "default"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        points = np.asarray(self.curve_points, dtype=np.float32)
        if points.size == 0:
            self.curve_points = np.empty((0, 2), dtype=np.float32)
        else:
            self.curve_points = points.reshape(-1, 2)
        self.model_params = np.asarray(self.model_params, dtype=np.float32).reshape(-1)

    def is_valid(self, min_points: int = 2) -> bool:
        return (
            self.score > 0.0
            and self.curve_points.shape[0] >= min_points
            and self.curve_length >= 0.0
        )

    def as_vector(self) -> np.ndarray:
        """
        Компактный вектор для грубого сравнения кандидатов между кадрами.
        """
        if self.curve_points.shape[0] == 0:
            center = np.zeros(2, dtype=np.float32)
        else:
            center = self.curve_points.mean(axis=0).astype(np.float32)
        return np.array([
            center[0],
            center[1],
            float(self.score),
            float(self.curve_length),
            float(self.support_mask_area),
        ], dtype=np.float32)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame_number": int(self.frame_number),
            "model_type": self.model_type,
            "curve_points": self.curve_points.tolist(),
            "model_params": self.model_params.tolist(),
            "score": float(self.score),
            "support_mask_area": int(self.support_mask_area),
            "curve_length": float(self.curve_length),
            "class_old": self.class_old,
            "class_new": self.class_new,
            "roi_id": self.roi_id,
            "metadata": dict(self.metadata),
        }


@dataclass
class TrackedBoundary:
    """
    Устойчивая граница, связанная между несколькими кадрами.

    state_params — сглаженное представление модели, а
    crossing_geometry — локальная геометрия в рабочей точке пересечения.
    """

    track_id: int
    model_type: BoundaryModelType = "unknown"
    state_params: np.ndarray = field(default_factory=lambda: np.empty((0,), dtype=np.float32))
    crossing_geometry: Optional[BoundaryCrossingGeometry] = None
    age: int = 0
    hits: int = 0
    misses: int = 0
    last_frame: int = -1
    mean_score: float = 0.0
    class_old: str = "unknown"
    class_new: str = "unknown"
    roi_id: str = "default"
    history: List[BoundaryCandidate] = field(default_factory=list)

    def __post_init__(self):
        self.state_params = np.asarray(self.state_params, dtype=np.float32).reshape(-1)

    def is_confirmed(self, min_hits: int = 3) -> bool:
        return self.hits >= min_hits and self.misses <= max(1, self.hits // 2)

    def append_candidate(self, candidate: BoundaryCandidate):
        self.history.append(candidate)
        self.last_frame = candidate.frame_number
        self.age += 1
        self.hits += 1
        if self.hits == 1:
            self.mean_score = candidate.score
        else:
            self.mean_score = (
                (self.mean_score * (self.hits - 1)) + candidate.score
            ) / self.hits
        self.class_old = candidate.class_old
        self.class_new = candidate.class_new
        self.roi_id = candidate.roi_id
        if candidate.model_params.size > 0:
            self.state_params = candidate.model_params.copy()

    def as_vector(self) -> np.ndarray:
        if self.crossing_geometry is None:
            return np.array([
                float(self.mean_score),
                float(self.age),
                float(self.hits),
                float(self.misses),
            ], dtype=np.float32)
        return np.array([
            float(self.crossing_geometry.u_cross),
            float(self.crossing_geometry.phi_cross),
            float(self.crossing_geometry.curvature),
            float(self.mean_score),
        ], dtype=np.float32)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "track_id": int(self.track_id),
            "model_type": self.model_type,
            "state_params": self.state_params.tolist(),
            "crossing_geometry": (
                None if self.crossing_geometry is None
                else self.crossing_geometry.to_dict()
            ),
            "age": int(self.age),
            "hits": int(self.hits),
            "misses": int(self.misses),
            "last_frame": int(self.last_frame),
            "mean_score": float(self.mean_score),
            "class_old": self.class_old,
            "class_new": self.class_new,
            "roi_id": self.roi_id,
            "history_size": len(self.history),
        }


@dataclass
class BoundaryMeasurement:
    """
    Финальное измерение для будущего map matching / фильтра состояния.

    s_cross может оставаться None до появления связки с маршрутом, картой
    или внешней локальной одометрией.
    """

    frame_number: int
    geometry: BoundaryCrossingGeometry
    confidence: float
    class_old: str = "unknown"
    class_new: str = "unknown"
    covariance: Optional[np.ndarray] = None
    s_cross: Optional[float] = None
    source_track_id: Optional[int] = None
    event_confirmed: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.confidence = float(np.clip(self.confidence, 0.0, 1.0))
        if self.covariance is not None:
            cov = np.asarray(self.covariance, dtype=np.float32)
            self.covariance = cov.reshape(3, 3)

    @property
    def u_cross(self) -> float:
        return float(self.geometry.u_cross)

    @property
    def phi_cross(self) -> float:
        return float(self.geometry.phi_cross)

    @property
    def curvature(self) -> float:
        return float(self.geometry.curvature)

    def is_valid(self) -> bool:
        return (
            np.isfinite(self.u_cross)
            and np.isfinite(self.phi_cross)
            and np.isfinite(self.curvature)
            and 0.0 <= self.confidence <= 1.0
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame_number": int(self.frame_number),
            "geometry": self.geometry.to_dict(),
            "confidence": float(self.confidence),
            "class_old": self.class_old,
            "class_new": self.class_new,
            "covariance": None if self.covariance is None else self.covariance.tolist(),
            "s_cross": None if self.s_cross is None else float(self.s_cross),
            "source_track_id": (
                None if self.source_track_id is None else int(self.source_track_id)
            ),
            "event_confirmed": bool(self.event_confirmed),
            "metadata": dict(self.metadata),
        }


def default_measurement_covariance(
    sigma_u: float = 3.0,
    sigma_phi: float = 0.2,
    sigma_kappa: float = 0.05
) -> np.ndarray:
    """
    Диагональная стартовая ковариация для локальной геометрии границы.
    """
    return np.diag([
        float(sigma_u) ** 2,
        float(sigma_phi) ** 2,
        float(sigma_kappa) ** 2,
    ]).astype(np.float32)
