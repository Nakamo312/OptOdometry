"""
boundaries.py
Извлечение границ между сегментами corridor-графа.
"""

from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np

from measurement import BoundaryCandidate, BoundaryCrossingGeometry


@dataclass
class SegmentBoundary:
    label_a: int
    label_b: int
    mask: np.ndarray
    points: np.ndarray
    candidate: Optional[BoundaryCandidate]
    geometry: Optional[BoundaryCrossingGeometry]
    score: float


def boundary_mask_from_segments(segment_image: np.ndarray) -> np.ndarray:
    """
    Пиксели границ между соседними сегментами.
    """
    h, w = segment_image.shape
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[:, 1:] |= (segment_image[:, 1:] != segment_image[:, :-1]).astype(np.uint8)
    mask[1:, :] |= (segment_image[1:, :] != segment_image[:-1, :]).astype(np.uint8)
    return mask * 255


def boundary_points(mask: np.ndarray) -> np.ndarray:
    ys, xs = np.nonzero(mask > 0)
    if xs.size == 0:
        return np.empty((0, 2), dtype=np.float32)
    return np.column_stack([xs, ys]).astype(np.float32)


def fit_boundary_candidate(
    points: np.ndarray,
    frame_number: int,
    roi_id: str,
    crossing_y: Optional[float] = None,
) -> tuple[Optional[BoundaryCandidate], Optional[BoundaryCrossingGeometry]]:
    if points.shape[0] < 2:
        return None, None

    pts = points[np.argsort(points[:, 1])]
    ys = pts[:, 1]
    xs = pts[:, 0]
    # Для corridor baseline предпочитаем прямую модель.
    coeffs = np.polyfit(ys, xs, deg=1).astype(np.float32)
    sample_y = np.linspace(float(ys.min()), float(ys.max()), num=max(16, len(np.unique(ys))))
    sample_x = np.polyval(coeffs, sample_y).astype(np.float32)
    curve_points = np.column_stack([sample_x, sample_y]).astype(np.float32)

    diffs = np.diff(curve_points, axis=0)
    curve_length = float(np.linalg.norm(diffs, axis=1).sum()) if len(diffs) > 0 else 0.0
    candidate = BoundaryCandidate(
        frame_number=frame_number,
        model_type="line",
        curve_points=curve_points,
        model_params=coeffs,
        score=float(len(points)),
        support_mask_area=int(len(points)),
        curve_length=curve_length,
        roi_id=roi_id,
    )

    geometry = None
    if crossing_y is not None and len(curve_points) >= 2:
        x_cross = float(np.interp(crossing_y, curve_points[:, 1], curve_points[:, 0]))
        idx = int(np.argmin(np.abs(curve_points[:, 1] - crossing_y)))
        lo = max(0, idx - 1)
        hi = min(len(curve_points) - 1, idx + 1)
        tangent = curve_points[hi] - curve_points[lo]
        tangent_norm = max(float(np.linalg.norm(tangent)), 1e-6)
        tangent = tangent / tangent_norm
        geometry = BoundaryCrossingGeometry(
            u_cross=x_cross,
            phi_cross=float(np.arctan2(tangent[1], tangent[0])),
            curvature=0.0,
            tangent=tangent.astype(np.float32),
            point=np.array([x_cross, crossing_y], dtype=np.float32),
        )

    return candidate, geometry


def _crosses_center(geometry: BoundaryCrossingGeometry, width: int, tolerance_ratio: float = 0.35) -> bool:
    center_x = width * 0.5
    tolerance = width * tolerance_ratio
    return abs(geometry.u_cross - center_x) <= tolerance


def _line_verticality(geometry: BoundaryCrossingGeometry) -> float:
    """
    1.0 = вертикальная граница, 0.0 = горизонтальная.
    Для synthetic demo именно это и нужно.
    """
    tangent = geometry.tangent.astype(np.float32)
    return float(abs(tangent[1]))


def extract_segment_boundaries(
    segment_image: np.ndarray,
    frame_number: int,
    roi_id: str = "motion_corridor",
    crossing_y: Optional[float] = None,
) -> List[SegmentBoundary]:
    mask = boundary_mask_from_segments(segment_image)
    kernel = np.ones((3, 3), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    boundaries: List[SegmentBoundary] = []
    h, w = segment_image.shape
    for component_id in range(1, num_labels):
        area = int(stats[component_id, cv2.CC_STAT_AREA])
        if area < max(12, int(0.10 * h)):
            continue
        comp_mask = np.where(labels == component_id, 255, 0).astype(np.uint8)
        pts = boundary_points(comp_mask)
        candidate, geometry = fit_boundary_candidate(
            pts,
            frame_number=frame_number,
            roi_id=roi_id,
            crossing_y=crossing_y,
        )
        if candidate is None or geometry is None:
            continue

        ys = pts[:, 1] if len(pts) else np.array([], dtype=np.float32)
        vertical_coverage = 0.0 if ys.size == 0 else float((ys.max() - ys.min()) / max(h, 1))
        if vertical_coverage < 0.45:
            continue
        if not _crosses_center(geometry, width=w):
            continue

        straight_score = _line_verticality(geometry)
        total_score = (
            1.2 * float(area)
            + 120.0 * vertical_coverage
            + 80.0 * straight_score
            - 2.0 * abs(geometry.u_cross - w * 0.5)
        )
        boundaries.append(
            SegmentBoundary(
                label_a=-1,
                label_b=-1,
                mask=comp_mask,
                points=pts,
                candidate=candidate,
                geometry=geometry,
                score=float(total_score),
            )
        )
    boundaries.sort(key=lambda item: item.score, reverse=True)
    return boundaries[:3]
