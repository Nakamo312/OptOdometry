"""
corridor.py
Выделение узкого corridor of interest вдоль направления движения.

Новая архитектура анализирует не весь кадр, а только информативную
полосу вдоль прогнозируемого направления движения.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass
class MotionCorridorSpec:
    """
    Параметры corridor of interest.
    """

    center_x_ratio: float = 0.5
    width_ratio: float = 0.30
    top_margin_ratio: float = 0.05
    bottom_margin_ratio: float = 0.05
    align_to_heading: bool = True
    heading_deg: float = 0.0
    corridor_id: str = "motion_corridor"


@dataclass
class CorridorROI:
    x0: int
    y0: int
    x1: int
    y1: int
    heading_deg: float
    corridor_id: str = "motion_corridor"

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0


def rotate_frame_to_heading(frame: np.ndarray, heading_deg: float) -> np.ndarray:
    """
    Повернуть кадр так, чтобы направление движения было вверх.

    Принято:
    heading_deg = 0 означает движение вправо в системе кадра.
    """
    h, w = frame.shape[:2]
    angle = -(heading_deg + 90.0)
    matrix = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    return cv2.warpAffine(
        frame,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def extract_motion_corridor(
    frame: np.ndarray,
    spec: MotionCorridorSpec,
    heading_deg: Optional[float] = None,
) -> Tuple[np.ndarray, CorridorROI]:
    """
    Вырезать corridor of interest из кадра.
    """
    used_heading = spec.heading_deg if heading_deg is None else float(heading_deg)
    working = frame
    if spec.align_to_heading:
        working = rotate_frame_to_heading(frame, used_heading)

    h, w = working.shape[:2]
    corridor_w = max(16, int(round(w * spec.width_ratio)))
    cx = int(round(w * spec.center_x_ratio))
    x0 = max(0, cx - corridor_w // 2)
    x1 = min(w, x0 + corridor_w)
    x0 = max(0, x1 - corridor_w)

    y0 = int(round(h * spec.top_margin_ratio))
    y1 = int(round(h * (1.0 - spec.bottom_margin_ratio)))
    y0 = int(np.clip(y0, 0, max(0, h - 2)))
    y1 = int(np.clip(y1, y0 + 1, h))

    corridor = working[y0:y1, x0:x1]
    roi = CorridorROI(
        x0=x0,
        y0=y0,
        x1=x1,
        y1=y1,
        heading_deg=used_heading,
        corridor_id=spec.corridor_id,
    )
    return corridor, roi


if __name__ == "__main__":
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    corridor, roi = extract_motion_corridor(frame, MotionCorridorSpec(), heading_deg=15.0)
    print("=" * 60)
    print("SMOKE TEST: corridor.py")
    print("=" * 60)
    print(f"corridor shape: {corridor.shape}")
    print(f"roi: ({roi.x0}, {roi.y0}) -> ({roi.x1}, {roi.y1})")
