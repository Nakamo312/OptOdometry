"""
simulator.py
Симулятор последовательностей для corridor-based spectral pipeline.

Старый simulator был завязан на detector.py и задачу event detection.
Этот вариант ориентирован на новую архитектуру:
- выдаёт кадры и heading;
- умеет строить синтетическую карту из зон;
- возвращает ground truth по границам для corridor-based оценки.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from dataclasses import dataclass
from typing import Iterator, List, Optional, Sequence, Tuple

import cv2
import numpy as np


@dataclass
class Waypoint:
    x: float
    y: float
    name: str = ""


@dataclass
class FrameMeta:
    frame_number: int
    position_x: float
    position_y: float
    yaw: float
    speed: float
    segment_name: str = ""


@dataclass
class BoundaryGroundTruth:
    frame_number: int
    x_map: float
    label: str
    true_u: float
    true_phi: float


def build_demo_map(width: int = 1200, height: int = 800) -> np.ndarray:
    """
    Синтетическая карта с тремя зонами и лёгкой текстурой.
    """
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:, :400] = [80, 200, 180]
    img[:, 400:800] = [30, 100, 30]
    img[:, 800:] = [100, 110, 120]

    noise = np.random.randint(0, 25, img.shape, dtype=np.uint8)
    return np.clip(img.astype(np.int32) + noise, 0, 255).astype(np.uint8)


def default_demo_route() -> List[Waypoint]:
    return [
        Waypoint(100, 400, "start"),
        Waypoint(1100, 400, "finish"),
    ]


class FlightSimulator:
    def __init__(
        self,
        image_path: Optional[str] = None,
        map_bgr: Optional[np.ndarray] = None,
        camera_w: int = 320,
        camera_h: int = 240,
        fps: float = 30.0,
        noise_level: float = 10.0,
        shake_pixels: float = 2.0,
        output_format: str = "hsv",
    ):
        if map_bgr is None:
            if image_path is None:
                raise ValueError("Either image_path or map_bgr must be provided")
            map_bgr = cv2.imread(image_path)
            if map_bgr is None:
                raise FileNotFoundError(f"Failed to load image: {image_path}")

        self._map_bgr = map_bgr
        self.map_h, self.map_w = self._map_bgr.shape[:2]
        self.camera_w = camera_w
        self.camera_h = camera_h
        self.fps = fps
        self.noise_level = noise_level
        self.shake_pixels = shake_pixels
        self.output_format = output_format.lower()

    @classmethod
    def from_demo_map(
        cls,
        width: int = 1200,
        height: int = 800,
        **kwargs,
    ) -> "FlightSimulator":
        return cls(map_bgr=build_demo_map(width=width, height=height), **kwargs)

    def fly(
        self,
        waypoints: Sequence[Waypoint],
        speed_pix_per_frame: float = 5.0,
    ) -> Iterator[Tuple[np.ndarray, FrameMeta]]:
        if len(waypoints) < 2:
            raise ValueError("Need at least 2 waypoints")

        frame_number = 0
        for seg_idx in range(len(waypoints) - 1):
            wp0 = waypoints[seg_idx]
            wp1 = waypoints[seg_idx + 1]

            dx = wp1.x - wp0.x
            dy = wp1.y - wp0.y
            distance = float(np.hypot(dx, dy))
            yaw = float(np.arctan2(dy, dx))
            n_frames = max(1, int(distance / speed_pix_per_frame))

            for step in range(n_frames):
                t = step / n_frames
                cx = wp0.x + t * dx
                cy = wp0.y + t * dy

                cx += np.random.uniform(-self.shake_pixels, self.shake_pixels)
                cy += np.random.uniform(-self.shake_pixels, self.shake_pixels)

                frame = self._capture_frame(cx, cy)
                yield frame, FrameMeta(
                    frame_number=frame_number,
                    position_x=float(cx),
                    position_y=float(cy),
                    yaw=yaw,
                    speed=float(speed_pix_per_frame),
                    segment_name=f"{wp0.name}->{wp1.name}",
                )
                frame_number += 1

    def _capture_frame(self, cx: float, cy: float) -> np.ndarray:
        x0 = int(cx - self.camera_w // 2)
        y0 = int(cy - self.camera_h // 2)
        x1 = x0 + self.camera_w
        y1 = y0 + self.camera_h

        pad_left = max(0, -x0)
        pad_top = max(0, -y0)
        pad_right = max(0, x1 - self.map_w)
        pad_bottom = max(0, y1 - self.map_h)

        x0c = max(0, x0)
        y0c = max(0, y0)
        x1c = min(self.map_w, x1)
        y1c = min(self.map_h, y1)

        crop = self._map_bgr[y0c:y1c, x0c:x1c]
        if pad_left or pad_top or pad_right or pad_bottom:
            crop = cv2.copyMakeBorder(
                crop,
                pad_top,
                pad_bottom,
                pad_left,
                pad_right,
                cv2.BORDER_REFLECT,
            )

        if self.noise_level > 0:
            noise = np.random.randint(
                -int(self.noise_level),
                int(self.noise_level) + 1,
                crop.shape,
                dtype=np.int16,
            )
            crop = np.clip(crop.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        if self.output_format == "hsv":
            return cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        return crop

    def estimate_vertical_boundary_gt(
        self,
        start_x: float,
        speed_pix_per_frame: float,
        boundaries_x: Sequence[float],
        corridor_width_ratio: float = 0.30,
    ) -> List[BoundaryGroundTruth]:
        """
        GT для demo-сценария с вертикальными границами и горизонтальным движением.
        """
        true_u = float(self.camera_w * corridor_width_ratio * 0.5)
        true_phi = 0.0
        items = []
        for bx in boundaries_x:
            frame_number = int(round((bx - start_x) / speed_pix_per_frame))
            items.append(
                BoundaryGroundTruth(
                    frame_number=frame_number,
                    x_map=float(bx),
                    label=f"x={bx:.0f}",
                    true_u=true_u,
                    true_phi=true_phi,
                )
            )
        return items


def save_demo_map_to_temp() -> str:
    img = build_demo_map()
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        path = f.name
    cv2.imwrite(path, img)
    return path


def parse_args():
    parser = argparse.ArgumentParser(description="Simulator for spectral corridor pipeline")
    parser.add_argument("--image", help="Path to map image")
    parser.add_argument("--demo", action="store_true", help="Use built-in synthetic map")
    parser.add_argument("--speed", type=float, default=10.0)
    parser.add_argument("--cam-w", type=int, default=200)
    parser.add_argument("--cam-h", type=int, default=150)
    parser.add_argument("--noise", type=float, default=8.0)
    parser.add_argument("--shake", type=float, default=1.5)
    parser.add_argument("--dump-meta", help="Optional JSON output for simulated trajectory")
    return parser.parse_args()


def main():
    args = parse_args()
    temp_path = None
    try:
        if args.demo or not args.image:
            temp_path = save_demo_map_to_temp()
            image_path = temp_path
        else:
            image_path = args.image

        sim = FlightSimulator(
            image_path=image_path,
            camera_w=args.cam_w,
            camera_h=args.cam_h,
            noise_level=args.noise,
            shake_pixels=args.shake,
            output_format="hsv",
        )
        route = default_demo_route()
        gt = sim.estimate_vertical_boundary_gt(
            start_x=route[0].x,
            speed_pix_per_frame=args.speed,
            boundaries_x=[400, 800],
        )

        print("=" * 60)
        print("SIMULATOR: spectral corridor pipeline")
        print("=" * 60)
        print(f"map: {sim.map_w}x{sim.map_h}")
        print(f"route: ({route[0].x}, {route[0].y}) -> ({route[1].x}, {route[1].y})")
        print(f"expected GT boundaries: {[item.frame_number for item in gt]}")

        meta_dump = []
        frames = 0
        for _, meta in sim.fly(route, speed_pix_per_frame=args.speed):
            frames += 1
            if len(meta_dump) < 8:
                meta_dump.append(
                    {
                        "frame": meta.frame_number,
                        "x": meta.position_x,
                        "y": meta.position_y,
                        "yaw_deg": float(np.degrees(meta.yaw)),
                    }
                )

        print(f"frames generated: {frames}")
        if args.dump_meta:
            with open(args.dump_meta, "w", encoding="utf-8") as f:
                json.dump({"sample_meta": meta_dump, "gt": [item.__dict__ for item in gt]}, f, indent=2)
            print(f"saved metadata: {args.dump_meta}")
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)


if __name__ == "__main__":
    main()
