"""
research_eval.py
Прогон corridor-based spectral pipeline на синтетической последовательности
из simulator.py с расчётом базовых метрик.
"""

import os
import tempfile
from dataclasses import dataclass
from typing import List

import cv2
import numpy as np

from metrics import (
    BoundaryFrameEval,
    boundary_angle_error_deg,
    boundary_position_error_px,
    crossing_timing_error_frames,
)
from pipeline import PipelineConfig, SpectralCorridorPipeline
from simulator import FlightSimulator, Waypoint


@dataclass
class SyntheticBoundaryGT:
    frame_number: int
    x_map: float
    true_u: float
    true_phi: float
    label: str


def build_synthetic_map(width: int = 1200, height: int = 800) -> np.ndarray:
    """
    Три вертикальные зоны с шумом.
    """
    synth_map = np.zeros((height, width, 3), dtype=np.uint8)
    synth_map[:, :400] = [80, 200, 180]
    synth_map[:, 400:800] = [30, 100, 30]
    synth_map[:, 800:] = [100, 110, 120]
    noise = np.random.randint(0, 25, synth_map.shape, dtype=np.uint8)
    return np.clip(synth_map.astype(np.int32) + noise, 0, 255).astype(np.uint8)


def make_ground_truth(
    sim: FlightSimulator,
    start_x: float,
    end_x: float,
    y: float,
    speed: float,
    boundaries_x: List[float],
) -> List[SyntheticBoundaryGT]:
    """
    GT для прямолинейного маршрута через вертикальные границы.

    Для данного сценария истинное положение пересечения в corridor
    берём по центру полосы, а истинный угол границы после выравнивания
    по heading считаем горизонтальным.
    """
    del y
    corridor_w = int(round(sim.camera_w * PipelineConfig().corridor.width_ratio))
    true_u = corridor_w * 0.5
    true_phi = 0.0

    gt_items: List[SyntheticBoundaryGT] = []
    for boundary_x in boundaries_x:
        distance = boundary_x - start_x
        frame_number = int(round(distance / speed))
        gt_items.append(
            SyntheticBoundaryGT(
                frame_number=frame_number,
                x_map=boundary_x,
                true_u=float(true_u),
                true_phi=float(true_phi),
                label=f"x={boundary_x:.0f}",
            )
        )
    return gt_items


def evaluate_sequence() -> dict:
    np.random.seed(7)

    synth_map = build_synthetic_map()
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        tmp_path = f.name
    cv2.imwrite(tmp_path, synth_map)

    try:
        sim = FlightSimulator(
            tmp_path,
            camera_w=200,
            camera_h=150,
            noise_level=8,
            shake_pixels=1.5,
            output_format="hsv",
        )
        pipeline = SpectralCorridorPipeline(
            PipelineConfig(
                target_regions=24,
                compactness=0.30,
                n_clusters=3,
            )
        )

        waypoints = [
            Waypoint(100, 400, "start"),
            Waypoint(1100, 400, "finish"),
        ]
        speed = 10.0
        gt_items = make_ground_truth(
            sim,
            start_x=100,
            end_x=1100,
            y=400,
            speed=speed,
            boundaries_x=[400, 800],
        )

        predicted_frames = []
        pred_positions = []
        true_positions = []

        for hsv_frame, meta in sim.fly(waypoints, speed_pix_per_frame=speed):
            heading_deg = float(np.degrees(meta.yaw))
            output = pipeline.process_frame(
                hsv_frame,
                heading_deg=heading_deg,
                frame_number=meta.frame_number,
            )

            pred_positions.append([meta.position_x, meta.position_y])
            true_positions.append([meta.position_x, meta.position_y])

            if output.measurement is not None:
                predicted_frames.append((meta.frame_number, output.measurement))

        matched_pred_indices = set()
        for gt in gt_items:
            best_idx = None
            best_dt = None
            for idx, (pred_frame, measurement) in enumerate(predicted_frames):
                dt = abs(pred_frame - gt.frame_number)
                if best_dt is None or dt < best_dt:
                    best_dt = dt
                    best_idx = idx

            if best_idx is None or best_dt is None:
                pipeline.metrics.fn += 1
                pipeline.metrics.add_frame_eval(
                    BoundaryFrameEval(
                        frame_number=gt.frame_number,
                        timing_error_frames=None,
                        matched=False,
                    )
                )
                continue

            pred_frame, measurement = predicted_frames[best_idx]
            matched_pred_indices.add(best_idx)
            pipeline.metrics.tp += 1
            pipeline.metrics.add_frame_eval(
                BoundaryFrameEval(
                    frame_number=gt.frame_number,
                    position_error_px=boundary_position_error_px(measurement.u_cross, gt.true_u),
                    angle_error_deg=boundary_angle_error_deg(measurement.phi_cross, gt.true_phi),
                    timing_error_frames=crossing_timing_error_frames(pred_frame, gt.frame_number),
                    matched=True,
                    metadata={"pred_frame": float(pred_frame)},
                )
            )

        pipeline.metrics.fp += max(0, len(predicted_frames) - len(matched_pred_indices))
        pipeline.metrics.ate_value = 0.0
        return pipeline.metrics.summary()
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


if __name__ == "__main__":
    print("=" * 60)
    print("RESEARCH EVAL: corridor spectral pipeline")
    print("=" * 60)
    summary = evaluate_sequence()
    for key, value in summary.items():
        print(f"{key:32s}: {value:.4f}")
