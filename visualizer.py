"""
visualizer.py
Десктопный визуализатор corridor-based spectral pipeline.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont, QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSlider,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from corridor import rotate_frame_to_heading
from pipeline import PipelineConfig, SpectralCorridorPipeline
from research_eval import build_synthetic_map
from simulator import FlightSimulator, default_demo_route


def np_to_pixmap(image_bgr: np.ndarray, target_width: int = 420) -> QPixmap:
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    qimg = QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888)
    pix = QPixmap.fromImage(qimg)
    return pix.scaledToWidth(target_width, Qt.SmoothTransformation)


def colorize_segments(segment_image: np.ndarray) -> np.ndarray:
    palette = np.array(
        [
            [50, 180, 255],
            [80, 220, 120],
            [240, 180, 60],
            [180, 90, 255],
            [255, 110, 110],
        ],
        dtype=np.uint8,
    )
    color = np.zeros((*segment_image.shape, 3), dtype=np.uint8)
    for idx in np.unique(segment_image):
        color[segment_image == idx] = palette[idx % len(palette)]
    return color


def draw_boundaries(base_bgr: np.ndarray, output) -> np.ndarray:
    img = base_bgr.copy()
    for boundary in output.boundaries:
        pts = boundary.points.astype(np.int32)
        if len(pts) > 1:
            for i in range(len(pts) - 1):
                p0 = tuple(pts[i])
                p1 = tuple(pts[i + 1])
                cv2.line(img, p0, p1, (0, 255, 255), 1, cv2.LINE_AA)
        if boundary.geometry is not None:
            px, py = boundary.geometry.point.astype(np.int32)
            cv2.circle(img, (int(px), int(py)), 3, (0, 0, 255), -1)

    if output.measurement is not None:
        px, py = output.measurement.geometry.point.astype(np.int32)
        cv2.circle(img, (int(px), int(py)), 6, (255, 255, 255), 2)
        cv2.putText(
            img,
            f"u={output.measurement.u_cross:.1f}, phi={np.degrees(output.measurement.phi_cross):.1f}",
            (8, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return img


@dataclass
class SequenceItem:
    frame_number: int
    heading_deg: float
    full_bgr: np.ndarray
    corridor_bgr: np.ndarray
    segments_bgr: np.ndarray
    boundaries_bgr: np.ndarray
    output: object
    meta: object


def build_demo_sequence(max_frames: int = 160) -> tuple[List[SequenceItem], dict]:
    np.random.seed(7)
    tmp_path = None
    try:
        img = build_synthetic_map()
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            tmp_path = f.name
        cv2.imwrite(tmp_path, img)

        sim = FlightSimulator(
            image_path=tmp_path,
            camera_w=200,
            camera_h=150,
            noise_level=8,
            shake_pixels=1.5,
            output_format="hsv",
        )
        pipeline = SpectralCorridorPipeline(
            PipelineConfig(
                cell_size=12,
                n_clusters=3,
            )
        )

        items: List[SequenceItem] = []
        route = default_demo_route()
        for hsv_frame, meta in sim.fly(route, speed_pix_per_frame=10.0):
            if meta.frame_number >= max_frames:
                break
            heading_deg = float(np.degrees(meta.yaw))
            output = pipeline.process_frame(
                hsv_frame,
                heading_deg=heading_deg,
                frame_number=meta.frame_number,
            )

            full_bgr = cv2.cvtColor(hsv_frame, cv2.COLOR_HSV2BGR)
            aligned_bgr = rotate_frame_to_heading(full_bgr, heading_deg)
            corridor_bgr = cv2.cvtColor(output.corridor_frame, cv2.COLOR_HSV2BGR)
            segments_bgr = colorize_segments(output.spectral.segment_image)
            boundaries_bgr = draw_boundaries(segments_bgr, output)

            items.append(
                SequenceItem(
                    frame_number=meta.frame_number,
                    heading_deg=heading_deg,
                    full_bgr=aligned_bgr,
                    corridor_bgr=corridor_bgr,
                    segments_bgr=segments_bgr,
                    boundaries_bgr=boundaries_bgr,
                    output=output,
                    meta=meta,
                )
            )

        return items, pipeline.metrics.summary()
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


class ImagePane(QLabel):
    def __init__(self, title: str):
        super().__init__()
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(320, 220)
        self.setStyleSheet("background:#1b1f26; border:1px solid #394150;")
        self._title = title

    def set_image(self, image_bgr: np.ndarray):
        self.setPixmap(np_to_pixmap(image_bgr))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Spectral Corridor Visualizer")
        self.resize(1400, 920)

        self.items, self.summary = build_demo_sequence()
        self.current_index = 0
        self.playing = False

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        controls = QHBoxLayout()
        self.play_btn = QPushButton("Play")
        self.play_btn.clicked.connect(self.toggle_play)
        self.reload_btn = QPushButton("Reload Demo")
        self.reload_btn.clicked.connect(self.reload_demo)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(max(len(self.items) - 1, 0))
        self.slider.valueChanged.connect(self.on_slider)
        controls.addWidget(self.play_btn)
        controls.addWidget(self.reload_btn)
        controls.addWidget(self.slider)
        root.addLayout(controls)

        grid = QGridLayout()
        self.frame_pane = ImagePane("Aligned Frame")
        self.corridor_pane = ImagePane("Corridor")
        self.segment_pane = ImagePane("Segments")
        self.boundary_pane = ImagePane("Boundaries")
        grid.addWidget(self._wrap("Aligned Frame", self.frame_pane), 0, 0)
        grid.addWidget(self._wrap("Corridor", self.corridor_pane), 0, 1)
        grid.addWidget(self._wrap("Spectral Segments", self.segment_pane), 1, 0)
        grid.addWidget(self._wrap("Boundaries / Measurement", self.boundary_pane), 1, 1)
        root.addLayout(grid, stretch=1)

        bottom = QHBoxLayout()
        self.info = QTextEdit()
        self.info.setReadOnly(True)
        self.info.setStyleSheet("background:#11151c; color:#e6edf3; border:1px solid #394150;")
        self.summary_box = QTextEdit()
        self.summary_box.setReadOnly(True)
        self.summary_box.setStyleSheet("background:#11151c; color:#e6edf3; border:1px solid #394150;")
        bottom.addWidget(self._wrap("Current Frame", self.info), 1)
        bottom.addWidget(self._wrap("Metrics Summary", self.summary_box), 1)
        root.addLayout(bottom)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.next_frame)
        self.timer.setInterval(120)

        self.update_view()

    def _wrap(self, title: str, widget: QWidget) -> QGroupBox:
        box = QGroupBox(title)
        box.setFont(QFont("Consolas", 10))
        layout = QVBoxLayout(box)
        layout.addWidget(widget)
        return box

    def toggle_play(self):
        self.playing = not self.playing
        self.play_btn.setText("Pause" if self.playing else "Play")
        if self.playing:
            self.timer.start()
        else:
            self.timer.stop()

    def reload_demo(self):
        self.items, self.summary = build_demo_sequence()
        self.current_index = 0
        self.slider.blockSignals(True)
        self.slider.setMaximum(max(len(self.items) - 1, 0))
        self.slider.setValue(0)
        self.slider.blockSignals(False)
        self.update_view()

    def on_slider(self, value: int):
        self.current_index = int(value)
        self.update_view()

    def next_frame(self):
        if not self.items:
            return
        self.current_index = (self.current_index + 1) % len(self.items)
        self.slider.blockSignals(True)
        self.slider.setValue(self.current_index)
        self.slider.blockSignals(False)
        self.update_view()

    def update_view(self):
        if not self.items:
            return
        item = self.items[self.current_index]
        self.frame_pane.set_image(item.full_bgr)
        self.corridor_pane.set_image(item.corridor_bgr)
        self.segment_pane.set_image(item.segments_bgr)
        self.boundary_pane.set_image(item.boundaries_bgr)

        measurement = item.output.measurement
        best_track = item.output.best_track
        self.info.setPlainText(
            "\n".join(
                [
                    f"frame: {item.frame_number}",
                    f"heading_deg: {item.heading_deg:.2f}",
                    f"position: ({item.meta.position_x:.1f}, {item.meta.position_y:.1f})",
                    f"boundaries detected: {len(item.output.boundaries)}",
                    f"best_track: {None if best_track is None else best_track.track_id}",
                    f"track_hits: {0 if best_track is None else best_track.hits}",
                    f"measurement_u: {None if measurement is None else round(measurement.u_cross, 2)}",
                    f"measurement_phi_deg: {None if measurement is None else round(np.degrees(measurement.phi_cross), 2)}",
                    f"measurement_conf: {None if measurement is None else round(measurement.confidence, 3)}",
                ]
            )
        )

        self.summary_box.setPlainText(
            "\n".join(f"{key}: {value:.4f}" for key, value in self.summary.items())
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Visualizer for spectral corridor pipeline")
    parser.add_argument("--demo", action="store_true", help="Kept for compatibility; demo is default")
    return parser.parse_args()


def main():
    parse_args()
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
