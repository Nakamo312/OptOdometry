"""
visualizer.py
Десктопный визуализатор corridor-based spectral pipeline
в стиле старого rich desktop UI.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from dataclasses import dataclass
from typing import List

import cv2
import numpy as np
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap
from PyQt5.QtWidgets import (
    QAction,
    QApplication,
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSplitter,
    QStatusBar,
    QTextEdit,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from corridor import rotate_frame_to_heading
from pipeline import PipelineConfig, SpectralCorridorPipeline
from research_eval import build_synthetic_map
from simulator import FlightSimulator, default_demo_route


C = {
    "bg": "#1a1d23",
    "panel": "#22262e",
    "widget": "#2a2f3a",
    "border": "#353a47",
    "text": "#e8eaf0",
    "muted": "#7b8299",
    "signal": "#4fc3f7",
    "event": "#ef5350",
    "track": "#7986cb",
    "pos": "#e91e63",
    "selected": "#ffd54f",
    "candidate": "#cfd8dc",
}

STYLESHEET = f"""
QMainWindow, QWidget {{
    background: {C['bg']};
    color: {C['text']};
    font-family: Consolas, 'Courier New', monospace;
    font-size: 11px;
}}
QGroupBox {{
    color: {C['signal']};
    border: 1px solid {C['border']};
    border-radius: 6px;
    margin-top: 8px;
    padding-top: 6px;
    font-size: 10px;
    font-weight: bold;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 8px;
    padding: 0 4px;
}}
QPushButton {{
    background: {C['widget']};
    color: {C['text']};
    border: 1px solid {C['border']};
    border-radius: 4px;
    padding: 5px 12px;
}}
QPushButton:hover {{ background: {C['signal']}; color: {C['bg']}; }}
QSlider::groove:horizontal {{
    background: {C['widget']};
    height: 4px;
    border-radius: 2px;
}}
QSlider::handle:horizontal {{
    background: {C['signal']};
    width: 12px; height: 12px;
    margin: -4px 0;
    border-radius: 6px;
}}
QSlider::sub-page:horizontal {{ background: {C['signal']}; border-radius: 2px; }}
QListWidget, QTextEdit {{
    background: {C['widget']};
    border: 1px solid {C['border']};
    border-radius: 4px;
}}
QStatusBar {{
    background: {C['panel']};
    border-top: 1px solid {C['border']};
    color: {C['muted']};
    font-size: 10px;
}}
QToolBar {{
    background: {C['panel']};
    border-bottom: 1px solid {C['border']};
    spacing: 4px;
    padding: 3px;
}}
QToolBar QToolButton {{
    background: transparent;
    color: {C['text']};
    border: 1px solid transparent;
    border-radius: 3px;
    padding: 4px 8px;
}}
QToolBar QToolButton:hover {{ background: {C['widget']}; border-color: {C['border']}; }}
QSplitter::handle {{ background: {C['border']}; }}
"""


def np_to_qimage(image_bgr: np.ndarray) -> QImage:
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    return QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888).copy()


def colorize_segments(segment_image: np.ndarray) -> np.ndarray:
    palette = np.array(
        [[50, 180, 255], [80, 220, 120], [240, 180, 60], [180, 90, 255], [255, 110, 110]],
        dtype=np.uint8,
    )
    out = np.zeros((*segment_image.shape, 3), dtype=np.uint8)
    for idx in np.unique(segment_image):
        out[segment_image == idx] = palette[idx % len(palette)]
    return out


@dataclass
class SequenceItem:
    frame_number: int
    heading_deg: float
    full_bgr: np.ndarray
    corridor_bgr: np.ndarray
    segments_bgr: np.ndarray
    output: object
    meta: object


def build_demo_sequence(waypoints=None, max_frames: int = 160) -> tuple[List[SequenceItem], dict]:
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
                target_regions=24,
                compactness=0.30,
                n_clusters=3,
            )
        )
        route = waypoints or default_demo_route()

        items: List[SequenceItem] = []
        for hsv_frame, meta in sim.fly(route, speed_pix_per_frame=10.0):
            if meta.frame_number >= max_frames:
                break
            heading_deg = float(np.degrees(meta.yaw))
            output = pipeline.process_frame(hsv_frame, heading_deg=heading_deg, frame_number=meta.frame_number)
            full_bgr = cv2.cvtColor(hsv_frame, cv2.COLOR_HSV2BGR)
            aligned_bgr = rotate_frame_to_heading(full_bgr, heading_deg)
            corridor_bgr = cv2.cvtColor(output.corridor_frame, cv2.COLOR_HSV2BGR)
            segments_bgr = colorize_segments(output.spectral.segment_image)
            items.append(
                SequenceItem(
                    frame_number=meta.frame_number,
                    heading_deg=heading_deg,
                    full_bgr=aligned_bgr,
                    corridor_bgr=corridor_bgr,
                    segments_bgr=segments_bgr,
                    output=output,
                    meta=meta,
                )
            )
        return items, pipeline.metrics.summary()
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


class FrameWidget(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(420, 300)
        self.setAlignment(Qt.AlignCenter)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet(f"background:{C['widget']}; border:1px solid {C['border']}; border-radius:4px;")
        self._full_qimg: QImage | None = None
        self._corridor_qimg: QImage | None = None
        self._frame_number = 0
        self._measurement = None
        self._boundaries = []
        self._heading_deg = 0.0

    def set_data(self, full_bgr: np.ndarray, corridor_bgr: np.ndarray, frame_number: int, heading_deg: float, boundaries, measurement):
        self._full_qimg = np_to_qimage(full_bgr)
        self._corridor_qimg = np_to_qimage(corridor_bgr)
        self._frame_number = frame_number
        self._heading_deg = heading_deg
        self._boundaries = boundaries
        self._measurement = measurement
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        w, h = self.width(), self.height()
        painter.fillRect(0, 0, w, h, QColor(C["widget"]))

        if self._full_qimg is None or self._corridor_qimg is None:
            painter.setPen(QColor(C["muted"]))
            painter.drawText(self.rect(), Qt.AlignCenter, "нет данных")
            return

        pix = QPixmap.fromImage(self._full_qimg)
        scaled = pix.scaled(w - 4, h - 4, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        ox = (w - scaled.width()) // 2
        oy = (h - scaled.height()) // 2
        painter.drawPixmap(ox, oy, scaled)

        fw = self._full_qimg.width()
        fh = self._full_qimg.height()
        cw = self._corridor_qimg.width()
        ch = self._corridor_qimg.height()

        sx = scaled.width() / max(fw, 1)
        sy = scaled.height() / max(fh, 1)

        roi_x0 = int(ox + (fw - cw) * 0.5 * sx)
        roi_x1 = int(ox + (fw + cw) * 0.5 * sx)
        roi_y0 = int(oy + fh * 0.05 * sy)
        roi_y1 = int(roi_y0 + ch * sy)

        painter.setPen(QPen(QColor(C["signal"]), 2))
        painter.drawRect(roi_x0, roi_y0, roi_x1 - roi_x0, roi_y1 - roi_y0)
        painter.setPen(QPen(QColor("white"), 1))
        painter.drawLine((roi_x0 + roi_x1) // 2, roi_y0, (roi_x0 + roi_x1) // 2, roi_y1)

        painter.setPen(QColor(C["text"]))
        painter.setFont(QFont("Consolas", 9))
        painter.fillRect(ox + 6, oy + 6, 180, 18, QColor(0, 0, 0, 150))
        painter.drawText(ox + 10, oy + 19, f"frame {self._frame_number:06d} | heading {self._heading_deg:.1f} deg")

        if self._measurement is not None:
            px = roi_x0 + int(self._measurement.geometry.point[0] / max(cw, 1) * (roi_x1 - roi_x0))
            py = roi_y0 + int(self._measurement.geometry.point[1] / max(ch, 1) * (roi_y1 - roi_y0))
            painter.setPen(QPen(QColor(C["selected"]), 2))
            painter.drawEllipse(px - 6, py - 6, 12, 12)
            painter.drawLine(roi_x0, py, roi_x1, py)


class CorridorWidget(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(300, 220)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet(f"background:{C['widget']}; border:1px solid {C['border']}; border-radius:4px;")
        self._base_qimg: QImage | None = None
        self._boundaries = []
        self._measurement = None

    def set_data(self, corridor_bgr: np.ndarray, boundaries, measurement):
        self._base_qimg = np_to_qimage(corridor_bgr)
        self._boundaries = boundaries
        self._measurement = measurement
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        w, h = self.width(), self.height()
        painter.fillRect(0, 0, w, h, QColor(C["widget"]))
        if self._base_qimg is None:
            painter.setPen(QColor(C["muted"]))
            painter.drawText(self.rect(), Qt.AlignCenter, "нет данных")
            return

        pix = QPixmap.fromImage(self._base_qimg)
        scaled = pix.scaled(w - 4, h - 4, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        ox = (w - scaled.width()) // 2
        oy = (h - scaled.height()) // 2
        painter.drawPixmap(ox, oy, scaled)

        cw = self._base_qimg.width()
        ch = self._base_qimg.height()
        sx = scaled.width() / max(cw, 1)
        sy = scaled.height() / max(ch, 1)
        painter.setPen(QPen(QColor("white"), 1))
        painter.drawLine(ox + scaled.width() // 2, oy, ox + scaled.width() // 2, oy + scaled.height())

        for idx, boundary in enumerate(self._boundaries):
            pts = boundary.points.astype(np.float32)
            if len(pts) < 2:
                continue
            color = QColor(C["candidate"]) if idx != 0 or self._measurement is None else QColor(C["selected"])
            pen = QPen(color, 1 if color != QColor(C["selected"]) else 2)
            painter.setPen(pen)
            for i in range(len(pts) - 1):
                p0x = ox + int(pts[i, 0] * sx)
                p0y = oy + int(pts[i, 1] * sy)
                p1x = ox + int(pts[i + 1, 0] * sx)
                p1y = oy + int(pts[i + 1, 1] * sy)
                painter.drawLine(p0x, p0y, p1x, p1y)

        if self._measurement is not None:
            px = ox + int(self._measurement.geometry.point[0] * sx)
            py = oy + int(self._measurement.geometry.point[1] * sy)
            painter.setPen(QPen(QColor("white"), 2))
            painter.drawEllipse(px - 5, py - 5, 10, 10)


class SignalPlot(QWidget):
    MAX_HISTORY = 240

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(420, 140)
        self.setStyleSheet(f"background:{C['widget']}; border:1px solid {C['border']}; border-radius:4px;")
        self._conf: List[float] = []
        self._n_boundaries: List[float] = []
        self._u_cross: List[float] = []

    def reset(self):
        self._conf.clear()
        self._n_boundaries.clear()
        self._u_cross.clear()
        self.update()

    def add_point(self, n_boundaries: int, confidence: float | None, u_cross: float | None):
        self._n_boundaries.append(float(min(n_boundaries, 10)) / 10.0)
        self._conf.append(0.0 if confidence is None else float(np.clip(confidence, 0.0, 1.0)))
        self._u_cross.append(0.0 if u_cross is None else float(min(max(u_cross / 100.0, 0.0), 1.0)))
        if len(self._conf) > self.MAX_HISTORY:
            excess = len(self._conf) - self.MAX_HISTORY
            self._conf = self._conf[excess:]
            self._n_boundaries = self._n_boundaries[excess:]
            self._u_cross = self._u_cross[excess:]
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        painter.fillRect(0, 0, w, h, QColor(C["widget"]))
        n = len(self._conf)
        if n < 2:
            painter.setPen(QColor(C["muted"]))
            painter.drawText(self.rect(), Qt.AlignCenter, "нет истории")
            return

        pad = (30, 8, 8, 18)
        pw = w - pad[0] - pad[2]
        ph = h - pad[1] - pad[3]

        def to_px(i, val):
            x = pad[0] + int(i / max(self.MAX_HISTORY - 1, 1) * pw)
            y = pad[1] + int((1.0 - val) * ph)
            return x, y

        for frac in [0.25, 0.5, 0.75, 1.0]:
            yg = pad[1] + int((1.0 - frac) * ph)
            painter.setPen(QPen(QColor(C["border"]), 1))
            painter.drawLine(pad[0], yg, w - pad[2], yg)

        curves = [
            (self._conf, QColor(C["selected"]), "conf"),
            (self._n_boundaries, QColor(C["signal"]), "N boundaries"),
            (self._u_cross, QColor(C["track"]), "u/100"),
        ]
        for values, color, _ in curves:
            painter.setPen(QPen(color, 2))
            pts = [to_px(i, values[i]) for i in range(len(values))]
            for i in range(len(pts) - 1):
                painter.drawLine(*pts[i], *pts[i + 1])

        painter.setFont(QFont("Consolas", 8))
        lx = pad[0]
        for _, color, label in curves:
            painter.setPen(color)
            painter.drawText(lx, h - 3, label)
            lx += len(label) * 7 + 16


class TrajectoryWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(240, 180)
        self.setStyleSheet(f"background:{C['widget']}; border:1px solid {C['border']}; border-radius:4px;")
        self._positions: List[tuple[float, float]] = []
        self._events: List[int] = []
        self._map_w = 1200
        self._map_h = 800

    def reset(self):
        self._positions.clear()
        self._events.clear()
        self.update()

    def add_position(self, x: float, y: float, is_event: bool):
        self._positions.append((x, y))
        if is_event:
            self._events.append(len(self._positions) - 1)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        pad = (8, 18, 8, 8)
        pw = w - pad[0] - pad[2]
        ph = h - pad[1] - pad[3]
        painter.fillRect(0, 0, w, h, QColor(C["widget"]))
        painter.setPen(QColor(C["muted"]))
        painter.setFont(QFont("Consolas", 8))
        painter.drawText(8, 12, "trajectory")

        if len(self._positions) < 2:
            painter.drawText(self.rect(), Qt.AlignCenter, "нет данных")
            return

        def to_screen(mx, my):
            sx = pad[0] + int(mx / max(self._map_w, 1) * pw)
            sy = pad[1] + int(my / max(self._map_h, 1) * ph)
            return sx, sy

        painter.setPen(QPen(QColor(C["track"]), 1.5))
        for i in range(len(self._positions) - 1):
            x0, y0 = to_screen(*self._positions[i])
            x1, y1 = to_screen(*self._positions[i + 1])
            painter.drawLine(x0, y0, x1, y1)

        painter.setPen(QPen(QColor("white"), 1.5))
        painter.setBrush(QColor(C["pos"]))
        cx, cy = to_screen(*self._positions[-1])
        painter.drawEllipse(cx - 5, cy - 5, 10, 10)

        painter.setPen(QPen(QColor(C["event"]), 1.5))
        painter.setBrush(QColor(C["event"]))
        for idx in self._events:
            ex, ey = to_screen(*self._positions[idx])
            painter.drawEllipse(ex - 4, ey - 4, 8, 8)


class EventListWidget(QListWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFont(QFont("Consolas", 9))

    def add_measurement(self, frame_number: int, track_id: int | None, confidence: float | None, phi_deg: float | None):
        text = (
            f"frame {frame_number:5d}  "
            f"track={track_id if track_id is not None else '-'}  "
            f"conf={0.0 if confidence is None else confidence:.2f}  "
            f"phi={0.0 if phi_deg is None else phi_deg:.1f}"
        )
        item = QListWidgetItem(text)
        item.setForeground(QColor(C["selected"]))
        self.addItem(item)
        self.scrollToBottom()


class RouteListWidget(QListWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFont(QFont("Consolas", 9))

    def set_waypoints(self, waypoints):
        self.clear()
        for i, wp in enumerate(waypoints):
            self.addItem(QListWidgetItem(f"{i:02d} | {wp.name or 'wp'} | x={wp.x:.1f} y={wp.y:.1f}"))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Spectral Corridor Visualizer")
        self.resize(1560, 980)
        self.setStyleSheet(STYLESHEET)

        self.route = default_demo_route()
        self.items, self.summary = build_demo_sequence(self.route)
        self.current_index = 0
        self.playing = False
        self.last_event_frame = -1

        self._build_toolbar()
        self._build_ui()
        self._build_statusbar()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.next_frame)
        self.timer.setInterval(120)

        self.signal_plot.reset()
        self.trajectory.reset()
        self.event_list.clear()
        self.route_list.set_waypoints(self.route)
        self.rebuild_history_to_current()

    def _build_toolbar(self):
        bar = QToolBar("Main", self)
        self.addToolBar(bar)

        act_reload = QAction("Reload Demo", self)
        act_reload.triggered.connect(self.reload_demo)
        bar.addAction(act_reload)

        act_open = QAction("Open Map...", self)
        act_open.triggered.connect(self.open_map_placeholder)
        bar.addAction(act_open)

        act_reset_route = QAction("Reset Route", self)
        act_reset_route.triggered.connect(self.reset_route)
        bar.addAction(act_reset_route)

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        controls = QHBoxLayout()
        self.play_btn = QPushButton("Play")
        self.play_btn.clicked.connect(self.toggle_play)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(max(len(self.items) - 1, 0))
        self.slider.valueChanged.connect(self.on_slider)
        controls.addWidget(self.play_btn)
        controls.addWidget(self.slider)
        root.addLayout(controls)

        splitter = QSplitter(Qt.Horizontal)
        root.addWidget(splitter, 1)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        self.frame_widget = FrameWidget()
        self.corridor_widget = CorridorWidget()
        self.signal_plot = SignalPlot()

        frame_box = QGroupBox("Frame")
        frame_layout = QVBoxLayout(frame_box)
        frame_layout.addWidget(self.frame_widget)

        corridor_box = QGroupBox("Corridor / Boundaries")
        corridor_layout = QVBoxLayout(corridor_box)
        corridor_layout.addWidget(self.corridor_widget)

        plot_box = QGroupBox("Signal History")
        plot_layout = QVBoxLayout(plot_box)
        plot_layout.addWidget(self.signal_plot)

        left_layout.addWidget(frame_box, 3)
        left_layout.addWidget(corridor_box, 2)
        left_layout.addWidget(plot_box, 1)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        self.trajectory = TrajectoryWidget()
        self.info_box = QTextEdit()
        self.info_box.setReadOnly(True)
        self.summary_box = QTextEdit()
        self.summary_box.setReadOnly(True)
        self.event_list = EventListWidget()
        self.route_list = RouteListWidget()

        traj_box = QGroupBox("Trajectory")
        traj_layout = QVBoxLayout(traj_box)
        traj_layout.addWidget(self.trajectory)

        route_box = QGroupBox("Route")
        route_layout = QVBoxLayout(route_box)
        route_layout.addWidget(self.route_list)
        route_buttons = QHBoxLayout()
        self.btn_add_wp = QPushButton("Add WP")
        self.btn_add_wp.clicked.connect(self.add_waypoint)
        self.btn_pop_wp = QPushButton("Pop WP")
        self.btn_pop_wp.clicked.connect(self.pop_waypoint)
        self.btn_apply_route = QPushButton("Apply Route")
        self.btn_apply_route.clicked.connect(self.apply_route)
        route_buttons.addWidget(self.btn_add_wp)
        route_buttons.addWidget(self.btn_pop_wp)
        route_buttons.addWidget(self.btn_apply_route)
        route_layout.addLayout(route_buttons)

        info_box = QGroupBox("Current Frame")
        info_layout = QVBoxLayout(info_box)
        info_layout.addWidget(self.info_box)

        summary_box = QGroupBox("Metrics")
        summary_layout = QVBoxLayout(summary_box)
        summary_layout.addWidget(self.summary_box)

        event_box = QGroupBox("Measurements")
        event_layout = QVBoxLayout(event_box)
        event_layout.addWidget(self.event_list)

        right_layout.addWidget(traj_box, 2)
        right_layout.addWidget(route_box, 2)
        right_layout.addWidget(info_box, 2)
        right_layout.addWidget(summary_box, 1)
        right_layout.addWidget(event_box, 2)

        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setSizes([1040, 420])

        legend = QLabel(
            "Legend: blue box = corridor ROI, white line = corridor center, "
            "gray = all candidates, yellow = selected boundary, white circle = crossing point"
        )
        legend.setStyleSheet(f"color:{C['muted']};")
        root.addWidget(legend)

    def _build_statusbar(self):
        status = QStatusBar(self)
        self.setStatusBar(status)
        status.showMessage("Ready")

    def open_map_placeholder(self):
        QFileDialog.getOpenFileName(self, "Open map", "", "Images (*.png *.jpg *.jpeg)")

    def toggle_play(self):
        self.playing = not self.playing
        self.play_btn.setText("Pause" if self.playing else "Play")
        if self.playing:
            self.timer.start()
        else:
            self.timer.stop()

    def reload_demo(self):
        self.items, self.summary = build_demo_sequence(self.route)
        self.current_index = 0
        self.playing = False
        self.play_btn.setText("Play")
        self.timer.stop()
        self.slider.blockSignals(True)
        self.slider.setMaximum(max(len(self.items) - 1, 0))
        self.slider.setValue(0)
        self.slider.blockSignals(False)
        self.signal_plot.reset()
        self.trajectory.reset()
        self.event_list.clear()
        self.last_event_frame = -1
        self.route_list.set_waypoints(self.route)
        self.rebuild_history_to_current()

    def reset_route(self):
        self.route = default_demo_route()
        self.reload_demo()

    def add_waypoint(self):
        if not self.route:
            self.route = default_demo_route()
        last = self.route[-1]
        idx = len(self.route)
        step = 120.0
        new_x = min(last.x + step, 1100.0)
        new_y = last.y + (60.0 if idx % 2 == 0 else -60.0)
        new_y = float(max(80.0, min(720.0, new_y)))
        from simulator import Waypoint
        self.route.append(Waypoint(new_x, new_y, f"wp_{idx}"))
        self.route_list.set_waypoints(self.route)
        self.statusBar().showMessage("Waypoint added. Click Apply Route to rebuild sequence.")

    def pop_waypoint(self):
        if len(self.route) > 2:
            self.route.pop()
            self.route_list.set_waypoints(self.route)
            self.statusBar().showMessage("Last waypoint removed. Click Apply Route to rebuild sequence.")

    def apply_route(self):
        if len(self.route) < 2:
            self.statusBar().showMessage("Need at least 2 waypoints")
            return
        self.reload_demo()

    def on_slider(self, value: int):
        self.current_index = int(value)
        self.rebuild_history_to_current()

    def next_frame(self):
        if not self.items:
            return
        self.current_index = (self.current_index + 1) % len(self.items)
        self.slider.blockSignals(True)
        self.slider.setValue(self.current_index)
        self.slider.blockSignals(False)
        self.rebuild_history_to_current()

    def rebuild_history_to_current(self):
        self.signal_plot.reset()
        self.trajectory.reset()
        self.event_list.clear()
        self.last_event_frame = -1
        for idx in range(self.current_index + 1):
            item = self.items[idx]
            measurement = item.output.measurement
            best_track = item.output.best_track
            self.signal_plot.add_point(
                n_boundaries=len(item.output.boundaries),
                confidence=None if measurement is None else measurement.confidence,
                u_cross=None if measurement is None else measurement.u_cross,
            )
            self.trajectory.add_position(item.meta.position_x, item.meta.position_y, measurement is not None)
            if measurement is not None and item.frame_number != self.last_event_frame:
                self.event_list.add_measurement(
                    frame_number=item.frame_number,
                    track_id=None if best_track is None else best_track.track_id,
                    confidence=measurement.confidence,
                    phi_deg=float(np.degrees(measurement.phi_cross)),
                )
                self.last_event_frame = item.frame_number
        self.update_view()

    def update_view(self):
        if not self.items:
            return
        item = self.items[self.current_index]
        self.frame_widget.set_data(
            item.full_bgr,
            item.corridor_bgr,
            frame_number=item.frame_number,
            heading_deg=item.heading_deg,
            boundaries=item.output.boundaries,
            measurement=item.output.measurement,
        )
        self.corridor_widget.set_data(
            item.corridor_bgr,
            boundaries=item.output.boundaries,
            measurement=item.output.measurement,
        )

        measurement = item.output.measurement
        best_track = item.output.best_track
        self.info_box.setPlainText(
            "\n".join(
                [
                    f"frame: {item.frame_number}",
                    f"heading_deg: {item.heading_deg:.2f}",
                    f"position: ({item.meta.position_x:.1f}, {item.meta.position_y:.1f})",
                    f"segment_name: {item.meta.segment_name}",
                    f"route_waypoints: {len(self.route)}",
                    "",
                    f"boundaries_detected: {len(item.output.boundaries)}",
                    f"best_track_id: {None if best_track is None else best_track.track_id}",
                    f"track_hits: {0 if best_track is None else best_track.hits}",
                    f"track_age: {0 if best_track is None else best_track.age}",
                    "",
                    f"measurement_u: {None if measurement is None else round(measurement.u_cross, 2)}",
                    f"measurement_phi_deg: {None if measurement is None else round(np.degrees(measurement.phi_cross), 2)}",
                    f"measurement_conf: {None if measurement is None else round(measurement.confidence, 3)}",
                ]
            )
        )
        self.summary_box.setPlainText("\n".join(f"{key}: {value:.4f}" for key, value in self.summary.items()))
        self.statusBar().showMessage(
            f"frame={item.frame_number} | boundaries={len(item.output.boundaries)} | "
            f"track={None if best_track is None else best_track.track_id}"
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Visualizer for spectral corridor pipeline")
    parser.add_argument("--demo", action="store_true", help="demo mode")
    return parser.parse_args()


def main():
    parse_args()
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
