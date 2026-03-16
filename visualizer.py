"""
visualizer.py — Задача 12
Десктопный визуализатор работы детектора переходов.

Запуск:
    python visualizer.py --image map.png
    python visualizer.py --frames frames/   (папка с PNG кадрами)
    python visualizer.py --demo             (синтетика без карты)
"""

import sys
import os
import argparse
import numpy as np
import cv2

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSplitter, QLabel, QPushButton, QSlider, QFileDialog,
    QGroupBox, QListWidget, QListWidgetItem, QStatusBar,
    QToolBar, QAction, QSpinBox, QDoubleSpinBox, QFormLayout,
    QSizePolicy, QFrame
)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QSize
from PyQt5.QtGui import (
    QImage, QPixmap, QPainter, QPen, QBrush, QColor,
    QFont, QPalette
)

# ── Наши модули ──────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from detector import TransitionDetector, TransitionEvent
from simulator import FlightSimulator, Waypoint


# ─────────────────────────────────────────────────────────────
#  ЦВЕТА
# ─────────────────────────────────────────────────────────────
C = {
    "bg":        "#1a1d23",
    "panel":     "#22262e",
    "widget":    "#2a2f3a",
    "border":    "#353a47",
    "text":      "#e8eaf0",
    "muted":     "#7b8299",
    "signal":    "#4fc3f7",    # Бхаттачарья
    "lr":        "#9575cd",    # LR
    "threshold": "#ffb74d",    # порог
    "event":     "#ef5350",    # момент перехода
    "old_zone":  "#81c784",    # старая зона
    "new_zone":  "#4db6ac",    # новая зона
    "track":     "#7986cb",    # траектория
    "pos":       "#e91e63",    # текущая позиция
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
QPushButton:disabled {{ color: {C['muted']}; }}
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
QListWidget {{
    background: {C['widget']};
    border: 1px solid {C['border']};
    border-radius: 4px;
    outline: none;
}}
QListWidget::item {{ padding: 3px 6px; border-radius: 2px; }}
QListWidget::item:selected {{ background: {C['event']}; color: white; }}
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
QScrollBar:vertical {{
    background: {C['bg']}; width: 6px; border-radius: 3px;
}}
QScrollBar::handle:vertical {{
    background: {C['border']}; border-radius: 3px; min-height: 16px;
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QLabel#val {{ color: {C['signal']}; font-weight: bold; }}
QLabel#title {{ color: {C['muted']}; font-size: 10px; }}
"""


# ─────────────────────────────────────────────────────────────
#  ВИДЖЕТ: КАДР КАМЕРЫ
# ─────────────────────────────────────────────────────────────
class FrameWidget(QLabel):
    """Показывает текущий BGR кадр с наложением информации."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(320, 240)
        self.setAlignment(Qt.AlignCenter)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet(f"background: {C['widget']}; border: 1px solid {C['border']}; border-radius:4px;")
        self._frame_bgr    = None
        self._frame_number = 0
        self._in_transition = False
        self._zone_color   = QColor(C['signal'])
        self._flash        = 0      # кол-во кадров вспышки

    def set_frame(self, bgr: np.ndarray, frame_number: int,
                  in_transition: bool, zone_color: QColor):
        self._frame_bgr     = bgr.copy()
        self._frame_number  = frame_number
        self._in_transition = in_transition
        self._zone_color    = zone_color
        if in_transition:
            self._flash = 8
        elif self._flash > 0:
            self._flash -= 1
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        w, h = self.width(), self.height()

        if self._frame_bgr is None:
            painter.fillRect(0, 0, w, h, QColor(C['widget']))
            painter.setPen(QColor(C['muted']))
            painter.setFont(QFont("Consolas", 11))
            painter.drawText(self.rect(), Qt.AlignCenter, "Нет данных\nзапустите воспроизведение")
            return

        # Конвертируем BGR → RGB для Qt
        rgb = cv2.cvtColor(self._frame_bgr, cv2.COLOR_BGR2RGB)
        fh, fw = rgb.shape[:2]
        qimg = QImage(rgb.data, fw, fh, fw * 3, QImage.Format_RGB888)
        pix  = QPixmap.fromImage(qimg)

        # Масштабируем с сохранением пропорций
        scaled = pix.scaled(w - 4, h - 4, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        ox = (w - scaled.width())  // 2
        oy = (h - scaled.height()) // 2
        painter.drawPixmap(ox, oy, scaled)

        # Рамка зоны
        border_color = QColor(C['event']) if self._flash > 0 else self._zone_color
        border_w     = 3 if self._flash > 0 else 2
        pen = QPen(border_color, border_w)
        painter.setPen(pen)
        painter.drawRect(ox, oy, scaled.width() - 1, scaled.height() - 1)

        # Оверлей: номер кадра
        painter.setPen(QColor(C['text']))
        painter.setFont(QFont("Consolas", 9))
        painter.fillRect(ox + 4, oy + 4, 90, 16,
                         QColor(0, 0, 0, 140))
        painter.drawText(ox + 7, oy + 16,
                         f"frame {self._frame_number:06d}")

        # Вспышка при переходе
        if self._flash > 0:
            alpha = int(self._flash / 8 * 80)
            painter.fillRect(ox, oy, scaled.width(), scaled.height(),
                             QColor(239, 83, 80, alpha))
            painter.setPen(QColor(C['event']))
            painter.setFont(QFont("Consolas", 11, QFont.Bold))
            painter.drawText(ox + 6, oy + scaled.height() - 8, "ПЕРЕХОД")


# ─────────────────────────────────────────────────────────────
#  ВИДЖЕТ: ГРАФИК СИГНАЛОВ
# ─────────────────────────────────────────────────────────────
class SignalPlot(QWidget):
    """
    График трёх сигналов:
    - Бхаттачарья (основной, голубой)
    - LR          (фильтрующий, фиолетовый, пунктир)
    - Порог       (адаптивный, оранжевый, пунктир)
    Вертикальные красные линии — моменты переходов.
    """

    MAX_HISTORY = 300

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(400, 120)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet(f"background: {C['widget']}; border: 1px solid {C['border']}; border-radius:4px;")

        self._bhat:      list[float] = []
        self._lr_norm:   list[float] = []    # LR нормализован в [0,1]
        self._threshold: list[float] = []
        self._events:    list[int]   = []    # индексы событий в истории

    def add_point(self, bhat: float, lr: float,
                  threshold: float, is_event: bool):
        # LR нормализуем: sigmoid(log(lr)) → [0,1]
        lr_norm = 1.0 / (1.0 + np.exp(-np.log(max(lr, 1e-9))))

        self._bhat.append(min(bhat, 1.0))
        self._lr_norm.append(lr_norm)
        self._threshold.append(min(threshold, 1.0))

        if is_event:
            self._events.append(len(self._bhat) - 1)

        # Срезаем историю
        if len(self._bhat) > self.MAX_HISTORY:
            excess = len(self._bhat) - self.MAX_HISTORY
            self._bhat      = self._bhat[excess:]
            self._lr_norm   = self._lr_norm[excess:]
            self._threshold = self._threshold[excess:]
            self._events    = [e - excess for e in self._events if e - excess >= 0]

        self.update()

    def reset(self):
        self._bhat.clear()
        self._lr_norm.clear()
        self._threshold.clear()
        self._events.clear()
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        pad  = (36, 6, 6, 18)    # left, top, right, bottom

        pw = w - pad[0] - pad[2]
        ph = h - pad[1] - pad[3]

        painter.fillRect(0, 0, w, h, QColor(C['widget']))

        n = len(self._bhat)
        if n < 2:
            painter.setPen(QColor(C['muted']))
            painter.drawText(self.rect(), Qt.AlignCenter, "нет данных")
            return

        def to_px(i, val):
            x = pad[0] + int(i / (self.MAX_HISTORY - 1) * pw)
            y = pad[1] + int((1.0 - val) * ph)
            return x, y

        # ── Сетка ────────────────────────────────────────────
        painter.setPen(QPen(QColor(C['border']), 0.5))
        for frac in [0.25, 0.5, 0.75, 1.0]:
            yg = pad[1] + int((1.0 - frac) * ph)
            painter.drawLine(pad[0], yg, w - pad[2], yg)
            painter.setFont(QFont("Consolas", 8))
            painter.setPen(QColor(C['muted']))
            painter.drawText(2, yg + 4, f"{frac:.2f}")
            painter.setPen(QPen(QColor(C['border']), 0.5))

        # ── Ось X (номера кадров) ─────────────────────────────
        painter.setPen(QColor(C['muted']))
        painter.setFont(QFont("Consolas", 8))
        for step in [0, n // 4, n // 2, 3 * n // 4, n - 1]:
            xi = pad[0] + int(step / max(n - 1, 1) * pw)
            painter.drawText(xi - 10, h - 3, f"{step}")

        # ── Вертикальные линии событий ───────────────────────
        for ei in self._events:
            xi = pad[0] + int(ei / max(self.MAX_HISTORY - 1, 1) * pw)
            painter.setPen(QPen(QColor(C['event']), 1,
                                Qt.SolidLine))
            painter.drawLine(xi, pad[1], xi, h - pad[3])

        # ── Линия порога ──────────────────────────────────────
        pen_thr = QPen(QColor(C['threshold']), 1, Qt.DashLine)
        pen_thr.setDashPattern([4, 3])
        painter.setPen(pen_thr)
        pts_thr = [to_px(i, self._threshold[i]) for i in range(n)]
        for i in range(len(pts_thr) - 1):
            painter.drawLine(*pts_thr[i], *pts_thr[i + 1])

        # ── Линия LR ──────────────────────────────────────────
        pen_lr = QPen(QColor(C['lr']), 1, Qt.DashLine)
        pen_lr.setDashPattern([5, 3])
        painter.setPen(pen_lr)
        pts_lr = [to_px(i, self._lr_norm[i]) for i in range(n)]
        for i in range(len(pts_lr) - 1):
            painter.drawLine(*pts_lr[i], *pts_lr[i + 1])

        # ── Линия Бхаттачарья ─────────────────────────────────
        painter.setPen(QPen(QColor(C['signal']), 1.5))
        pts_b = [to_px(i, self._bhat[i]) for i in range(n)]
        for i in range(len(pts_b) - 1):
            painter.drawLine(*pts_b[i], *pts_b[i + 1])

        # ── Легенда ───────────────────────────────────────────
        ly = h - 3
        painter.setFont(QFont("Consolas", 8))
        items = [
            (C['signal'],    "─  Бхатт"),
            (C['lr'],        "╌  LR"),
            (C['threshold'], "╌  порог"),
            (C['event'],     "│  переход"),
        ]
        lx = pad[0]
        for color, label in items:
            painter.setPen(QColor(color))
            painter.drawText(lx, ly, label)
            lx += len(label) * 6 + 10


# ─────────────────────────────────────────────────────────────
#  ВИДЖЕТ: ГИСТОГРАММА H
# ─────────────────────────────────────────────────────────────
class HistWidget(QWidget):
    """Гистограмма канала H (16 бинов) с цветовой раскраской."""

    def __init__(self, title="Гистограмма H", parent=None):
        super().__init__(parent)
        self._title = title
        self._bins  = np.zeros(16, dtype=np.float32)
        self.setMinimumSize(120, 80)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet(
            f"background: {C['widget']}; border:1px solid {C['border']}; border-radius:4px;"
        )

    def set_histogram(self, bins: np.ndarray):
        self._bins = bins.copy()
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        pad  = (4, 18, 4, 14)

        painter.fillRect(0, 0, w, h, QColor(C['widget']))

        # Заголовок
        painter.setPen(QColor(C['muted']))
        painter.setFont(QFont("Consolas", 8))
        painter.drawText(pad[0], 11, self._title)

        pw = w - pad[0] - pad[2]
        ph = h - pad[1] - pad[3]
        n  = len(self._bins)

        if not np.any(self._bins):
            painter.setPen(QColor(C['muted']))
            painter.drawText(pad[0], pad[1] + ph // 2 + 4, "нет данных")
            return

        max_val    = float(self._bins.max()) or 1.0
        bar_w      = pw / n
        hue_step   = 180.0 / n    # H в OpenCV: 0-180

        for i, val in enumerate(self._bins):
            bh = int((val / max_val) * ph)
            bx = pad[0] + int(i * bar_w)
            by = pad[1] + ph - bh

            # Цвет бара соответствует оттенку H
            hue_qt = int(i * hue_step * 2)   # Qt H: 0-360
            color  = QColor.fromHsv(hue_qt, 200, 220)
            painter.fillRect(bx, by, max(int(bar_w) - 1, 1), bh, color)

        # Ось X
        painter.setPen(QColor(C['border']))
        painter.drawLine(pad[0], pad[1] + ph, w - pad[2], pad[1] + ph)
        painter.setPen(QColor(C['muted']))
        painter.setFont(QFont("Consolas", 7))
        painter.drawText(pad[0], h - 2, "0°")
        painter.drawText(w // 2 - 8, h - 2, "90°")
        painter.drawText(w - pad[2] - 20, h - 2, "180°")


# ─────────────────────────────────────────────────────────────
#  ВИДЖЕТ: МИНИ-КАРТА ТРАЕКТОРИИ
# ─────────────────────────────────────────────────────────────
class TrajectoryWidget(QWidget):
    """
    Горизонтальная полоса: траектория БПЛА, позиция, метки переходов,
    цветные зоны снизу (по доминирующему цвету каждого отрезка).
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(400, 80)
        self.setFixedHeight(90)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setStyleSheet(
            f"background: {C['widget']}; border:1px solid {C['border']}; border-radius:4px;"
        )
        self._positions:   list[tuple] = []   # (x, y) пикселей карты
        self._map_w:       int  = 1
        self._map_h:       int  = 1
        self._events_pos:  list[int]   = []   # индексы позиций переходов
        self._zone_colors: list[QColor] = []   # цвет каждого шага

    def set_map_size(self, w: int, h: int):
        self._map_w = w
        self._map_h = h

    def add_position(self, x: float, y: float, zone_color: QColor):
        self._positions.append((x, y))
        self._zone_colors.append(zone_color)
        self.update()

    def add_event(self):
        if self._positions:
            self._events_pos.append(len(self._positions) - 1)
        self.update()

    def reset(self):
        self._positions.clear()
        self._events_pos.clear()
        self._zone_colors.clear()
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        pad  = (6, 20, 6, 22)

        painter.fillRect(0, 0, w, h, QColor(C['widget']))

        pw = w - pad[0] - pad[2]

        # Заголовок
        painter.setPen(QColor(C['muted']))
        painter.setFont(QFont("Consolas", 8))
        painter.drawText(pad[0], 12, "траектория БПЛА над картой")

        n = len(self._positions)
        if n < 2:
            return

        def pos_to_x(idx):
            # Проецируем X позиции карты на ширину виджета
            px = self._positions[idx][0]
            return pad[0] + int((px / max(self._map_w, 1)) * pw)

        track_y = pad[1] + (h - pad[1] - pad[3]) // 2

        # ── Цветные зоны под траекторией ──────────────────────
        zone_y  = h - pad[3] + 2
        zone_h  = 10
        for i in range(n - 1):
            x0 = pos_to_x(i)
            x1 = pos_to_x(i + 1)
            if x1 > x0:
                c = self._zone_colors[i]
                c.setAlpha(120)
                painter.fillRect(x0, zone_y, x1 - x0, zone_h, c)

        # ── Линия трека ───────────────────────────────────────
        painter.setPen(QPen(QColor(C['track']), 1.5))
        for i in range(n - 1):
            painter.drawLine(pos_to_x(i), track_y, pos_to_x(i + 1), track_y)

        # ── Метки переходов ───────────────────────────────────
        for j, ei in enumerate(self._events_pos):
            ex = pos_to_x(ei)
            painter.setPen(QPen(QColor(C['event']), 1.5))
            painter.drawLine(ex, pad[1] + 2, ex, h - pad[3] + zone_h + 2)
            painter.setFont(QFont("Consolas", 7, QFont.Bold))
            painter.setPen(QColor(C['event']))
            painter.drawText(ex - 6, pad[1], f"T{j+1}")

        # ── Текущая позиция ───────────────────────────────────
        cx = pos_to_x(n - 1)
        painter.setBrush(QBrush(QColor(C['pos'])))
        painter.setPen(QPen(QColor("white"), 1))
        painter.drawEllipse(cx - 5, track_y - 5, 10, 10)


# ─────────────────────────────────────────────────────────────
#  ВИДЖЕТ: МАТРИЦА РАССТОЯНИЙ БУФЕРА
# ─────────────────────────────────────────────────────────────
class DistanceMatrixWidget(QWidget):
    """
    Квадратная тепловая карта N×N — расстояния Бхаттачарьи
    между каждой парой кадров в буфере.

    Что читать на матрице
    ─────────────────────
    Однородная синяя матрица
        Все кадры в буфере похожи → буфер стабилен → переходов нет.

    Два синих блока по диагонали + яркая крестовина между ними
        В буфере есть переход: старые кадры похожи между собой,
        новые похожи между собой, но старые ≠ новые.
        Именно это состояние должно давать высокий LR и Бхаттачарью.

    Хаотичная матрица без блочной структуры
        Буфер содержит шум или быстро меняющуюся сцену.
        Ложное срабатывание — LR высокий но блоков нет.

    Красная диагональная полоса (не блок)
        Плавный градиентный переход — освещение меняется постепенно.

    Белая вертикальная/горизонтальная линия
        Один аномальный кадр (засветка, тень) — не настоящий переход.

    Цветовая шкала: синий=0 (идентичны) → красный=1 (максимально различны)
    """

    # Цветовая карта: синий → голубой → зелёный → жёлтый → красный
    _COLORMAP = [
        (0.00, ( 20,  60, 180)),   # тёмно-синий
        (0.20, ( 40, 140, 220)),   # голубой
        (0.40, ( 30, 190, 160)),   # циан-зелёный
        (0.60, (200, 210,  40)),   # жёлто-зелёный
        (0.80, (240, 120,  20)),   # оранжевый
        (1.00, (220,  30,  30)),   # красный
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(200, 200)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet(
            f"background:{C['widget']}; border:1px solid {C['border']}; border-radius:4px;"
        )

        # Последняя матрица N×N расстояний Бхаттачарьи
        self._matrix:     np.ndarray | None = None
        self._n:          int  = 0
        self._m:          int  = 0     # текущий M (граница старое/новое)
        self._is_trigger: bool = False

        # Кэш QImage чтобы не перерисовывать каждый раз
        self._img_cache:  QImage | None = None
        self._cache_n:    int = -1

    # ── Публичный интерфейс ───────────────────────────────────

    def update_matrix(self, matrix: np.ndarray, m: int, is_trigger: bool):
        """
        Принять новую матрицу расстояний.

        Параметры
        ----------
        matrix     : np.ndarray  shape (N, N), dtype float32, значения [0,1]
        m          : int          текущий M — граница старое/новое в буфере
        is_trigger : bool         детектор сейчас в состоянии перехода
        """
        self._matrix     = matrix
        self._n          = matrix.shape[0]
        self._m          = m
        self._is_trigger = is_trigger

        # Инвалидируем кэш только если размер изменился
        if self._n != self._cache_n:
            self._img_cache = None
            self._cache_n   = self._n

        self.update()

    def clear(self):
        self._matrix     = None
        self._n          = 0
        self._img_cache  = None
        self.update()

    # ── Отрисовка ─────────────────────────────────────────────

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()

        painter.fillRect(0, 0, w, h, QColor(C['widget']))

        if self._matrix is None or self._n < 2:
            painter.setPen(QColor(C['muted']))
            painter.setFont(QFont("Consolas", 9))
            painter.drawText(self.rect(), Qt.AlignCenter,
                             "Нет данных\nбуфер заполняется...")
            return

        pad_top   = 14
        pad_right = 52   # место для шкалы
        pad_bot   = 14
        pad_left  = 14

        cell_w = (w - pad_left - pad_right) / self._n
        cell_h = (h - pad_top  - pad_bot)   / self._n

        # ── Рисуем ячейки ─────────────────────────────────────
        for i in range(self._n):
            for j in range(self._n):
                val   = float(self._matrix[i, j])
                color = self._val_to_color(val)
                x = pad_left + j * cell_w
                y = pad_top  + i * cell_h
                painter.fillRect(
                    int(x), int(y),
                    max(int(cell_w), 1), max(int(cell_h), 1),
                    color
                )

        # ── Линия границы M (старое/новое) ────────────────────
        split = self._n - self._m
        if 0 < split < self._n:
            lx = pad_left + split * cell_w
            ly = pad_top  + split * cell_h
            mx_end = pad_left + self._n * cell_w
            my_end = pad_top  + self._n * cell_h

            pen = QPen(QColor("white"), 1.5, Qt.DashLine)
            pen.setDashPattern([3, 2])
            painter.setPen(pen)
            # Вертикальная линия
            painter.drawLine(int(lx), pad_top, int(lx), int(my_end))
            # Горизонтальная линия
            painter.drawLine(pad_left, int(ly), int(mx_end), int(ly))

            # Метки
            painter.setFont(QFont("Consolas", 7))
            painter.setPen(QColor("white"))
            painter.drawText(int(lx) + 2, pad_top + 9, f"M={self._m}")

        # ── Рамка матрицы ─────────────────────────────────────
        mat_w = int(self._n * cell_w)
        mat_h = int(self._n * cell_h)
        border_color = QColor(C['event']) if self._is_trigger else QColor(C['border'])
        painter.setPen(QPen(border_color, 1.5 if self._is_trigger else 0.5))
        painter.drawRect(pad_left, pad_top, mat_w, mat_h)

        # ── Шкала цветов (правый край) ────────────────────────
        sc_x  = w - pad_right + 6
        sc_y  = pad_top
        sc_h  = mat_h
        sc_w  = 10

        steps = 64
        for si in range(steps):
            frac  = si / (steps - 1)
            color = self._val_to_color(frac)
            sy    = sc_y + int(frac * sc_h)
            painter.fillRect(sc_x, sy, sc_w,
                             max(int(sc_h / steps) + 1, 1), color)

        # Метки шкалы
        painter.setPen(QColor(C['muted']))
        painter.setFont(QFont("Consolas", 7))
        painter.drawText(sc_x + sc_w + 2, sc_y + 6,        "1.0")
        painter.drawText(sc_x + sc_w + 2, sc_y + sc_h // 2, "0.5")
        painter.drawText(sc_x + sc_w + 2, sc_y + sc_h - 2,  "0.0")

        # ── Подпись состояния ─────────────────────────────────
        painter.setFont(QFont("Consolas", 8, QFont.Bold))
        if self._is_trigger:
            painter.setPen(QColor(C['event']))
            painter.drawText(pad_left + 2, pad_top - 2, "ПЕРЕХОД")
        else:
            painter.setPen(QColor(C['muted']))
            painter.drawText(pad_left + 2, pad_top - 2,
                             f"N={self._n}  M={self._m}")

    # ── Вспомогательные ───────────────────────────────────────

    def _val_to_color(self, val: float) -> QColor:
        """Интерполировать значение [0,1] в цвет по colormap."""
        val = max(0.0, min(1.0, val))
        cm  = self._COLORMAP
        for k in range(len(cm) - 1):
            t0, c0 = cm[k]
            t1, c1 = cm[k + 1]
            if t0 <= val <= t1:
                frac = (val - t0) / (t1 - t0)
                r = int(c0[0] + frac * (c1[0] - c0[0]))
                g = int(c0[1] + frac * (c1[1] - c0[1]))
                b = int(c0[2] + frac * (c1[2] - c0[2]))
                return QColor(r, g, b)
        return QColor(*cm[-1][1])


# ─────────────────────────────────────────────────────────────
#  СПИСОК СОБЫТИЙ
# ─────────────────────────────────────────────────────────────
class EventListWidget(QListWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFont(QFont("Consolas", 9))

    def add_event(self, event: TransitionEvent):
        text = (
            f"кадр {event.frame_center:5d}  "
            f"conf={event.confidence:.2f}  "
            f"Δ={event.bhattacharyya_max:.3f}  "
            f"LR={event.lr_max:.1f}"
        )
        item = QListWidgetItem(text)
        item.setForeground(QColor(C['event']))
        self.addItem(item)
        self.scrollToBottom()


# ─────────────────────────────────────────────────────────────
#  ПОТОК ОБРАБОТКИ КАДРОВ
# ─────────────────────────────────────────────────────────────
class ProcessingThread(QThread):
    """
    Фоновый поток: генерирует кадры из симулятора или файлов,
    прогоняет через детектор, отправляет результаты в UI.
    """
    frame_ready  = pyqtSignal(np.ndarray, dict)   # (bgr_frame, info)
    event_found  = pyqtSignal(object)              # TransitionEvent
    finished_all = pyqtSignal()

    def __init__(self, source, detector_params: dict, parent=None):
        super().__init__(parent)
        self.source          = source    # FlightSimulator | list[str]
        self.detector_params = detector_params
        self._running        = True
        self._paused         = False
        self._delay_ms       = 33        # ~30 fps

    def run(self):
        det = TransitionDetector(**self.detector_params)

        if isinstance(self.source, FlightSimulator):
            self._run_simulator(det)
        elif isinstance(self.source, list):
            self._run_files(det)

        self.finished_all.emit()

    def _run_simulator(self, det: TransitionDetector):
        # Берём пользовательский маршрут если задан, иначе дефолтный
        wps = getattr(self.source, '_user_waypoints', None)
        if not wps:
            wps = self.source.make_default_route(n_segments=4)
        for hsv_frame, meta in self.source.fly(wps, speed_pix_per_frame=8):
            if not self._running:
                break
            while self._paused and self._running:
                self.msleep(50)

            bgr = cv2.cvtColor(hsv_frame, cv2.COLOR_HSV2BGR)
            ev  = det.process_frame(hsv_frame, meta.frame_number)

            n   = len(det.signal_history)
            bhat = det.signal_history[-1]   if n else 0.0
            lr   = det.lr_history[-1]       if n else 1.0
            thr  = det.threshold_history[-1] if n else 0.3

            # Матрица расстояний буфера (каждые 3 кадра для производительности)
            buf_matrix = None
            buf_m      = 0
            if meta.frame_number % 3 == 0 and det._buffer.is_full():
                import cv2 as _cv2
                states = det._buffer.get_window()
                buf_m  = det._buffer.dynamic_m(min_m=det.min_m)
                n_s    = len(states)
                mat    = np.zeros((n_s, n_s), dtype=np.float32)
                for ii in range(n_s):
                    for jj in range(ii + 1, n_s):
                        d = states[ii].bhattacharyya_distance(states[jj])
                        mat[ii, jj] = d
                        mat[jj, ii] = d
                buf_matrix = mat

            info = {
                "frame_number":  meta.frame_number,
                "bhat":          bhat,
                "lr":            lr,
                "threshold":     thr,
                "in_transition": det._in_transition,
                "position_x":    meta.position_x,
                "position_y":    meta.position_y,
                "map_w":         self.source.map_w,
                "map_h":         self.source.map_h,
                "is_event":      ev is not None,
                "old_hist":      ev.old_state.histogram() if ev and ev.old_state else None,
                "new_hist":      ev.new_state.histogram() if ev and ev.new_state else None,
                "buf_matrix":    buf_matrix,
                "buf_m":         buf_m,
            }
            self.frame_ready.emit(bgr, info)
            if ev:
                self.event_found.emit(ev)
            self.msleep(self._delay_ms)

    def _run_files(self, det: TransitionDetector):
        for i, path in enumerate(self.source):
            if not self._running:
                break
            while self._paused and self._running:
                self.msleep(50)

            bgr = cv2.imread(path)
            if bgr is None:
                continue
            hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
            ev  = det.process_frame(hsv, i)

            n   = len(det.signal_history)
            # Матрица расстояний
            buf_matrix = None
            buf_m      = 0
            if i % 3 == 0 and det._buffer.is_full():
                states = det._buffer.get_window()
                buf_m  = det._buffer.dynamic_m(min_m=det.min_m)
                n_s    = len(states)
                mat    = np.zeros((n_s, n_s), dtype=np.float32)
                for ii in range(n_s):
                    for jj in range(ii + 1, n_s):
                        d = states[ii].bhattacharyya_distance(states[jj])
                        mat[ii, jj] = d
                        mat[jj, ii] = d
                buf_matrix = mat

            info = {
                "frame_number":  i,
                "bhat":          det.signal_history[-1]    if n else 0.0,
                "lr":            det.lr_history[-1]        if n else 1.0,
                "threshold":     det.threshold_history[-1] if n else 0.3,
                "in_transition": det._in_transition,
                "position_x":    float(i),
                "position_y":    0.0,
                "map_w":         len(self.source),
                "map_h":         1,
                "is_event":      ev is not None,
                "old_hist":      ev.old_state.histogram() if ev and ev.old_state else None,
                "new_hist":      ev.new_state.histogram() if ev and ev.new_state else None,
                "buf_matrix":    buf_matrix,
                "buf_m":         buf_m,
            }
            self.frame_ready.emit(bgr, info)
            if ev:
                self.event_found.emit(ev)
            self.msleep(self._delay_ms)

    def pause(self):   self._paused = True
    def resume(self):  self._paused = False
    def stop(self):
        self._running = False
        self._paused  = False


# ─────────────────────────────────────────────────────────────
#  ВИДЖЕТ: РЕДАКТОР МАРШРУТА
# ─────────────────────────────────────────────────────────────
class RouteEditorWidget(QWidget):
    """
    Миниатюра карты с рисованием маршрута мышью.

    Управление:
        ЛКМ           — добавить точку маршрута
        ПКМ           — удалить последнюю точку
        Двойной ЛКМ   — завершить маршрут (фиксировать)
        Колесо мыши   — зум миниатюры
        Alt + ЛКМ     — панорамирование

    Сигналы:
        route_changed(list[Waypoint]) — маршрут изменился
    """

    route_changed = pyqtSignal(list)

    # Цвета элементов маршрута
    CLR_WP        = QColor("#4fc3f7")   # waypoint
    CLR_WP_FIRST  = QColor("#81c784")   # стартовая точка
    CLR_WP_LAST   = QColor("#ef5350")   # конечная точка
    CLR_LINE      = QColor("#7986cb")   # линия маршрута
    CLR_PREVIEW   = QColor("#ffb74d")   # линия к курсору
    CLR_DONE      = QColor("#4db6ac")   # финальный маршрут

    WP_RADIUS     = 6    # радиус кружка waypoint в экранных пикселях
    MIN_WP_DIST   = 8    # минимальное расстояние между точками (пкс экрана)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(280, 180)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet(
            f"background:{C['widget']}; border:1px solid {C['border']}; border-radius:4px;"
        )
        self.setMouseTracking(True)

        # Данные
        self._map_pixmap:  QPixmap | None = None   # оригинальный снимок
        self._map_w:       int = 1
        self._map_h:       int = 1
        self._waypoints:   list  = []   # [(map_x, map_y, name), ...]
        self._route_done:  bool  = False
        self._cursor_pos:  tuple | None = None   # позиция курсора в map-coords

        # Вид (для зума/панорамы)
        self._view_offset  = [0.0, 0.0]   # смещение в map-coords
        self._view_scale   = 1.0           # масштаб (map_px / screen_px)
        self._panning      = False
        self._pan_start    = None

        # Кнопки управления
        btn_layout = QHBoxLayout()
        btn_layout.setContentsMargins(0, 0, 0, 0)
        btn_layout.setSpacing(4)

        self.btn_clear   = QPushButton("Очистить")
        self.btn_undo    = QPushButton("← Отмена")
        self.btn_done    = QPushButton("Готово ✓")
        self.btn_fit     = QPushButton("По экрану")

        for btn in [self.btn_clear, self.btn_undo, self.btn_done, self.btn_fit]:
            btn.setFixedHeight(22)
            btn_layout.addWidget(btn)

        self.btn_clear.clicked.connect(self._clear_route)
        self.btn_undo.clicked.connect(self._undo_last)
        self.btn_done.clicked.connect(self._finish_route)
        self.btn_fit.clicked.connect(self._fit_view)

        # Подсказка
        self.lbl_hint = QLabel("ЛКМ — точка  |  ПКМ — отмена  |  2×ЛКМ — готово")
        self.lbl_hint.setStyleSheet(f"color:{C['muted']}; font-size:9px;")
        self.lbl_hint.setAlignment(Qt.AlignCenter)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(2, 2, 2, 2)
        outer.setSpacing(2)
        outer.addLayout(btn_layout)
        self._canvas = _RouteCanvas(self)
        outer.addWidget(self._canvas, stretch=1)
        outer.addWidget(self.lbl_hint)

    # ── Публичные методы ──────────────────────────────────────

    def load_map(self, image_path: str):
        """Загрузить изображение карты в виджет."""
        pix = QPixmap(image_path)
        if pix.isNull():
            return
        self._map_pixmap = pix
        self._map_w      = pix.width()
        self._map_h      = pix.height()
        self._clear_route()
        self._fit_view()
        self._canvas.update()

    def get_waypoints(self) -> list:
        """
        Вернуть список Waypoint в координатах карты.
        Пустой список если маршрут не задан.
        """
        if len(self._waypoints) < 2:
            return []
        return [
            Waypoint(x, y, name=f"wp_{i:02d}")
            for i, (x, y, _) in enumerate(self._waypoints)
        ]

    def has_route(self) -> bool:
        return len(self._waypoints) >= 2

    # ── Слоты кнопок ─────────────────────────────────────────

    def _clear_route(self):
        self._waypoints  = []
        self._route_done = False
        self.lbl_hint.setText("ЛКМ — точка  |  ПКМ — отмена  |  2×ЛКМ — готово")
        self._canvas.update()
        self.route_changed.emit([])

    def _undo_last(self):
        if self._waypoints:
            self._waypoints.pop()
            self._route_done = False
            self._canvas.update()
            self.route_changed.emit(self.get_waypoints())

    def _finish_route(self):
        if len(self._waypoints) >= 2:
            self._route_done = True
            n = len(self._waypoints)
            self.lbl_hint.setText(
                f"Маршрут задан: {n} точек  |  нажмите Старт"
            )
            self._canvas.update()
            self.route_changed.emit(self.get_waypoints())

    def _fit_view(self):
        """Подогнать масштаб чтобы вся карта влезла в виджет."""
        if self._map_pixmap is None:
            return
        cw = self._canvas.width()
        ch = self._canvas.height()
        if cw < 1 or ch < 1:
            return
        sx = self._map_w / cw
        sy = self._map_h / ch
        self._view_scale  = max(sx, sy)
        self._view_offset = [0.0, 0.0]
        self._canvas.update()

    # ── Преобразования координат ──────────────────────────────

    def _screen_to_map(self, sx: float, sy: float) -> tuple:
        """Экранные координаты (canvas) → координаты карты."""
        mx = self._view_offset[0] + sx * self._view_scale
        my = self._view_offset[1] + sy * self._view_scale
        return mx, my

    def _map_to_screen(self, mx: float, my: float) -> tuple:
        """Координаты карты → экранные координаты (canvas)."""
        sx = (mx - self._view_offset[0]) / self._view_scale
        sy = (my - self._view_offset[1]) / self._view_scale
        return sx, sy

    # ── Обработка событий мыши (делегируем в canvas) ─────────

    def _mouse_press(self, event):
        if self._map_pixmap is None:
            return

        if event.button() == Qt.MiddleButton or (
            event.button() == Qt.LeftButton and
            event.modifiers() & Qt.AltModifier
        ):
            self._panning   = True
            self._pan_start = (event.x(), event.y())
            return

        mx, my = self._screen_to_map(event.x(), event.y())

        if event.button() == Qt.RightButton:
            self._undo_last()
            return

        if event.button() == Qt.LeftButton and not self._route_done:
            # Проверяем минимальное расстояние до последней точки
            if self._waypoints:
                lx, ly, _ = self._waypoints[-1]
                lsx, lsy  = self._map_to_screen(lx, ly)
                dist = ((event.x() - lsx) ** 2 + (event.y() - lsy) ** 2) ** 0.5
                if dist < self.MIN_WP_DIST:
                    return
            name = f"wp_{len(self._waypoints):02d}"
            self._waypoints.append((mx, my, name))
            self._canvas.update()
            self.route_changed.emit(self.get_waypoints())

    def _mouse_release(self, event):
        if self._panning:
            self._panning   = False
            self._pan_start = None

    def _mouse_move(self, event):
        mx, my = self._screen_to_map(event.x(), event.y())
        self._cursor_pos = (mx, my)

        if self._panning and self._pan_start:
            dx = (event.x() - self._pan_start[0]) * self._view_scale
            dy = (event.y() - self._pan_start[1]) * self._view_scale
            self._view_offset[0] -= dx
            self._view_offset[1] -= dy
            # Ограничиваем смещение
            self._view_offset[0] = max(0, min(self._view_offset[0],
                                              self._map_w - self._canvas.width() * self._view_scale))
            self._view_offset[1] = max(0, min(self._view_offset[1],
                                              self._map_h - self._canvas.height() * self._view_scale))
            self._pan_start = (event.x(), event.y())

        self._canvas.update()

    def _mouse_double_click(self, event):
        if event.button() == Qt.LeftButton:
            self._finish_route()

    def _wheel(self, event):
        if self._map_pixmap is None:
            return
        factor = 0.85 if event.angleDelta().y() > 0 else 1.18
        new_scale = self._view_scale * factor

        # Ограничиваем зум
        min_scale = min(self._map_w / max(self._canvas.width(), 1),
                        self._map_h / max(self._canvas.height(), 1))
        new_scale = max(min_scale * 0.5, min(new_scale, float(max(self._map_w, self._map_h))))

        # Зум вокруг курсора
        cx, cy = self._screen_to_map(event.x(), event.y())
        self._view_scale    = new_scale
        self._view_offset[0] = cx - event.x() * new_scale
        self._view_offset[1] = cy - event.y() * new_scale
        self._canvas.update()

    def _paint_canvas(self, painter: QPainter, w: int, h: int):
        """Отрисовка содержимого canvas."""
        painter.fillRect(0, 0, w, h, QColor(C['widget']))

        if self._map_pixmap is None:
            painter.setPen(QColor(C['muted']))
            painter.setFont(QFont("Consolas", 9))
            painter.drawText(
                0, 0, w, h, Qt.AlignCenter,
                "Загрузите карту\n(кнопка  🗺  Загрузить карту)"
            )
            return

        # ── Рисуем миниатюру карты ────────────────────────────
        vis_w = int(w * self._view_scale)
        vis_h = int(h * self._view_scale)
        ox    = int(self._view_offset[0])
        oy    = int(self._view_offset[1])

        src_rect  = (ox, oy, min(vis_w, self._map_w - ox),
                              min(vis_h, self._map_h - oy))
        dest_rect = (0, 0, w, h)

        # Вырезаем нужный кусок карты и масштабируем
        cropped = self._map_pixmap.copy(ox, oy, src_rect[2], src_rect[3])
        scaled  = cropped.scaled(w, h, Qt.IgnoreAspectRatio,
                                  Qt.SmoothTransformation)
        painter.drawPixmap(0, 0, scaled)

        # ── Тёмный оверлей для читаемости ─────────────────────
        painter.fillRect(0, 0, w, h, QColor(0, 0, 0, 60))

        # ── Линии маршрута ────────────────────────────────────
        if len(self._waypoints) >= 2:
            color = self.CLR_DONE if self._route_done else self.CLR_LINE
            pen   = QPen(color, 2)
            painter.setPen(pen)
            for i in range(len(self._waypoints) - 1):
                x0, y0, _ = self._waypoints[i]
                x1, y1, _ = self._waypoints[i + 1]
                sx0, sy0  = self._map_to_screen(x0, y0)
                sx1, sy1  = self._map_to_screen(x1, y1)
                painter.drawLine(int(sx0), int(sy0), int(sx1), int(sy1))

        # ── Линия предпросмотра к курсору ─────────────────────
        if (self._waypoints and not self._route_done
                and self._cursor_pos):
            lx, ly, _ = self._waypoints[-1]
            sx0, sy0  = self._map_to_screen(lx, ly)
            sx1, sy1  = self._map_to_screen(*self._cursor_pos)
            pen = QPen(self.CLR_PREVIEW, 1, Qt.DashLine)
            pen.setDashPattern([4, 3])
            painter.setPen(pen)
            painter.drawLine(int(sx0), int(sy0), int(sx1), int(sy1))

        # ── Точки маршрута ────────────────────────────────────
        for i, (mx, my, name) in enumerate(self._waypoints):
            sx, sy = self._map_to_screen(mx, my)
            sx, sy = int(sx), int(sy)

            if i == 0:
                color = self.CLR_WP_FIRST
            elif i == len(self._waypoints) - 1:
                color = self.CLR_WP_LAST
            else:
                color = self.CLR_WP

            painter.setBrush(QBrush(color))
            painter.setPen(QPen(QColor("white"), 1.5))
            painter.drawEllipse(sx - self.WP_RADIUS, sy - self.WP_RADIUS,
                                self.WP_RADIUS * 2, self.WP_RADIUS * 2)

            # Номер точки
            painter.setPen(QColor("white"))
            painter.setFont(QFont("Consolas", 7, QFont.Bold))
            painter.drawText(sx + self.WP_RADIUS + 2, sy + 4, str(i))

        # ── Информация ────────────────────────────────────────
        painter.setPen(QColor(C['muted']))
        painter.setFont(QFont("Consolas", 8))
        info = f"карта {self._map_w}×{self._map_h}  |  точек: {len(self._waypoints)}"
        painter.drawText(4, h - 4, info)

        # ── Координаты курсора ────────────────────────────────
        if self._cursor_pos:
            cx, cy = self._cursor_pos
            cx = max(0, min(cx, self._map_w))
            cy = max(0, min(cy, self._map_h))
            painter.setPen(QColor(C['signal']))
            painter.drawText(w - 120, h - 4, f"x={cx:.0f} y={cy:.0f}")


class _RouteCanvas(QWidget):
    """Внутренний холст — делегирует все события в RouteEditorWidget."""

    def __init__(self, editor: RouteEditorWidget):
        super().__init__(editor)
        self._editor = editor
        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        painter.setRenderHint(QPainter.Antialiasing)
        self._editor._paint_canvas(painter, self.width(), self.height())

    def mousePressEvent(self, e):      self._editor._mouse_press(e)
    def mouseReleaseEvent(self, e):    self._editor._mouse_release(e)
    def mouseMoveEvent(self, e):       self._editor._mouse_move(e)
    def mouseDoubleClickEvent(self, e): self._editor._mouse_double_click(e)
    def wheelEvent(self, e):           self._editor._wheel(e)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._editor._fit_view()


# ─────────────────────────────────────────────────────────────
#  ПАНЕЛЬ ПАРАМЕТРОВ ДЕТЕКТОРА
# ─────────────────────────────────────────────────────────────
class ParamPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        grp = QGroupBox("Параметры детектора")
        form = QFormLayout(grp)
        form.setSpacing(5)

        self.spin_window = QSpinBox()
        self.spin_window.setRange(10, 100)
        self.spin_window.setValue(30)
        form.addRow("Окно:", self.spin_window)

        self.spin_minm = QSpinBox()
        self.spin_minm.setRange(2, 30)
        self.spin_minm.setValue(5)
        form.addRow("Min M:", self.spin_minm)

        self.spin_alpha = QDoubleSpinBox()
        self.spin_alpha.setRange(0.01, 0.3)
        self.spin_alpha.setSingleStep(0.01)
        self.spin_alpha.setValue(0.05)
        form.addRow("α (FA):", self.spin_alpha)

        self.spin_lr = QDoubleSpinBox()
        self.spin_lr.setRange(1.0, 10.0)
        self.spin_lr.setSingleStep(0.1)
        self.spin_lr.setValue(1.2)
        form.addRow("LR фильтр:", self.spin_lr)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(grp)

    def get_params(self) -> dict:
        return {
            "window_size":      self.spin_window.value(),
            "min_m":            self.spin_minm.value(),
            "false_alarm_rate": self.spin_alpha.value(),
            "lr_filter_thresh": self.spin_lr.value(),
        }


# ─────────────────────────────────────────────────────────────
#  ГЛАВНОЕ ОКНО
# ─────────────────────────────────────────────────────────────
class VisualizerWindow(QMainWindow):
    def __init__(self, args=None):
        super().__init__()
        self.setWindowTitle("Детектор переходов — визуализатор")
        self.setMinimumSize(1200, 720)
        self.resize(1440, 860)
        self.setStyleSheet(STYLESHEET)

        self._thread:     ProcessingThread | None = None
        self._sim:        FlightSimulator  | None = None
        self._zone_color  = QColor(C['signal'])
        self._last_old_hist = None
        self._last_new_hist = None

        self._build_toolbar()
        self._build_ui()
        self._build_statusbar()

        # Автозапуск демо если указано
        if args and args.demo:
            self._start_demo()
        elif args and args.image:
            self._start_from_image(args.image)
        elif args and args.frames:
            self._start_from_frames(args.frames)

    # ── Тулбар ───────────────────────────────────────────────

    def _build_toolbar(self):
        tb = QToolBar("Управление")
        tb.setMovable(False)
        self.addToolBar(tb)

        self.act_demo   = QAction("▶  Демо (синтетика)", self)
        self.act_image  = QAction("🗺  Загрузить карту...", self)
        self.act_frames = QAction("📂  Папка кадров...", self)
        self.act_play   = QAction("▶  Старт", self)
        self.act_pause  = QAction("⏸  Пауза", self)
        self.act_stop   = QAction("⏹  Стоп", self)

        self.act_pause.setEnabled(False)
        self.act_stop.setEnabled(False)

        for a in [self.act_demo, self.act_image, self.act_frames,
                  None, self.act_play, self.act_pause, self.act_stop]:
            if a:
                tb.addAction(a)
            else:
                tb.addSeparator()

        self.act_demo.triggered.connect(self._start_demo)
        self.act_image.triggered.connect(self._open_image_dialog)
        self.act_frames.triggered.connect(self._open_frames_dialog)
        self.act_play.triggered.connect(self._on_play)
        self.act_pause.triggered.connect(self._on_pause)
        self.act_stop.triggered.connect(self._on_stop)

    # ── UI ───────────────────────────────────────────────────

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(6)

        # ── Левая колонка: видео + параметры + события ────────
        left = QVBoxLayout()
        left.setSpacing(6)

        # Видео
        vg = QGroupBox("КАДР КАМЕРЫ")
        vgl = QVBoxLayout(vg)
        vgl.setContentsMargins(4, 4, 4, 4)
        self.frame_widget = FrameWidget()
        self.frame_widget.setFixedSize(320, 240)
        vgl.addWidget(self.frame_widget)
        left.addWidget(vg)

        # Редактор маршрута
        rg = QGroupBox("МАРШРУТ  (ЛКМ=точка  ПКМ=отмена  2×ЛКМ=готово)")
        rgl = QVBoxLayout(rg)
        rgl.setContentsMargins(4, 4, 4, 4)
        self.route_editor = RouteEditorWidget()
        self.route_editor.setMinimumHeight(200)
        rgl.addWidget(self.route_editor)
        left.addWidget(rg)

        # Параметры
        self.param_panel = ParamPanel()
        left.addWidget(self.param_panel)

        # Список событий
        eg = QGroupBox("ОБНАРУЖЕННЫЕ ПЕРЕХОДЫ")
        egl = QVBoxLayout(eg)
        egl.setContentsMargins(4, 4, 4, 4)
        self.event_list = EventListWidget()
        self.event_list.setMaximumHeight(130)
        egl.addWidget(self.event_list)
        left.addWidget(eg)

        left.addStretch()

        lw = QWidget()
        lw.setLayout(left)
        lw.setFixedWidth(380)
        root.addWidget(lw)

        # ── Правая колонка: графики ────────────────────────────
        right = QVBoxLayout()
        right.setSpacing(6)

        # График сигналов
        sg = QGroupBox("СИГНАЛ ДЕТЕКТОРА  (Бхаттачарья / LR / Порог)")
        sgl = QVBoxLayout(sg)
        sgl.setContentsMargins(4, 4, 4, 4)
        self.signal_plot = SignalPlot()
        self.signal_plot.setMinimumHeight(160)
        sgl.addWidget(self.signal_plot)
        right.addWidget(sg, stretch=2)

        # Гистограммы
        hrow = QHBoxLayout()
        hrow.setSpacing(6)

        hog = QGroupBox("ГИСТОГРАММА H — старая зона")
        hogl = QVBoxLayout(hog)
        hogl.setContentsMargins(4, 4, 4, 4)
        self.hist_old = HistWidget("старая зона")
        hogl.addWidget(self.hist_old)
        hrow.addWidget(hog)

        hng = QGroupBox("ГИСТОГРАММА H — новая зона")
        hngl = QVBoxLayout(hng)
        hngl.setContentsMargins(4, 4, 4, 4)
        self.hist_new = HistWidget("новая зона")
        hngl.addWidget(self.hist_new)
        hrow.addWidget(hng)

        hw = QWidget()
        hw.setLayout(hrow)
        right.addWidget(hw, stretch=1)

        # Матрица расстояний
        mg = QGroupBox("МАТРИЦА РАССТОЯНИЙ БУФЕРА  (синий=похожи  красный=различны)")
        mgl = QHBoxLayout(mg)
        mgl.setContentsMargins(4, 4, 4, 4)
        self.dist_matrix = DistanceMatrixWidget()
        mgl.addWidget(self.dist_matrix)

        # Легенда справа от матрицы
        leg = QVBoxLayout()
        leg.setSpacing(4)
        leg.setContentsMargins(6, 0, 0, 0)
        for txt, color in [
            ("Синий блок\n= стабильная зона",    C['signal']),
            ("2 блока + крест\n= настоящий переход", C['event']),
            ("Хаос\n= шум/ложняк",               C['threshold']),
            ("Диагональ\n= плавный градиент",     C['muted']),
        ]:
            lbl = QLabel(txt)
            lbl.setStyleSheet(f"color:{color}; font-size:9px;")
            lbl.setWordWrap(True)
            lbl.setMaximumWidth(110)
            leg.addWidget(lbl)
        leg.addStretch()
        mgl.addLayout(leg)

        right.addWidget(mg, stretch=2)

        # Траектория
        tg = QGroupBox("ТРАЕКТОРИЯ НАД КАРТОЙ")
        tgl = QVBoxLayout(tg)
        tgl.setContentsMargins(4, 4, 4, 4)
        self.traj_widget = TrajectoryWidget()
        tgl.addWidget(self.traj_widget)
        right.addWidget(tg, stretch=0)

        rw = QWidget()
        rw.setLayout(right)
        root.addWidget(rw, stretch=1)

    def _build_statusbar(self):
        sb = QStatusBar()
        self.setStatusBar(sb)
        self.lbl_status  = QLabel("Готов")
        self.lbl_frame   = QLabel("кадр: —")
        self.lbl_bhat    = QLabel("Бхатт: —")
        self.lbl_events  = QLabel("событий: 0")
        for lbl in [self.lbl_status, self.lbl_frame,
                    self.lbl_bhat,   self.lbl_events]:
            sb.addPermanentWidget(lbl)

    # ── Запуск ────────────────────────────────────────────────

    def _start_demo(self):
        """Синтетическая карта: три вертикальные полосы."""
        W, H = 2000, 600
        synth = np.zeros((H, W, 3), dtype=np.uint8)
        synth[:, :650]     = [80,  200, 180]
        synth[:, 650:1300] = [30,  100,  30]
        synth[:, 1300:]    = [100, 110, 120]
        noise = np.random.randint(0, 20, synth.shape, dtype=np.uint8)
        synth = np.clip(synth.astype(np.int32) + noise, 0, 255).astype(np.uint8)

        import tempfile
        self._tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        cv2.imwrite(self._tmp.name, synth)
        self._start_from_image(self._tmp.name, title="Демо — синтетическая карта")

    def _open_image_dialog(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Открыть карту",
            "", "Изображения (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"
        )
        if path:
            self._start_from_image(path)

    def _open_frames_dialog(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Папка с кадрами (PNG)"
        )
        if folder:
            self._start_from_frames(folder)

    def _start_from_image(self, path: str, title: str = None):
        self._on_stop()
        try:
            self._sim = FlightSimulator(
                path,
                camera_w=320, camera_h=240,
                noise_level=8, shake_pixels=1.5,
                output_format="hsv"
            )
        except FileNotFoundError as e:
            self.lbl_status.setText(str(e))
            return

        # Загружаем миниатюру карты в редактор маршрута
        self.route_editor.load_map(path)
        self._map_path = path

        self.traj_widget.set_map_size(self._sim.map_w, self._sim.map_h)
        self.setWindowTitle(title or f"Детектор — {os.path.basename(path)}")
        self.lbl_status.setText(
            f"Карта: {self._sim.map_w}×{self._sim.map_h}  |  нарисуйте маршрут и нажмите Старт"
        )

    def _start_from_frames(self, folder: str):
        self._on_stop()
        paths = sorted([
            os.path.join(folder, f) for f in os.listdir(folder)
            if f.lower().endswith(".png")
        ])
        if not paths:
            self.lbl_status.setText("Папка не содержит PNG файлов")
            return

        self._sim = None
        self._frame_paths = paths
        self.traj_widget.set_map_size(len(paths), 1)
        self.lbl_status.setText(f"Кадров: {len(paths)}  |  готов к запуску")

    # ── Управление воспроизведением ───────────────────────────

    def _on_play(self):
        if self._thread and self._thread.isRunning():
            return

        # Сбрасываем UI
        self.signal_plot.reset()
        self.traj_widget.reset()
        self.dist_matrix.clear()
        self.event_list.clear()
        self.lbl_events.setText("событий: 0")

        params = self.param_panel.get_params()

        if self._sim:
            # Берём маршрут из редактора или дефолтный
            wps = self.route_editor.get_waypoints()
            if not wps:
                wps = self._sim.make_default_route(n_segments=3)
                self.lbl_status.setText(
                    "Маршрут не задан — используется автоматический"
                )
            # Передаём waypoints в симулятор через атрибут
            self._sim._user_waypoints = wps
            source = self._sim
        elif hasattr(self, "_frame_paths"):
            source = self._frame_paths
        else:
            self._start_demo()
            return

        self._thread = ProcessingThread(source, params)
        self._thread.frame_ready.connect(self._on_frame)
        self._thread.event_found.connect(self._on_event)
        self._thread.finished_all.connect(self._on_finished)
        self._thread.start()

        self.act_play.setEnabled(False)
        self.act_pause.setEnabled(True)
        self.act_stop.setEnabled(True)
        self.lbl_status.setText("Воспроизведение...")

    def _on_pause(self):
        if self._thread:
            if self._thread._paused:
                self._thread.resume()
                self.act_pause.setText("⏸  Пауза")
                self.lbl_status.setText("Воспроизведение...")
            else:
                self._thread.pause()
                self.act_pause.setText("▶  Продолжить")
                self.lbl_status.setText("Пауза")

    def _on_stop(self):
        if self._thread:
            self._thread.stop()
            self._thread.wait(2000)
            self._thread = None
        self.act_play.setEnabled(True)
        self.act_pause.setEnabled(False)
        self.act_stop.setEnabled(False)
        self.act_pause.setText("⏸  Пауза")
        self.lbl_status.setText("Остановлено")

    def _on_finished(self):
        self._on_stop()
        self.lbl_status.setText("Завершено")

    # ── Обновление UI ─────────────────────────────────────────

    def _on_frame(self, bgr: np.ndarray, info: dict):
        fn   = info["frame_number"]
        bhat = info["bhat"]
        lr   = info["lr"]
        thr  = info["threshold"]

        # Цвет зоны по текущему HSV (грубая оценка по средней яркости)
        hsv_f = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        h_mean = int(hsv_f[:, :, 0].mean())
        s_mean = int(hsv_f[:, :, 1].mean())
        v_mean = int(hsv_f[:, :, 2].mean())
        self._zone_color = QColor.fromHsv(
            h_mean * 2, max(s_mean, 80), max(v_mean, 120)
        )

        # Кадр
        self.frame_widget.set_frame(
            bgr, fn, info["in_transition"], self._zone_color
        )

        # График
        self.signal_plot.add_point(bhat, lr, thr, info["is_event"])

        # Траектория
        self.traj_widget.add_position(
            info["position_x"], info["position_y"], QColor(self._zone_color)
        )
        if info["is_event"]:
            self.traj_widget.add_event()

        # Матрица расстояний
        if info.get("buf_matrix") is not None:
            self.dist_matrix.update_matrix(
                info["buf_matrix"],
                info["buf_m"],
                info["in_transition"]
            )

        # Гистограммы при переходе
        if info["old_hist"] is not None:
            self.hist_old.set_histogram(info["old_hist"])
        if info["new_hist"] is not None:
            self.hist_new.set_histogram(info["new_hist"])

        # Статусбар
        self.lbl_frame.setText(f"кадр: {fn:6d}")
        self.lbl_bhat.setText(f"Бхатт: {bhat:.3f}  |  порог: {thr:.3f}")

    def _on_event(self, event: TransitionEvent):
        self.event_list.add_event(event)
        n = self.event_list.count()
        self.lbl_events.setText(f"событий: {n}")

    def closeEvent(self, event):
        self._on_stop()
        super().closeEvent(event)


# ─────────────────────────────────────────────────────────────
#  ЗАПУСК
# ─────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Визуализатор детектора переходов")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--demo",   action="store_true",  help="Синтетика (без файлов)")
    g.add_argument("--image",  type=str, default=None, help="Путь к изображению карты")
    g.add_argument("--frames", type=str, default=None, help="Папка с PNG кадрами")
    return p.parse_args()


def main():
    args = parse_args()
    app  = QApplication(sys.argv)
    app.setApplicationName("TransitionDetector Visualizer")
    win  = VisualizerWindow(args)
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
