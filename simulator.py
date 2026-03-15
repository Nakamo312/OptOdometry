"""
simulator.py — Задача 13
Симулятор полёта БПЛА над спутниковым снимком.

Использование
-------------
python simulator.py --image map.png --output frames/ --speed 5

Или программно:
    sim = FlightSimulator("map.png")
    for frame, meta in sim.fly(waypoints):
        detector.process_frame(frame, meta.frame_number)
"""

import cv2
import numpy as np
import json
import os
import argparse
from dataclasses import dataclass, field
from typing import List, Tuple, Iterator, Optional


# ─────────────────────────────────────────────────────────────
#  СТРУКТУРЫ ДАННЫХ
# ─────────────────────────────────────────────────────────────

@dataclass
class Waypoint:
    """Точка маршрута в пикселях карты."""
    x: float
    y: float
    name: str = ""


@dataclass
class FrameMeta:
    """Метаданные одного кадра симулятора."""
    frame_number:  int
    position_x:   float          # центр окна камеры, пиксели карты
    position_y:   float
    yaw:          float          # курс в радианах
    speed:        float          # пикс/кадр
    # Истинный момент пересечения границы (если есть)
    true_transition: bool = False
    segment_name:    str  = ""   # название участка ("поле_1 → лесополка")


@dataclass
class GroundTruth:
    """Истинные моменты переходов для оценки детектора."""
    frame_number: int
    position_x:  float
    position_y:  float
    from_zone:   str
    to_zone:     str


# ─────────────────────────────────────────────────────────────
#  СИМУЛЯТОР
# ─────────────────────────────────────────────────────────────

class FlightSimulator:
    """
    Симулирует полёт камеры над спутниковым снимком.

    Параметры
    ----------
    image_path    : str   — путь к изображению карты (PNG/JPG)
    camera_w      : int   — ширина кадра камеры в пикселях
    camera_h      : int   — высота кадра камеры в пикселях
    fps           : float — кадров в секунду (для расчёта метрик)
    noise_level   : float — уровень шума яркости [0, 50]
    shake_pixels  : float — амплитуда дрожания камеры в пикселях
    output_format : str   — "hsv" или "bgr" — формат возвращаемых кадров
    """

    def __init__(self,
                 image_path:    str,
                 camera_w:      int   = 320,
                 camera_h:      int   = 240,
                 fps:           float = 30.0,
                 noise_level:   float = 10.0,
                 shake_pixels:  float = 2.0,
                 output_format: str   = "hsv"):

        self.camera_w      = camera_w
        self.camera_h      = camera_h
        self.fps           = fps
        self.noise_level   = noise_level
        self.shake_pixels  = shake_pixels
        self.output_format = output_format.lower()

        # Загрузка карты
        self._map_bgr = cv2.imread(image_path)
        if self._map_bgr is None:
            raise FileNotFoundError(f"Не удалось загрузить: {image_path}")

        self.map_h, self.map_w = self._map_bgr.shape[:2]
        print(f"  Карта загружена: {self.map_w}×{self.map_h} пикселей")

        # Для цветовой квантизации зон (задача 9 в будущем)
        self._zone_map: Optional[np.ndarray] = None

        # История полёта
        self._ground_truth: List[GroundTruth] = []

    # ── Публичные методы ──────────────────────────────────────

    def fly(self,
            waypoints: List[Waypoint],
            speed_pix_per_frame: float = 5.0
            ) -> Iterator[Tuple[np.ndarray, FrameMeta]]:
        """
        Генератор кадров по маршруту.

        Параметры
        ----------
        waypoints            : список точек маршрута
        speed_pix_per_frame  : скорость движения в пикселях на кадр

        Yields
        ------
        (frame, meta) — HSV (или BGR) кадр и его метаданные
        """
        if len(waypoints) < 2:
            raise ValueError("Нужно минимум 2 waypoint")

        frame_number = 0
        self._ground_truth.clear()

        for seg_idx in range(len(waypoints) - 1):
            wp_start = waypoints[seg_idx]
            wp_end   = waypoints[seg_idx + 1]

            # Вектор сегмента
            dx       = wp_end.x - wp_start.x
            dy       = wp_end.y - wp_start.y
            dist     = np.sqrt(dx**2 + dy**2)
            yaw      = np.arctan2(dy, dx)
            n_frames = max(1, int(dist / speed_pix_per_frame))

            for step in range(n_frames):
                t   = step / n_frames
                cx  = wp_start.x + t * dx
                cy  = wp_start.y + t * dy

                # Дрожание камеры
                cx += np.random.uniform(-self.shake_pixels, self.shake_pixels)
                cy += np.random.uniform(-self.shake_pixels, self.shake_pixels)

                frame = self._capture_frame(cx, cy)

                meta = FrameMeta(
                    frame_number=frame_number,
                    position_x=cx,
                    position_y=cy,
                    yaw=yaw,
                    speed=speed_pix_per_frame,
                    segment_name=f"{wp_start.name}→{wp_end.name}"
                )

                yield frame, meta
                frame_number += 1

    def fly_linear(self,
                   x0: float, y0: float,
                   x1: float, y1: float,
                   speed: float = 5.0,
                   name: str = "route"
                   ) -> Iterator[Tuple[np.ndarray, FrameMeta]]:
        """
        Упрощённый вариант: прямолинейный полёт от (x0,y0) до (x1,y1).
        """
        wp = [
            Waypoint(x0, y0, f"{name}_start"),
            Waypoint(x1, y1, f"{name}_end"),
        ]
        yield from self.fly(wp, speed_pix_per_frame=speed)

    def make_default_route(self,
                           margin: float = 0.1,
                           n_segments: int = 3
                           ) -> List[Waypoint]:
        """
        Сгенерировать маршрут по умолчанию: зигзаг через карту.

        Используется когда нет заданного маршрута.
        """
        wps = []
        mx  = int(self.map_w * margin)
        my  = int(self.map_h * margin)

        for i in range(n_segments + 1):
            t = i / n_segments
            x = mx + t * (self.map_w - 2 * mx)
            # Зигзаг по Y
            y = (my if i % 2 == 0 else self.map_h - my)
            wps.append(Waypoint(x, y, name=f"wp_{i}"))

        return wps

    def save_ground_truth(self, path: str):
        """Сохранить истинные моменты переходов в JSON."""
        data = [
            {
                "frame": gt.frame_number,
                "x":     gt.position_x,
                "y":     gt.position_y,
                "from":  gt.from_zone,
                "to":    gt.to_zone,
            }
            for gt in self._ground_truth
        ]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"  Ground truth сохранён: {path}")

    def save_frames(self,
                    waypoints: List[Waypoint],
                    output_dir: str,
                    speed: float = 5.0,
                    max_frames: int = 5000):
        """
        Сохранить все кадры маршрута в папку output_dir.
        Также сохраняет ground_truth.json.
        """
        os.makedirs(output_dir, exist_ok=True)
        meta_list = []

        for frame_bgr, meta in self.fly(waypoints, speed):
            if meta.frame_number >= max_frames:
                break
            path = os.path.join(output_dir, f"frame_{meta.frame_number:06d}.png")
            # Сохраняем BGR (стандарт OpenCV)
            if self.output_format == "hsv":
                save_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_HSV2BGR)
            else:
                save_frame = frame_bgr
            cv2.imwrite(path, save_frame)
            meta_list.append({
                "frame": meta.frame_number,
                "x":     meta.position_x,
                "y":     meta.position_y,
                "yaw":   meta.yaw,
            })

        with open(os.path.join(output_dir, "trajectory.json"), "w") as f:
            json.dump(meta_list, f, indent=2)

        print(f"  Сохранено {len(meta_list)} кадров в {output_dir}/")

    # ── Приватные методы ──────────────────────────────────────

    def _capture_frame(self, cx: float, cy: float) -> np.ndarray:
        """
        Вырезать кадр камеры из карты с центром в (cx, cy).
        Добавить шум яркости.
        """
        x0 = int(cx - self.camera_w // 2)
        y0 = int(cy - self.camera_h // 2)
        x1 = x0 + self.camera_w
        y1 = y0 + self.camera_h

        # Паддинг если вышли за границы карты
        pad_left   = max(0, -x0)
        pad_top    = max(0, -y0)
        pad_right  = max(0, x1 - self.map_w)
        pad_bottom = max(0, y1 - self.map_h)

        x0c = max(0, x0);  y0c = max(0, y0)
        x1c = min(self.map_w, x1);  y1c = min(self.map_h, y1)

        crop = self._map_bgr[y0c:y1c, x0c:x1c]

        if pad_left or pad_top or pad_right or pad_bottom:
            crop = cv2.copyMakeBorder(
                crop,
                pad_top, pad_bottom, pad_left, pad_right,
                cv2.BORDER_REFLECT
            )

        # Шум яркости
        if self.noise_level > 0:
            noise = np.random.randint(
                -int(self.noise_level),
                int(self.noise_level) + 1,
                crop.shape,
                dtype=np.int16
            )
            crop = np.clip(crop.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # Формат вывода
        if self.output_format == "hsv":
            return cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        return crop


# ─────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Симулятор полёта БПЛА")
    p.add_argument("--image",   required=True,  help="Путь к снимку карты")
    p.add_argument("--output",  default="frames", help="Папка для кадров")
    p.add_argument("--speed",   type=float, default=5.0, help="Скорость (пкс/кадр)")
    p.add_argument("--noise",   type=float, default=10.0, help="Уровень шума")
    p.add_argument("--shake",   type=float, default=2.0,  help="Дрожание (пкс)")
    p.add_argument("--cam-w",   type=int,   default=320,  help="Ширина кадра")
    p.add_argument("--cam-h",   type=int,   default=240,  help="Высота кадра")
    p.add_argument("--max-frames", type=int, default=2000, help="Макс кадров")
    p.add_argument("--preview", action="store_true", help="Показать в окне")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────
#  БЫСТРЫЙ ТЕСТ (без реального изображения — синтетика)
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import tempfile

    print("=" * 60)
    print("ТЕСТ: Симулятор на синтетической карте")
    print("=" * 60)

    # Создаём синтетическую карту: три вертикальные полосы
    # Поле (жёлто-зелёное) | Лес (тёмно-зелёный) | Поле снова
    W, H = 1200, 800
    synth_map = np.zeros((H, W, 3), dtype=np.uint8)

    # Поле
    synth_map[:, :400]    = [80, 200, 180]   # BGR
    # Лес
    synth_map[:, 400:800] = [30, 100, 30]
    # Дорога
    synth_map[:, 800:]    = [100, 110, 120]

    # Добавляем текстуру
    noise = np.random.randint(0, 25, synth_map.shape, dtype=np.uint8)
    synth_map = np.clip(synth_map.astype(np.int32) + noise, 0, 255).astype(np.uint8)

    # Сохраняем во временный файл
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        tmp_path = f.name
    cv2.imwrite(tmp_path, synth_map)

    sim = FlightSimulator(
        tmp_path,
        camera_w=200, camera_h=150,
        noise_level=8, shake_pixels=1.5,
        output_format="hsv"
    )

    # Маршрут: прямо слева направо через все три зоны
    waypoints = [
        Waypoint(100,  400, "старт"),
        Waypoint(1100, 400, "финиш"),
    ]

    print(f"\n  Маршрут: ({waypoints[0].x},{waypoints[0].y}) → "
          f"({waypoints[1].x},{waypoints[1].y})")
    print(f"  Ожидаемые переходы: ~кадр 60 (Поле→Лес), ~кадр 120 (Лес→Дорога)\n")

    # Прогоняем через детектор
    from detector import TransitionDetector

    detector = TransitionDetector(
        window_size=30,
        false_alarm_rate=0.05,
        min_m=5,
        lr_filter_thresh=1.2
    )

    frames_processed = 0
    for hsv_frame, meta in sim.fly(waypoints, speed_pix_per_frame=10.0):
        event = detector.process_frame(hsv_frame, meta.frame_number)
        if event:
            print(f"  ✓ Переход: кадр {event.frame_center:4d}  "
                  f"| Бхатт={event.bhattacharyya_max:.3f}  "
                  f"| conf={event.confidence:.2f}  "
                  f"| сегмент: {meta.segment_name}")
        frames_processed += 1

    print(f"\n  Обработано кадров: {frames_processed}")
    events = detector.get_completed_events()
    print(f"  Обнаружено переходов: {len(events)}")

    # Убираем временный файл
    os.unlink(tmp_path)

    print("\nТест завершён ✓")
    print("\nДля запуска на реальном снимке:")
    print("  python simulator.py --image your_map.png --output frames/ --preview")
