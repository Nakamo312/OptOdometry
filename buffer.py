"""
buffer.py — Задача 4
Кольцевой буфер состояний со скользящим окном.
"""

from collections import deque
import numpy as np
from state import SurfaceState


class StateBuffer:
    """
    Кольцевой буфер последних N состояний поверхности.

    Параметры
    ----------
    window_size : int
        Максимальное число хранимых состояний.
    stability_threshold : float
        Порог дисперсии для is_stable(). По умолчанию 0.1.
    min_stable_frames : int
        Минимальное число кадров для оценки стабильности.
    """

    def __init__(self,
                 window_size: int = 30,
                 stability_threshold: float = 0.1,
                 min_stable_frames: int = 5):
        self.window_size         = window_size
        self.stability_threshold = stability_threshold
        self.min_stable_frames   = min_stable_frames
        self._buffer: deque      = deque(maxlen=window_size)

        # Счётчик кадров с момента последней зафиксированной стабильности
        self._frames_since_stable = 0
        self._last_stable_frame   = -1

    # ── Основные методы ───────────────────────────────────────

    def add_state(self, state: SurfaceState) -> None:
        """Добавить новое состояние в буфер."""
        self._buffer.append(state)
        if self.is_stable():
            self._last_stable_frame   = state.frame_number
            self._frames_since_stable = 0
        else:
            self._frames_since_stable += 1

    def get_window(self) -> list:
        """Вернуть все состояния в буфере (от старых к новым)."""
        return list(self._buffer)

    def get_first_half(self) -> list:
        """Первая (старая) половина окна."""
        states = list(self._buffer)
        mid    = len(states) // 2
        return states[:mid]

    def get_last_half(self) -> list:
        """Вторая (новая) половина окна."""
        states = list(self._buffer)
        mid    = len(states) // 2
        return states[mid:]

    def get_last_n(self, n: int) -> list:
        """Последние n состояний из буфера."""
        states = list(self._buffer)
        return states[-n:] if n <= len(states) else states

    def is_full(self) -> bool:
        """Буфер заполнен до window_size."""
        return len(self._buffer) == self.window_size

    def is_stable(self, threshold: float = None) -> bool:
        """
        Проверить стабильность: дисперсия признаковых векторов
        внутри буфера мала.

        Параметры
        ----------
        threshold : float, optional
            Если None — используется self.stability_threshold.

        Алгоритм
        --------
        Берём матрицу признаков всех состояний в буфере,
        считаем среднее стандартное отклонение по всем измерениям.
        Если оно меньше порога — буфер стабилен.
        """
        if threshold is None:
            threshold = self.stability_threshold

        states = list(self._buffer)
        if len(states) < self.min_stable_frames:
            return False

        matrix = np.array([s.feature_vector() for s in states])  # (N, 5)
        mean_std = float(matrix.std(axis=0).mean())
        return mean_std < threshold

    def mean_std(self) -> float:
        """
        Текущее среднее стандартное отклонение по буферу.
        Полезно для отладки и визуализации.
        """
        states = list(self._buffer)
        if len(states) < 2:
            return 0.0
        matrix = np.array([s.feature_vector() for s in states])
        return float(matrix.std(axis=0).mean())

    def dynamic_m(self, min_m: int = 5) -> int:
        """
        Динамически определить M — число "новых" кадров для LR.

        M = количество кадров с момента последнего стабильного
        состояния буфера. Ограничено снизу min_m и сверху
        половиной window_size.

        Параметры
        ----------
        min_m : int
            Минимальное значение M.
        """
        max_m = self.window_size // 2
        m     = max(min_m, self._frames_since_stable)
        return min(m, max_m)

    def feature_matrix(self) -> np.ndarray:
        """
        Вернуть матрицу признаков всех состояний в буфере.
        Shape: (N, 5), dtype float32.
        """
        states = list(self._buffer)
        if not states:
            return np.empty((0, 5), dtype=np.float32)
        return np.array([s.feature_vector() for s in states],
                        dtype=np.float32)

    def reset(self) -> None:
        """Очистить буфер и сбросить счётчики."""
        self._buffer.clear()
        self._frames_since_stable = 0
        self._last_stable_frame   = -1

    def __len__(self) -> int:
        return len(self._buffer)

    def __repr__(self):
        return (
            f"StateBuffer(size={len(self._buffer)}/{self.window_size}, "
            f"stable={self.is_stable()}, "
            f"mean_std={self.mean_std():.4f})"
        )


# ─────────────────────────────────────────────────────────────
#  БЫСТРЫЙ ТЕСТ
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import cv2

    np.random.seed(42)

    def make_state(rgb, frame_num, noise=8):
        img = np.full((100, 100, 3), rgb, dtype=np.uint8)
        img = np.clip(
            img + np.random.randint(-noise, noise, img.shape), 0, 255
        ).astype(np.uint8)
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        return SurfaceState(hsv, frame_number=frame_num)

    buf = StateBuffer(window_size=20, stability_threshold=0.05)

    print("=" * 55)
    print("ТЕСТ 1: Заполнение буфера стабильными кадрами (поле)")
    print("=" * 55)
    for i in range(20):
        s = make_state([180, 200, 80], frame_num=i, noise=5)
        buf.add_state(s)

    print(f"  {buf}")
    assert len(buf) == 20,    "Буфер должен содержать 20 состояний"
    assert buf.is_full(),     "Буфер должен быть полным"
    print(f"  is_stable: {buf.is_stable()}")
    print(f"  first_half: {len(buf.get_first_half())} состояний")
    print(f"  last_half:  {len(buf.get_last_half())} состояний")
    print("  OK\n")

    print("=" * 55)
    print("ТЕСТ 2: Кольцевое поведение — добавляем > window_size")
    print("=" * 55)
    for i in range(20, 35):
        s = make_state([180, 200, 80], frame_num=i, noise=5)
        buf.add_state(s)

    assert len(buf) == 20, "Буфер не должен превышать window_size"
    oldest_frame = buf.get_window()[0].frame_number
    print(f"  Самый старый кадр в буфере: {oldest_frame}  (ожидается 15)")
    assert oldest_frame == 15
    print("  OK\n")

    print("=" * 55)
    print("ТЕСТ 3: Нестабильность после резкой смены зоны")
    print("=" * 55)
    buf2 = StateBuffer(window_size=20, stability_threshold=0.05)

    # 15 стабильных кадров поля
    for i in range(15):
        buf2.add_state(make_state([180, 200, 80], i, noise=5))

    print(f"  После 15 кадров поля:  stable={buf2.is_stable()}, "
          f"std={buf2.mean_std():.4f}")

    # 5 кадров леса (резкий переход)
    for i in range(15, 20):
        buf2.add_state(make_state([30, 100, 30], i, noise=5))

    print(f"  После 5 кадров леса:   stable={buf2.is_stable()}, "
          f"std={buf2.mean_std():.4f}")
    assert not buf2.is_stable(), "Смешанный буфер не должен быть стабильным"
    print("  OK\n")

    print("=" * 55)
    print("ТЕСТ 4: dynamic_m")
    print("=" * 55)
    buf3 = StateBuffer(window_size=30, stability_threshold=0.05)
    for i in range(10):
        buf3.add_state(make_state([180, 200, 80], i, noise=5))
    print(f"  Стабильных кадров: M = {buf3.dynamic_m()}")

    for i in range(10, 17):
        buf3.add_state(make_state([30, 100, 30], i, noise=5))
    print(f"  После 7 кадров перехода: M = {buf3.dynamic_m()}")
    assert buf3.dynamic_m() >= 5, "M должен быть не менее min_m=5"
    print("  OK\n")

    print("Все тесты пройдены ✓")
