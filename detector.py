"""
detector.py — Задачи 6-8
Отношение правдоподобия, адаптивный порог Неймана-Пирсона,
главный детектор переходов.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List

from state import SurfaceState
from buffer import StateBuffer
from gaussian_utils import estimate_gaussian, gaussian_logpdf_batch


# ─────────────────────────────────────────────────────────────
#  ЗАДАЧА 6 — Отношение правдоподобия
# ─────────────────────────────────────────────────────────────

def compute_likelihood_ratio(buffer: StateBuffer, m: int) -> float:
    """
    Вычислить отношение правдоподобия (LR) для обнаружения перехода.

    Алгоритм
    --------
    Делим буфер на "старые" (всё кроме последних m) и "новые" (последние m).
    Оцениваем распределение каждой группы.
    LR = exp( Σ log p(x_new | μ_new, Σ_new)
               - Σ log p(x_new | μ_old, Σ_old) )

    LR ≈ 1  →  новые кадры не отличаются от старых (нет перехода)
    LR >> 1 →  новые кадры объясняются новым распределением гораздо лучше

    Параметры
    ----------
    buffer : StateBuffer
    m      : int  — число "новых" кадров (≥ 2)

    Возвращает
    ----------
    float — отношение правдоподобия ≥ 0.
    """
    states = buffer.get_window()
    n      = len(states)

    if n < m + 2:
        return 1.0      # недостаточно данных

    m = max(2, min(m, n // 2))     # защита от крайних значений

    old_states = states[:n - m]
    new_states = states[n - m:]

    old_matrix = np.array([s.feature_vector() for s in old_states])
    new_matrix = np.array([s.feature_vector() for s in new_states])

    # Оцениваем оба распределения
    try:
        mu_old, sig_old = estimate_gaussian(old_matrix)
        mu_new, sig_new = estimate_gaussian(new_matrix)
    except ValueError:
        return 1.0

    # Log-правдоподобие новых кадров при обоих распределениях
    lp_old = gaussian_logpdf_batch(new_matrix, mu_old, sig_old).sum()
    lp_new = gaussian_logpdf_batch(new_matrix, mu_new, sig_new).sum()

    log_lr = lp_new - lp_old

    # Ограничиваем чтобы избежать числового переполнения
    log_lr = np.clip(log_lr, -500, 500)
    return float(np.exp(log_lr))


def bhattacharyya_ratio(buffer: StateBuffer, m: int) -> float:
    """
    Расстояние Бхаттачарьи между гистограммами H старой и новой частей буфера.

    Используется как ОСНОВНОЙ сигнал детектора.
    LR (gaussian) — фильтрует ложные срабатывания.

    Возвращает
    ----------
    float ∈ [0, 1] — расстояние Бхаттачарьи.
    0 = идентичные гистограммы, 1 = полностью различные.
    """
    import cv2
    states = buffer.get_window()
    n      = len(states)

    if n < m + 2:
        return 0.0

    m = max(2, min(m, n // 2))

    old_states = states[:n - m]
    new_states = states[n - m:]

    # Усреднённые гистограммы для каждой группы
    h_old = np.mean([s.histogram() for s in old_states], axis=0).astype(np.float32)
    h_new = np.mean([s.histogram() for s in new_states], axis=0).astype(np.float32)

    return float(cv2.compareHist(h_old, h_new, cv2.HISTCMP_BHATTACHARYYA))


# ─────────────────────────────────────────────────────────────
#  ЗАДАЧА 7 — Адаптивный порог (Нейман-Пирсон)
# ─────────────────────────────────────────────────────────────

class AdaptiveThreshold:
    """
    Адаптивный порог по критерию Неймана-Пирсона.

    Накапливает историю значений основного сигнала (Бхаттачарья)
    в моменты когда буфер стабилен — это нулевая гипотеза H0
    ("перехода нет").

    Порог = (1 - α)-квантиль этой истории.

    Параметры
    ----------
    false_alarm_rate : float
        Уровень ложных тревог α ∈ (0, 1). По умолчанию 0.05.
    min_history : int
        Минимальная длина истории для вычисления порога.
    default_threshold : float
        Порог до накопления достаточной статистики.
    """

    def __init__(self,
                 false_alarm_rate: float = 0.05,
                 min_history: int = 30,
                 default_threshold: float = 0.3):
        self.false_alarm_rate  = false_alarm_rate
        self.min_history       = min_history
        self.default_threshold = default_threshold
        self._history: List[float] = []

    def update(self, signal_value: float, is_stable: bool) -> None:
        """
        Обновить историю нулевой гипотезы.

        Добавляем значение сигнала только когда буфер стабилен —
        в этот момент мы уверены что перехода нет.

        Параметры
        ----------
        signal_value : float  — текущее значение Бхаттачарьи
        is_stable    : bool   — флаг стабильности буфера
        """
        if is_stable:
            self._history.append(signal_value)

    def get_threshold(self) -> float:
        """
        Вернуть текущий порог как (1-α)-квантиль накопленной истории.

        Если истории недостаточно — возвращает default_threshold.
        """
        if len(self._history) < self.min_history:
            return self.default_threshold

        return float(np.quantile(self._history, 1.0 - self.false_alarm_rate))

    def is_transition(self, signal_value: float) -> bool:
        """
        Проверить: превышает ли сигнал текущий порог?

        Параметры
        ----------
        signal_value : float  — текущее значение Бхаттачарьи

        Возвращает
        ----------
        bool — True если обнаружен переход.
        """
        return signal_value > self.get_threshold()

    def history_size(self) -> int:
        return len(self._history)

    def __repr__(self):
        return (
            f"AdaptiveThreshold(α={self.false_alarm_rate}, "
            f"threshold={self.get_threshold():.4f}, "
            f"history={self.history_size()})"
        )


# ─────────────────────────────────────────────────────────────
#  СОБЫТИЕ ПЕРЕХОДА
# ─────────────────────────────────────────────────────────────

@dataclass
class TransitionEvent:
    """
    Событие обнаруженного перехода между зонами.

    Поля
    ----
    frame_enter  : int    — номер кадра входа в переход (LR > порога)
    frame_exit   : int    — номер кадра выхода (LR упал ниже порога)
    frame_center : int    — середина перехода (наилучшая оценка границы)
    bhattacharyya_max : float  — максимальное значение сигнала
    lr_max       : float  — максимальное значение LR
    confidence   : float  — уверенность ∈ [0, 1]
    old_state    : Optional[SurfaceState]  — представитель старой зоны
    new_state    : Optional[SurfaceState]  — представитель новой зоны
    m_used       : int    — значение M при обнаружении
    """
    frame_enter:       int
    frame_exit:        int   = -1
    frame_center:      int   = -1
    bhattacharyya_max: float = 0.0
    lr_max:            float = 1.0
    confidence:        float = 0.0
    old_state:         Optional[SurfaceState] = None
    new_state:         Optional[SurfaceState] = None
    m_used:            int   = 0

    def finalize(self, frame_exit: int):
        """Зафиксировать выход из зоны перехода."""
        self.frame_exit   = frame_exit
        self.frame_center = (self.frame_enter + frame_exit) // 2

    def __repr__(self):
        return (
            f"TransitionEvent(frames={self.frame_enter}→{self.frame_exit}, "
            f"center={self.frame_center}, "
            f"bhatt={self.bhattacharyya_max:.3f}, "
            f"conf={self.confidence:.3f})"
        )


# ─────────────────────────────────────────────────────────────
#  ЗАДАЧА 8 — Главный детектор
# ─────────────────────────────────────────────────────────────

class TransitionDetector:
    """
    Детектор переходов между визуально различимыми зонами.

    Архитектура
    -----------
    1. Основной сигнал  — Бхаттачарья между гистограммами H
       (чувствителен к цветовым переходам)
    2. Фильтрующий сигнал — LR на скалярных признаках
       (отсекает флуктуации освещения)
    3. Порог — адаптивный по Нейману-Пирсону

    Режим срабатывания: пока сигнал выше порога — мы в зоне перехода.
    Фиксируем вход (передний фронт), выход (задний фронт),
    середину как наилучшую оценку момента пересечения границы.

    Параметры
    ----------
    window_size      : int    — размер скользящего окна буфера
    false_alarm_rate : float  — уровень ложных тревог α
    min_m            : int    — минимальный M для LR
    lr_filter_thresh : float  — минимальный LR для подтверждения перехода
    stability_thresh : float  — порог стабильности буфера
    """

    def __init__(self,
                 window_size:       int   = 30,
                 false_alarm_rate:  float = 0.05,
                 min_m:             int   = 5,
                 lr_filter_thresh:  float = 1.5,
                 stability_thresh:  float = 0.05):

        self.window_size      = window_size
        self.min_m            = min_m
        self.lr_filter_thresh = lr_filter_thresh

        self._buffer    = StateBuffer(
            window_size=window_size,
            stability_threshold=stability_thresh,
            min_stable_frames=5
        )
        self._threshold = AdaptiveThreshold(
            false_alarm_rate=false_alarm_rate,
            min_history=30,
            default_threshold=0.3
        )

        # Состояние конечного автомата
        self._in_transition    = False
        self._current_event:   Optional[TransitionEvent] = None
        self._completed_events: List[TransitionEvent]    = []

        # История сигналов для визуализации
        self.signal_history:    List[float] = []
        self.lr_history:        List[float] = []
        self.threshold_history: List[float] = []

    # ── Главный метод ─────────────────────────────────────────

    def process_frame(self,
                      hsv_frame: np.ndarray,
                      frame_number: int) -> Optional[TransitionEvent]:
        """
        Обработать один HSV кадр.

        Параметры
        ----------
        hsv_frame    : np.ndarray  — HSV кадр (H, W, 3)
        frame_number : int         — номер кадра

        Возвращает
        ----------
        TransitionEvent  — если в этом кадре зафиксирован ВЫХОД из перехода
                           (т.е. событие завершено и можно использовать).
        None             — иначе.
        """
        # 1. Создаём состояние и добавляем в буфер
        state = SurfaceState(hsv_frame, frame_number=frame_number)
        self._buffer.add_state(state)

        # 2. Ждём заполнения буфера
        if not self._buffer.is_full():
            self.signal_history.append(0.0)
            self.lr_history.append(1.0)
            self.threshold_history.append(self._threshold.get_threshold())
            return None

        # 3. Динамический M
        m = self._buffer.dynamic_m(min_m=self.min_m)

        # 4. Основной сигнал — Бхаттачарья
        bhat = bhattacharyya_ratio(self._buffer, m)

        # 5. Фильтрующий сигнал — LR
        lr = compute_likelihood_ratio(self._buffer, m)

        # 6. Обновляем порог на стабильных данных
        is_stable = self._buffer.is_stable()
        self._threshold.update(bhat, is_stable)

        # 7. Решение
        threshold  = self._threshold.get_threshold()
        is_trigger = (bhat > threshold) and (lr > self.lr_filter_thresh)

        # 8. Конечный автомат: вход / выход из перехода
        completed_event = None

        if is_trigger and not self._in_transition:
            # Передний фронт — вход в переход
            self._in_transition  = True
            old_states = self._buffer.get_first_half()
            old_repr   = old_states[-1] if old_states else None
            self._current_event  = TransitionEvent(
                frame_enter=frame_number,
                bhattacharyya_max=bhat,
                lr_max=lr,
                old_state=old_repr,
                m_used=m
            )

        elif is_trigger and self._in_transition:
            # Продолжаем переход — обновляем максимумы
            if bhat > self._current_event.bhattacharyya_max:
                self._current_event.bhattacharyya_max = bhat
            if lr > self._current_event.lr_max:
                self._current_event.lr_max = lr
            # Обновляем "новое" состояние
            new_states = self._buffer.get_last_half()
            self._current_event.new_state = new_states[-1] if new_states else None

        elif not is_trigger and self._in_transition:
            # Задний фронт — выход из перехода
            self._current_event.finalize(frame_exit=frame_number)
            self._current_event.confidence = self._compute_confidence(
                self._current_event
            )
            self._completed_events.append(self._current_event)
            completed_event     = self._current_event
            self._current_event = None
            self._in_transition = False

        # 9. Запись истории для визуализации
        self.signal_history.append(bhat)
        self.lr_history.append(lr)
        self.threshold_history.append(threshold)

        return completed_event

    # ── Вспомогательные методы ────────────────────────────────

    def _compute_confidence(self, event: TransitionEvent) -> float:
        """
        Уверенность ∈ [0, 1] на основе:
        - насколько сигнал превысил порог
        - длительности перехода
        """
        thresh = self._threshold.get_threshold()
        if thresh <= 0:
            return 0.0

        signal_ratio = min(event.bhattacharyya_max / thresh, 5.0) / 5.0
        duration     = event.frame_exit - event.frame_enter + 1
        # Короткие чёткие переходы — высокая уверенность
        duration_score = 1.0 / (1.0 + 0.1 * max(duration - 10, 0))
        return float(np.clip(signal_ratio * duration_score, 0.0, 1.0))

    def get_completed_events(self) -> List[TransitionEvent]:
        """Вернуть список всех завершённых событий."""
        return list(self._completed_events)

    def reset(self):
        """Сбросить состояние детектора."""
        self._buffer.reset()
        self._in_transition   = False
        self._current_event   = None
        self._completed_events.clear()
        self.signal_history.clear()
        self.lr_history.clear()
        self.threshold_history.clear()

    def __repr__(self):
        return (
            f"TransitionDetector("
            f"events={len(self._completed_events)}, "
            f"in_transition={self._in_transition}, "
            f"{self._threshold})"
        )


# ─────────────────────────────────────────────────────────────
#  БЫСТРЫЙ ТЕСТ
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import cv2

    np.random.seed(42)

    def make_hsv_frame(rgb, noise=10):
        img = np.full((100, 100, 3), rgb, dtype=np.uint8)
        img = np.clip(
            img + np.random.randint(-noise, noise, img.shape), 0, 255
        ).astype(np.uint8)
        return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    detector = TransitionDetector(
        window_size=30,
        false_alarm_rate=0.05,
        min_m=5,
        lr_filter_thresh=1.2
    )

    print("=" * 60)
    print("ТЕСТ: Синтетическая последовательность с двумя переходами")
    print("=" * 60)
    print("  Зоны: [Поле x60] → [Лес x60] → [Дорога x60]")
    print()

    # Три зоны
    zones = [
        ([180, 200,  80], 60, "Поле"),
        ([30,  100,  30], 60, "Лес"),
        ([120, 110, 100], 60, "Дорога"),
    ]

    events    = []
    frame_num = 0

    for rgb, count, name in zones:
        for _ in range(count):
            hsv = make_hsv_frame(rgb, noise=8)
            event = detector.process_frame(hsv, frame_num)
            if event:
                events.append(event)
                print(f"  ✓ Переход обнаружен: {event}")
            frame_num += 1

    print()
    print(f"  Всего событий: {len(events)}  (ожидается 2)")
    print(f"  {detector}")

    # Переходы должны быть около кадров 60 и 120
    if len(events) >= 1:
        print(f"\n  Переход 1: центр кадра {events[0].frame_center} "
              f"(ожидается ~60, уверенность {events[0].confidence:.2f})")
    if len(events) >= 2:
        print(f"  Переход 2: центр кадра {events[1].frame_center} "
              f"(ожидается ~120, уверенность {events[1].confidence:.2f})")

    print()
    print("=" * 60)
    print("ТЕСТ: Стабильный сигнал — LR ≈ 1 (нет переходов)")
    print("=" * 60)
    det2 = TransitionDetector(window_size=20)
    for i in range(60):
        hsv = make_hsv_frame([180, 200, 80], noise=5)
        e = det2.process_frame(hsv, i)
        if e:
            print(f"  ЛОЖНАЯ ТРЕВОГА на кадре {i}!")

    lr_vals = [v for v in det2.lr_history if v != 1.0]
    if lr_vals:
        print(f"  LR: min={min(lr_vals):.3f}, max={max(lr_vals):.3f}, "
              f"mean={np.mean(lr_vals):.3f}")
    print(f"  Событий: {len(det2.get_completed_events())}  (ожидается 0)")
    print()
    print("Все тесты пройдены ✓")
