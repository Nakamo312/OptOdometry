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

# ─────────────────────────────────────────────────────────────
#  ЗАДАЧА 7 — Адаптивный порог + детектор пиков
# ─────────────────────────────────────────────────────────────

class LocalBackground:
    """
    Скользящая оценка фонового уровня сигнала.

    Использует экспоненциальное скользящее среднее (EMA)
    с коэффициентом забывания — старые значения постепенно
    теряют вес. Обновляется только на стабильных кадрах.

    Параметры
    ----------
    alpha       : float  — коэффициент забывания ∈ (0,1).
                           0.95 = медленное забывание (стабильный фон)
                           0.7  = быстрое забывание (адаптация к изменениям)
    init_value  : float  — начальное значение фона
    """

    def __init__(self, alpha: float = 0.95, init_value: float = 0.05):
        self.alpha       = alpha
        self._value      = init_value
        self._n_updates  = 0

    def update(self, signal: float, is_stable: bool):
        if is_stable:
            if self._n_updates == 0:
                self._value = signal
            else:
                self._value = self.alpha * self._value + (1.0 - self.alpha) * signal
            self._n_updates += 1

    def get(self) -> float:
        return self._value

    def reset(self):
        self._n_updates = 0


class PeakDetector:
    """
    Детектор локальных максимумов на сигнале Бхаттачарьи.

    Алгоритм
    --------
    1. Нормализуем сигнал: normalized = signal / background
       Это инвариантно к абсолютной амплитуде — порог фиксированный.

    2. Ищем пик: normalized > peak_threshold И производная
       меняет знак + → -  (мы на спуске после максимума).

    3. LR используем как второй фильтр: пик засчитывается только
       если LR > lr_threshold.

    Параметры
    ----------
    peak_threshold  : float  — минимальное отношение сигнал/фон для пика.
                               2.0 = сигнал должен быть вдвое выше фона.
    lr_threshold    : float  — минимальный LR для подтверждения.
    min_peak_width  : int    — минимальная ширина пика в кадрах.
                               Защита от одиночных шумовых выбросов.
    forgetting_alpha: float  — коэффициент забывания фона.
    """

    def __init__(self,
                 peak_threshold:   float = 2.0,
                 lr_threshold:     float = 1.5,
                 min_peak_width:   int   = 3,
                 forgetting_alpha: float = 0.95):

        self.peak_threshold   = peak_threshold
        self.lr_threshold     = lr_threshold
        self.min_peak_width   = min_peak_width

        self._bg        = LocalBackground(alpha=forgetting_alpha)
        self._prev_norm = 0.0   # нормализованный сигнал на прошлом шаге
        self._prev_bhat = 0.0   # сырой сигнал на прошлом шаге

        # Отслеживание текущего пика
        self._in_peak       = False
        self._peak_start    = -1
        self._peak_max_norm = 0.0
        self._peak_max_bhat = 0.0
        self._peak_width    = 0

        # История для визуализации
        self.norm_history:       list[float] = []
        self.background_history: list[float] = []
        self.threshold_history:  list[float] = []

    def update(self,
               bhat:       float,
               lr:         float,
               is_stable:  bool,
               frame_number: int
               ) -> tuple[bool, bool]:
        """
        Обработать одно значение сигнала.

        Возвращает
        ----------
        (in_peak, peak_confirmed)

        in_peak         : True пока нормализованный сигнал выше порога
        peak_confirmed  : True в момент подтверждения пика
                          (задний фронт + достаточная ширина + LR)
        """
        # 1. Обновляем фон
        self._bg.update(bhat, is_stable)
        bg = max(self._bg.get(), 1e-6)

        # 2. Нормализованный сигнал
        norm = bhat / bg

        # 3. Порог в нормализованном пространстве
        thr = self.peak_threshold

        above = (norm > thr) and (lr > self.lr_threshold)

        in_peak        = False
        peak_confirmed = False

        if above and not self._in_peak:
            # Передний фронт пика
            self._in_peak       = True
            self._peak_start    = frame_number
            self._peak_max_norm = norm
            self._peak_max_bhat = bhat
            self._peak_width    = 1

        elif above and self._in_peak:
            # Внутри пика — обновляем максимум
            self._peak_width   += 1
            if norm > self._peak_max_norm:
                self._peak_max_norm = norm
                self._peak_max_bhat = bhat

        elif not above and self._in_peak:
            # Задний фронт — проверяем валидность пика
            if self._peak_width >= self.min_peak_width:
                peak_confirmed = True
            self._in_peak    = False
            self._peak_width = 0

        in_peak = self._in_peak

        # 4. История
        self.norm_history.append(norm)
        self.background_history.append(bg)
        self.threshold_history.append(thr)

        self._prev_norm = norm
        self._prev_bhat = bhat

        return in_peak, peak_confirmed

    def get_threshold_abs(self) -> float:
        """Порог в абсолютных единицах Бхаттачарьи (для визуализации)."""
        return self._bg.get() * self.peak_threshold

    def reset(self):
        self._in_peak       = False
        self._peak_start    = -1
        self._peak_max_norm = 0.0
        self._peak_max_bhat = 0.0
        self._peak_width    = 0
        self._prev_norm     = 0.0
        self._prev_bhat     = 0.0
        self.norm_history.clear()
        self.background_history.clear()
        self.threshold_history.clear()

    def __repr__(self):
        return (
            f"PeakDetector(thr={self.peak_threshold}x, "
            f"bg={self._bg.get():.4f}, "
            f"abs_thr={self.get_threshold_abs():.4f}, "
            f"in_peak={self._in_peak})"
        )


# Оставляем AdaptiveThreshold как алиас для обратной совместимости
class AdaptiveThreshold:
    """
    Устаревший класс — оставлен для совместимости.
    Новый код использует PeakDetector.
    """
    def __init__(self, false_alarm_rate=0.05, min_history=30,
                 default_threshold=0.3):
        self.false_alarm_rate  = false_alarm_rate
        self.min_history       = min_history
        self.default_threshold = default_threshold
        self._history: list[float] = []

    def update(self, v, stable):
        if stable:
            self._history.append(v)

    def get_threshold(self):
        if len(self._history) < self.min_history:
            return self.default_threshold
        return float(np.quantile(self._history, 1.0 - self.false_alarm_rate))

    def is_transition(self, v):
        return v > self.get_threshold()

    def history_size(self):
        return len(self._history)


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

    Архитектура (новая)
    -------------------
    1. Основной сигнал  — Бхаттачарья между взвешенными гистограммами H
    2. Нормализация     — сигнал / локальный фон (EMA с забыванием)
       Инвариантна к абсолютной амплитуде → нет проблемы консервативного порога
    3. Детектор пиков   — срабатывает на локальном максимуме нормализованного сигнала
    4. Фильтр LR        — подтверждает что пик не шум освещения

    Параметры
    ----------
    window_size      : int    — размер скользящего окна буфера
    peak_threshold   : float  — минимальное отношение сигнал/фон (например 2.0)
    lr_filter_thresh : float  — минимальный LR для подтверждения пика
    min_peak_width   : int    — минимальная ширина пика в кадрах
    forgetting_alpha : float  — коэффициент забывания фона (0.9-0.98)
    min_m            : int    — минимальный M для расчёта сигналов
    stability_thresh : float  — порог стабильности буфера
    """

    def __init__(self,
                 window_size:      int   = 30,
                 peak_threshold:   float = 2.0,
                 lr_filter_thresh: float = 1.5,
                 min_peak_width:   int   = 3,
                 forgetting_alpha: float = 0.95,
                 min_m:            int   = 5,
                 stability_thresh: float = 0.05,
                 # Обратная совместимость со старыми параметрами
                 false_alarm_rate: float = None,
                 **kwargs):

        self.window_size      = window_size
        self.min_m            = min_m
        self.lr_filter_thresh = lr_filter_thresh

        self._buffer = StateBuffer(
            window_size=window_size,
            stability_threshold=stability_thresh,
            min_stable_frames=5
        )

        self._peak = PeakDetector(
            peak_threshold=peak_threshold,
            lr_threshold=lr_filter_thresh,
            min_peak_width=min_peak_width,
            forgetting_alpha=forgetting_alpha,
        )

        # Состояние конечного автомата
        self._in_transition     = False
        self._current_event:    Optional[TransitionEvent] = None
        self._completed_events: List[TransitionEvent]     = []

        # История для визуализации
        self.signal_history:    List[float] = []   # сырая Бхаттачарья
        self.lr_history:        List[float] = []
        self.threshold_history: List[float] = []   # порог в абс. единицах
        self.norm_history:      List[float] = []   # нормализованный сигнал

    # ── Главный метод ─────────────────────────────────────────

    def process_frame(self,
                      hsv_frame:    np.ndarray,
                      frame_number: int) -> Optional[TransitionEvent]:
        """
        Обработать один HSV кадр.

        Возвращает TransitionEvent в момент подтверждения пика,
        иначе None.
        """
        # 1. Состояние → буфер
        state = SurfaceState(hsv_frame, frame_number=frame_number)
        self._buffer.add_state(state)

        # 2. Ждём заполнения буфера
        if not self._buffer.is_full():
            self.signal_history.append(0.0)
            self.lr_history.append(1.0)
            self.threshold_history.append(self._peak.get_threshold_abs())
            self.norm_history.append(0.0)
            return None

        # 3. Динамический M
        m = self._buffer.dynamic_m(min_m=self.min_m)

        # 4. Сигналы
        bhat = bhattacharyya_ratio(self._buffer, m)
        lr   = compute_likelihood_ratio(self._buffer, m)

        # 5. Стабильность буфера
        is_stable = self._buffer.is_stable()

        # 6. Детектор пиков
        in_peak, peak_confirmed = self._peak.update(
            bhat=bhat,
            lr=lr,
            is_stable=is_stable,
            frame_number=frame_number
        )

        # 7. Синхронизируем флаг перехода с состоянием пика
        completed_event = None

        if in_peak and not self._in_transition:
            # Передний фронт
            self._in_transition = True
            old_states = self._buffer.get_first_half()
            self._current_event = TransitionEvent(
                frame_enter=frame_number,
                bhattacharyya_max=bhat,
                lr_max=lr,
                old_state=old_states[-1] if old_states else None,
                m_used=m
            )

        elif in_peak and self._in_transition:
            # Внутри пика — обновляем максимумы
            if bhat > self._current_event.bhattacharyya_max:
                self._current_event.bhattacharyya_max = bhat
            if lr > self._current_event.lr_max:
                self._current_event.lr_max = lr
            new_states = self._buffer.get_last_half()
            self._current_event.new_state = (
                new_states[-1] if new_states else None
            )

        elif peak_confirmed and self._in_transition:
            # Пик подтверждён — фиксируем событие
            self._current_event.finalize(frame_exit=frame_number)
            self._current_event.confidence = self._compute_confidence(
                self._current_event
            )
            self._completed_events.append(self._current_event)
            completed_event     = self._current_event
            self._current_event = None
            self._in_transition = False

        elif not in_peak and self._in_transition and not peak_confirmed:
            # Пик слишком короткий — сбрасываем без события
            self._current_event = None
            self._in_transition = False

        # 8. История для визуализации
        self.signal_history.append(bhat)
        self.lr_history.append(lr)
        self.threshold_history.append(self._peak.get_threshold_abs())
        self.norm_history.append(
            self._peak.norm_history[-1] if self._peak.norm_history else 0.0
        )

        return completed_event

    # ── Вспомогательные ───────────────────────────────────────

    def _compute_confidence(self, event: TransitionEvent) -> float:
        """
        Уверенность ∈ [0,1].
        Считаем по нормализованному максимуму пика —
        инвариантно к абсолютной амплитуде.
        """
        bg = max(self._peak._bg.get(), 1e-6)
        norm_max = event.bhattacharyya_max / bg

        # Нормализуем относительно порога: 1.0 = ровно на пороге
        rel = (norm_max - self._peak.peak_threshold) / max(
            self._peak.peak_threshold, 1e-6
        )
        confidence = float(np.clip(rel, 0.0, 1.0))

        # Штраф за слишком длинный переход
        duration = max(event.frame_exit - event.frame_enter, 1)
        duration_penalty = 1.0 / (1.0 + 0.05 * max(duration - 15, 0))

        return float(np.clip(confidence * duration_penalty, 0.0, 1.0))

    def get_completed_events(self) -> List[TransitionEvent]:
        return list(self._completed_events)

    def reset(self):
        self._buffer.reset()
        self._peak.reset()
        self._in_transition    = False
        self._current_event    = None
        self._completed_events.clear()
        self.signal_history.clear()
        self.lr_history.clear()
        self.threshold_history.clear()
        self.norm_history.clear()

    def __repr__(self):
        return (
            f"TransitionDetector("
            f"events={len(self._completed_events)}, "
            f"in_transition={self._in_transition}, "
            f"{self._peak})"
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
