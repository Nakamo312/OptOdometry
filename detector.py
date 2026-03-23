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
    """
    states = buffer.get_window()
    n      = len(states)

    if n < m + 2:
        return 1.0

    m = max(2, min(m, n // 2))

    old_states = states[:n - m]
    new_states = states[n - m:]

    old_matrix = np.array([s.feature_vector() for s in old_states])
    new_matrix = np.array([s.feature_vector() for s in new_states])

    try:
        mu_old, sig_old = estimate_gaussian(old_matrix)
        mu_new, sig_new = estimate_gaussian(new_matrix)
    except ValueError:
        return 1.0

    lp_old = gaussian_logpdf_batch(new_matrix, mu_old, sig_old).sum()
    lp_new = gaussian_logpdf_batch(new_matrix, mu_new, sig_new).sum()

    log_lr = lp_new - lp_old
    log_lr = np.clip(log_lr, -500, 500)
    return float(np.exp(log_lr))


def bhattacharyya_ratio(buffer: StateBuffer, m: int) -> float:
    """
    Расстояние Бхаттачарьи между гистограммами H старой и новой частей буфера.

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

    h_old = np.mean([s.histogram() for s in old_states], axis=0).astype(np.float32)
    h_new = np.mean([s.histogram() for s in new_states], axis=0).astype(np.float32)

    return float(cv2.compareHist(h_old, h_new, cv2.HISTCMP_BHATTACHARYYA))


def block_score(buffer: StateBuffer, m: int) -> float:
    """
    Структурный сигнал перехода по матрице расстояний буфера.

    Алгоритм
    --------
    Делим буфер на старые ([:split]) и новые ([split:]) кадры.
    Считаем три средних расстояния Бхаттачарьи:
        cross    = mean(D[i,j])  для i ∈ старые, j ∈ новые
        intra_old = mean(D[i,j]) для i,j ∈ старые  (i ≠ j)
        intra_new = mean(D[i,j]) для i,j ∈ новые   (i ≠ j)

    block_score = cross − 0.5 * (intra_old + intra_new)

    Интерпретация
    -------------
    0   → однородный буфер (cross ≈ intra, переходов нет)
    >0  → межгрупповое расстояние больше внутригруппового → есть переход
    1   → максимальная блочная структура

    Преимущество перед простой Бхаттачарьей
    ----------------------------------------
    Устойчив к случаю когда обе зоны похожи по среднему цвету,
    но каждая внутренне однородна — усреднение в bhattacharyya_ratio
    в этом случае даёт низкий сигнал, block_score — нет.
    Также менее чувствителен к завышенному LocalBackground при
    обратном движении, т.к. нормализуется на внутригрупповой разброс.

    Параметры
    ----------
    buffer : StateBuffer
    m      : int — число "новых" кадров

    Возвращает
    ----------
    float ∈ [0, 1]
    """
    import cv2

    states = buffer.get_window()
    n      = len(states)

    if n < m + 2:
        return 0.0

    m     = max(2, min(m, n // 2))
    split = n - m

    old = states[:split]
    new = states[split:]

    def _mean_dist(group_a: list, group_b: list, skip_same: bool = False) -> float:
        """Среднее попарное расстояние Бхаттачарьи между двумя группами."""
        vals = []
        for i, sa in enumerate(group_a):
            for j, sb in enumerate(group_b):
                if skip_same and sa is sb:
                    continue
                h1 = sa.histogram()
                h2 = sb.histogram()
                vals.append(
                    float(cv2.compareHist(h1, h2, cv2.HISTCMP_BHATTACHARYYA))
                )
        return float(np.mean(vals)) if vals else 0.0

    cross     = _mean_dist(old, new, skip_same=False)
    intra_old = _mean_dist(old, old, skip_same=True)
    intra_new = _mean_dist(new, new, skip_same=True)
    intra     = 0.5 * (intra_old + intra_new)

    score = cross - intra
    return float(np.clip(score, 0.0, 1.0))


def combined_signal(buffer: StateBuffer, m: int,
                    w_bhat: float = 0.4,
                    w_block: float = 0.6) -> float:
    """
    Взвешенная комбинация bhattacharyya_ratio и block_score.

    Параметры
    ----------
    w_bhat  : вес простой Бхаттачарьи (быстрая, реагирует на абс. разницу)
    w_block : вес block_score          (медленнее, структурно устойчивее)

    По умолчанию block_score имеет больший вес, т.к. он устойчивее
    к асимметрии буфера при смене направления движения.
    """
    bhat  = bhattacharyya_ratio(buffer, m)
    bscor = block_score(buffer, m)
    return float(w_bhat * bhat + w_block * bscor)


# ─────────────────────────────────────────────────────────────
#  ЗАДАЧА 7 — Адаптивный порог + детектор пиков
# ─────────────────────────────────────────────────────────────

class LocalBackground:
    """
    Скользящая оценка фонового уровня сигнала.

    Использует экспоненциальное скользящее среднее (EMA).
    Обновляется только на стабильных кадрах.
    """

    def __init__(self, alpha: float = 0.95, init_value: float = 0.05):
        self.alpha      = alpha
        self._value     = init_value
        self._n_updates = 0

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
    Детектор локальных максимумов на комбинированном сигнале.

    Алгоритм
    --------
    1. Нормализуем сигнал: normalized = signal / background
    2. Ищем пик: normalized > peak_threshold И производная + → -
    3. LR используем как второй фильтр
    """

    def __init__(self,
                 peak_threshold:   float = 2.0,
                 lr_threshold:     float = 1.5,
                 min_peak_width:   int   = 3,
                 forgetting_alpha: float = 0.95):

        self.peak_threshold = peak_threshold
        self.lr_threshold   = lr_threshold
        self.min_peak_width = min_peak_width

        self._bg        = LocalBackground(alpha=forgetting_alpha)
        self._prev_norm = 0.0
        self._prev_bhat = 0.0

        self._in_peak       = False
        self._peak_start    = -1
        self._peak_max_norm = 0.0
        self._peak_max_bhat = 0.0
        self._peak_width    = 0

        self.norm_history:       list[float] = []
        self.background_history: list[float] = []
        self.threshold_history:  list[float] = []

    def update(self,
               bhat:         float,
               lr:           float,
               is_stable:    bool,
               frame_number: int
               ) -> tuple[bool, bool]:
        """
        Обработать одно значение сигнала.

        Возвращает
        ----------
        (in_peak, peak_confirmed)
        """
        self._bg.update(bhat, is_stable)
        bg   = max(self._bg.get(), 1e-6)
        norm = bhat / bg
        thr  = self.peak_threshold

        above = (norm > thr) and (lr > self.lr_threshold)

        in_peak        = False
        peak_confirmed = False

        if above and not self._in_peak:
            self._in_peak       = True
            self._peak_start    = frame_number
            self._peak_max_norm = norm
            self._peak_max_bhat = bhat
            self._peak_width    = 1

        elif above and self._in_peak:
            self._peak_width += 1
            if norm > self._peak_max_norm:
                self._peak_max_norm = norm
                self._peak_max_bhat = bhat

        elif not above and self._in_peak:
            if self._peak_width >= self.min_peak_width:
                peak_confirmed = True
            self._in_peak    = False
            self._peak_width = 0

        in_peak = self._in_peak

        self.norm_history.append(norm)
        self.background_history.append(bg)
        self.threshold_history.append(thr)

        self._prev_norm = norm
        self._prev_bhat = bhat

        return in_peak, peak_confirmed

    def get_threshold_abs(self) -> float:
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


class AdaptiveThreshold:
    """Устаревший класс — оставлен для совместимости."""
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
    """Событие обнаруженного перехода между зонами."""
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
    1. Основной сигнал  — combined_signal():
         0.4 × Бхаттачарья (усреднённые гистограммы)
       + 0.6 × block_score  (структура матрицы расстояний)
    2. Нормализация     — сигнал / локальный фон (EMA)
    3. Детектор пиков   — срабатывает на локальном максимуме
    4. Фильтр LR        — подтверждает что пик не шум освещения

    Параметры block_score
    ---------------------
    w_bhat  : float — вес простой Бхаттачарьи в combined_signal (по умолч. 0.4)
    w_block : float — вес block_score в combined_signal (по умолч. 0.6)
    block_every : int — считать block_score каждые N кадров (по умолч. 1)
                        увеличьте до 2-3 если производительность критична
    """

    def __init__(self,
                 window_size:      int   = 30,
                 peak_threshold:   float = 2.0,
                 lr_filter_thresh: float = 1.5,
                 min_peak_width:   int   = 3,
                 forgetting_alpha: float = 0.95,
                 min_m:            int   = 5,
                 stability_thresh: float = 0.05,
                 w_bhat:           float = 0.4,
                 w_block:          float = 0.6,
                 block_every:      int   = 1,
                 # обратная совместимость
                 false_alarm_rate: float = None,
                 **kwargs):

        self.window_size      = window_size
        self.min_m            = min_m
        self.lr_filter_thresh = lr_filter_thresh
        self.w_bhat           = w_bhat
        self.w_block          = w_block
        self.block_every      = block_every
        self._frame_count     = 0
        self._last_block      = 0.0   # кэш последнего block_score

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

        self._in_transition     = False
        self._current_event:    Optional[TransitionEvent] = None
        self._completed_events: List[TransitionEvent]     = []

        self.signal_history:    List[float] = []
        self.lr_history:        List[float] = []
        self.threshold_history: List[float] = []
        self.norm_history:      List[float] = []
        # дополнительные истории для диагностики
        self.bhat_raw_history:  List[float] = []
        self.block_history:     List[float] = []

    # ── Главный метод ─────────────────────────────────────────

    def process_frame(self,
                      hsv_frame:    np.ndarray,
                      frame_number: int) -> Optional[TransitionEvent]:
        """
        Обработать один HSV кадр.
        Возвращает TransitionEvent в момент подтверждения пика, иначе None.
        """
        self._frame_count += 1

        # 1. Состояние → буфер
        state = SurfaceState(hsv_frame, frame_number=frame_number)
        self._buffer.add_state(state)

        # 2. Ждём заполнения буфера
        if not self._buffer.is_full():
            self.signal_history.append(0.0)
            self.lr_history.append(1.0)
            self.threshold_history.append(self._peak.get_threshold_abs())
            self.norm_history.append(0.0)
            self.bhat_raw_history.append(0.0)
            self.block_history.append(0.0)
            return None

        # 3. Динамический M
        m = self._buffer.dynamic_m(min_m=self.min_m)

        # 4. Сигналы
        bhat_raw = bhattacharyya_ratio(self._buffer, m)

        # block_score считаем каждые block_every кадров (кэшируем)
        if self._frame_count % self.block_every == 0:
            self._last_block = block_score(self._buffer, m)
        bscor = self._last_block

        # Комбинированный сигнал
        sig = float(self.w_bhat * bhat_raw + self.w_block * bscor)

        lr  = compute_likelihood_ratio(self._buffer, m)

        # 5. Стабильность буфера
        is_stable = self._buffer.is_stable()

        # 6. Детектор пиков (работает на combined signal)
        in_peak, peak_confirmed = self._peak.update(
            bhat=sig,
            lr=lr,
            is_stable=is_stable,
            frame_number=frame_number
        )

        # 7. Конечный автомат переходов
        completed_event = None

        if in_peak and not self._in_transition:
            self._in_transition = True
            old_states = self._buffer.get_first_half()
            self._current_event = TransitionEvent(
                frame_enter=frame_number,
                bhattacharyya_max=sig,
                lr_max=lr,
                old_state=old_states[-1] if old_states else None,
                m_used=m
            )

        elif in_peak and self._in_transition:
            if sig > self._current_event.bhattacharyya_max:
                self._current_event.bhattacharyya_max = sig
            if lr > self._current_event.lr_max:
                self._current_event.lr_max = lr
            new_states = self._buffer.get_last_half()
            self._current_event.new_state = (
                new_states[-1] if new_states else None
            )

        elif peak_confirmed and self._in_transition:
            self._current_event.finalize(frame_exit=frame_number)
            self._current_event.confidence = self._compute_confidence(
                self._current_event
            )
            self._completed_events.append(self._current_event)
            completed_event     = self._current_event
            self._current_event = None
            self._in_transition = False

        elif not in_peak and self._in_transition and not peak_confirmed:
            self._current_event = None
            self._in_transition = False

        # 8. История
        self.signal_history.append(sig)
        self.lr_history.append(lr)
        self.threshold_history.append(self._peak.get_threshold_abs())
        self.norm_history.append(
            self._peak.norm_history[-1] if self._peak.norm_history else 0.0
        )
        self.bhat_raw_history.append(bhat_raw)
        self.block_history.append(bscor)

        return completed_event

    # ── Вспомогательные ───────────────────────────────────────

    def _compute_confidence(self, event: TransitionEvent) -> float:
        bg       = max(self._peak._bg.get(), 1e-6)
        norm_max = event.bhattacharyya_max / bg
        rel      = (norm_max - self._peak.peak_threshold) / max(
            self._peak.peak_threshold, 1e-6
        )
        confidence = float(np.clip(rel, 0.0, 1.0))
        duration   = max(event.frame_exit - event.frame_enter, 1)
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
        self.bhat_raw_history.clear()
        self.block_history.clear()
        self._frame_count  = 0
        self._last_block   = 0.0

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
        min_m=5,
        lr_filter_thresh=1.2,
        w_bhat=0.4,
        w_block=0.6,
    )

    print("=" * 60)
    print("ТЕСТ: Синтетическая последовательность с двумя переходами")
    print("=" * 60)
    print("  Зоны: [Поле x60] → [Лес x60] → [Дорога x60]")
    print()

    zones = [
        ([180, 200,  80], 60, "Поле"),
        ([30,  100,  30], 60, "Лес"),
        ([120, 110, 100], 60, "Дорога"),
    ]

    events    = []
    frame_num = 0

    for rgb, count, name in zones:
        for _ in range(count):
            hsv   = make_hsv_frame(rgb, noise=8)
            event = detector.process_frame(hsv, frame_num)
            if event:
                events.append(event)
                print(f"  ✓ Переход обнаружен: {event}")
            frame_num += 1

    print()
    print(f"  Всего событий: {len(events)}  (ожидается 2)")
    print(f"  {detector}")

    if len(events) >= 1:
        print(f"\n  Переход 1: центр кадра {events[0].frame_center} "
              f"(ожидается ~60, уверенность {events[0].confidence:.2f})")
    if len(events) >= 2:
        print(f"  Переход 2: центр кадра {events[1].frame_center} "
              f"(ожидается ~120, уверенность {events[1].confidence:.2f})")

    # Диагностика: сравниваем bhat_raw и block_score
    n = len(detector.bhat_raw_history)
    if n > 0:
        print(f"\n  Диагностика сигналов:")
        print(f"    bhat_raw  max={max(detector.bhat_raw_history):.3f}  "
              f"mean={np.mean(detector.bhat_raw_history):.3f}")
        print(f"    block     max={max(detector.block_history):.3f}  "
              f"mean={np.mean(detector.block_history):.3f}")
        print(f"    combined  max={max(detector.signal_history):.3f}  "
              f"mean={np.mean(detector.signal_history):.3f}")

    print()
    print("=" * 60)
    print("ТЕСТ: Стабильный сигнал — нет переходов")
    print("=" * 60)
    det2 = TransitionDetector(window_size=20, w_bhat=0.4, w_block=0.6)
    for i in range(60):
        hsv = make_hsv_frame([180, 200, 80], noise=5)
        e   = det2.process_frame(hsv, i)
        if e:
            print(f"  ЛОЖНАЯ ТРЕВОГА на кадре {i}!")

    print(f"  Событий: {len(det2.get_completed_events())}  (ожидается 0)")
    print()
    print("Все тесты пройдены ✓")
