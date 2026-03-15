"""
state.py — Задача 3
Класс SurfaceState: хранит признаки одного кадра, умеет сравниваться.
"""

import numpy as np
from features import extract_frame_features, feature_vector_from_dict


class SurfaceState:
    """
    Состояние поверхности в один момент времени.

    Хранит:
    - скалярный вектор признаков (5-мерный)
    - нормализованную гистограмму H (16 бинов)
    - исходные скалярные признаки в виде словаря

    Параметры
    ----------
    hsv_frame : np.ndarray
        HSV кадр (H, W, 3), dtype uint8.
    frame_number : int, optional
        Номер кадра для логирования.
    """

    def __init__(self, hsv_frame: np.ndarray, frame_number: int = -1):
        self.frame_number = frame_number
        self._features    = extract_frame_features(hsv_frame)
        self._vec         = feature_vector_from_dict(self._features)

    # ── Публичные методы ──────────────────────────────────────

    def feature_vector(self) -> np.ndarray:
        """
        Вернуть вектор скалярных признаков.
        Порядок: [h_mean, h_std, s_mean, v_mean, edge_ratio]
        Shape: (5,), dtype float32.
        """
        return self._vec.copy()

    def histogram(self) -> np.ndarray:
        """
        Вернуть нормализованную гистограмму канала H.
        Shape: (16,), сумма = 1.0.
        """
        return self._features["h_hist"].copy()

    def distance_to(self, other: "SurfaceState") -> float:
        """
        Евклидово расстояние между скалярными векторами признаков.

        Параметры
        ----------
        other : SurfaceState

        Возвращает
        ----------
        float — расстояние ≥ 0.
        """
        return float(np.linalg.norm(self._vec - other._vec))

    def bhattacharyya_distance(self, other: "SurfaceState") -> float:
        """
        Расстояние Бхаттачарьи между гистограммами H.

        Чем больше — тем сильнее различаются цветовые профили.
        Диапазон: [0, +inf), 0 = идентичные гистограммы.

        Используется как ОСНОВНОЙ сигнал детектора перехода
        (скалярный LR — фильтрующий).
        """
        h1 = self._features["h_hist"]
        h2 = other._features["h_hist"]
        # cv2.compareHist возвращает коэффициент Бхаттачарьи ∈ [0,1]
        # где 0 = идентичны. Мы возвращаем -ln(coeff) чтобы
        # расстояние росло при различии.
        import cv2
        coeff = cv2.compareHist(h1, h2, cv2.HISTCMP_BHATTACHARYYA)
        return float(coeff)

    def to_dict(self) -> dict:
        """
        Сериализовать состояние в словарь для логирования / JSON.
        """
        return {
            "frame_number": self.frame_number,
            "h_mean":       float(self._features["h_mean"]),
            "h_std":        float(self._features["h_std"]),
            "s_mean":       float(self._features["s_mean"]),
            "v_mean":       float(self._features["v_mean"]),
            "edge_ratio":   float(self._features["edge_ratio"]),
            "h_hist":       self._features["h_hist"].tolist(),
        }

    def __repr__(self):
        v = self._vec
        return (
            f"SurfaceState(frame={self.frame_number}, "
            f"h={v[0]:.3f}, s={v[2]:.3f}, v={v[3]:.3f}, "
            f"edge={v[4]:.3f})"
        )


# ─────────────────────────────────────────────────────────────
#  БЫСТРЫЙ ТЕСТ
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import cv2 as _cv2

    np.random.seed(7)

    def make_hsv(rgb_color, noise=10):
        """Создать синтетический HSV кадр заданного RGB цвета."""
        img = np.full((200, 200, 3), rgb_color, dtype=np.uint8)
        img = np.clip(
            img + np.random.randint(-noise, noise, img.shape), 0, 255
        ).astype(np.uint8)
        return _cv2.cvtColor(img, _cv2.COLOR_RGB2HSV)

    # Три "зоны"
    field_hsv  = make_hsv([180, 200, 80])    # поле — жёлто-зелёное
    forest_hsv = make_hsv([30,  100, 30])    # лес  — тёмно-зелёное
    road_hsv   = make_hsv([120, 110, 100])   # дорога — серая

    s_field  = SurfaceState(field_hsv,  frame_number=0)
    s_forest = SurfaceState(forest_hsv, frame_number=1)
    s_road   = SurfaceState(road_hsv,   frame_number=2)

    print("=" * 55)
    print("ТЕСТ 1: repr и feature_vector")
    print("=" * 55)
    for s in [s_field, s_forest, s_road]:
        print(f"  {s}")
        assert s.feature_vector().shape == (5,)
        assert s.histogram().shape == (16,)
    print("  OK\n")

    print("=" * 55)
    print("ТЕСТ 2: Евклидово расстояние")
    print("=" * 55)
    d_ff = s_field.distance_to(s_field)
    d_fl = s_field.distance_to(s_forest)
    d_fr = s_field.distance_to(s_road)
    print(f"  Поле  ↔ Поле  : {d_ff:.4f}  (должно быть 0)")
    print(f"  Поле  ↔ Лес   : {d_fl:.4f}")
    print(f"  Поле  ↔ Дорога: {d_fr:.4f}")
    assert d_ff == 0.0, "Расстояние до себя != 0"
    assert d_fl > 0.01, "Поле и лес слишком похожи"
    print("  OK\n")

    print("=" * 55)
    print("ТЕСТ 3: Расстояние Бхаттачарьи")
    print("=" * 55)
    b_ff = s_field.bhattacharyya_distance(s_field)
    b_fl = s_field.bhattacharyya_distance(s_forest)
    b_fr = s_field.bhattacharyya_distance(s_road)
    print(f"  Поле  ↔ Поле  : {b_ff:.4f}  (должно быть ≈ 0)")
    print(f"  Поле  ↔ Лес   : {b_fl:.4f}")
    print(f"  Поле  ↔ Дорога: {b_fr:.4f}")
    assert b_ff < 0.05, "Бхаттачарья до себя должна быть близка к 0"
    assert b_fl > b_ff, "Лес должен быть дальше чем сам от себя"
    print("  OK\n")

    print("=" * 55)
    print("ТЕСТ 4: to_dict")
    print("=" * 55)
    import json
    d = s_field.to_dict()
    print(json.dumps({k: v for k, v in d.items() if k != "h_hist"},
                     indent=2, ensure_ascii=False))
    assert "h_hist" in d and len(d["h_hist"]) == 16
    print("  OK\n")

    print("Все тесты пройдены ✓")
