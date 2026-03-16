"""
features.py — Задачи 1-2
Конвертация RGB→HSV и извлечение признаков из кадра.
"""

import cv2
import numpy as np


# ─────────────────────────────────────────────────────────────
#  ЗАДАЧА 1 — RGB → HSV
# ─────────────────────────────────────────────────────────────

def rgb_to_hsv(frame: np.ndarray) -> np.ndarray:
    """
    Конвертировать кадр из RGB в HSV.

    Параметры
    ----------
    frame : np.ndarray
        Изображение в формате RGB, shape (H, W, 3), dtype uint8.

    Возвращает
    ----------
    np.ndarray
        HSV изображение той же формы.
        H: 0–180, S: 0–255, V: 0–255  (соглашение OpenCV)

    Примечание
    ----------
    OpenCV хранит H в диапазоне 0–180 (не 0–360),
    чтобы уместить в uint8. Учитывать при интерпретации.
    """
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"Ожидается (H, W, 3), получено {frame.shape}")
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)

    return cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)


def bgr_to_hsv(frame: np.ndarray) -> np.ndarray:
    """
    Конвертировать кадр из BGR (формат OpenCV по умолчанию) в HSV.
    Удобна когда кадры читаются через cv2.VideoCapture / cv2.imread.
    """
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)


# ─────────────────────────────────────────────────────────────
#  ЗАДАЧА 2 — Извлечение признаков
# ─────────────────────────────────────────────────────────────

# Количество бинов гистограммы H
H_BINS = 16

# Диапазоны каналов OpenCV HSV
H_MAX = 180.0
S_MAX = 255.0
V_MAX = 255.0


def extract_frame_features(hsv_frame: np.ndarray) -> dict:
    """
    Извлечь числовые признаки из HSV кадра.

    Параметры
    ----------
    hsv_frame : np.ndarray
        HSV изображение (H, W, 3), dtype uint8.

    Возвращает
    ----------
    dict со следующими ключами:

    Скалярные признаки (все нормализованы в [0, 1]):
        h_mean      — среднее значение канала H
        h_std       — стандартное отклонение канала H
        s_mean      — среднее канала S
        v_mean      — среднее канала V
        edge_ratio  — доля пикселей-границ по Canny на V-канале
        s_confidence — средняя насыщенность [0,1], доверие к H

    Векторные признаки:
        h_hist      — нормализованная гистограмма H, shape (H_BINS,)
                      взвешена по насыщенности S каждого пикселя.
                      Серые пиксели (низкое S) вносят малый вклад —
                      их H нестабилен и не несёт информации о зоне.
                      Сумма элементов = 1.0 (или 0 если кадр полностью серый).
        h_hist_raw  — обычная гистограмма H без взвешивания (для сравнения)
    """
    h, s, v = hsv_frame[:, :, 0], hsv_frame[:, :, 1], hsv_frame[:, :, 2]

    # ── Скалярные статистики ──────────────────────────────────
    h_mean = float(h.mean()) / H_MAX
    h_std  = float(h.std())  / H_MAX
    s_mean = float(s.mean()) / S_MAX
    v_mean = float(v.mean()) / V_MAX

    # Доверие к H: насколько насыщен кадр в среднем
    # 0.0 = полностью серый (H бессмысленен)
    # 1.0 = полностью насыщенный
    s_confidence = s_mean   # уже нормализовано в [0,1]

    # ── Доля граничных пикселей (Canny на V-канале) ───────────
    edges      = cv2.Canny(v, threshold1=30, threshold2=100)
    edge_ratio = float(np.count_nonzero(edges)) / edges.size

    # ── Взвешенная гистограмма H ──────────────────────────────
    # Идея: каждый пиксель вносит вклад в бин пропорционально
    # своей насыщенности S. Серые пиксели (S≈0) почти не влияют
    # на гистограмму — их H случаен и зашумлён.
    #
    # Реализация: используем S как маску весов.
    # calcHist не поддерживает веса напрямую, поэтому
    # считаем вручную через np.histogram.
    s_float  = s.astype(np.float32) / S_MAX      # веса ∈ [0, 1]
    h_flat   = h.flatten().astype(np.float32)
    s_flat   = s_float.flatten()

    h_hist_w, _ = np.histogram(
        h_flat,
        bins=H_BINS,
        range=(0, H_MAX),
        weights=s_flat
    )
    h_hist_w = h_hist_w.astype(np.float32)

    total_w = h_hist_w.sum()
    if total_w > 1e-6:
        h_hist_w /= total_w
    else:
        # Кадр полностью серый — равномерное распределение
        # (неопределённость максимальна, не провоцируем ложняк)
        h_hist_w = np.ones(H_BINS, dtype=np.float32) / H_BINS

    # ── Обычная гистограмма H (без весов, для визуализации) ───
    h_hist_raw = cv2.calcHist([hsv_frame], [0], None,
                              [H_BINS], [0, H_MAX])
    h_hist_raw = h_hist_raw.flatten().astype(np.float32)
    total_raw  = h_hist_raw.sum()
    if total_raw > 0:
        h_hist_raw /= total_raw

    return {
        "h_mean":       h_mean,
        "h_std":        h_std,
        "s_mean":       s_mean,
        "v_mean":       v_mean,
        "edge_ratio":   edge_ratio,
        "s_confidence": s_confidence,
        "h_hist":       h_hist_w,      # взвешенная — используется в детекторе
        "h_hist_raw":   h_hist_raw,    # сырая — для визуализации
    }


def feature_vector_from_dict(features: dict) -> np.ndarray:
    """
    Собрать скалярные признаки в numpy-вектор (без гистограммы).
    Порядок: [h_mean, h_std, s_mean, v_mean, edge_ratio]
    """
    return np.array([
        features["h_mean"],
        features["h_std"],
        features["s_mean"],
        features["v_mean"],
        features["edge_ratio"],
    ], dtype=np.float32)


# ─────────────────────────────────────────────────────────────
#  БЫСТРЫЙ ТЕСТ
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    np.random.seed(42)

    print("=" * 60)
    print("ТЕСТ 1: rgb_to_hsv — случайное изображение")
    print("=" * 60)
    rgb = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    hsv = rgb_to_hsv(rgb)
    assert hsv.shape == rgb.shape
    assert hsv.dtype == np.uint8
    assert hsv[:, :, 0].max() <= 180
    print(f"  shape : {hsv.shape}")
    print(f"  H range: [{hsv[:,:,0].min()}, {hsv[:,:,0].max()}]")
    print("  OK\n")

    print("=" * 60)
    print("ТЕСТ 2: Взвешивание по S — поле vs дорога")
    print("=" * 60)

    # Поле — насыщенный жёлто-зелёный
    field_rgb = np.full((200, 200, 3), [180, 200, 80], dtype=np.uint8)
    field_rgb += np.random.randint(0, 15, field_rgb.shape, dtype=np.uint8)

    # Дорога — серая (низкое S, H нестабилен)
    road_rgb = np.full((200, 200, 3), [120, 118, 115], dtype=np.uint8)
    road_rgb += np.random.randint(0, 8, road_rgb.shape, dtype=np.uint8)

    # Другая серая дорога — другой оттенок серого но тот же тип
    road2_rgb = np.full((200, 200, 3), [100, 98, 95], dtype=np.uint8)
    road2_rgb += np.random.randint(0, 8, road2_rgb.shape, dtype=np.uint8)

    field_hsv = rgb_to_hsv(field_rgb)
    road_hsv  = rgb_to_hsv(road_rgb)
    road2_hsv = rgb_to_hsv(road2_rgb)

    f_field = extract_frame_features(field_hsv)
    f_road  = extract_frame_features(road_hsv)
    f_road2 = extract_frame_features(road2_hsv)

    print(f"  Поле   s_confidence = {f_field['s_confidence']:.3f}  (ожидается высокое ~0.5+)")
    print(f"  Дорога s_confidence = {f_road['s_confidence']:.3f}   (ожидается низкое ~0.05)")

    # Бхаттачарья между двумя серыми дорогами
    # взвешенная должна быть МЕНЬШЕ чем сырая
    # (серые объекты не должны казаться разными из-за шума H)
    import cv2 as _cv2
    bhat_weighted = float(_cv2.compareHist(
        f_road["h_hist"], f_road2["h_hist"],
        _cv2.HISTCMP_BHATTACHARYYA
    ))
    bhat_raw = float(_cv2.compareHist(
        f_road["h_hist_raw"], f_road2["h_hist_raw"],
        _cv2.HISTCMP_BHATTACHARYYA
    ))
    print(f"\n  Дорога1 vs Дорога2:")
    print(f"    Бхаттачарья взвешенная : {bhat_weighted:.4f}")
    print(f"    Бхаттачарья сырая      : {bhat_raw:.4f}")
    print(f"    Улучшение: {'ДА ✓' if bhat_weighted < bhat_raw else 'НЕТ ✗'}")

    # Бхаттачарья поле vs дорога должна быть высокой в обоих случаях
    bhat_fd_w = float(_cv2.compareHist(
        f_field["h_hist"], f_road["h_hist"],
        _cv2.HISTCMP_BHATTACHARYYA
    ))
    bhat_fd_r = float(_cv2.compareHist(
        f_field["h_hist_raw"], f_road["h_hist_raw"],
        _cv2.HISTCMP_BHATTACHARYYA
    ))
    print(f"\n  Поле vs Дорога:")
    print(f"    Бхаттачарья взвешенная : {bhat_fd_w:.4f}  (должна быть высокой)")
    print(f"    Бхаттачарья сырая      : {bhat_fd_r:.4f}")
    print("  OK\n")

    print("=" * 60)
    print("ТЕСТ 3: Серый кадр → равномерное распределение")
    print("=" * 60)
    grey_rgb = np.full((100, 100, 3), [128, 128, 128], dtype=np.uint8)
    grey_hsv = rgb_to_hsv(grey_rgb)
    f_grey   = extract_frame_features(grey_hsv)
    hist_std = f_grey["h_hist"].std()
    print(f"  std гистограммы серого кадра: {hist_std:.6f}  (должно быть ≈ 0 — равномерное)")
    assert hist_std < 0.01, "Серый кадр должен давать равномерную гистограмму"
    print("  OK\n")

    print("=" * 60)
    print("ТЕСТ 4: Гистограмма нормализована (сумма = 1)")
    print("=" * 60)
    for name, feat in [("Поле", f_field), ("Дорога", f_road), ("Серый", f_grey)]:
        s_w = feat["h_hist"].sum()
        s_r = feat["h_hist_raw"].sum()
        print(f"  {name}: взвешенная={s_w:.6f}  сырая={s_r:.6f}")
        assert abs(s_w - 1.0) < 1e-4
        assert abs(s_r - 1.0) < 1e-4
    print("  OK\n")

    print("Все тесты пройдены ✓")

    print("=" * 55)
    print("ТЕСТ 1: rgb_to_hsv — случайное изображение")
    print("=" * 55)
    rgb = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    hsv = rgb_to_hsv(rgb)
    assert hsv.shape == rgb.shape, "Форма изменилась!"
    assert hsv.dtype == np.uint8,  "dtype должен быть uint8"
    assert hsv[:, :, 0].max() <= 180, "H выходит за 180"
    print(f"  shape : {hsv.shape}")
    print(f"  H range: [{hsv[:,:,0].min()}, {hsv[:,:,0].max()}]  (ожидается ≤ 180)")
    print(f"  S range: [{hsv[:,:,1].min()}, {hsv[:,:,1].max()}]")
    print(f"  V range: [{hsv[:,:,2].min()}, {hsv[:,:,2].max()}]")
    print("  OK\n")

    # ─────────────────────────────────────────────────────────
    print("=" * 55)
    print("ТЕСТ 2: extract_frame_features — поле vs лес")
    print("=" * 55)

    # Синтетическое "поле": жёлто-зелёный, малонасыщенный
    field_rgb = np.zeros((200, 200, 3), dtype=np.uint8)
    field_rgb[:, :] = [180, 200, 80]   # RGB жёлто-зелёный
    field_rgb += np.random.randint(0, 15, field_rgb.shape, dtype=np.uint8)

    # Синтетический "лес": тёмно-зелёный, насыщенный
    forest_rgb = np.zeros((200, 200, 3), dtype=np.uint8)
    forest_rgb[:, :] = [30, 100, 30]   # RGB тёмно-зелёный
    forest_rgb += np.random.randint(0, 15, forest_rgb.shape, dtype=np.uint8)

    field_hsv  = rgb_to_hsv(field_rgb)
    forest_hsv = rgb_to_hsv(forest_rgb)

    f_field  = extract_frame_features(field_hsv)
    f_forest = extract_frame_features(forest_hsv)

    print(f"{'Признак':<14} {'Поле':>10} {'Лес':>10} {'Разница':>10}")
    print("-" * 46)
    for key in ["h_mean", "h_std", "s_mean", "v_mean", "edge_ratio"]:
        diff = abs(f_field[key] - f_forest[key])
        print(f"  {key:<12} {f_field[key]:>10.4f} {f_forest[key]:>10.4f} {diff:>10.4f}")

    print()

    # Проверяем что признаки различаются
    vec_field  = feature_vector_from_dict(f_field)
    vec_forest = feature_vector_from_dict(f_forest)
    dist = np.linalg.norm(vec_field - vec_forest)
    print(f"  Евклидово расстояние между векторами: {dist:.4f}")
    assert dist > 0.05, "Признаки поля и леса слишком похожи — что-то не так!"
    print("  OK\n")

    # ─────────────────────────────────────────────────────────
    print("=" * 55)
    print("ТЕСТ 3: Гистограмма H нормализована (сумма = 1)")
    print("=" * 55)
    for name, feat in [("Поле", f_field), ("Лес", f_forest)]:
        s = feat["h_hist"].sum()
        print(f"  {name}: сумма гистограммы = {s:.6f}")
        assert abs(s - 1.0) < 1e-5, f"Сумма != 1 для {name}"
    print("  OK\n")

    print("Все тесты пройдены ✓")
