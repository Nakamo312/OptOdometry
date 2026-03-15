"""
gaussian_utils.py — Задача 5
Функции для работы с многомерным нормальным распределением.
"""

import numpy as np
from scipy import linalg


# Малое число для регуляризации ковариационной матрицы
REGULARIZATION = 1e-6


def estimate_gaussian(feature_matrix: np.ndarray) -> tuple:
    """
    Оценить параметры многомерного нормального распределения.

    Параметры
    ----------
    feature_matrix : np.ndarray
        Матрица признаков, shape (N, D), где N — число наблюдений.

    Возвращает
    ----------
    mu : np.ndarray
        Вектор средних, shape (D,).
    sigma : np.ndarray
        Ковариационная матрица, shape (D, D).
        Регуляризована — гарантированно невырождена.
    """
    if feature_matrix.ndim != 2:
        raise ValueError(f"Ожидается (N, D), получено {feature_matrix.shape}")

    n, d = feature_matrix.shape
    if n < 2:
        raise ValueError(f"Нужно минимум 2 наблюдения, получено {n}")

    mu    = feature_matrix.mean(axis=0)                    # (D,)
    sigma = np.cov(feature_matrix, rowvar=False)           # (D, D)

    # Если D=1 — np.cov возвращает скаляр, приводим к матрице
    if sigma.ndim == 0:
        sigma = np.array([[float(sigma)]])

    # Регуляризация: добавляем eps * I чтобы матрица была
    # положительно определённой даже при малом числе наблюдений
    sigma += REGULARIZATION * np.eye(d)

    return mu, sigma


def gaussian_logpdf(feature_vector: np.ndarray,
                    mu: np.ndarray,
                    sigma: np.ndarray) -> float:
    """
    Логарифм плотности многомерного нормального распределения.

    log p(x | μ, Σ) = -0.5 * [D*ln(2π) + ln|Σ| + (x-μ)ᵀ Σ⁻¹ (x-μ)]

    Параметры
    ----------
    feature_vector : np.ndarray  shape (D,)
    mu             : np.ndarray  shape (D,)
    sigma          : np.ndarray  shape (D, D)

    Возвращает
    ----------
    float — log-правдоподобие (≤ 0 для нормированного распределения,
            но может быть > 0 если плотность > 1).
    """
    x   = feature_vector - mu           # (D,)
    d   = len(x)

    # Знак и логарифм определителя через разложение Холецкого
    # (численно стабильнее чем np.linalg.det)
    try:
        chol  = linalg.cholesky(sigma, lower=True)
        # log|Σ| = 2 * sum(log(diag(L)))
        log_det = 2.0 * np.sum(np.log(np.diag(chol)))
        # Σ⁻¹ x через решение треугольной системы
        sol   = linalg.solve_triangular(chol, x, lower=True)
        maha2 = float(np.dot(sol, sol))          # Mahalonobis² = xᵀ Σ⁻¹ x
    except linalg.LinAlgError:
        # Если матрица всё равно плохая — добавляем регуляризацию
        sigma_reg = sigma + 1e-4 * np.eye(d)
        chol  = linalg.cholesky(sigma_reg, lower=True)
        log_det = 2.0 * np.sum(np.log(np.diag(chol)))
        sol   = linalg.solve_triangular(chol, x, lower=True)
        maha2 = float(np.dot(sol, sol))

    log_2pi = d * np.log(2.0 * np.pi)
    return -0.5 * (log_2pi + log_det + maha2)


def gaussian_logpdf_batch(feature_matrix: np.ndarray,
                           mu: np.ndarray,
                           sigma: np.ndarray) -> np.ndarray:
    """
    Логарифм плотности для нескольких наблюдений сразу.

    Параметры
    ----------
    feature_matrix : np.ndarray  shape (N, D)

    Возвращает
    ----------
    np.ndarray shape (N,) — log-правдоподобие каждого наблюдения.
    """
    return np.array([
        gaussian_logpdf(row, mu, sigma)
        for row in feature_matrix
    ], dtype=np.float64)


# ─────────────────────────────────────────────────────────────
#  БЫСТРЫЙ ТЕСТ
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    np.random.seed(0)

    print("=" * 55)
    print("ТЕСТ 1: estimate_gaussian на известном распределении")
    print("=" * 55)
    true_mu    = np.array([0.3, 0.5, 0.2, 0.7, 0.1])
    true_sigma = np.diag([0.01, 0.02, 0.01, 0.03, 0.005])

    samples = np.random.multivariate_normal(true_mu, true_sigma, size=200)
    mu_est, sigma_est = estimate_gaussian(samples)

    print(f"  Истинное   μ: {true_mu}")
    print(f"  Оценённое  μ: {np.round(mu_est, 3)}")
    print(f"  Максимальная ошибка μ: {np.abs(mu_est - true_mu).max():.4f}")
    assert np.abs(mu_est - true_mu).max() < 0.05, "Оценка μ слишком далека"

    diag_true = np.diag(true_sigma)
    diag_est  = np.diag(sigma_est)
    print(f"  Диагональ истинной Σ:   {diag_true}")
    print(f"  Диагональ оценённой Σ:  {np.round(diag_est, 4)}")
    print("  OK\n")

    print("=" * 55)
    print("ТЕСТ 2: gaussian_logpdf — точка в центре vs далеко")
    print("=" * 55)
    mu    = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
    sigma = np.eye(5) * 0.01

    x_center = mu.copy()
    x_far    = mu + np.ones(5) * 0.5

    lp_center = gaussian_logpdf(x_center, mu, sigma)
    lp_far    = gaussian_logpdf(x_far,    mu, sigma)

    print(f"  log p(центр) = {lp_center:.2f}")
    print(f"  log p(далеко)= {lp_far:.2f}")
    assert lp_center > lp_far, "Центр должен иметь выше log p"
    print("  OK\n")

    print("=" * 55)
    print("ТЕСТ 3: Регуляризация при малом числе наблюдений")
    print("=" * 55)
    tiny = np.random.randn(3, 5).astype(np.float32) * 0.001
    mu_t, sig_t = estimate_gaussian(tiny)
    lp = gaussian_logpdf(tiny[0], mu_t, sig_t)
    print(f"  3 наблюдения: log p = {lp:.2f}  (не должно падать)")
    print("  OK\n")

    print("=" * 55)
    print("ТЕСТ 4: batch vs поштучно")
    print("=" * 55)
    X = np.random.multivariate_normal(mu, sigma, 10)
    batch  = gaussian_logpdf_batch(X, mu, sigma)
    single = np.array([gaussian_logpdf(x, mu, sigma) for x in X])
    assert np.allclose(batch, single, atol=1e-10)
    print(f"  Максимальное расхождение batch vs single: "
          f"{np.abs(batch - single).max():.2e}")
    print("  OK\n")

    print("Все тесты пройдены ✓")
