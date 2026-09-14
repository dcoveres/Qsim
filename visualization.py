"""Опциональная визуализация (гистограммы, сфера Блоха) через matplotlib.

matplotlib — необязательная зависимость: этот модуль НЕ импортируется
автоматически из qsim/__init__.py (чтобы наличие/отсутствие matplotlib
никак не влияло на возможность использовать сам qsim), импортируйте
явно: ``from qsim.visualization import plot_histogram, plot_bloch``.
"""
from __future__ import annotations


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError(
            "Для визуализации нужен matplotlib: pip install matplotlib"
        ) from e
    return plt


def plot_histogram(counts, ax=None, title=None):
    """Столбчатая диаграмма распределения измерений — принимает counts,
    как их возвращает Circuit.run(shots=...) (первый элемент кортежа)."""
    plt = _require_matplotlib()
    if ax is None:
        _fig, ax = plt.subplots()
    labels = sorted(counts.keys())
    values = [counts[k] for k in labels]
    ax.bar(labels, values)
    ax.set_xlabel("Исход")
    ax.set_ylabel("Число измерений")
    if title:
        ax.set_title(title)
    if len(labels) > 6:
        for tick in ax.get_xticklabels():
            tick.set_rotation(45)
            tick.set_ha('right')
    return ax


def plot_bloch(vector, ax=None, title=None):
    """Рисует вектор Блоха (x, y, z) — например, из
    Simulator.bloch_vector(qubit) — на сфере Блоха."""
    plt = _require_matplotlib()
    import numpy as np

    x, y, z = vector
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

    u = np.linspace(0, 2 * np.pi, 40)
    v = np.linspace(0, np.pi, 40)
    xs = np.outer(np.cos(u), np.sin(v))
    ys = np.outer(np.sin(u), np.sin(v))
    zs = np.outer(np.ones_like(u), np.cos(v))
    ax.plot_wireframe(xs, ys, zs, color='lightgray', linewidth=0.3, alpha=0.5)

    # оси
    ax.plot([-1, 1], [0, 0], [0, 0], color='gray', linewidth=0.5)
    ax.plot([0, 0], [-1, 1], [0, 0], color='gray', linewidth=0.5)
    ax.plot([0, 0], [0, 0], [-1, 1], color='gray', linewidth=0.5)

    ax.quiver(0, 0, 0, x, y, z, color='crimson', linewidth=2)
    ax.set_xlim([-1, 1]); ax.set_ylim([-1, 1]); ax.set_zlim([-1, 1])
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    if title:
        ax.set_title(title)
    return ax
