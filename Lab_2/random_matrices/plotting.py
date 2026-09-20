import numpy as np
import matplotlib.pyplot as plt

from .spectrum import wigner_surmise


def plot_histogram(spacings, ax, title, bins=50, s_max=3.0):
    """
    Гистограмма нормированных разностей + теоретическая кривая.
    """
    ax.hist(
        spacings,
        bins=bins,
        range=(0.0, s_max),
        density=True,
        alpha=0.6,
        color="steelblue",
        edgecolor="black",
        linewidth=0.3,
        label="гистограмма",
    )

    s = np.linspace(0.0, s_max, 400)
    ax.plot(s, wigner_surmise(s), "r-", linewidth=2,
            label="Вигнер (формула 15)")

    ax.set_xlim(0.0, s_max)
    ax.set_xlabel("s (нормированная разность)")
    ax.set_ylabel("плотность вероятности")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)


def make_figure(results, fname):
    """
    results: список (title, spacings).
    Строит сетку графиков и сохраняет в файл.
    """
    n = len(results)
    cols = 2
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows))
    axes = np.atleast_1d(axes).ravel()

    for ax, (title, sp) in zip(axes, results):
        plot_histogram(sp, ax, title)

    for ax in axes[len(results):]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(fname, dpi=130)
    plt.close(fig)
    print("сохранил", fname)