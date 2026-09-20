"""
Прогон эксперимента по лабораторной 2:
- ансамбли 2x2, 4x4, 16x16
- гауссовы и бернуллиевы элементы
- гистограммы разностей + кривая Вигнера
"""

import os
import numpy as np

from random_matrices.ensembles import gaussian_ensemble, bernoulli_ensemble
from random_matrices.spectrum import eigenvalues_sorted, unfold_all
from random_matrices.plotting import make_figure


SIZE = 2000       # число матриц в ансамбле
SEED = 42
SIZES = [2, 4, 16]


def run_family(name, generator, sizes, out_path):
    rng = np.random.default_rng(SEED)
    results = []

    for n in sizes:
        A = generator(n, SIZE, rng)
        vals = eigenvalues_sorted(A)
        sp = unfold_all(vals)

        # убираем возможные отрицательные/нулевые (не должны появляться,
        # но на всякий случай)
        sp = sp[sp > 0]

        title = "{} {}x{}".format(name, n, n)
        print(title, "— среднее s = {:.4f}".format(np.mean(sp)))

        results.append((title, sp))

    make_figure(results, out_path)


def main():
    os.makedirs("figures", exist_ok=True)

    run_family(
        "гаусс",
        gaussian_ensemble,
        SIZES,
        "figures/gaussian.png",
    )

    run_family(
        "бернулли ±1",
        bernoulli_ensemble,
        SIZES,
        "figures/bernoulli.png",
    )


if __name__ == "__main__":
    main()