import numpy as np


def gaussian_ensemble(n, size, rng):
    """
    Ансамбль симметричных матриц размера n x n.
    Элементы: нормальное распределение N(0, 1).
    Возвращает массив формы (size, n, n).
    """
    A = rng.normal(0.0, 1.0, size=(size, n, n))
    # симметризуем
    A = (A + A.transpose(0, 2, 1)) / np.sqrt(2.0)
    return A


def bernoulli_ensemble(n, size, rng):
    """
    Ансамбль симметричных матриц размера n x n.
    Элементы: ±1 с вероятностью 1/2.
    """
    A = rng.choice([-1.0, 1.0], size=(size, n, n))
    A = (A + A.transpose(0, 2, 1)) / np.sqrt(2.0)
    return A