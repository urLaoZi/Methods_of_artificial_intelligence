import numpy as np


def eigenvalues_sorted(A):
    """
    Собственные числа симметричных матриц из ансамбля.
    A: (size, n, n) -> (size, n), отсортированы по возрастанию.
    """
    vals = np.linalg.eigvalsh(A)
    return np.sort(vals, axis=1)


def level_spacings(vals):
    """
    Разности между соседними собственными числами.
    vals: (size, n) -> (size, n-1)
    """
    return np.diff(vals, axis=1)


def normalize_spacings(spacings):
    """
    Нормировка так, чтобы среднее было равно 1.
    """
    return spacings / np.mean(spacings)


def unfold_all(vals):
    """
    Полный конвейер: сортировка -> разности -> нормировка.
    """
    sp = level_spacings(vals)
    return normalize_spacings(sp)


def wigner_surmise(s):
    """
    Формула (15): rho(s) = (pi*s/2) * exp(-pi*s^2/4).
    """
    return (np.pi * s / 2.0) * np.exp(-np.pi * s * s / 4.0)