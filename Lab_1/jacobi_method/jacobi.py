import math
import numpy as np

from .exceptions import (
    BadToleranceError,
    ConvergenceError,
    NotSymmetricMatrixError,
    NotSquareMatrixError,
)


def jacobi_eigen(A, tol=1e-10, max_iter=1000):
    # --- проверки входных данных ---
    if not isinstance(A, np.ndarray):
        raise TypeError("A должен быть numpy.ndarray")

    if A.ndim != 2:
        raise NotSquareMatrixError("матрица должна быть двумерной")

    if A.shape[0] != A.shape[1]:
        raise NotSquareMatrixError(
            "матрица должна быть квадратной, а тут {}".format(A.shape)
        )

    if not np.all(np.isfinite(A)):
        raise ValueError("в матрице есть NaN или inf")

    if not np.allclose(A, A.T, atol=1e-12):
        raise NotSymmetricMatrixError("матрица должна быть симметричной")

    if isinstance(tol, bool) or not isinstance(tol, (int, float)):
        raise BadToleranceError("tol должно быть числом")

    if tol <= 0:
        raise BadToleranceError("tol должно быть больше нуля")

    if isinstance(max_iter, bool) or not isinstance(max_iter, int):
        raise ValueError("max_iter должен быть целым")

    if max_iter <= 0:
        raise ValueError("max_iter должен быть положительным")

    # --- собственно метод ---
    A = np.array(A, dtype=float, copy=True)
    n = A.shape[0]

    V = np.eye(n)

    S = float(np.sum(A * A) - np.sum(np.diag(A) ** 2))

    if S == 0.0:
        vals = np.diag(A).copy()
        order = np.argsort(vals)
        return vals[order], V

    converged = False

    for _ in range(max_iter):
        if S < tol:
            converged = True
            break

        p = 0
        q = 1
        mx = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                v = abs(A[i, j])
                if v > mx:
                    mx = v
                    p, q = i, j

        if mx < 1e-15:
            converged = True
            break

        apq = A[p, q]
        app = A[p, p]
        aqq = A[q, q]

        C = (aqq - app) / (2.0 * apq)

        if C > 0:
            t = 1.0 / (C + math.sqrt(C * C + 1.0))
        else:
            t = 1.0 / (C - math.sqrt(C * C + 1.0))

        cos_phi = 1.0 / math.sqrt(1.0 + t * t)
        sin_phi = t * cos_phi

        col_p = A[:, p].copy()
        col_q = A[:, q].copy()

        A[p, p] = app - apq * t
        A[q, q] = aqq + apq * t
        A[p, q] = 0.0
        A[q, p] = 0.0

        factor = sin_phi / (1.0 + cos_phi)
        for r in range(n):
            if r == p or r == q:
                continue

            new_rp = col_p[r] - sin_phi * (col_q[r] + factor * col_p[r])
            new_rq = col_q[r] + sin_phi * (col_p[r] - factor * col_q[r])

            A[r, p] = new_rp
            A[p, r] = new_rp
            A[r, q] = new_rq
            A[q, r] = new_rq

        vp = V[:, p].copy()
        vq = V[:, q].copy()
        V[:, p] = vp * cos_phi - vq * sin_phi
        V[:, q] = vp * sin_phi + vq * cos_phi

        S -= apq * apq

    if not converged:
        raise ConvergenceError(
            "метод Якоби не сошёлся за {} итераций, "
            "остаточная сумма квадратов = {:.3e}".format(max_iter, S)
        )

    vals = np.diag(A).copy()

    order = np.argsort(vals)
    vals = vals[order]
    V = V[:, order]

    for j in range(n):
        nrm = np.linalg.norm(V[:, j])
        if nrm > 0:
            V[:, j] /= nrm

    return vals, V