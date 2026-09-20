import numpy as np

from jacobi_method import jacobi_eigen


def main():
    A = np.array([
        [1.0,  4.0,  7.0, 10.0, 13.0],
        [4.0,  7.0, 10.0, 13.0, 16.0],
        [7.0, 10.0, 13.0, 16.0, 19.0],
        [10.0, 13.0, 16.0, 19.0, 22.0],
        [13.0, 16.0, 19.0, 22.0, 25.0],
    ])

    vals, vecs = jacobi_eigen(A, tol=1e-12)

    print("собственные числа:")
    print(vals)
    print()
    print("собственные векторы:")
    print(vecs)
    print()
    print("A @ V == V @ diag(vals):",
          np.allclose(A @ vecs, vecs @ np.diag(vals), atol=1e-10))
    print("V ортонормирована:",
          np.allclose(vecs.T @ vecs, np.eye(5), atol=1e-10))

    ref, _ = np.linalg.eigh(A)
    print("совпадает с numpy:", np.allclose(vals, ref, atol=1e-8))


if __name__ == "__main__":
    main()