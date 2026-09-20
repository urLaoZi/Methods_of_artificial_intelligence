# jacobi_method

Метод Якоби для поиска собственных чисел и собственных векторов
действительной симметричной матрицы.

## Установка

    pip install -e ".[dev]"

## Использование

    import numpy as np
    from jacobi_method import jacobi_eigen

    A = np.array([[4.0, 1.0], [1.0, 3.0]])
    vals, vecs = jacobi_eigen(A)
    print(vals)
    print(vecs)

## Что возвращает

`jacobi_eigen(A, tol=1e-10, max_iter=1000)` возвращает пару
`(vals, vecs)`. `vals` — собственные числа по возрастанию, `vecs` —
матрица, столбцы которой являются соответствующими собственными
векторами.

## Ошибки

- `NotSquareMatrixError` — матрица не квадратная
- `NotSymmetricMatrixError` — матрица не симметричная
- `BadToleranceError` — tol <= 0
- `ConvergenceError` — не сошлось за max_iter
- `ValueError` — плохой max_iter или NaN/inf в матрице

## Тесты

    pytest tests -v

## Как работает

На каждом шаге ищем максимальный по модулю недиагональный элемент,
строим вращение Гивенса, зануляем этот элемент. Сумма квадратов
недиагональных элементов уменьшается, по ней и проверяем сходимость.