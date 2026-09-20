class JacobiError(Exception):
    pass


class NotSquareMatrixError(JacobiError):
    pass


class NotSymmetricMatrixError(JacobiError):
    pass


class BadToleranceError(JacobiError):
    pass


class ConvergenceError(JacobiError):
    pass