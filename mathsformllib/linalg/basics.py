"""
Basic matrix and scalar operations.
"""
import numpy as np

def matrix_addition(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Returns the sum of two matrices.

    The matrices must share the same dimensions for the operation
    to be defined.

    Vectorized version of this operation using numpy arrays would be: A+B.

    :param A: Matrix of shape m*n.
    :param B: Matrix of shape m*n.
    :return: Sum of two matrices of shape m*n.
    """
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError("A and B must be 2D matrices")
    if A.shape != B.shape:
        raise ValueError("A and B must have the same shape")

    result = np.zeros_like(A)  # We could take A or B, its the same
    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            result[i, j] = A[i, j] + B[i, j]
    return result

def matrix_multiplication(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Returns the product of two 2D matrices.

    The "inner" dimensions of the matrices must match for the operation
    to be defined, e.g. multiplying A of shape m*n with B of shape p*k, with n=p, gives the
    result of shape m*k.

    Vectorized version of this operation using numpy arrays would be: A*B.

    :param a: Matrix of shape m*n.
    :param b: Matrix of shape p*k (with p=n).
    :return: Product of two matrices of shape m*k.
    """
    assert A.ndim == 2 and B.ndim == 2, "A and B must be 2D matrices"

    n, m = A.shape
    m2, p = B.shape
    if m != m2:
        raise ValueError(f"inner dimensions must match: {A.shape} vs {B.shape}")

    result = np.zeros_like(A, dtype=float)
    # We iterate of A's dimensions since the order of the multiplication matters,
    # it's first matrix multiplied by the second, not the other way around:
    for i in range(n):
        for j in range(p):
            total = 0.0
            for k in range(m):
                total += A[i, k] * B[k, j]
            result[i, j] = total
    return result

def matrix_by_scalar_multiplication(A: np.ndarray, b: int | float) -> np.ndarray:
    """
    Element wise multiplication of a matrix by a scalar.

    Works regarding of the matrix's dimensions.

    :param A: Matrix to be multiplied.
    :param b: Scalar to multiply the matrix by.
    :return: Result of the multiplication.
    """
    # Forcing dtype so that we don't truncate result to int
    result = np.zeros_like(A, dtype=float)
    for idx in np.ndindex(result.shape):
        result[idx] = A[idx]*b
    return result