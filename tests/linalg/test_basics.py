import numpy as np
import pytest
from hypothesis import given
from hypothesis.extra.numpy import arrays
from hypothesis import strategies as st

from mathsformllib.linalg.basics import (
    matrix_addition,
    matrix_multiplication,
    matrix_by_scalar_multiplication,
)


# --- matrix_addition ---

def test_add_known_value():
    A = np.array([[1.0, 2.0], [3.0, 4.0]])
    B = np.array([[5.0, 6.0], [7.0, 8.0]])
    expected = np.array([[6.0, 8.0], [10.0, 12.0]])
    np.testing.assert_allclose(matrix_addition(A, B), expected)


def test_add_matches_numpy():
    A = np.array([[1.0, -2.0], [3.5, 0.0]])
    B = np.array([[0.5, 2.0], [-1.0, 4.0]])
    np.testing.assert_allclose(matrix_addition(A, B), A + B)


def test_add_rejects_shape_mismatch():
    with pytest.raises(AssertionError):
        matrix_addition(np.zeros((2, 2)), np.zeros((3, 2)))


@given(arrays(dtype=np.float64, shape=(3, 3), elements=st.floats(-1e3, 1e3, allow_nan=False)),
       arrays(dtype=np.float64, shape=(3, 3), elements=st.floats(-1e3, 1e3, allow_nan=False)))
def test_add_is_commutative(A, B):
    np.testing.assert_allclose(matrix_addition(A, B), matrix_addition(B, A))


# --- matrix_by_scalar_multiplication ---

def test_scalar_multiply_known_value():
    A = np.array([[1.0, 2.0], [3.0, 4.0]])
    expected = np.array([[2.0, 4.0], [6.0, 8.0]])
    np.testing.assert_allclose(matrix_by_scalar_multiplication(A, 2.0), expected)


def test_scalar_multiply_preserves_precision_on_int_input():
    A = np.array([[1, 2], [3, 4]])  # int dtype on purpose
    result = matrix_by_scalar_multiplication(A, 0.5)
    np.testing.assert_allclose(result, np.array([[0.5, 1.0], [1.5, 2.0]]))


def test_scalar_multiply_works_on_1d():
    v = np.array([1.0, 2.0, 3.0])
    np.testing.assert_allclose(matrix_by_scalar_multiplication(v, 2.0), np.array([2.0, 4.0, 6.0]))


def test_scalar_multiply_works_on_3d():
    A = np.ones((2, 2, 2))
    np.testing.assert_allclose(matrix_by_scalar_multiplication(A, 5.0), 5.0 * np.ones((2, 2, 2)))


@given(arrays(dtype=np.float64, shape=st.tuples(st.integers(1, 4), st.integers(1, 4)),
              elements=st.floats(-1e3, 1e3, allow_nan=False)),
       st.floats(-100, 100, allow_nan=False))
def test_scalar_multiply_matches_numpy_any_shape(A, scalar):
    np.testing.assert_allclose(matrix_by_scalar_multiplication(A, scalar), scalar * A, rtol=1e-6)


# --- matrix_multiplication ---

def test_multiply_known_value():
    A = np.array([[1.0, 2.0], [3.0, 4.0]])
    B = np.array([[5.0, 6.0], [7.0, 8.0]])
    expected = np.array([[19.0, 22.0], [43.0, 50.0]])
    np.testing.assert_allclose(matrix_multiplication(A, B), expected)


def test_multiply_matches_numpy():
    A = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    B = np.array([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]])
    np.testing.assert_allclose(matrix_multiplication(A, B), A @ B)


def test_multiply_rejects_dimension_mismatch():
    with pytest.raises(ValueError):
        matrix_multiplication(np.zeros((2, 3)), np.zeros((2, 2)))


def test_multiply_by_identity_is_unchanged():
    A = np.array([[1.0, 2.0], [3.0, 4.0]])
    np.testing.assert_allclose(matrix_multiplication(A, np.eye(2)), A)


def test_multiply_is_not_generally_commutative():
    A = np.array([[1.0, 2.0], [0.0, 1.0]])
    B = np.array([[1.0, 0.0], [3.0, 1.0]])
    assert not np.allclose(matrix_multiplication(A, B), matrix_multiplication(B, A))