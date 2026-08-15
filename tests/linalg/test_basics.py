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


# Property: addition is commutative (A+B == B+A) for any matrix, any shape.
# We generate one random shape first, then reuse it for both A and B via
# flatmap — generating A and B independently would almost always give them
# different shapes, which matrix_addition would just reject.
_shared_shape_pair = st.tuples(st.integers(1, 6), st.integers(1, 6)).flatmap(
    lambda shape: st.tuples(
        arrays(dtype=np.float64, shape=shape, elements=st.floats(-1e3, 1e3, allow_nan=False)),
        arrays(dtype=np.float64, shape=shape, elements=st.floats(-1e3, 1e3, allow_nan=False)),
    )
)


@given(_shared_shape_pair)
def test_add_is_commutative(matrices):
    A, B = matrices
    np.testing.assert_allclose(matrix_addition(A, B), matrix_addition(B, A))


# Property: our addition must agree with NumPy's own `+` across many random
# shapes, not just the one fixed example above.
@given(_shared_shape_pair)
def test_add_matches_numpy_any_shape(matrices):
    A, B = matrices
    np.testing.assert_allclose(matrix_addition(A, B), A + B)


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


# Property: matches NumPy's `scalar * A` regardless of shape (1D, 2D, 3D...)
# - this is the test that would have caught a dimension-specific bug, since
# the shape strategy below isn't restricted to 2D.
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


# Property: A @ B must match NumPy for any *compatible* shapes (n×m and m×p),
# not just one fixed example. n, m, p are generated independently, then A and
# B are built so their shared inner dimension m always lines up.
_compatible_matrix_pair = st.tuples(
    st.integers(1, 5), st.integers(1, 5), st.integers(1, 5)
).flatmap(
    lambda dims: st.tuples(
        arrays(dtype=np.float64, shape=(dims[0], dims[1]),
               elements=st.floats(-1e2, 1e2, allow_nan=False)),
        arrays(dtype=np.float64, shape=(dims[1], dims[2]),
               elements=st.floats(-1e2, 1e2, allow_nan=False)),
    )
)


@given(_compatible_matrix_pair)
def test_multiply_matches_numpy_any_compatible_shape(matrices):
    A, B = matrices
    np.testing.assert_allclose(matrix_multiplication(A, B), A @ B, rtol=1e-6, atol=1e-8)


# Property: multiplying by the identity matrix leaves any matrix unchanged
# (A @ I == A), for any square-compatible size, not just the 2x2 example above.
@given(st.integers(1, 6).flatmap(
    lambda n: arrays(dtype=np.float64, shape=(n, n), elements=st.floats(-1e2, 1e2, allow_nan=False))
))
def test_multiply_by_identity_is_unchanged_any_size(A):
    n = A.shape[0]
    np.testing.assert_allclose(matrix_multiplication(A, np.eye(n)), A, rtol=1e-6, atol=1e-8)