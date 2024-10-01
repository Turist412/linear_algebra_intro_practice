from typing import Sequence

import numpy as np
from scipy import sparse


def get_vector(dim: int) -> np.ndarray:
    """Create random column vector with dimension dim.

        Args:
            dim (int): vector dimension.

        Returns:
            np.ndarray: column vector.
        """
    return np.random.rand(dim, 1)


def get_sparce_vector(dim: int) -> sparse.coo_matrix:
    """Create random sparse column vector with dimension dim.

        Args:
            dim (int): vector dimension.

        Returns:
            sparse.coo_matrix: sparse column vector.
        """
    n_zero = np.random.randint(1, dim)
    row_indices = np.random.choice(dim, n_zero, replace=False)
    values = np.random.rand(np.random.randint(1, dim))
    return sparse.coo_matrix((values, (row_indices, np.zeros(n_zero))), shape=(dim, 1))


def add(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Vector addition.

        Args:
            x (np.ndarray): 1th vector.
            y (np.ndarray): 2nd vector.

        Returns:
            np.ndarray: vector sum.
        """
    if x.shape != y.shape:
        raise ValueError("Shape is not the same.")
    return np.add(x, y)


def scalar_multiplication(x: np.ndarray, a: float) -> np.ndarray:
    """Vector multiplication by scalar.

    Args:
        x (np.ndarray): vector.
        a (float): scalar.

    Returns:
        np.ndarray: multiplied vector.
    """
    max_len = len(x)
    result = np.zeros(max_len)
    result[:max_len] = x * a
    return result


def linear_combination(vectors: Sequence[np.ndarray], coeffs: Sequence[float]) -> np.ndarray:
    """Linear combination of vectors.

    Args:
        vectors (Sequence[np.ndarray]): list of vectors of len N.
        coeffs (Sequence[float]): list of coefficients of len N.

    Returns:
        np.ndarray: linear combination of vectors.
    """
    if len(vectors) != len(coeffs):
        raise ValueError("Number of vectors and coeffs must be the same.")

    result = np.zeros_like(vectors[0])
    for vec, coeff in zip(vectors, coeffs):
        result += vec * coeff

    return result

def dot_product(x: np.ndarray, y: np.ndarray) -> float:
    """Vectors dot product.

    Args:
        x (np.ndarray): 1th vector.
        y (np.ndarray): 2nd vector.

    Returns:
        float: dot product.
    """
    if x.shape != y.shape:
        raise ValueError("Shapes are incompatible for dot product.")
    return float(np.dot(x.T, y).item())

def norm(x: np.ndarray, order: int | float) -> float:
    """Vector norm: Manhattan, Euclidean or Max.

    Args:
        x (np.ndarray): vector
        order (int | float): norm's order: 1, 2 or inf.

    Returns:
        float: vector norm
    """
    if order == 1:
        return np.sum(np.abs(x))
    elif order == 2:
        return np.sqrt(np.sum(np.square(x)))
    elif order == float('inf'):
        return np.max(np.abs(x))
    else:
        raise ValueError("Order must be 1, 2, or inf.")

def distance(x: np.ndarray, y: np.ndarray) -> float:
    """L2 distance between vectors.

    Args:
        x (np.ndarray): 1th vector.
        y (np.ndarray): 2nd vector.

    Returns:
        float: distance.
    """
    if x.shape != y.shape:
        raise ValueError("Vectors must have the same shape.")
    return np.sqrt(np.sum(np.square(x - y)))


def cos_between_vectors(x: np.ndarray, y: np.ndarray) -> float:
    """Cosine between vectors in degrees.

    Args:
        x (np.ndarray): 1st vector.
        y (np.ndarray): 2nd vector.

    Returns:
        float: angle in degrees.
    """
    if x.shape != y.shape:
        raise ValueError("Vectors must have the same shape.")
    return np.degrees(np.arccos(dot_product(x, y) / (norm(x, 2) * norm(y, 2))))


def is_orthogonal(x: np.ndarray, y: np.ndarray) -> bool:
    """Check is vectors orthogonal.

    Args:
        x (np.ndarray): 1th vector.
        y (np.ndarray): 2nd vector.


    Returns:
        bool: are vectors orthogonal.
    """
    if x.shape != y.shape:
        raise ValueError("Vectors must have the same shape.")
    return dot_product(x, y) == 0


def solves_linear_systems(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve system of linear equations.

    Args:
        a (np.ndarray): coefficient matrix.
        b (np.ndarray): ordinate values.

    Returns:
        np.ndarray: sytems solution
    """
    if a.shape[0] != b.shape[0]:
        raise ValueError("Number of rows in matrix a and number of elements in vector b must be the same.")
    return np.linalg.solve(a, b)

