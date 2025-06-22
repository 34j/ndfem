import numpy as np
import pytest
from numpy.testing import assert_allclose

from ndfem.simplex import (
    all_simplex_permutations,
    barycentric_to_cartesian,
    cartesian_to_barycentric,
    reference_simplex,
)


@pytest.mark.parametrize("n", [1, 4])
def test_reference_simplex(n: int) -> None:
    simplex = reference_simplex(n)
    assert simplex.shape == (n + 1, n)


@pytest.mark.parametrize("n", [1, 4])
def test_simplex(n: int) -> None:
    rng = np.random.default_rng()
    simplex = rng.random((n + 1, n))
    cartesian = rng.random((100, n))
    barycentric = cartesian_to_barycentric(cartesian, simplex)
    cartesian2 = barycentric_to_cartesian(barycentric, simplex)
    assert_allclose(cartesian, cartesian2)


@pytest.mark.parametrize("n", [1, 3])
def test_rotated_simplex(n: int) -> None:
    print(all_simplex_permutations(n, max(1, n - 1)))
