from ndfem.simplex import reference_simplex, barycentric_to_cartesian, cartesian_to_barycentric
import numpy as np
from numpy.testing import assert_allclose
import pytest
@pytest.mark.parametrize("n", [1, 4])
def test_reference_simplex(n: int) -> None:
    simplex = reference_simplex(n, ref=np.array(0))
    assert simplex.shape == (n + 1, n)


@pytest.mark.parametrize("n", [1, 4])
def test_simplex(n: int) -> None:
    rng = np.random.default_rng()
    simplex = rng.random((n + 1, n))
    cartesian = rng.random((100, n))
    barycentric = cartesian_to_barycentric(cartesian, simplex)
    cartesian2 = barycentric_to_cartesian(barycentric, simplex)
    assert_allclose(cartesian, cartesian2)