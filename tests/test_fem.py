import numpy as np
from numpy.testing import assert_allclose

from ndfem.fem import P1Element, evaluate_basis
from ndfem.simplex import cartesian_to_barycentric, reference_simplex


def test_p1_element() -> None:
    element = P1Element(d=2, bubble=True)
    basis = evaluate_basis(element, np.array([[0,0,1],[0.1, 0.4, 0.5]]), 0)
    assert_allclose(basis, [[0,0,1,0],[0.1, 0.4, 0.5, 2**2*0.5*0.1*0.4]])
