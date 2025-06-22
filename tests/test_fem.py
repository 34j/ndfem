import numpy as np
from array_api._2024_12 import Array
from array_api_compat import array_namespace
from numpy.testing import assert_allclose

from ndfem.fem import BilinearDataProtocol, DataProtocol, P1Element, evaluate_basis, fem
from ndfem.mesh import cuboid


def test_p1_element() -> None:
    element = P1Element(d=2, bubble=True)
    basis = evaluate_basis(element, np.array([[0, 0, 1], [0.1, 0.4, 0.5]]), 0)
    assert_allclose(basis, [[0, 0, 1, 0], [0.1, 0.4, 0.5, 2**2 * 0.5 * 0.1 * 0.4]])


def test_laplace() -> None:
    element = P1Element(d=2, bubble=False)
    mesh = cuboid(np.array([10, 10]), np.array(1))

    def bilinear_form[TArray: Array](d: BilinearDataProtocol[TArray]) -> TArray:
        gradu = d.u(1)
        gradv = d.v(1)
        xp = array_namespace(gradu, gradv)
        return xp.tensordot(gradu, gradv, axes=-2)

    def linear_form[TArray: Array](d: DataProtocol[TArray]) -> TArray:
        return d.v(0)

    fem(mesh.vertices, mesh.simplex, bilinear_form, linear_form, element)
