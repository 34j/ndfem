import numpy as np
from numpy.testing import assert_allclose

from ndfem.fem import P1Element



def test_p1_element() -> None:
    element = P1Element(n=2)
    assert_allclose(element(np.array([0.5, 0.5]), 0), [0.5, 0.5, 0])
