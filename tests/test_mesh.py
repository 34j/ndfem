import numpy as np
from numpy.testing import assert_array_equal

from ndfem.mesh import cuboid, traiangulate_cube


def test_traiangulate():
    assert_array_equal(traiangulate_cube(2), [[0, 1, 3], [0, 2, 3]])


def test_cuboid():
    actual = cuboid(np.asarray([2, 5]), 1)
    print(actual)
