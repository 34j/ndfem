from pathlib import Path

import meshio
import numpy as np
import pytest
from numpy.testing import assert_array_equal

from ndfem.mesh import cuboid, simplex_planes, traiangulate_cube


@pytest.fixture(scope="module", autouse=True)
def setup():
    Path("tests/.cache").mkdir(parents=True, exist_ok=True)


def test_traiangulate():
    assert_array_equal(traiangulate_cube(2), [[0, 1, 3], [0, 2, 3]])
    print(traiangulate_cube(3))


def test_cuboid():
    actual = cuboid(np.asarray([1, 1, 1]), 1)
    meshio.Mesh(actual.vertices, {"triangle": simplex_planes(actual.simplex)}).write(
        "tests/.cache/cuboid.obj"
    )
