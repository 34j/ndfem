from ndfem.mesh import traiangulate_cube, cuboid
from numpy.testing import assert_array_equal

def test_traiangulate():
    assert_array_equal(traiangulate_cube(2),[[0, 1, 3], [0, 2, 3]])
    
def test_cuboid():
    actual = cuboid([])