from typing import Callable, Protocol

from array_api._2024_12 import Array
from array_api_compat import array_namespace
import numpy as np

def traiangulate_cube[TArray: Array](n: int, stride: TArray | None = None) -> TArray:
    """
    Triangulate hypercube [0, 1]^n into n! simplexes.

    Simplexes are {{x | 0 <= x_sigma(1) <= ... <= x_sigma(n) <= 1} | sigma in S_n}.

    Note that this is not a minimal traiangulation.

    Parameters
    ----------
    n : int
        The dimension of the hypercube.
    stride : TArray, optional
        The stride to use for the vertices. If None, defaults to np.ones(n).

    Returns
    -------
    Array
        The vertice numbers of the simplexes in the triangulation of shape (n!, n).

        The first axis corresponds to the simplex number, the second axis corresponds to the vertex number.

        The vertice numbers are ordered lexicographically,
        e.g. 0 -> (0, ..., 0), 1 -> (0, ..., 1), 2 -> (0, ..., 1, 0), 3 -> (0, ..., 1, 1), ...
        2^n - 1 -> (1, ..., 1).

        Since the simplexes are surrounded by n + 1 planes, which are
        0 = x_sigma(1), x_sigma(i) = x_sigma(i + 1) for i = 1, ..., n - 1, x_sigma(n) = 1,
        the vertices are the points where n of these planes intersect.

    """
    from itertools import permutations
    if stride is None:
        stride = np.ones(n, dtype=int)
    xp = array_namespace(n, stride)
    diff = xp.cumulative_sum(stride)
    print(diff)
    # (n!, n)
    diff_perm = xp.asarray(list(permutations(diff)))
    print(diff_perm)
    return xp.cumulative_sum(diff_perm, axis=1, include_initial=True)


def cuboid[TArray: Array](lengths: TArray, units: TArray) -> TArray:
    xp = array_namespace(lengths, units)
    lengths, units = xp.broadcast_arrays(lengths, units)
    if lengths.ndim != 1:
        raise ValueError()
    nums = -lengths // -units
    vertices = xp.meshgrid(
        *[xp.linspace(0, length, num=scale) for length, scale in zip(lengths, nums)],
        indexing="ij",
    )