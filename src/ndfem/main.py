from typing import Callable, Protocol

from array_api._2024_12 import Array
from array_api_compat import array_namespace


class Data[TArray: Array](Protocol):
    def v(self, derv: int) -> TArray: ...

    @property
    def x(self) -> TArray: ...


class BilinearData[TArray: Array](Data[TArray], Protocol):
    def u(self, derv: tuple[int, int]) -> TArray:
        """
        Returns the basis functions evaluated at x for a polynomial of degree k.

        Parameters
        ----------
        derv : tuple[int, int]
            _description_

        Returns
        -------
        TArray
            _description_

        """
        ...


# def pk_element[TArray: Array](x: TArray, k: int) -> TArray:
#     """Returns the basis functions evaluated at x for a polynomial of degree k.
#     """
#     n = x.shape[-1]


def traiangulate_cube(n: int) -> Array:
    """
    Triangulate hypercube [0, 1]^n into n! simplexes.

    Simplexes are {{x | 0 <= x_sigma(1) <= ... <= x_sigma(n) <= 1} | sigma in S_n}.

    Note that this is not a minimal traiangulation.

    Parameters
    ----------
    n : int
        The dimension of the hypercube.

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

    xp = array_namespace(n)
    diff = 2 ** xp.arange(n)
    # (n!, n)
    diff_perm = xp.asarray(permutations(diff))
    return xp.cumulative_sum(diff_perm, axis=1)


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


def fem[TArray: Array](
    vertices: TArray,
    simplex: TArray,
    bilinear_form: Callable[[BilinearData[TArray]], TArray],
    linear_form: Callable[[Data[TArray]], TArray],
) -> TArray:
    """
    Finite element method for a triangle mesh.

    Parameters
    ----------
    vertices : TArray
        The vertices of the mesh of shape (n, d).
    simplex : TArray
        The indices of the vertices that form simplex of shape (m, d).
    bilinear_form : Callable[[BilinearData[TArray]], TArray]
        The bilinear form to be integrated over simplex.
    linear_form : Callable[[Data[TArray]], TArray]
        The linear form to be integrated over simplex.

    """
    # (n, d, d)
    simplex_vertices = vertices[simplex, :]


fem()


import numpy as np
import quadpy

dim = 4
scheme = quadpy.tn.grundmann_moeller(dim, 3)
val = scheme.integrate(
    lambda x: np.exp(x[0]),
    np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 2.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 3.0, 1.0, 0.0],
            [0.0, 0.0, 4.0, 1.0],
        ]
    ),
)
print(val)
