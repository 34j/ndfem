from itertools import combinations
from typing import Protocol

import attrs
import numpy as np
from array_api._2024_12 import Array
from array_api_compat import array_namespace


class MeshProtocol[TArray: Array](Protocol):
    """
    A mesh for finite element methods.

    Parameters
    ----------
    vertices : TArray
        The vertices of the mesh of shape (n_vertices, d).
    simplex : TArray
        The indices of the vertices that form simplex of shape (n_simpex, d + 1).

    """

    @property
    def vertices(self) -> TArray:
        """The vertices of the mesh of shape (n_vertices, d)."""

    @property
    def simplex(self) -> TArray:
        """The indices of the vertices that form simplex of shape (n_simpex, d + 1)."""


@attrs.frozen(kw_only=True)
class Mesh[TArray: Array](MeshProtocol[TArray]):
    """Simplical mesh."""

    vertices: TArray
    """The vertices of the mesh of shape (n_vertices, d)."""
    simplex: TArray
    """The indices of the vertices that form simplex of shape (n_simplex, d + 1)."""

    def __attrs_post_init__(self) -> None:
        if self.vertices.ndim != 2:
            raise ValueError("Vertices must be a 2D array.")
        if self.simplex.ndim != 2:
            raise ValueError("Simplex must be a 2D array.")
        if self.simplex.shape[1] != self.vertices.shape[1] + 1:
            raise ValueError(
                f"simplex.shape[1]={self.simplex.shape[1]} "
                f"!= vertices.shape[1]={self.vertices.shape[1]} + 1"
            )


def traiangulate_cube[TArray: Array](n: int, /, *, stride: TArray | None = None) -> TArray:
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

        The first axis corresponds to the simplex number,
        the second axis corresponds to the vertex number.

        The vertice numbers are ordered lexicographically,
        e.g. 0 -> (0, ..., 0), 1 -> (0, ..., 1),
        2 -> (0, ..., 1, 0), 3 -> (0, ..., 1, 1), ...
        2^n - 1 -> (1, ..., 1).

        Since the simplexes are surrounded by n + 1 planes, which are
        0 = x_sigma(1), x_sigma(i) = x_sigma(i + 1) for i = 1, ..., n - 1, x_sigma(n) = 1,
        the vertices are the points where n of these planes intersect.

    """
    from itertools import permutations

    if stride is None:
        stride = np.ones(n, dtype=int)
    xp = array_namespace(n, stride)
    diff = xp.concat((xp.ones((1,), dtype=int), xp.cumulative_prod(stride[:-1] + 1)))
    # (n!, n)
    diff_perm = xp.asarray(list(permutations(diff)))
    return xp.cumulative_sum(diff_perm, axis=1, include_initial=True)


def cuboid[TArray: Array](lengths: TArray, units: TArray, /) -> MeshProtocol[TArray]:
    """
    Create a cuboid mesh.

    Returns
    -------
    lengths : TArray
        The lengths of the cuboid in each dimension.
    units : TArray
        The mesh lengths in each dimension.

    Returns
    -------
    MeshProtocol[TArray]
        The mesh of the cuboid with vertices and simplexes.

    """
    xp = array_namespace(lengths, units)
    lengths, units = xp.broadcast_arrays(lengths, units)
    if lengths.ndim != 1:
        raise ValueError()
    n = lengths.shape[0]
    nums = -lengths // -units
    vertices = xp.reshape(
        xp.stack(
            xp.meshgrid(
                *[
                    xp.linspace(0, length, num=num + 1)
                    for length, num in zip(lengths, nums, strict=False)
                ],
                indexing="ij",
            )
        ),
        (n, -1),
    ).T
    vertice_indices_touching_right_edge = xp.reshape(
        xp.stack(
            xp.meshgrid(
                *[
                    xp.concat(
                        (
                            xp.zeros((num,), dtype=xp.bool),
                            xp.ones((1,), dtype=xp.bool),
                        )
                    )
                    for num in nums
                ],
                indexing="ij",
            )
        ),
        (n, -1),
    )
    vertice_indices_start = xp.nonzero(
        xp.all(
            ~vertice_indices_touching_right_edge,
            axis=0,
        )
    )[0]
    # need to fli nums as meshgrid orders in counter-lexicographic order
    simplexes = xp.reshape(
        traiangulate_cube(len(lengths), stride=xp.flip(nums))[None, ...]  # type: ignore
        + vertice_indices_start[:, None, None],
        shape=(-1, n + 1),
    )
    return Mesh(vertices=vertices, simplex=simplexes)


def mesh_subentities[TArray: Array](simplex: TArray, d1_subentities: int, /) -> TArray:
    """
    Convert a mesh to a triangle mesh.

    Parameters
    ----------
    simplex : TArray
        The indices of the vertices that form simplex of shape (n_simplex, d + 1).
    d1_subentities : int
        The number of subentities in the simplex.
        Must be 0 <= d1_subentities <= d.

    Returns
    -------
    TArray
        The triangle mesh of shape (n_triangles, d).

    """
    xp = array_namespace(simplex)
    d = simplex.shape[-1] - 1
    if not (0 <= d1_subentities <= d):
        raise ValueError(
            f"d1_subentities must be in [0, {d=}], got {d1_subentities}."
        )
    # (n_comb, d1_subentities + 1)
    comb = xp.asarray(
        list(combinations(range(d + 1), d1_subentities + 1)),
        dtype=xp.int16,
    )
    # (n_simplex, n_comb, d1_subentities + 1)
    subentities = simplex[:, comb]
    # (n_simplex * n_comb, d1_subentities + 1)
    subentities = xp.reshape(subentities, (-1, subentities.shape[-1]))
    subentities = xp.sort(subentities, axis=1)
    subentities = xp.unique(subentities, axis=0)
    return subentities
