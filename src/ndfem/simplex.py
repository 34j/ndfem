from collections.abc import Mapping, Sequence
from typing import Any, Callable, Protocol

from array_api._2024_12 import Array, ArrayNamespace
from array_api_compat import array_namespace
import attrs

def barycentric_to_cartesian[TArray: Array](
    x: TArray,
    simplex: TArray,
    /,
) -> TArray:
    """
    Transform the points `x` from the barycentric coordinates
    to cartesian coordinates in the general simplex.

    Parameters
    ----------
    x : TArray
        The points in barycentric coordinates of shape (n_points, d + 1).
    simplex : TArray
        The vertices of the simplex of shape (d + 1, d).

    Returns
    -------
    TArray
        The points transformed to the general simplex.
    """
    xp = array_namespace(x, simplex)
    return x @ simplex


def cartesian_to_barycentric[TArray: Array](
    x: TArray,
    simplex: TArray,
    /,
) -> TArray:
    """
    Transform the points `x` in cartesian coordinates
    to the barycentric coordinates.

    Parameters
    ----------
    x : TArray
        The points in the general simplex of shape (n_points, d).
    simplex : TArray
        The vertices of the simplex of shape (d + 1, d).

    Returns
    -------
    TArray
        The points transformed to barycentric coordinates of shape (n_points, d + 1).
    """
    xp = array_namespace(x, simplex)
    simplex = xp.concat((simplex, xp.ones_like(simplex[:, 0])), axis=-1)
    x = xp.concat((x, xp.ones_like(x[:, 0])), axis=-1)
    return xp.linalg.solve(simplex.T, x.T).T


def reference_simplex[TArray: Array](n: int, /, ref: TArray) -> TArray:
    """
    Transform the points `x` from the general simplex to the barycentric coordinates
    of the reference simplex.

    Parameters
    ----------
    x : TArray
        The points in the general simplex of shape (n_points, d).

    Returns
    -------
    TArray
        The points transformed to barycentric coordinates of shape (n_points, d + 1).
    """
    xp = array_namespace(ref)
    simplex = xp.concat(
        (xp.zeros((n, 1), dtype=ref.dtype, device=ref.device), xp.eye(n, dtype=ref.dtype, device=ref.device)), axis=-1
    )
    return simplex