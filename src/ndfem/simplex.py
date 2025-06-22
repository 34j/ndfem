from itertools import combinations
from typing import Any

import numpy as np
from array_api._2024_12 import Array, ArrayNamespace
from array_api_compat import array_namespace


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
        The points in barycentric coordinates of shape (..., n_points, d + 1).
    simplex : TArray
        The vertices of the simplex of shape (..., d + 1, d).

    Returns
    -------
    TArray
        The points transformed to the general simplex
        of shape (..., n_points, d)

    References
    ----------
    齊藤宣一. (2023年). 偏微分方程式の計算数理 (pp. xii, 544p). 共立出版.
    https://ci.nii.ac.jp/ncid/BD04524053 p.160

    """
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

    References
    ----------
    齊藤宣一. (2023年). 偏微分方程式の計算数理 (pp. xii, 544p). 共立出版.
    https://ci.nii.ac.jp/ncid/BD04524053 p.160

    """
    xp = array_namespace(x, simplex)
    simplex = xp.concat((simplex, xp.ones_like(simplex[:, 0, None])), axis=-1)
    x = xp.concat((x, xp.ones_like(x[:, 0, None])), axis=-1)
    return xp.linalg.solve(simplex.T, x.T).T


def reference_simplex[TArray: Array](
    d: int,
    /,
    *,
    xp: ArrayNamespace[TArray, Any, Any] = np,
    dtype: Any = None,
    device: Any = None,
) -> TArray:
    """
    Transform the points `x` from the general simplex to the barycentric coordinates
    of the reference simplex.

    Parameters
    ----------
    d : int
        The dimension of the simplex.
    xp : ArrayNamespace, optional
        The array namespace to use, by default numpy.
    dtype : Any, optional
        The data type of the output array, by default None.
    device : Any, optional
        The device to use for the output array, by default None.


    Returns
    -------
    TArray
        The points transformed to barycentric coordinates of shape (n_points, d + 1).


    """
    simplex = xp.concat(
        (
            xp.zeros((1, d), dtype=dtype, device=device),
            xp.eye(d, dtype=dtype, device=device),
        ),
        axis=0,
    )
    return simplex


def all_simplex_permutations[TArray: Array](
    n: int, d1_subentities: int, /, *, xp: ArrayNamespace[TArray, Any, Any] = np
) -> TArray:
    """
    All permutations for subentities.

    Returns
    -------
    TArray
        All combinations of subentities in the simplex of shape (nCn_subentities, n).

    Examples
    --------
    >>> from ndfem.simplex import all_rotated_simplex
    >>> all_rotated_simplex(3, 2)
    array([[0, 1, 2], [0, 2, 1], [1, 2, 0]])
            ----       ----       ----

    """
    result = []
    vertices = xp.arange(n)
    for comb in combinations(vertices, d1_subentities + 1):
        line = list(comb) + list(set(vertices) - set(comb))
        result.append(line)
    return xp.asarray(result)
