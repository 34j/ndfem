from array_api._2024_12 import Array
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
        The points in barycentric coordinates of shape (n_points, d + 1).
    simplex : TArray
        The vertices of the simplex of shape (d + 1, d).

    Returns
    -------
    TArray
        The points transformed to the general simplex.

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

    """
    xp = array_namespace(x, simplex)
    simplex = xp.concat((simplex, xp.ones_like(simplex[:, 0, None])), axis=-1)
    x = xp.concat((x, xp.ones_like(x[:, 0, None])), axis=-1)
    return xp.linalg.solve(simplex.T, x.T).T


def reference_simplex[TArray: Array](n: int, /, ref: TArray) -> TArray:
    """
    Transform the points `x` from the general simplex to the barycentric coordinates
    of the reference simplex.

    Parameters
    ----------
    n : int
        The dimension of the simplex.
    ref : TArray
        The reference array to call `array_namespace` on.

    Returns
    -------
    TArray
        The points transformed to barycentric coordinates of shape (n_points, d + 1).

    References
    ----------
    齊藤宣一. (2023年). 偏微分方程式の計算数理 (pp. xii, 544p). 共立出版.
    https://ci.nii.ac.jp/ncid/BD04524053 p.160

    """
    xp = array_namespace(ref)
    simplex = xp.concat(
        (
            xp.zeros((1, n), dtype=ref.dtype, device=ref.device),
            xp.eye(n, dtype=ref.dtype, device=ref.device),
        ),
        axis=0,
    )
    return simplex
