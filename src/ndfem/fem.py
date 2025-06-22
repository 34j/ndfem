from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal, Protocol

import attrs
import numpy as np
import quadpy
from array_api._2024_12 import Array, ArrayNamespace
from array_api_compat import array_namespace

from .mesh import mesh_subentities
from .simplex import (
    all_simplex_permutations,
    barycentric_to_cartesian,
    reference_simplex,
)


class DataProtocol[TArray: Array](Protocol):
    """A protocol for data used when evaluating linear form."""

    def v(self, derv: int) -> TArray:
        """
        Returns the basis functions evaluated at x for a polynomial of degree k.

        Parameters
        ----------
        derv : int
            The drivative order of the basis functions.

        Returns
        -------
        TArray
            The basis functions evaluated at x.

        """
        ...

    @property
    def x(self) -> TArray:
        """The points at which the basis functions are evaluated."""
        ...


class BilinearDataProtocol[TArray: Array](DataProtocol[TArray], Protocol):
    """A protocol for data used when evaluating bilinear form."""

    def u(self, derv: int) -> TArray:
        """
        Returns the basis functions evaluated at x for a polynomial of degree k.

        Parameters
        ----------
        derv : int
            The drivative order of the basis functions.

        Returns
        -------
        TArray
            The basis functions evaluated at x.

        """
        ...


class ElementProtocol[TArray: Array, TBC: str](Protocol):
    """A protocol for finite element basis functions."""

    def __call__(self, x: TArray, d1_subentity: int, derv: int, /) -> TArray | None:
        """
        Evaluate the element at x with derivatives specified by derv.

        The element must be a reference simplex,
        i.e. the vertices are
        (0, 0, ..., 0), (1, 0, ..., 0), ..., (0, 0, ..., 1).

        Parameters
        ----------
        x : TArray
            The points at which to evaluate the element of shape (..., n).
        d1_subentity : int
            The subentity number.
            The basis functions belong to the d1-subentity of the simplex.
            The basis functions are shared among other simplexes which
            share the one of the n-subentities in the simplex.
        derv : int
            The derivative order of the basis functions.

        Returns
        -------
        TArray
            The basis functions evaluated at x
            of shape (..., *derv_shape, n_basis_d1_subentity),
            where derv_shape = (d,) * derv.

            The basis functions must not be repeated
            for each d1-subentity but must be for
            the first d1-subentity in terms of lexicographic order.

            The basis functions must be symmetric under
            permutation of the vertices of the d1-subentity.

        """
        ...

    def essentical_bc(self, bc: Mapping[TBC, TArray], /) -> TArray:
        """
        The essential boundary conditions for the element.

        Parameters
        ----------
        bc : Mapping[TBC, TArray]
            The essential boundary conditions to apply to the element.
            The keys are the boundary condition types, e.g. 'dirichelet',
            and the values are array of shape (n_surfaces)
            which specify the surface number (the vertice number
            which is not on the surface).

        Returns
        -------
        TArray
            The basis functions which should be omitted of shape (n_basis_to_omit,).

        """
        ...


@attrs.frozen(kw_only=True)
class P1Element[TArray: Array](ElementProtocol[TArray, Literal["dirichelet"]]):
    """n-dimensional P1 element with bubble function (optional)."""

    d: int
    bubble: bool = False

    def __call__(self, x: TArray, d1_subentity: int, derv: int, /) -> TArray | None:  # noqa: D102
        xp = array_namespace(x)
        if x.shape[-1] != self.d:
            raise ValueError(f"Expected last dimension of x to be {self.d=}, got {x.shape[-1]=}.")
        if derv == 0:
            if d1_subentity == 0:
                return (1 - xp.sum(x, axis=-1))[..., None]
            elif d1_subentity == self.d and self.bubble:
                return ((self.d**self.d) * xp.prod(x, axis=-1) * (1 - xp.sum(x, axis=-1)))[
                    ..., None
                ]
            else:
                return None
        elif derv == 1:
            if d1_subentity == 0:
                return -xp.ones((self.d, 1), dtype=x.dtype, device=x.device)[
                    (None,) * (x.ndim - 1) + (...,)
                ]
            elif d1_subentity == self.d and self.bubble:
                coef = self.d**self.d
                idx = xp.arange(self.d, dtype=xp.int16)
                d1 = xp.prod(x[..., idx[:, None] + idx[None, :-1]], axis=-1) * (
                    1 - xp.sum(x, axis=-1)
                )
                d2 = xp.prod(x, axis=-1)[..., None] * -1
                return coef * (d1 + d2)[..., None]
            else:
                return None
        else:
            raise ValueError(f"Unsupported derivative order {derv} for P1Element.")

    def essentical_bc(self, bc: Mapping[Literal["dirichelet"], TArray]) -> TArray | None:  # noqa: D102
        if bc.keys() != {"dirichelet"}:
            raise ValueError("Only 'dirichelet' boundary condition is supported.")
        if "dirichelet" not in bc:
            return None
        dv = bc["dirichelet"]
        xp = array_namespace(dv)
        if dv.size is None or dv.size > 1:
            return xp.arange(self.d + 1)
        elif dv.size == 0:
            return xp.empty((0,), dtype=xp.int64, device=dv.device)
        else:
            res = xp.arange(self.d + 1)
            res = res[res != dv[0]]
            return res.astype(xp.int64, device=dv.device)


def transform_derivatives[TArray: Array](
    funcs: TArray, simplex_vertices: TArray, derv: int
) -> TArray:
    """
    Convert the (derivatives of) basis functions to a general simplex.

    Parameters
    ----------
    funcs : TArray
        The basis functions evaluated of shape (..., *derv_shape, n_basis),
        where derv_shape = (d,) * derv.
    simplex_vertices : TArray
        The vertices of the simplex of shape (n_simplex, d + 1, d).
    derv : int
        The derivative order of the basis functions to evaluate.

    Returns
    -------
    TArray
        The basis functions transformed to the general simplex,
        of shape (..., n_simplex, *derv_shape, n_basis),
        where derv_shape = (d,) * derv.

    """
    xp = array_namespace(simplex_vertices)
    # (n_simplex, d, d)
    jaccobian = simplex_vertices[:, 1:, :] - simplex_vertices[:, 0:1, :]
    x_extra_ndim = funcs.ndim - 1 - derv
    # (..., n_simplex, *derv_shape, d)
    jaccobian = jaccobian[
        (None,) * x_extra_ndim + (slice(None),) + (None,) * (derv - 1) + (slice(None), slice(None))
    ]
    # (..., n_simplex, *derv_shape, n_basis)
    funcs = funcs[(..., None) + (slice(None),) * (derv + 1)]
    for _ in range(derv):
        funcs = jaccobian @ funcs
        funcs = xp.moveaxis(funcs, -2, -derv - 1)
    return funcs


def evaluate_basis[TArray: Array](
    element: ElementProtocol[TArray, Any],
    x_barycentric: TArray,
    derv: int,
) -> tuple[Sequence[Sequence[int]], TArray]:
    """
    Evaluate the basis functions at the given barycentric coordinates.

    Parameters
    ----------
    element : ElementProtocol[TArray, Any]
        The element to evaluate the basis functions for.
    x_barycentric : TArray
        The barycentric coordinates of the points to evaluate the basis functions at,
        of shape (..., d + 1).
    derv : int
        The derivative order of the basis functions to evaluate.

    Returns
    -------
    Sequence[Sequence[int]]
        Sequence of (d_subentity_vertices)
    TArray
        The basis functions evaluated at the barycentric coordinates,
        of shape (..., *derv_shape, n_basis),
        where derv_shape = (d,) * derv and
        n_basis = sum_d1_subentity n_basis_d1_subentity.

    """
    xp = array_namespace(x_barycentric)
    d1 = x_barycentric.shape[-1]
    simplex = reference_simplex(d1 - 1)
    indices = []
    results = []
    for d1_subentity in range(d1):
        # (d1Cd_subentity+1, d+1)
        permutations = all_simplex_permutations(d1, d1_subentity)
        # (..., d1Cd_subentity+1, d+1)
        x_barycentric_perm = x_barycentric[..., permutations]
        # (..., d1Cd_subentity+1, d)
        x_reference = barycentric_to_cartesian(x_barycentric_perm, simplex)
        # (... or 1, d1Cd_subentity+1 or 1, *derv, n_basis_d1_subentity)
        value = element(x_reference, d1_subentity, derv)
        if value is None or value.shape[-1] == 0:
            continue
        try:
            xp.broadcast_shapes(x_reference.shape[:2], value.shape[:2])
            if len(value.shape) != 3 + derv:
                raise ValueError()
            if not xp.all(xp.asarray(value.shape[-1 - derv : -1]) == (d1 - 1)):
                raise ValueError()
        except ValueError:
            raise ValueError(
                "Element must return basis functions "
                f"with shape (...={x_reference.shape[:2]}, *derv_shape={(d1 - 1,) * derv}, n_basis_n), "
                f"got {value.shape=}"
            )
        n_basis_d1_subentity = value.shape[-1]
        # (..., d1Cd_subentity+1, *derv, n_basis_d1_subentity)
        value = xp.broadcast_to(value, (*x_barycentric.shape[:2], *value.shape[2:]))
        # (..., *derv, d1Cd_subentity+1, n_basis_d1_subentity)
        value = xp.moveaxis(value, 1, -1)
        # (..., *derv, d1Cd_subentity+1 * n_basis_d1_subentity)
        value = xp.reshape(value, (*value.shape[:-2], -1))
        results.append(value)
        for perm in permutations:
            for _ in range(n_basis_d1_subentity):
                indices.append(perm[: d1_subentity + 1])
    return indices, xp.concat(results, axis=-1)


def get_basis_info[TArray: Array](
    element: ElementProtocol[TArray, Any], d: int, /, *, xp: ArrayNamespace[TArray, Any, Any] = np
) -> Sequence[Sequence[int]]:
    """
    Get the basis information for the element.

    Parameters
    ----------
    element : ElementProtocol[TArray, Any]
        The element to get the basis information for.
    d : int
        The dimension of the element.
    xp : ArrayNamespace[TArray, Any, Any]
        The array namespace to use for the element.

    Returns
    -------
    Sequence[Sequence[int]]
        Sequence of (d_subentity_vertices)

    """
    indices, _ = evaluate_basis(element, xp.zeros((1, d + 1)), 0)
    return indices


def evaluate_basis_and_transform_derivatives[TArray: Array](
    element: ElementProtocol[TArray, Any],
    x_barycentric: TArray,
    simplex_vertices: TArray,
    derv: int,
) -> TArray:
    """
    Evaluate the basis functions at the given barycentric coordinates
    and transform them to the general simplex.

    Parameters
    ----------
    element : ElementProtocol[TArray, Any]
        The element to evaluate the basis functions for.
    x_barycentric : TArray
        The barycentric coordinates of the points to evaluate the basis functions at,
        of shape (..., d + 1).
    simplex_vertices : TArray
        The vertices of the simplex of shape (n_simplex, d + 1, d).
    derv : int
        The derivative order of the basis functions to evaluate.

    Returns
    -------
    TArray
        The basis functions evaluated at the barycentric coordinates,
        transformed to the general simplex
        of shape (..., n_simplex, *derv_shape, n_basis),
        where derv_shape = (d,) * derv.

    """
    funcs = evaluate_basis(element, x_barycentric, derv)
    return transform_derivatives(funcs[1], simplex_vertices, derv)


@attrs.frozen(kw_only=True)
class BilinearData[TArray: Array, TElement: ElementProtocol](BilinearDataProtocol[TArray]):
    """Data used for bilinear form evaluation."""

    element: TElement
    """The finite element to use for the bilinear form."""
    simplex_vertices: TArray
    """The vertices of the mesh of shape (n_vertices, d, d + 1)."""
    x_barycentric: TArray
    """The integration points in barycentric coordinates of shape (n_points, d + 1)."""

    @property
    def x(self) -> TArray:  # noqa: D102
        return barycentric_to_cartesian(self.x_barycentric, self.simplex_vertices)

    def _funcs(self, derv: int) -> TArray:
        # (n_points, n_simplex, *derv_shape, n_basis)
        funcs = evaluate_basis_and_transform_derivatives(
            self.element, self.x_barycentric, self.simplex_vertices, derv
        )
        # (n_points, n_simplex, n_basis, *derv_shape)
        xp = array_namespace(funcs)
        funcs = xp.moveaxis(funcs, -1, -derv - 1)
        return funcs

    def v(self, derv: int) -> TArray:  # noqa: D102
        # (n_points, n_simplex, n_basis_u, n_basis_v, *derv_shape)
        return self._funcs(derv)[(..., None, slice(None)) + (slice(None),) * derv]

    def u(self, derv: int) -> TArray:  # noqa: D102
        # (n_points, n_simplex, n_basis_u, n_basis_v, *derv_shape)
        return self._funcs(derv)[(..., slice(None), None) + (slice(None),) * derv]


def fem[TArray: Array, TBC: str](
    vertices: TArray,
    simplex: TArray,
    bilinear_form: Callable[[BilinearDataProtocol[TArray]], TArray],
    linear_form: Callable[[DataProtocol[TArray]], TArray],
    element: ElementProtocol[TArray, TBC],
    # essential_bc: Mapping[TBC, TArray] | None = None,
) -> TArray:
    """
    Finite element method for a triangle mesh.

    Parameters
    ----------
    vertices : TArray
        The vertices of the mesh of shape (n_vertices, d).
    simplex : TArray
        The indices of the vertices that form simplex of shape (n_simpex, d + 1).
    bilinear_form : Callable[[BilinearData[TArray]], TArray]
        The bilinear form to be integrated over simplex.
    linear_form : Callable[[Data[TArray]], TArray]
        The linear form to be integrated over simplex.
    element : ElementProtocol[TArray, TBC]
        The element to use for the finite element method.

    """
    # (n_simplex, d, d + 1)
    d = simplex.shape[-1] - 1
    n_vertices = vertices.shape[0]
    n_simplex = simplex.shape[0]
    xp = array_namespace(vertices, simplex)
    simplex_vertices = vertices[simplex, :]
    scheme = quadpy.tn.grundmann_moeller(d, 2)
    n_points = scheme.points.shape[1]
    bilinear_data = BilinearData(
        element=element,
        simplex_vertices=simplex_vertices,
        x_barycentric=scheme.points.T,
    )
    # (n_points, n_simplex, n_basis_u, n_basis_v)
    bilinear_val = bilinear_form(bilinear_data)
    linear_val = linear_form(bilinear_data)
    if bilinear_val.shape[0] not in {1, n_points} or bilinear_val.shape[1] not in {1, n_simplex}:
        raise ValueError(
            f"Expected bilinear form to have shape (n_points={n_points}, n_simplex={n_simplex}, "
            f"n_basis_u, n_basis_v), got {bilinear_val.shape=}"
        )
    if linear_val.shape[0] not in {1, n_points} or linear_val.shape[1] not in {1, n_simplex}:
        raise ValueError(
            f"Expected linear form to have shape (n_points={n_points}, n_simplex={n_simplex}), "
            f"got {linear_val.shape=}"
        )
    # (n_simplex, n_basis_u, n_basis_v)
    bilinear = xp.vecdot(scheme.weights, bilinear_val, axis=0)
    linear = xp.vecdot(scheme.weights, linear_val, axis=0)
    subentities = {
        d1_subentities: mesh_subentities(simplex, d1_subentities)
        for d1_subentities in range(0, d + 1)
    }
    for subentity_vertices_i in get_basis_info(element, d):
        print(subentity_vertices_i)
        subentity_vertices = simplex[:, subentity_vertices_i]
        print(subentity_vertices)
