from collections.abc import Mapping
from typing import Any, Callable, Literal, Protocol

import attrs
from array_api._2024_12 import Array
from array_api_compat import array_namespace

from .simplex import (
    all_simplex_permutations,
    barycentric_to_cartesian,
    reference_simplex,
)


class DataProtocol[TArray: Array](Protocol):
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
    def x(self) -> TArray: ...


class BilinearDataProtocol[TArray: Array](DataProtocol[TArray], Protocol):
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
    def __call__(self, x: TArray, d_subentity: int, derv: int, /) -> TArray | None:
        """
        Evaluate the element at x with derivatives specified by derv.

        The element must be a reference simplex,
        i.e. the vertices are
        (0, 0, ..., 0), (1, 0, ..., 0), ..., (0, 0, ..., 1).

        Parameters
        ----------
        x : TArray
            The points at which to evaluate the element of shape (..., n).
        subentity : int
            The subentity number.
            The basis functions belong to the n-subentity of the simplex.
            The basis functions are shared among other simplexes which
            share the one of the n-subentities in the simplex.
        derv : int
            The derivative order of the basis functions.

        Returns
        -------
        TArray
            The basis functions evaluated at x
            of shape (..., *derv_shape, n_basis_n),
            where derv_shape = (n,) * derv.

            The basis functions must not be repeated
            for each n-subentity but must be for
            the first n-subentity in terms of lexicographic order.

            The basis functions must be symmetric under
            permutation of the vertices of the n-subentity.

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
    n: int
    bubble: bool = False

    def __call__(self, x: TArray, d_subentity: int, derv: int, /) -> TArray | None:
        xp = array_namespace(x)
        if x.shape[-1] != self.n:
            raise ValueError(
                f"Expected last dimension of x to be {self.n=}, got {x.shape[-1]=}."
            )
        if derv == 0:
            if d_subentity == 0:
                return (1 - xp.sum(x, axis=-1))[..., None]
            if d_subentity == self.n and self.bubble:
                return ((self.n ** self.n) * xp.prod(x, axis=-1) * (1 - xp.sum(x, axis=-1)))[..., None]
            else:
                return None
        elif derv == 1:
            if d_subentity == 0:
                return -xp.ones((self.n + 1, 1), dtype=x.dtype, device=x.device)[
                    (None,) * (x.ndim - 1), ...
                ]
            else:
                return None
        else:
            raise ValueError(f"Unsupported derivative order {derv} for P1Element.")

    def essentical_bc(
        self, bc: Mapping[Literal["dirichelet"], TArray]
    ) -> TArray | None:
        if bc.keys() != {"dirichelet"}:
            raise ValueError("Only 'dirichelet' boundary condition is supported.")
        if "dirichelet" not in bc:
            return None
        dv = bc["dirichelet"]
        xp = array_namespace(dv)
        if dv.size is None or dv.size > 1:
            return xp.arange(self.n + 1)
        elif dv.size == 0:
            return xp.empty((0,), dtype=xp.int64, device=dv.device)
        else:
            res = xp.arange(self.n + 1)
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
        The basis functions evaluated of shape (..., *derv_shape, n_basis_n).
    simplex_vertices : TArray
        The vertices of the simplex of shape (n_simplex, d, d + 1).
    derv : int
        The derivative order of the basis functions to evaluate.
        
    Returns
    -------
    TArray
        The basis functions transformed to the general simplex,
        of shape (..., *derv_shape, n_basis_n).

    """
    xp = array_namespace(simplex_vertices)
    jaccobian = simplex_vertices[:, :, 1:] - simplex_vertices[:, :, 0:1]
    return funcs @ xp.linalg.matrix_power(jaccobian, derv)


def evaluate_basis[TArray: Array](
    element: ElementProtocol[TArray, Any],
    x_barycentric: TArray,
    derv: int,
) -> TArray:
    """
    Evaluate the basis functions at the given barycentric coordinates.

    Parameters
    ----------
    element : ElementProtocol[TArray, Any]
        The element to evaluate the basis functions for.
    x_barycentric : TArray
        The barycentric coordinates of the points to evaluate the basis functions at,
        of shape (..., n_points, d + 1).
    derv : int
        The derivative order of the basis functions to evaluate.

    Returns
    -------
    TArray
        The basis functions evaluated at the barycentric coordinates,
        of shape (..., *derv_shape, n_basis_n),

    """
    xp = array_namespace(x_barycentric)
    d1 = x_barycentric.shape[-1]
    simplex = reference_simplex(d1 - 1)
    results = []
    for d1_subentity in range(d1):
        # (d1Cd_subentity+1, d+1)
        permutation = all_simplex_permutations(d1, d1_subentity)
        # (n_points, d1Cd_subentity+1, d)
        x_reference = barycentric_to_cartesian(x_barycentric[..., permutation], simplex)
        # (n_points, d1Cd_subentity+1, *derv, n_basis_d1_subentity)
        value = element(x_reference, d1_subentity, derv)
        if value is None:
            continue
        value = xp.moveaxis(value, 1, -1)
        value = xp.reshape(value, (*value.shape[:-2], -1))
        results.append(value)
    return xp.concat(results, axis=-1)

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
        of shape (..., n_points, d + 1).
    simplex_vertices : TArray
        The vertices of the simplex of shape (n_simplex, d, d + 1).
    derv : int
        The derivative order of the basis functions to evaluate.

    Returns
    -------
    TArray
        The basis functions evaluated at the barycentric coordinates,
        transformed to the general simplex, of shape (..., *derv_shape, n_basis_n).

    """
    funcs = evaluate_basis(element, x_barycentric, derv)
    return transform_derivatives(funcs, simplex_vertices, derv)

@attrs.frozen(kw_only=True)
class BilinearData[TArray: Array, TElement: ElementProtocol](
    BilinearDataProtocol[TArray]
):
    element: TElement
    simplex_vertices: TArray
    """The vertices of the mesh of shape (n_vertices, d, d + 1)."""
    x_barycentric: TArray
    """The integration points in barycentric coordinates of shape (n_points, d + 1)."""

    @property
    def x(self) -> TArray:
        return barycentric_to_cartesian(self.x_barycentric, self.simplex_vertices)

    def _funcs(self, derv: int) -> TArray:
        return evaluate_basis_and_transform_derivatives(
            self.element, self.x_barycentric, self.simplex_vertices, derv
        )

    def v(self, derv: int) -> TArray:
        return self._funcs(derv)[None, :]

    def u(self, derv: int) -> TArray:
        return self._funcs(derv)[:, None]


def fem[TArray: Array, TBC: str](
    vertices: TArray,
    simplex: TArray,
    bilinear_form: Callable[[BilinearDataProtocol[TArray]], TArray],
    linear_form: Callable[[DataProtocol[TArray]], TArray],
    element: ElementProtocol[TArray, TBC] | None = None,
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

    """
    # (n_simplex, d, d + 1)
    simplex_vertices = vertices[simplex, :]


# fem()


import quadpy

scheme = quadpy.tn.grundmann_moeller(3, 2)
# scheme = quadpy.tn.stroud_1969(3)
print(scheme.weights.shape, scheme.points.shape)
