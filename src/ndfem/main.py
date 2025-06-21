from collections.abc import Mapping, Sequence
from typing import Callable, Protocol

import attrs
from array_api._2024_12 import Array
from array_api_compat import array_namespace
from .simplex import barycentric_to_cartesian

class DataProtocol[TArray: Array](Protocol):
    def v(self, derv: int) -> TArray: ...

    @property
    def x(self) -> TArray: ...


class BilinearDataProtocol[TArray: Array](DataProtocol[TArray], Protocol):
    def u(self, derv: Sequence[tuple[int, int]]) -> TArray:
        """
        Returns the basis functions evaluated at x for a polynomial of degree k.

        Parameters
        ----------
        derv : Sequence[tuple[int, int]]
            (i, j) where returns the j-th derivative
            of the i-th coordinate.

        Returns
        -------
        TArray
            The basis functions evaluated at x.

        """
        ...


class ElementProtocol[TArray: Array](Protocol):
    def __call__(self, x: TArray, derv: Sequence[tuple[int, int]], /) -> TArray:
        """
        Evaluate the element at x with derivatives specified by derv.

        The element must be a reference simplex,
        i.e. the vertices are
        (0, 0, ..., 0), (1, 0, ..., 0), ..., (0, 0, ..., 1).

        Parameters
        ----------
        x : TArray
            The points at which to evaluate the element.
        derv : Sequence[tuple[int, int]]
            (i, j) where returns the j-th derivative
            of the i-th coordinate.

        Returns
        -------
        TArray
            The evaluated element at x.

        """
        ...


@attrs.frozen(kw_only=True)
class Data[TElement: ElementProtocol, TArray: Array](DataProtocol[TArray]):
    element: TElement
    simplex: TArray
    x_barycentric: TArray

    @property
    def x(self) -> TArray:
        return barycentric_to_cartesian(self.x_barycentric, self.simplex)

    def v(self, derv: int) -> TArray:
        return self.element(self.x_barycentric, [(0, derv)])


@attrs.frozen(kw_only=True)
class BilinearData[TElement: ElementProtocol, TArray: Array](
    BilinearDataProtocol[TArray]
):
    element: TElement
    simplex: TArray
    x_barycentric: TArray
    
    @property
    def x(self) -> TArray:
        return barycentric_to_cartesian(self.x_barycentric, self.simplex)

    def v(self, derv: int) -> TArray:
        return self.element(self.x_barycentric, [(0, derv)])[None, :]

    def u(self, derv: Sequence[tuple[int, int]]) -> TArray:
        return self.element(self.x_barycentric, derv)[:, None]


# def pk_element[TArray: Array](x: TArray, k: int) -> TArray:
#     """Returns the basis functions evaluated at x for a polynomial of degree k.
#     """
#     n = x.shape[-1]


def fem[TArray: Array, TBC: str](
    vertices: TArray,
    simplex: TArray,
    bilinear_form: Callable[[BilinearDataProtocol[TArray]], TArray],
    linear_form: Callable[[DataProtocol[TArray]], TArray],
    element: ElementProtocol[TArray] | None = None,
    essential_bc: Mapping[TBC, TArray] | None = None,
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
