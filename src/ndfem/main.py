from collections.abc import Mapping, Sequence
from typing import Callable, Protocol

from array_api._2024_12 import Array


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


# import numpy as np
# import quadpy

# dim = 4
# scheme = quadpy.tn.grundmann_moeller(dim, 3)
# val = scheme.integrate(
#     lambda x: np.exp(x[0]),
#     np.array(
#         [
#             [0.0, 0.0, 0.0, 0.0],
#             [1.0, 2.0, 0.0, 0.0],
#             [0.0, 1.0, 0.0, 0.0],
#             [0.0, 3.0, 1.0, 0.0],
#             [0.0, 0.0, 4.0, 1.0],
#         ]
#     ),
# )
# print(val)
