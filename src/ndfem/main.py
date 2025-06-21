from typing import Callable, Protocol

from array_api._2024_12 import Array


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
