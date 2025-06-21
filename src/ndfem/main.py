from abc import ABCMeta
from collections.abc import Mapping, Sequence
from typing import Callable, Protocol

import attrs
from array_api._2024_12 import Array
from array_api_compat import array_namespace
from .simplex import barycentric_to_cartesian, reference_simplex

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
        
    # def essentical_bc(
    #     self, bc: Mapping[TBC, TArray], /
    # ) -> TArray:
    #     ...
        

def par
    



@attrs.frozen(kw_only=True)
class BilinearData[TElement: ElementProtocol, TArray: Array](
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
        xp = array_namespace(self.x_barycentric, self.simplex_vertices)
        funcs = self.element(self.x_barycentric, derv)[None, :]
        mat = self.simplex_vertices[:, :, 1:] - self.simplex_vertices[:, :, 0:1]
        return funcs @ xp.linalg.matrix_power(mat, derv)

    def v(self, derv: int) -> TArray:
        return self._funcs(derv)[None, :]

    def u(self, derv: int) -> TArray:
        return self._funcs(derv)[:, None]


def fem[TArray: Array, TBC: str](
    vertices: TArray,
    simplex: TArray,
    bilinear_form: Callable[[BilinearDataProtocol[TArray]], TArray],
    linear_form: Callable[[DataProtocol[TArray]], TArray],
    element: ElementProtocol[TArray] | None = None,
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
