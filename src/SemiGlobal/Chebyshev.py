# -*- coding: utf-8 -*-
"""
Functions for a Chebishev polynomial approximation.
Author: Ido Schaefer
"""
from typing import Callable

import numpy as np
from scipy.fftpack import dct


def chebc(
    f: Callable[[float], float],
    leftb: float,
    rightb: float,
    N: int,
) -> np.ndarray:  # Chebyshev coefficient computation of a given function
    """
    The function returns the Chebychev coefficients of the function f in a given domain.

    Parameters
    ----------
    f: Callable[[float], float]
        A function object of the form: f(x).
    leftb: float
        The lower boundary of the approximation domain.
    rightb: float
        The upper boundary of the approximation domain.
    N: int
        The number of Chebychev coefficients.

    Returns
    -------
    np.ndaarray
        An ndarray containing the Chebyshev coefficients
    """
    # The Chebyshev points in the Chebyshev polynomial domain, [-1, 1]:
    xcheb = np.cos((np.arange(1, N + 1) * 2 - 1) * np.pi / (2 * N))
    # The Chebyshev points transformed to the approximation domain, [leftb, rightb]:
    x = 0.5 * (xcheb * (rightb - leftb) + rightb + leftb)
    c = dct(f(x)) / N
    c[0] = c[0] / 2
    return c


def chebcM(
    fM: np.ndarray,
) -> np.ndarray:  # Chebyshev coefficient computation from sampling points
    """
    The function computes the Chebychev coefficients of a set of functions
    from their samplings at the Chebychev points.

    Parameters
    ----------
    fM: np.ndarray
        Two dimensional ndarray; contains the sampled values of several functions in
        its columns. For faster computation, fM should be stored in the memory
        in a column major fasion (Fortran like).

    Returns
    -------
    np.ndarray
        Two dimensional ndarray; the Chebyshev coefficients of each function
        are the corresponding columns of the output ndarray.
    """

    # The number of Chebyshev sampling points:
    N = fM.shape[0]
    C = dct(fM, axis=0) / N
    C[0, :] = C[0, :] / 2
    return C


def chebc2result(
    Ccheb: np.ndarray, xdomain: np.ndarray, xresult: np.ndarray
) -> np.ndarray:  # Computation of the Chebyshev approximation from the coefficients
    """
    The function computes the Chebyshev polynomial approximation of a function
    from the corresponding Chebyshev coefficients, at a given set of points.

    Parameters
    ----------
    Ccheb: np.ndarray
        The Chebyshev coefficients of the function (see the function chebc); ndarray
    xdomain: np.ndarray
        The approximation domain; ndarray of the form: np.array([xmin, xmax])
    xresult: np.ndarray
        An ndarray; the set of points in which the function is to be evaluated

    Returns
    -------
    np.ndarray
        An ndarray of the shape of xresult with the approximated function
        values at xresult
    """

    # Transforming xresult to the Chebyshev domain [-1 1]:
    xrcheb = (2 * xresult - xdomain[0] - xdomain[1]) / (xdomain[1] - xdomain[0])
    m = Ccheb.size
    # Tk represents the Chebyshev polynomial of the k'th degree.
    # T0 represents the Chebyshev polynomial of the (k-2)'th degree.
    # T1 represents the Chebyshev polynomial of the (k-1)'th degree.
    # Computing the Chebyshev polynomial terms iteraively by the Chebyshev
    # polynomial recurrence relation:
    T0 = np.ones(xresult.shape)
    T1 = xrcheb.copy()
    result = Ccheb[0] * T0 + Ccheb[1] * T1
    for k in range(2, m):
        Tk = 2 * xrcheb * T1 - T0
        result = result + Ccheb[k] * Tk
        T0 = T1
        T1 = Tk
    return result


def vchebMop(operator, u0, leftb, rightb, Ncheb, data_type=np.complex128):
    """
    The function computes the vectors: :math:`v_k = T_k(operator)u0`, where :math:`u0` is a vector,
    :math:`T_k(x)` is the k'th degree Chebyshev polynomial, and operator is a linear operator.
    These vectors can be used for construction of a Chebyshev expansion of any
    function of operator which operates on the vector `u0`, as follows:
    :math:`f(operator)u0 \\approx \\sum_{k=0}^{Ncheb - 1} c_k*v_k`,
    where the :math:`c_k`'s are the Chebyshev coefficients (see chebc).
    The :math:`c_k`'s are f dependent, and can be computed by the program chebc.
    The program is useful when it is required to compute several functions
    of the same operator, which operate on the same vector.

    Parameters
    ----------
    operator: np.ndarray
        A function object of the form: operator(v); the function returns
        the operation of the operator on the one-dimensional ndarray v.
    u0: np.ndarray
        A one dimensional ndarray; mathematically defined above
    Ncheb: np.ndarray
        The number of expansion terms/Chebyshev coefficients/output vectors
    leftb:  float
        Scalar; defines the lower boundary of the eigenvalue domain of the operator
    rightb:  float
        Scalar; defines the upper boundary of the eigenvalue domain of the operator

    Returns
    -------
    np.ndarray
        Two dimensional ndarray which contains the v_k vectors defined above.
        Let M be a view of the output ndarray; its columns M[:, k] represent the corresponding v_k's.
    """
    # Defining the operator transformed to the domain of the Chebyshev
    # polynomial approximation, [-1, 1]
    def chebop(v):
        return (2 * operator(v) - (leftb + rightb) * v) / (rightb - leftb)

    dim = u0.size
    M = np.empty((dim, Ncheb), dtype=data_type, order="F")
    M[:, 0] = u0.copy()
    M[:, 1] = chebop(u0)
    for k in range(2, Ncheb):
        M[:, k] = 2 * chebop(M[:, k - 1]) - M[:, k - 2]
    return M
