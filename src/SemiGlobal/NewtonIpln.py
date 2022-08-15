# -*- coding: utf-8 -*-
"""
Functions for a Newton interpolation

Author: Ido Schaefer
"""

import numpy as np


def divdif(z, fz):  # Divided difference computation
    """
    The function computes the divided difference coefficients for a Newton interpolation.
    The routine is based on a divided difference table, where each
    coefficient is given by the last term of a new diagonal in the table.
    The program applies also for an interpolated function with multiple output values.
    Input:
    z: A 1D ndarray; contains the sampling points
    fz: The function values at z.
        For a function which returns a single value: fz can be either a 1D ndarray or a
        2D ndarray with dimensions (1,N), where N is the number of interpolation points.
        For a function which returns multiple values: fz is a 2D ndarray, where
        function values of different sampling points are represented by different columns.
    Output: A tuple which contains two ndarrays. For example, for:
    polcoef, diagonal = divdif(z, fz)
    the output contains the following data:
    polcoef: The coefficients of the Newton basis polynomials for the Newton
    interpolation.
    diagonal: The last diagonal, for continuing the process to higher orders,
    if necessary.
    For a 1D fz, polcoef and diagonal are 1D ndarrays.
    For a 2D fz, polcoef and diagonal are 2D ndarrays. The different columns of
    polcoef represent the coefficients of different Newton basis polynomials.
    The different columns of diagonal represent different divided differences."""

    fz_is_1D = fz.ndim == 1
    if fz_is_1D:
        # The program is built for a 2D fz:
        fz = fz[np.newaxis, :]
    dim, Npoints = fz.shape
    output_type = (z[0] + fz[0, 0] + 0.0).dtype.type
    polcoef = np.empty((dim, Npoints), dtype=output_type, order="F")
    diagonal = np.empty((dim, Npoints), dtype=output_type, order="F")
    polcoef[:, 0] = fz[:, 0]
    diagonal[:, 0] = fz[:, 0]
    # coefi indexes the Newton interpolation coefficients.
    # dtermi indexes the terms of the new diagonal in the divided difference
    # table. dtermi coincides with the index of the sampling point with the lowest
    # index of the divided difference. For example, the index of f[z[2], z[3], z[4]]
    # is dtermi == 2.
    for coefi in range(1, Npoints):
        diagonal[:, coefi] = fz[:, coefi]
        for dtermi in range(coefi - 1, -1, -1):
            # diagonal[:, dtermi] belongs to the previous diagonal.
            # diagonal[:, dtermi + 1] belongs to the new diagonal.
            diagonal[:, dtermi] = (diagonal[:, dtermi + 1] - diagonal[:, dtermi]) / (
                z[coefi] - z[dtermi]
            )
        polcoef[:, coefi] = diagonal[:, 0]
    if fz_is_1D:
        # Preparing a 1D output:
        polcoef = np.squeeze(polcoef)
        diagonal = np.squeeze(diagonal)
    return polcoef, diagonal


def dvd2fun(sp, polcoef, resultp):
    """
    The program computes the Newton interpolation polynomial of a function from its divided
    differences and sampling points, evaluated at a set of points specified by resultp.
    It applies also for functions which return a vector.
    sp: 1D ndarray; the set of sampling points; the last sampling point is
    unrequired, but can be nevertheless included in sp.
    polcoef: The divided differences;
        For a function which returns a single value: polcoef can be either a 1D ndarray or a
        2D ndarray with dimensions (1,N), where N is the number of sampling points.
        For a function which returns multiple values: polcoef is a 2D ndarray, where
        the different vector divided differences are represented by different columns.
    resultp: 1D ndarray; the points at which the desired function is to be evaluated.
    Output: 2D ndarray for functions which return a vector with resultp.size>1;
    1D ndarray for functions which return a vector with resultp.size==1 or
    functions which return a scalar with resultp.size>1;
    0D ndarray for functions which return a scalar with resultp.size==1"""

    polcoef_is_1D = polcoef.ndim == 1
    if polcoef_is_1D:
        # The program is built for a 2D polcoef:
        polcoef = polcoef[np.newaxis, :]
    # The number of sampling points:
    N = polcoef.shape[1]
    # The degree of the interpolation polynomial is N-1.
    Nrp = resultp.size
    result = np.tile(polcoef[:, N - 1][:, np.newaxis], (1, Nrp))
    for spi in range(N - 2, -1, -1):
        result = polcoef[:, spi][:, np.newaxis] + result * (resultp - sp[spi])
    return np.squeeze(result)
    # This applies also for the case of a non 2D output, where either the
    # input polcoef is 1D, or Nrp == 1


def get_capacity(sp, testp):
    """
    Computation of the capacity of the interpolation domain.
    sp: 1D ndarray which contains the sampling points
    testp: The test point"""

    capacity = 1
    sp_comp = sp[sp != testp]
    Nsp = sp_comp.size
    for zi in range(0, Nsp):
        capacity = capacity * np.abs(testp - sp_comp[zi]) ** (1 / Nsp)
    return capacity
