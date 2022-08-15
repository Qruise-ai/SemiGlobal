# -*- coding: utf-8 -*-
"""
Functions for Arnoldi approximation

Author: Ido Schaefer
"""

import numpy as np
from scipy.linalg import norm


def createKrop(Op, v0, Nv, data_type=np.complex128):
    """
    % The function creates an orthonormal Krylov basis of dimension Nv+1, from
    % a vector v0, and a function handle Op, which represents the operation of
    % an operator on a vector. It uses the Arnoldi algorithm.
    % The columns of V are the vectors of the Krylov space.
    % H is the extended Hessenberg matrix."""
    v_dim = v0.size
    V = np.empty((v_dim, Nv + 1), dtype=data_type, order="F")
    H = np.zeros((Nv + 1, Nv), dtype=data_type, order="F")
    V[:, 0] = v0 / norm(v0)
    for vj in range(0, Nv):
        V[:, vj + 1] = Op(V[:, vj])
        for vi in range(0, vj + 1):
            H[vi, vj] = np.conj(V[:, vi]) @ V[:, vj + 1]
            V[:, vj + 1] = V[:, vj + 1] - H[vi, vj] * V[:, vi]
        H[vj + 1, vj] = norm(V[:, vj + 1], check_finite=False)
        V[:, vj + 1] = V[:, vj + 1] / H[vj + 1, vj]
    return V, H


def getRvKr(Hessenberg, v, samplingp, Nkr, capacity):
    """
    For the Newton approximation of a function of matrix which multiplies a vector:
    f(A)v \approx \sum_{n=0}^Nkr a_n*R_n(A)v,
    the function computes the R_n(A)v vectors represented in the Krylov space of
    dimension Nkr+1, where the R_n(z)'s are the Newton basis polynomials, with
    samplingp as the sampling points. The Newton approximation is performed in a
    space of capacity 1.
    Input:
    Hessenberg: 2D ndarray of dimension (Nkr + 1, Nkr); represents the extended
    Hessenberg matrix of the problem.
    v: 1D ndarray; defined mathematically above.
    samplingp: The sampling points; should be the eigenvalues of the Hessenberg matrix.
    Nkr: The dimension of the Krylov space which is actually used for the Arnoldi approximation.
    capacity: The capacity of the approximation domain (see NewtonIpln.get_capacity).
    Output: 2D ndarray of dimension (Nkr + 1, Nkr + 1); the required vectors are
    placed in seperate columns."""

    Rv = np.zeros((Nkr + 1, Nkr + 1), order="F", dtype=Hessenberg.dtype.type)
    # Rv(:, 0) is v in the Krylov space.
    Rv[0, 0] = norm(v)
    for spi in range(0, Nkr):
        # Rv[:, spi] belongs to a Krylov space of dimension spi+1, and
        # the terms with row indices larger than spi vanish.
        Rv[0 : (spi + 2), spi + 1] = (
            Hessenberg[0 : (spi + 2), 0 : (spi + 1)] @ Rv[0 : (spi + 1), spi]
            - samplingp[spi] * Rv[0 : (spi + 2), spi]
        ) / capacity
    return Rv
