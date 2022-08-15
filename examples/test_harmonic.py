# -*- coding: utf-8 -*-
"""
Created on Sun Mar 13 14:31:59 2022

@author: Ido Schaefer
"""

import numpy as np
from SemiGlobal.SGfuns import SemiGlobal
from SemiGlobal.FourierGrid import Hpsi, xp_grid


# The program tests the efficiency of Hillel Tal-Ezer's
# semi-global propagator, for a forced harmonic oscillator.
# You may play with the following parameters:
T = 10
Nts = 200
Nt_ts = 9
Nfm = 9
tol = 1e-5
# Constructing the grid:
L = 16 * np.sqrt(np.pi)
Nx = 128
dx = L / Nx
x, p = xp_grid(L, Nx)
xcolumn = x[:, np.newaxis]
# The kinetic energy matrix diagonal in the p domain:
K = p**2 / 2
# The potential energy matrix diagonal in the x domain:
V = x**2 / 2
fi0 = np.pi ** (-1 / 4) * np.exp(-(x**2) / 2) * np.sqrt(dx)
# The output time grid:
dt = 0.1
t = np.r_[0 : (T + dt) : dt]


def Gop(u, t, v):
    return -1j * Hpsi(K, V + x * np.cos(t), v)


def Gdiff_op(u1, t1, u2, t2):
    return -1j * (xcolumn * (np.cos(t1) - np.cos(t2))) * u1


print("Chebyshev algorithm:")
U, history = SemiGlobal(
    Gop, Gdiff_op, 0, fi0, t, Nts, Nt_ts, Nfm, tol, ev_domain=np.r_[-188 * 1j, 1j]
)
print("The mean number of iterations per time-step:", history["mniter"])
# (should be close to 1, for ideal efficiency)
print("The number of matrix vector multiplications:", history["matvecs"])
# Computation of the maximal error - the deviation from the analytical
# result of the expectation value.
# Computing the expectation value of x in all the time points:
mx = np.sum(np.conj(U) * xcolumn * U, axis=0)
error = mx - (-0.5 * np.sin(t) * t)
maxer = np.max(np.abs(error))
print("The maximal error from the analytic solution expectation value of x:", maxer)
print("\nArnoldi algorithm:")
Uar, history_ar = SemiGlobal(Gop, Gdiff_op, 0, fi0, t, Nts, Nt_ts, Nfm, tol)
print("The mean number of iterations per time-step:", history_ar["mniter"])
# (should be close to 1, for ideal efficiency)
print("The number of matrix vector multiplications:", history_ar["matvecs"])
# Computation of the maximal error - the deviation from the analytical
# result of the expectation value.
# Computing the expectation value of x in all the time points:
mx_ar = np.sum(np.conj(Uar) * Uar * xcolumn, axis=0)
error_ar = mx_ar - (-0.5 * np.sin(t) * t)
maxer_ar = np.max(np.abs(error_ar))
print("The maximal error from the analytic solution expectation value of x:", maxer_ar)
