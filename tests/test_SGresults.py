# -*- coding: utf-8 -*-
"""
The module contains a set of pytest tests for the function SemiGlobal.
Created on Thu Apr  7 16:30:27 2022

@author: Ido Schaefer
"""
import numpy as np
from scipy.fftpack import ifft
from scipy.linalg import eig
from SemiGlobal.FourierGrid import Hpsi, xp_grid
from SemiGlobal.SGfuns import SemiGlobal


# Required for the BEC tests:
def gsNLHdiag(H0, Vnl, x, tol):
    """
    The function finds the ground state of a non-linear Hamiltonian by an iterative
    process.
    Input:
    H0: A 2D ndarray; the linear part of the Hamiltonian, represented as a matrix.
    Vnl: Function object of the form: Vnl(u, x); the nonlinear purterbation, where
    u is the state (1D ndarray) and x is the x grid (1D ndarray)
    x: 1D ndarray; the space grid
    tol: The desired tolerance of the iterative process
    Output:
    gs: 1D ndarray; the resulting ground state
    niter: The required number of iterations"""
    eigval, P = eig(H0)
    iminE = np.argmin(eigval)
    gs = P[:, iminE]
    H = H0 + np.diag(Vnl(gs, x))
    Hgs = H @ gs
    niter = 1
    maxNiter = 100
    while np.abs(Hgs @ Hgs - (gs @ Hgs) ** 2) > tol and niter <= maxNiter:
        eigval, P = eig(H)
        iminE = np.argmin(eigval)
        gs = P[:, iminE]
        H = H0 + np.diag(Vnl(gs, x))
        niter += 1
        Hgs = H @ gs
    if niter > maxNiter:
        print("The program has failed to achieve the desired tolerance.")
    return gs, niter


# Creating problem details for the hatmonic oscillator tests:
L = 16 * np.sqrt(np.pi)
Nx = 128
dx = L / Nx
x, p = xp_grid(L, Nx)
xcolumn = x[:, np.newaxis]
# The kinetic energy matrix diagonal in the p domain:
K = p**2 / 2
# The potential energy matrix diagonal in the x domain:
V = x**2 / 2
fi0_harmonic = np.pi ** (-1 / 4) * np.exp(-(x**2) / 2) * np.sqrt(dx)


# The output time grid:
def Gop(u, t, v):
    return -1j * Hpsi(K, V + x * np.cos(t), v)


def Gdiff_op(u1, t1, u2, t2):
    return -1j * (xcolumn * (np.cos(t1) - np.cos(t2))) * u1


# For the BEC tests:
Vmat = np.diag(V)
# The kinetic energy matrix in the x domain:
Kmat = Nx * np.conj(ifft(np.conj(ifft(np.diag(K))).T)).T
# The Hamiltonian:
H = Kmat + Vmat
# The ground state, found by an iterative process:
gsBEC, _ = gsNLHdiag(H, lambda u, x: np.conj(u) * u, x, 2e-12)


def GopBEC(u, t, v):
    return -1j * Hpsi(K, V + x * np.cos(t) + np.conj(u) * u, v)


def Gdiff_opBEC(u1, t1, u2, t2):
    u2column = u2[:, None]
    return (
        -1j
        * (
            xcolumn * (np.cos(t1) - np.cos(t2))
            + np.conj(u1) * u1
            - np.conj(u2column) * u2column
        )
        * u1
    )


# For the inhomogeneous problem tests:
def ihfun(t):
    return np.exp(-(xcolumn**2) / 2) * np.cos(t)


def test_result_harmonic_cheb():
    U, _ = SemiGlobal(
        Gop,
        Gdiff_op,
        0,
        fi0_harmonic,
        np.r_[0, 10],
        200,
        9,
        9,
        1e-5,
        ev_domain=np.r_[-188 * 1j, 1j],
        save_memory=True,
    )
    mx = np.sum(np.conj(U[:, 1]) * U[:, 1] * x)
    assert np.abs(mx - 2.720105556) < 1e-8


def test_result_harmonic_arnoldi():
    U, _ = SemiGlobal(
        Gop, Gdiff_op, 0, fi0_harmonic, np.r_[0, 10], 200, 9, 9, 1e-5, save_memory=True
    )
    mx = np.sum(np.conj(U[:, 1]) * U[:, 1] * x)
    assert np.abs(mx - 2.720105556) < 1e-8


def test_resultBECcheb():
    U, _ = SemiGlobal(
        GopBEC,
        Gdiff_opBEC,
        0,
        gsBEC,
        np.r_[0, 10],
        200,
        9,
        9,
        1e-5,
        ev_domain=np.r_[-188 * 1j, 1j],
        save_memory=True,
    )
    mx = np.sum(np.conj(U[:, 1]) * U[:, 1] * x)
    assert np.abs(mx - 2.720105556) < 1e-8


def test_resultBECarnoldi():
    U, _ = SemiGlobal(
        GopBEC, Gdiff_opBEC, 0, gsBEC, np.r_[0, 10], 200, 9, 9, 1e-5, save_memory=True
    )
    mx = np.sum(np.conj(U[:, 1]) * U[:, 1] * x)
    assert np.abs(mx - 2.720105556) < 1e-8


def test_result_ih_cheb():
    U, _ = SemiGlobal(
        Gop,
        Gdiff_op,
        0,
        fi0_harmonic,
        np.r_[0, 10],
        200,
        9,
        9,
        1e-5,
        ihfun,
        ev_domain=np.r_[-188 * 1j, 1j],
        save_memory=True,
    )
    mx = np.sum(np.conj(U[:, 1]) * U[:, 1] * x)
    assert np.abs(mx - 39.98876716) < 1e-7


def test_result_ih_arnoldi():
    U, _ = SemiGlobal(
        Gop,
        Gdiff_op,
        0,
        fi0_harmonic,
        np.r_[0, 10],
        200,
        9,
        9,
        1e-5,
        ihfun,
        save_memory=True,
    )
    mx = np.sum(np.conj(U[:, 1]) * U[:, 1] * x)
    assert np.abs(mx - 39.98876716) < 1e-7
