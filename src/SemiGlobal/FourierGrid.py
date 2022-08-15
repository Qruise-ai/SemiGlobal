# -*- coding: utf-8 -*-
"""
Functions for Fourier grid calculations

Author: Ido Schaefer
"""
import numpy as np
from scipy.fftpack import fft, ifft


def Hpsi(K, V, psi):  # Hamiltonian operation in the Fourier grid
    """
    The function returns the operation of the Hamiltonian on the wave function psi.
    V: 1D ndarray; represents the potential energy vector in the x domain.
    K: 1D ndarray; represents the kinetic energy vector in the p domain.
    psi: 1D ndarray; represents the state vector."""

    return ifft(K * fft(psi)) + V * psi


def xp_grid(L, Nx):  # Creation of the x and p grid
    """
    The function creates the x grid and the p grid from the length of the x grid L
    and the number of grid points Nx. Nx is assumed to be even.
    Output: Tuple with the x and p grids as 1D ndarrays.
    """
    dx = L / Nx
    x = np.arange(-L / 2, L / 2, dx)
    p = np.r_[0 : 2 * np.pi / dx : 2 * np.pi / L]
    # Nx is assumed to be even.
    p[int(Nx / 2) : Nx] -= 2 * np.pi / dx
    return x, p


def xp2VK(x, p, Vfun, m=1):
    """ """
    V = Vfun(x)
    K = p**2 / (2 * m)
    return V, K
