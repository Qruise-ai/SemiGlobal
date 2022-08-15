# -*- coding: utf-8 -*-
"""
Functions for Fourier grid calculations

Author: Ido Schaefer
"""
from typing import Tuple

import numpy as np
from scipy.fftpack import fft, ifft


def Hpsi(
    K: np.ndarray,
    V: np.ndarray,
    psi: np.ndarray,
) -> np.ndarray:  # Hamiltonian operation in the Fourier grid
    """
    The function returns the operation of the Hamiltonian on the wave function psi.

    Parameters
    ----------
    V: np.ndarray
        1D ndarray; represents the potential energy vector in the x domain.
    K: np.ndarray
        1D ndarray; represents the kinetic energy vector in the p domain.
    psi: np.ndarray
        1D ndarray; represents the state vector.

    Returns
    -------
    np.ndarray
        Hamiltonian
    """

    return ifft(K * fft(psi)) + V * psi


def xp_grid(
    L: int, Nx: int
) -> Tuple[np.ndarray, np.ndarray]:  # Creation of the x and p grid
    """
    The function creates the x grid and the p grid

    Parameters
    ----------
    L: int
        the length of the x grid
    Nx: int
        and the number of grid points Nx. Nx is assumed to be even.

    Returns
    -------
    Tuple[np.ndarray,np.ndarray]
        Tuple with the x and p grids as 1D ndarrays.
    """
    dx = L / Nx
    x = np.arange(-L / 2, L / 2, dx)
    p = np.r_[0 : 2 * np.pi / dx : 2 * np.pi / L]
    # Nx is assumed to be even.
    p[int(Nx / 2) : Nx] -= 2 * np.pi / dx
    return x, p
