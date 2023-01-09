import math

import numpy as np
from scipy.linalg import expm
from scipy.stats import ortho_group

from SGfuns import SemiGlobal

_DIM = 9
_CHANGE_BASIS = ortho_group.rvs(dim=_DIM, random_state=42)

_EIGENVALUES = [1.77e8, 1.26e8, 8.17e7, 7.62e7, 3.77e7]


def ideal_final_state(initial_state: np.ndarray, drift_final: float =0, t_final: float =1):

    e1 = _EIGENVALUES[0] * (1 - drift_final / 2) * t_final
    e2 = _EIGENVALUES[1] * (1 - 1 / 6 * drift_final ** 2) * t_final
    e3 = _EIGENVALUES[2] * t_final
    e4 = _EIGENVALUES[3] * (1 + 1 / 6 * drift_final ** 2) * t_final
    e5 = _EIGENVALUES[4] * (1 + drift_final / 2) * t_final

    H = np.diag([e1, e2, e2, e3, e2, e3, e4, e5, e2])
    A = _CHANGE_BASIS @ (-1j * H) @ _CHANGE_BASIS.T
    return expm(A) @ initial_state


def gen_A(t, drift_final: float=0, t_final: float = 1):
    """Generate matrix differential A: y` = A(t)y. Note: A = -iH."""
    drift_per_s = drift_final / t_final
    eigenvalue_drift = t * drift_per_s

    e1 = _EIGENVALUES[0] * (1 - eigenvalue_drift)
    e2 = _EIGENVALUES[1] * (1 - 0.5 * eigenvalue_drift ** 2)
    e3 = _EIGENVALUES[2] * (1 + 0 * eigenvalue_drift)
    e4 = _EIGENVALUES[3] * (1 + 0.5 * eigenvalue_drift ** 2)
    e5 = _EIGENVALUES[4] * (1 + eigenvalue_drift)

    eigvalue_array = np.array([e1, e2, e2, e3, e2, e3, e4, e5, e2]).T

    if isinstance(t, np.ndarray):
        H = np.stack([np.diag(eigvalue_array[i]) for i in range(t.shape[0])])
        A = _CHANGE_BASIS[np.newaxis,...] @ (-1j * H) @ _CHANGE_BASIS.T[np.newaxis,...]
        return A
    A = _CHANGE_BASIS @ (-1j * np.diag(eigvalue_array)) @ _CHANGE_BASIS.T
    return A


def overlap(u1, u2):
    return np.real(np.dot(np.conj(u1),u2))


def test_SG_monotonically_converges():
    drift_final = 0.1
    t_final = 3.5e-7

    def Gop(u, t, v):
        A = gen_A(t, drift_final, t_final)
        return A @ v

    def Gdiff_op(u1, t1, u2, t2):
        dA = gen_A(t1, drift_final, t_final) - gen_A(t2, drift_final, t_final)[np.newaxis, ...]
        du = np.einsum("...ij,j...->i...", dA, u1)
        return du

    ui = np.zeros(_DIM,dtype=complex)
    ui[0] = 1 + 0j
    final_state_ideal = ideal_final_state(ui, drift_final, t_final)

    infidelity_last = 1e-8
    for Nts in range(35, 70):
        U, _ = SemiGlobal(Gop, Gdiff_op, 0,
                          ui=ui, tgrid=np.r_[0, t_final], Nts=Nts, Nt_ts=9, Nfm=4, tol=1e-4, save_memory=True,
                          Niter=1,
                          Niter1st=16)

        infidelity = math.fabs(1 - np.sqrt(np.abs(overlap(U[:, 1], final_state_ideal))))
        if infidelity >= infidelity_last:
            raise AssertionError(f"{infidelity} >= {infidelity_last}; error not monotonic for {Nts} and {Nts-1}")
        infidelity_last = infidelity
