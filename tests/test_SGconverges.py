import math

import numpy as np
from scipy.linalg import expm

from SGfuns import SemiGlobal

_DIM = 9
_CHANGE_BASIS = np.array([
    [2.325132348030061191e-01, -6.240146823757646705e-02, 3.650705152027148293e-01, 8.215014282924514299e-01, 3.500650774075048655e-01, -6.762238638287411396e-02, -5.534151241376996516e-02, -3.256995871319006558e-02, -5.158239892700285101e-02],
    [2.539738198403615477e-01, -1.643805837735528019e-01, -1.206033031663028381e-02, 1.199589443996213617e-01, -5.580435028337020764e-01, -5.291607830178689520e-01, -3.741518017580076072e-01, 2.102281211604481448e-01, 3.439936097963689488e-01],
    [-4.250485189490289528e-01, -3.987799119235876932e-01, 5.391429252754265794e-02, -2.652887268614205485e-02, 3.433770688486796618e-02, -4.735128316894299805e-01, -3.223139297453277846e-02, -6.279506467719083718e-01, -1.895875760637258378e-01],
    [1.758652557513412928e-01, -1.999895011503644016e-01, -2.265508816633948416e-02, -3.134658758368248366e-01, 4.560977871895434133e-01, -2.125442091553885471e-01, -4.848192573334594302e-01, 3.668717977409576836e-01, -4.554740852468280865e-01],
    [9.776961218586907587e-02, -6.119284029407866532e-01, -5.355154032292093191e-01, 1.115492155491280002e-01, 2.180490114923778477e-01, 3.345449470771080236e-01, -1.091746119764404888e-01, -7.582141284620577681e-02, 3.736724266629336544e-01],
    [-3.369610155728297585e-01, -1.131651571241703708e-01, 6.589979638986936950e-02, 2.332401031303544814e-01, -4.448353571683523744e-01, 4.937334065197407362e-01, -4.727987883694231841e-01, 3.959852827075178994e-02, -3.849016535418067209e-01],
    [4.826136573774396643e-01, 2.457145084428303539e-01, 1.499330325412348841e-01, -2.235142417596189812e-01, 2.036054594904881943e-02, 2.073055364404151613e-01, -4.339185211496683903e-01, -6.254963996153335426e-01, 1.065304714080855669e-01],
    [-5.599475472394894737e-01, 2.979842022398933032e-01, 1.160752372407830996e-01, -5.537201210857644988e-03, 3.318642812693729516e-01, -2.897478550058198848e-02, -4.251948296106590197e-01, 1.014460956801842700e-01, 5.311352421225198350e-01],
    [-1.677026554032066269e-02, 4.833538395000680499e-01, -7.321783650151268974e-01, 3.081349842314809462e-01, 2.941600784915174294e-02, -2.156815105513300579e-01, -1.277711037848289488e-01, -1.302782320885791656e-01, -2.331819227570740560e-01],
])

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
