from SemiGlobal.SGfuns import SemiGlobal
import numpy as np


def test_SemiGlobal_SC():
    """Test SemiGlobal function with a simple driven 4-level anharmonic oscillator"""
    sim_res = 1e12
    t_final = 7e-9
    t_mid = t_final / 2
    t_sigma = t_final / 4
    lo_freq = 5e9
    # qubit_levels = 4
    # qubit_frequency = 5e9
    # qubit_anharm = -200e6

    h0 = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 3.14159265e10, 0.0, 0.0],
            [0.0, 0.0, 6.15752160e10, 0.0],
            [0.0, 0.0, 0.0, 9.04778684e10],
        ],
        dtype=np.complex128,
    )

    hks = np.array(
        [
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 1.41421356, 0.0],
            [0.0, 1.41421356, 0.0, 1.73205081],
            [0.0, 0.0, 1.73205081, 0.0],
        ],
        dtype=np.complex128,
    )

    amp = 0.45 * 1e9  # 0.45 V * 1e9 Hz/V

    def signal(t):
        return (
            amp
            * np.exp(-0.5*(((t - t_mid) / t_sigma) ** 2))
            * np.cos(t * 2 * np.pi * lo_freq)
        )

    def Gop(u, t, v):
        s = signal(t)
        hk = s * hks
        Gs = -1j * (hk + h0)
        g = (Gs @ v.T).T
        return g

    def Gdiff_op(u1, t1, u2, t2):
        """Non-efficient but simple"""
        gop2 = Gop(u2, t2, u2)
        gop1 = np.array([Gop(u, t, u) for u, t in zip(u1.T, t1)])
        diff = gop1 - gop2
        return diff.T

    # def Gdiff_op(u1, t1, u2, t2):
    #     s = signal(t1) - signal(t2)
    #     s_e = s[:, np.newaxis, np.newaxis]
    #     h = s_e * hks
    #     u1_T = np.expand_dims(u1.T, axis=2)
    #     m = np.squeeze(h @ u1_T, axis=2)
    #     u = (-1j * m).T

    #     u1 = Gdiff_op_1(u1, t1, u2, t2)
    #     dg = np.abs(u - u1)
    #     assert np.all(dg < 1e-10)
    #     return u

    ui = np.zeros((4,))
    ui[0] = 1.0

    def norm(u):
        return np.abs(np.sum(np.conj(u) * u))

    assert np.abs(norm(ui) - 1) < 1e-5

    tgrid = np.r_[0, t_final]

    U, _ = SemiGlobal(
        Gop, Gdiff_op, 0, ui, tgrid, t_final * sim_res, 9, 9, 1e-5, save_memory=True
    )
    u_final = U[:, -1]
    norm = np.sum(np.conj(u_final) * u_final)
    assert np.abs(norm - 1) < 1e-5


def test_SemiGlobal_SC_3Levels():
    """Test SemiGlobal function with a simple driven 3-level anharmonic oscillator"""
    sim_res = 1e12
    t_final = 7e-9
    t_mid = t_final / 2
    t_sigma = t_final / 4
    lo_freq = 5e9
    # qubit_levels = 4
    # qubit_frequency = 5e9
    # qubit_anharm = -200e6

    h0 = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 3.14159265e10, 0.0],
            [0.0, 0.0, 6.15752160e10],
        ],
        dtype=np.complex128,
    )

    hks = np.array(
        [
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 1.41421356],
            [0.0, 1.41421356, 0.0],
        ],
        dtype=np.complex128,
    )

    amp = 0.45 * 1e9  # 0.45 V * 1e9 Hz/V

    def signal(t):
        return (
            amp
            * np.exp(-0.5*(((t - t_mid) / t_sigma) ** 2))
            * np.cos(t * 2 * np.pi * lo_freq)
        )

    def Gop(u, t, v):
        s = signal(t)
        hk = s * hks
        Gs = -1j * (hk + h0)
        g = (Gs @ v.T).T
        return g

    def Gdiff_op(u1, t1, u2, t2):
        """Non-efficient but simple"""
        gop2 = Gop(u2, t2, u2)
        gop1 = np.array([Gop(u, t, u) for u, t in zip(u1.T, t1)])
        diff = gop1 - gop2
        return diff.T

    # def Gdiff_op(u1, t1, u2, t2):
    #     s = signal(t1) - signal(t2)
    #     s_e = s[:, np.newaxis, np.newaxis]
    #     h = s_e * hks
    #     u1_T = np.expand_dims(u1.T, axis=2)
    #     m = np.squeeze(h @ u1_T, axis=2)
    #     u = (-1j * m).T

    #     u1 = Gdiff_op_1(u1, t1, u2, t2)
    #     dg = np.abs(u - u1)
    #     assert np.all(dg < 1e-10)
    #     return u

    ui = np.zeros((3,))
    ui[0] = 1.0

    def norm(u):
        return np.abs(np.sum(np.conj(u) * u))

    assert np.abs(norm(ui) - 1) < 1e-5

    tgrid = np.r_[0, t_final]

    U, _ = SemiGlobal(
        Gop, Gdiff_op, 0, ui, tgrid, t_final * sim_res, 9, 9, 1e-5, save_memory=True
    )
    u_final = U[:, -1]
    norm = np.sum(np.conj(u_final) * u_final)
    assert np.abs(norm - 1) < 1e-5
