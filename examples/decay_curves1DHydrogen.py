# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 15:05:03 2022

@author: Ido Schaefer

This script reproduces the results of the paper:
"Semi-global approach for propagation of the time-dependent Schr\"odinger 
equation for time-dependent and nonlinear problems", 2017, Sec. 4.3, Fig. 8.
The results were originally obtained by a MATLAB implementation of the algorithm.
Minor differences from the original results (the relative deviation from the
exact solution) have been observed in the large time-step regime and the small
time-step regime. These regimes are more sensitive to roundoff errors: The large
time-step regime is characterized by a slight instability in the propagation
process, and the error in the small time-step regime is close to the machine
precision regime.
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy import io
from scipy.linalg import norm
from SGfuns import SemiGlobal
from FourierGrid import Hpsi


Mdata = io.loadmat('H1Dpython', squeeze_me=True)
xabs_col = Mdata['xabs240'][:, None]

def Gop(u, t, v):
    return -1j*Hpsi(Mdata['K240'], Mdata['Vabs240'] - Mdata['xabs240']*0.1/(np.cosh((t-500)/(170))**2)*np.cos(0.06*(t-500)), v)
    
    
def Gdiff_op(u1, t1, u2, t2):
    return 1j*xabs_col*0.1*(np.cos(0.06*(t1-500))/np.cosh((t1-500)/(170))**2
                           - np.cos(0.06*(t2-500))/np.cosh((t2-500)/(170))**2)*u1

def errorSGarticle(T, Nt_ts, Nkr, minNt, Nsamp, Niter_input=1):
# The function computes the data of the error decay with reduction of the
# time-step.
    er_decay = {'max_ers': {}}
    er_decay['allNt'] = np.zeros(Nsamp)
    er_decay['allmv'] = np.zeros(Nsamp)
    er_decay['aller'] = np.zeros(Nsamp)
    er_decay['max_ers']['texp'] = np.zeros(Nsamp)
    er_decay['max_ers']['fU'] = np.zeros(Nsamp)
    er_decay['max_ers']['conv'] = np.zeros(Nsamp)
    for degi in range(0, Nsamp):
        deg = np.log10(minNt) + degi*0.1
        Nt = int(np.round(10**deg))
        er_decay['allNt'][degi] = Nt
        u, history = SemiGlobal(Gop, Gdiff_op, 0, Mdata['fi0240'], np.r_[0, T], Nt, Nt_ts, Nkr, np.spacing(1),
                                Niter=Niter_input, Niter1st=20, display_mode=False, save_memory=True)
        print('\n')
        er_decay['allmv'][degi] = history['matvecs']
        er_decay['max_ers']['texp'][degi] = history['max_errors']['texp']
        er_decay['max_ers']['fU'][degi] = history['max_errors']['fU']
        er_decay['max_ers']['conv'][degi] = history['max_errors']['conv']
        er_decay['aller'][degi] = norm(u[:, 1] - Mdata['Uex'][:, -1])/norm(Mdata['Uex'][:, -1])
    plt.figure()
    plt.plot(np.log10(er_decay['allmv']), np.log10(er_decay['aller']), marker='o')
    plt.xlabel('log(matvecs)')
    plt.ylabel('log(error)')
    plt.show()
    return er_decay


T = 1e3
er_decay5 = errorSGarticle(T, 5, 5, 4e3, 17)
er_decay7 = errorSGarticle(T, 7, 7, 2.85e3, 14)
er_decay9 = errorSGarticle(T, 9, 9, 2.85e3, 11)