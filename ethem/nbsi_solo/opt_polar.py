9#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Creating the alorithm to find the optimal polarization point for
a given configuration Theta of the nbsi detector.

@author: misiak
"""

import sympy as sy

# adding ethem module path to the pythonpath
import sys
from os.path import dirname
sys.path.append( dirname(dirname(dirname(__file__))) )

import ethem as eth

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from scipy.integrate import odeint
from scipy.optimize import root
import numpy as np
import scipy.signal as sgl
import scipy.linalg as LA

from config_nbsi_solo import evad, nbsi, cryo, per, time, freq, E

from scipy.optimize import minimize


plt.close('all')


ib = 2.e-10
tc = 0.018

edict = evad.copy()
edict.pop(nbsi.current)
edict.pop(cryo.temperature)

phi_vect = eth.System.phi_vect
param = [nbsi.current, cryo.temperature] + list(phi_vect)

eteq = eth.System.eteq
eteq_num = eteq.subs(edict)
eteq_list = list(eteq_num)
eteq_fun = sy.lambdify(param, eteq_list, 'numpy')



def ss_solve(current, temp, t0=0.0):
    """ Solve the steady-state.

    Parameters
    ==========
    current : float
        Bias current.
    temp : float
        Temperature of the cryostat.
    t0 : float, optional
        Starting point for the nbsi temperature search (by default 0K,
        this is working great)

    Return
    ======
    sol.x : float
        Nbsi temperature solution of the steady-state.
    """
    eteq_aux0 = lambda y: eteq_fun(current, temp, *y)
    eteq_aux1 = lambda y,t: eteq_aux0(y)

    time_ss_array = np.linspace(0., 10., 10)
    inte = odeint(eteq_aux1, [t0], time_ss_array)

    t_conv = inte[-1]

    sol = root(eteq_aux0, t_conv)

    return sol.x


def response_solve(current, temp, L=10., fs=1e3):
    """ Solve the frequency response of the detector.

    Parameters
    ==========
    current : float
        Bias current.
    temp : float
        Temperature of the cryostat.
    L : float
        Time window in seconds.
    fs : float
        Sampling frequency in Hertz.

    Return
    ======

    RESPONSE
    """
    t_init = ss_solve(current, temp, t0=0.0)

    freq_fft = np.fft.fftfreq(int(L*fs), fs**-1)
    time_array = np.arange(0, L, fs**-1)
    freq_fftshift = np.fft.fftshift(freq_fft)

    edict_freqinv = edict.copy()
    edict_freqinv.update({
            nbsi.current:current,
            cryo.temperature:temp,
            nbsi.temperature:t_init,
    })

    # inversion of complex impedance matrix
    cimeq = eth.System.admittance_matrix
    cimeq_num = cimeq.subs(edict_freqinv)
    cimeq_funk = sy.lambdify(freq, cimeq_num, 'numpy')
    cimeq_fun = lambda f: np.linalg.inv(eth.lambda_fun_mat(cimeq_funk, f))

    # perturbation vector
    perf = eth.per_fft(per)
    perf_num = perf.subs(edict_freqinv) * fs
    perf_funk = sy.lambdify(freq, perf_num, 'numpy')
    perf_fun = lambda frange: eth.lambda_fun(perf_funk, frange)

    freq_array = np.flip(np.arange(fs/2, 0., -L**-1), axis=0)

    cimeq_array = cimeq_fun(freq_fft)
    perf_array = perf_fun(freq_fft)
    sv_array = np.einsum('ijk, ik -> ij', cimeq_array, perf_array)

    pulse_array = np.real(np.fft.ifft(sv_array, axis=0))
    pulse_array = pulse_array.T[0] #only one main quant, so this trick is needed

    sv_array = sv_array.T[0]
    sv_shift = np.fft.fftshift(sv_array)

    return time_array, pulse_array

#i_array = 10**np.linspace(-11, -9, 10)
#t_array = np.linspace(0.015, 0.020, 10)
#
##plt.figure()
#for i,t in zip(i_array, t_array):
#    plt.plot(*response_solve(i,t))

#plt.figure()

@np.vectorize
def reduce_solve(current, temp):

    time, pulse = response_solve(current, temp)

#    plt.plot(pulse)

    max_pulse = max(abs(pulse))

    energy = float(E.subs(edict)) / (1.6e-19 * 1e3) #J to keV

    sens = max_pulse / energy

### BACKLINE IS THE RATIO BETWEEN THE END OF PULSE AND ITS MAX ?
#    last = np.mean(pulse[-100:])
#    backline = last / max_pulse

### BACKLINE IS THE TIME WHERE THE PULSE IS 5% OF ITS MAX ?
# not implemented

### BACKLINE IS A CONDITON ON THE PULSE. IF PULSE NOT BACK TO BASELINE,
### THEN ITS SENS IS ZERO
    last = np.mean(pulse[-100:])
    if last > max_pulse*0.01:
        sens = 1e-10 # 0.1 nV/keV

    print sens

    return sens

#print reduce_solve(ib, tc)

i_array = 10**np.linspace(-11, -8, 10)
t_array = np.linspace(0.010, 0.020, 10)

i_mesh, t_mesh = np.meshgrid(i_array, t_array)

sens_mesh = reduce_solve(i_mesh, t_mesh)

fig = plt.figure('sens meshplot')
ax = plt.subplot(projection='3d')
ax.plot_wireframe(np.log10(i_mesh), t_mesh, np.log10(sens_mesh), color='red', alpha=0.3)

#
scat = lambda x: ax.scatter(np.log10(x[0]), x[1], np.log10(reduce_solve(*x)),
                            c='b', marker='o')
#
sol = minimize(lambda x: -reduce_solve(*x), (1e-11, 0.020), method='nelder-mead',
               callback=scat)
