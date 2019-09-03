#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Illustrate the different methods for solving the response of the system.
NI : Numerical Integration
FI : Frequency Inversion
TD : Temporal Diagonalization

TD not completely understood in order to obtain the correct time constants
    with an exponential decaying energy deposit. So for now, working
    with a Dirac energy deposit.
All in all, the results of the three methods are coherent.

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

from config_nbsi_solo import evad, nbsi, cryo, per, time, freq, energy

plt.close('all')


ib = 1.5e-10
tc = 0.016

edict = evad.copy()
edict.pop(nbsi.current)
edict.pop(cryo.temperature)

phi_vect = eth.System.phi_vect
param = [nbsi.current, cryo.temperature] + list(phi_vect)

eteq = eth.System.eteq
eteq_num = eteq.subs(edict)
eteq_list = list(eteq_num)
eteq_fun = sy.lambdify(param, eteq_list, 'numpy')

#t0 = np.random.uniform(0.010, 0.030)
t0 = 0.0

time_ss_array = np.linspace(0., 10., 10)


eteq_aux0 = lambda y: eteq_fun(ib, tc, *y)
eteq_aux1 = lambda y,t: eteq_aux0(y)

inte = odeint(eteq_aux1, [t0], time_ss_array)

t_conv = inte[-1]

sol = root(eteq_aux0, t_conv)

t_init = sol.x[0]

L = 10.
fs = 1e3
freq_fft = np.fft.fftfreq(int(L*fs), fs**-1)
time_array = np.arange(0, L, fs**-1)

#==============================================================================
# NUMERICAL INTEGRATION (NI)
#==============================================================================
capa_matrix = eth.System.capacity_matrix
per_arg = capa_matrix**-1 * per / sy.Heaviside(time)

niteq = eteq + per_arg
niteq_num = niteq.subs(edict).subs({nbsi.current: ib, cryo.temperature:tc})
niteq_list = list(niteq_num)
niteq_fun = sy.lambdify([time]+list(phi_vect), niteq_list, 'numpy')

niteq_aux = lambda y,t: niteq_fun(t, *y)

pulse_array = odeint(niteq_aux, [t_init], time_array) - t_init
pulse_array = pulse_array.T[0] #only one main quant, so this trick is needed

pulse_fft = np.fft.fft(pulse_array)
freq_fftshift = np.fft.fftshift(freq_fft)
pulse_fftshift = np.fft.fftshift(pulse_fft)

freq_psd, pulse_psd = eth.psd(pulse_fft, fs)

fig, ax = plt.subplots(ncols=3, num='NUMERICAL INTEGRATION', figsize=(17,3))

ax[0].plot(time_array, pulse_array, label='NI')
ax[0].grid(True)
ax[0].legend()
ax[0].set_title('Temporal')
ax[0].set_xlabel('Time [s]')
ax[0].set_ylabel('Temperature [K]')

ax[1].plot(freq_fftshift, np.real(pulse_fftshift), label='Real NI')
ax[1].plot(freq_fftshift, np.imag(pulse_fftshift), label='Imag NI')
ax[1].set_xscale('symlog')
ax[1].grid(True)
ax[1].legend()
ax[1].set_title('FFT')
ax[1].set_xlabel('Frequency [Hz]')
ax[1].set_ylabel('Temperature [K]')

ax[2].loglog(freq_psd, pulse_psd, label='NI')
ax[2].grid(True)
ax[2].legend()
ax[2].set_title('PSD')
ax[2].set_xlabel('Frequency [Hz]')
ax[2].set_ylabel('PSD [$K^2/Hz$]')

fig.tight_layout()


#==============================================================================
# FREQUENTIAL INVERSION (FI)
#==============================================================================

edict_freqinv = edict.copy()
edict_freqinv.update({
        nbsi.current:ib,
        cryo.temperature:tc,
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

freq_psd, sv_psd = eth.psd(sv_array, fs)

#fig, ax = plt.subplots(ncols=3, num='FREQUENTIAL INVERSION', figsize=(17,3))

ax[0].plot(time_array, pulse_array, label='FI')
ax[0].grid(True)
ax[0].legend()
ax[0].set_title('Temporal')
ax[0].set_xlabel('Time [s]')
ax[0].set_ylabel('Temperature [K]')

ax[1].plot(freq_fftshift, np.real(sv_shift), label='Real FI')
ax[1].plot(freq_fftshift, np.imag(sv_shift), label='Imag FI')
ax[1].set_xscale('symlog')
ax[1].grid(True)
ax[1].legend()
ax[1].set_title('FFT')
ax[1].set_xlabel('Frequency [Hz]')
ax[1].set_ylabel('Temperature [K]')

ax[2].loglog(freq_psd, sv_psd, label='FI')
ax[2].grid(True)
ax[2].legend()
ax[2].set_title('PSD')
ax[2].set_xlabel('Frequency [Hz]')
ax[2].set_ylabel('PSD [$K^2/Hz$]')

fig.tight_layout()


#==============================================================================
# TEMPORAL DIAGONALIZATION without thermalization time
#==============================================================================

coup_mat = eth.System.coupling_matrix
coup_mat_num = coup_mat.subs(edict)
coup_mat_fun = sy.lambdify(param, coup_mat_num, 'numpy')

coup_mat_eval = coup_mat_fun(ib, tc, t_init)

#eigen-values and vectors
eig, P = LA.eig(coup_mat_eval)
P_1 = LA.inv(P)

capa_matrix = eth.System.capacity_matrix
per_arg = capa_matrix**-1 * per / sy.Heaviside(time)
per_arg_num = per_arg.subs(edict)
per_arg_fun = sy.lambdify(time, per_arg_num, 'numpy')
phi_array = per_arg_fun(time_array)

#phi0 = np.array([(1-eps)*E/s['Ca'], 0, eps*E/s['Ce'], 0])  #perturbation

phi0 = sy.Matrix([[energy/nbsi.th_capacity]]).subs(edict)
phi0 = np.array(phi0).astype(np.float64)

A = P_1.dot(phi0)
tau = 1.0/eig

#sol.in eigenvector basis
#exp_vec = map(lambda x,y: y*np.exp(-time_array/x), tau, A)
exp_vec = [a*np.exp(-time_array/t) for a,t in zip(tau, A)]
pulse_array = P.dot(exp_vec)[0]

### CONVOLUTION WITH EVENT SHAPE
##    exp_phonon = np.exp(-t/taup) / taup
#
#dTa, dTp, dTe, dV = map(np.array, deltaT_exp)
#
##tweak
#dTa = np.insert(dTa, 0, [0, 0])
#dTp = np.insert(dTp, 0, [0, 0])
#dTe = np.insert(dTe, 0, [0, 0])
#dV = np.insert(dV, 0, [0, 0])
#t = np.insert(t, 0, [-0.4, -1e-3])
#
##    def buff_convolve(dT, exp):
##        numa = 2*len(dT)-1
##        A=np.zeros(numa)
##        A[numa/2:] = dT.real
##        B=np.zeros(numa)
##        B[numa/2:] = exp
##        return np.convolve(A, B, mode='same') / numa
##
##    (dTa, dTp, dTe, dV) = map(lambda L: buff_convolve(L, exp_phonon),
##                              (dTa, dTp, dTe, dV)
##                              )
##    t = np.linspace(-t_display, t_display, 2*num-1)

pulse_fft = np.fft.fft(pulse_array)
freq_fftshift = np.fft.fftshift(freq_fft)
pulse_fftshift = np.fft.fftshift(pulse_fft)

freq_psd, pulse_psd = eth.psd(pulse_fft, fs)

ax[0].plot(time_array, pulse_array, label='TD')
ax[0].grid(True)
ax[0].legend()
ax[0].set_title('Temporal')
ax[0].set_xlabel('Time [s]')
ax[0].set_ylabel('Temperature [K]')

ax[1].plot(freq_fftshift, np.real(pulse_fftshift), label='Real TD')
ax[1].plot(freq_fftshift, np.imag(pulse_fftshift), label='Imag TD')
ax[1].set_xscale('symlog')
ax[1].grid(True)
ax[1].legend(loc=1)
ax[1].set_title('FFT')
ax[1].set_xlabel('Frequency [Hz]')
ax[1].set_ylabel('Temperature [K]')

ax[2].loglog(freq_psd, pulse_psd, label='TD')
ax[2].grid(True)
ax[2].legend()
ax[2].set_title('PSD')
ax[2].set_xlabel('Frequency [Hz]')
ax[2].set_ylabel('PSD [$K^2/Hz$]')

