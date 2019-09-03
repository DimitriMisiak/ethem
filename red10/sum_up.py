9#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Sum up all the possibility of the ethem package applied to the nbsi_solo simulation.

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

from config import (evad, abso, elntd, cryo, per, time, freq,
                              energy, tau_therm)

from scipy.optimize import minimize

plt.close('all')

#==============================================================================
# PARAMETERS
#==============================================================================
L = 100.
fs = 1e3

#==============================================================================
# STEADY STATE SOLUTION
#==============================================================================
#sol_ss = eth.solve_sse(evad, [0.016, 0.016])
sol_ss = eth.solve_sse(evad)

#==============================================================================
# NUMERICAL INTEGRATION (NI)
#==============================================================================
# temporal
ni_time_array, ni_pulse_array = eth.num_int(per, evad, sol_ss, L=L, fs=fs)

# fft freq
ni_freq_fft = eth.temp_to_fft(ni_time_array)
ni_pulse_fft = np.fft.fft(ni_pulse_array)

ni_freq_fftshift = np.fft.fftshift(ni_freq_fft)
ni_pulse_fftshift = np.fft.fftshift(ni_pulse_fft)

# psd freq
ni_freq_psd, ni_pulse_psd = eth.psd(ni_pulse_fft, fs)
##==============================================================================
## FREQUENCY INVERSION
##==============================================================================
#edict = evad.copy()
#edict.update({
#        elntd.temperature: sol_ss[eth.System.bath_list.index(elntd)],
#        abso.temperature: sol_ss[eth.System.bath_list.index(abso)],
#})
edict = eth.dict_sse(evad)

# time and freq array
fi_time_array = np.arange(0, L, fs**-1)
fi_freq_fft = np.fft.fftfreq(int(L*fs), fs**-1)
fi_freq_fftshift = np.fft.fftshift(fi_freq_fft)

# fft freq
fi_pulse_fft = eth.response_event(per, edict, fs)(fi_freq_fft)
fi_pulse_fftshift = np.fft.fftshift(fi_pulse_fft)

# temporal
fi_pulse_array = np.real(np.fft.ifft(fi_pulse_fft, axis=1))

# fft psd
fi_freq_psd, fi_pulse_psd = eth.psd(fi_pulse_fft, fs)

#==============================================================================
# TEMPORAL DIAGONALIZATION
#==============================================================================
coup_mat = eth.System.coupling_matrix
coup_mat_num = coup_mat.subs(edict)
coup_mat_eval = np.array(coup_mat_num).astype('float64')

#eigen-values and vectors
eig = LA.eigvals(coup_mat_eval)
tau_coup = 1.0/np.real(eig)

tau_therm_eval = float(tau_therm.subs(edict))

tau_array = np.sort(np.append(tau_therm_eval, tau_coup))
tau_msg = '$\\tau$ = ['
for t in tau_array:
    tau_msg += ' {:.3e},'.format(t)
tau_msg += '] s'

f0_array = tau_array**-1 / (2*np.pi)
f0_msg = '$f_0$ = ['
for f in f0_array:
    f0_msg += ' {:.2f},'.format(f)
f0_msg += '] Hz'

#==============================================================================
# NOISE RESPONSE
#==============================================================================
ref_bath = elntd
ref_ind = eth.System.bath_list.index(ref_bath)

inf = 1.
sup = 100.

ref_freq_fft = np.fft.fftfreq(int(L*fs), fs**-1)
ref_pulse_fft = eth.response_event(per, edict, fs)(fi_freq_fft)
ref_freq_psd, ref_pulse_psd = eth.psd(ref_pulse_fft, fs)

psd_fun_dict = eth.response_noise(edict)
psd_eval_dict = {k:v(ref_freq_psd) for k,v in psd_fun_dict.iteritems()}

obs_fun_dict = eth.measure_noise(ref_bath, edict)
obs_eval_dict = {k:v(ref_freq_psd) for k,v in obs_fun_dict.iteritems()}

#full_array = (
#        np.sum(obs_eval_dict.values(), axis=0)
#        + np.sum(psd_eval_dict.values(), axis=0)[ref_ind]
#)

full_array = eth.noise_tot_fun(ref_bath, edict)(ref_freq_psd)

#nep_array = full_array / fi_pulse_psd[ref_ind]
nep_freq_array, nep_array = eth.nep_ref(per, edict, fs, L, ref_bath)

invres_array = 4. / nep_array

inf_index = max(np.where(fi_freq_psd<inf)[0])
sup_index = min(np.where(fi_freq_psd>sup)[0])

invres_trapz = invres_array[inf_index:sup_index]
freq_trapz = fi_freq_psd[inf_index:sup_index]

#invres_int = np.trapz(invres_trapz, freq_trapz)
#res = (invres_int)**-0.5

#res = eth.nep_to_res(nep_freq_array, nep_array, (inf, sup))
res = eth.res_ref(per, edict, fs, L, ref_bath, (inf, sup))

res_msg = 'Resolution : {:.0f} eV'.format(
        res * energy.subs(edict) / (1.6e-19)
)
print(res_msg)

#%%
##==============================================================================
## PLOT
##==============================================================================
num = len(eth.System.bath_list)
fig, ax = plt.subplots(ncols=3, nrows=num, num='NUMERICAL INTEGRATION', figsize=(17,7))

for i in range(num):
    # NI
    ax[i,0].plot(ni_time_array, ni_pulse_array[i], label='NI')
    ax[i,0].plot(fi_time_array, fi_pulse_array[i], label='FI')
    ax[i,0].grid(True)
    ax[i,0].legend(title=tau_msg)
    ax[i,0].set_title('Temporal')
    ax[i,0].set_xlabel('Time [s]')
    ax[i,0].set_ylabel('Temperature [K]')

    ax[i,1].plot(ni_freq_fftshift, np.real(ni_pulse_fftshift)[i], label='Real NI')
    ax[i,1].plot(ni_freq_fftshift, np.imag(ni_pulse_fftshift)[i], label='Imag NI')
    ax[i,1].plot(fi_freq_fftshift, np.real(fi_pulse_fftshift)[i], label='Real FI')
    ax[i,1].plot(fi_freq_fftshift, np.imag(fi_pulse_fftshift)[i], label='Imag FI')
    ax[i,1].set_xscale('symlog')
    ax[i,1].grid(True)
    ax[i,1].legend()
    ax[i,1].set_title('FFT')
    ax[i,1].set_xlabel('Frequency [Hz]')
    ax[i,1].set_ylabel('Temperature [K]')

    ax[i,2].plot(ni_freq_psd, ni_pulse_psd[i], label='NI')
    ax[i,2].plot(fi_freq_psd, fi_pulse_psd[i], label='FI')
    #ax[i,2].loglog(freq_welch, psd_welch, label='welch')
    for k,v in psd_eval_dict.iteritems():
        ax[i,2].plot(fi_freq_psd, v[i], label=k)

    if i == ref_ind:

        for k,v in obs_eval_dict.iteritems():
            ax[i,2].plot(fi_freq_psd, v, label=k)

        ax[i,2].plot(fi_freq_psd, full_array, label='full')

    for f in f0_array:
        ax[i,2].axvline(f, ls=':', color='k')
    ax[i,2].set_xscale('log')
    ax[i,2].set_yscale('log')
    ax[i,2].grid(True)
    ax[i,2].legend(title=f0_msg)
    ax[i,2].set_title('PSD')
    ax[i,2].set_xlabel('Frequency [Hz]')
    ax[i,2].set_ylabel('PSD [$K^2/Hz$]')

fig.tight_layout()
#
fig_ref, ax_ref = plt.subplots(nrows=3, num='REF BATH PLOT', figsize=(7,10))

ax_ref[0].set_title('Response PSD ; NEP ; 1/NEP^2')

ax_ref[0].plot(fi_freq_psd, fi_pulse_psd[ref_ind], label='1keV pulse')

for k,v in psd_eval_dict.iteritems():
    ax_ref[0].plot(fi_freq_psd, v[ref_ind], label=k)

for k,v in obs_eval_dict.iteritems():
    ax_ref[0].plot(fi_freq_psd, v, label=k)
ax_ref[0].plot(fi_freq_psd, full_array, label='Tot Noise')

ax_ref[1].plot(fi_freq_psd, nep_array, label='nep')
ax_ref[1].set_ylabel('NEP')

ax_ref[2].plot(fi_freq_psd, invres_array, label='invres')
ax_ref[2].set_ylabel('$1/NEP^2$')
ax_ref[2].fill_between(freq_trapz, invres_trapz, color='slateblue', alpha=0.4)

for i in range(3):
    ax_ref[i].set_xlabel('Frequency [Hz]')
    ax_ref[i].set_xscale('log')
    ax_ref[i].set_yscale('log')
    ax_ref[i].grid(True)
    ax_ref[i].legend()
    if i == 2:
        ax_ref[i].legend(title=res_msg)

fig_ref.tight_layout()

