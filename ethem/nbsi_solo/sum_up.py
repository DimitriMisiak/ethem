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

from config_nbsi_solo import (evad, nbsi, cryo, per, time, freq,
                              energy, tau_therm)

from scipy.optimize import minimize

plt.close('all')

#==============================================================================
# PARAMETERS
#==============================================================================
L = 30.
fs = 1e3

#==============================================================================
# STEADY STATE SOLUTION
#==============================================================================
sol_ss = eth.solve_sse(evad, [0.016])

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
#==============================================================================
# FREQUENCY INVERSION
#==============================================================================
edict = evad.copy()
edict.update({
        nbsi.temperature:sol_ss[0],
})

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

# only one main quant, so this trick is needed
fi_pulse_fftshift = fi_pulse_fftshift[0]
fi_pulse_psd = fi_pulse_psd[0]
fi_pulse_array = fi_pulse_array[0]


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

psd_fun_dict = eth.response_noise(edict)

tfn_array = psd_fun_dict['TFN ep'](fi_freq_psd).T[0]
obs_array = eth.measure_noise(nbsi, edict)['flat'](fi_pulse_psd)
full_array = tfn_array + obs_array

nep_array = full_array / fi_pulse_psd
invres_array = 4. / nep_array

inf = 1.
sup = 100.

inf_index = max(np.where(fi_freq_psd<inf)[0])
sup_index = min(np.where(fi_freq_psd>100)[0])

invres_int = np.trapz(invres_array[inf_index:sup_index],
                      fi_freq_psd[inf_index:sup_index])
res = (invres_int)**-0.5

res_msg = 'Resolution : {:.3e} keV'.format(res)
print res_msg

plt.figure('nep')
plt.plot(fi_freq_psd, nep_array, label='nep')
plt.legend()
plt.grid(True)
plt.xscale('log')
plt.yscale('log')

plt.figure('invres')
plt.plot(fi_freq_psd, invres_array, label='invres')
plt.legend(title=res_msg)
plt.grid(True)
plt.xscale('log')
plt.yscale('log')
plt.axvspan(0., inf, color='r', alpha=0.3)
plt.axvspan(sup, 1e3, color='r', alpha=0.3)

#==============================================================================
# PLOT
#==============================================================================
fig, ax = plt.subplots(ncols=3, num='NUMERICAL INTEGRATION', figsize=(17,3))

# NI
ax[0].plot(ni_time_array, ni_pulse_array, label='NI')
ax[0].plot(fi_time_array, fi_pulse_array, label='FI')
ax[0].grid(True)
ax[0].legend(title=tau_msg)
ax[0].set_title('Temporal')
ax[0].set_xlabel('Time [s]')
ax[0].set_ylabel('Temperature [K]')

ax[1].plot(ni_freq_fftshift, np.real(ni_pulse_fftshift), label='Real NI')
ax[1].plot(ni_freq_fftshift, np.imag(ni_pulse_fftshift), label='Imag NI')
ax[1].plot(fi_freq_fftshift, np.real(fi_pulse_fftshift), label='Real FI')
ax[1].plot(fi_freq_fftshift, np.imag(fi_pulse_fftshift), label='Imag FI')
ax[1].set_xscale('symlog')
ax[1].grid(True)
ax[1].legend()
ax[1].set_title('FFT')
ax[1].set_xlabel('Frequency [Hz]')
ax[1].set_ylabel('Temperature [K]')

ax[2].plot(ni_freq_psd, ni_pulse_psd, label='NI')
ax[2].plot(fi_freq_psd, fi_pulse_psd, label='FI')
#ax[2].loglog(freq_welch, psd_welch, label='welch')
ax[2].plot(fi_freq_psd, tfn_array, label='tfn')
ax[2].plot(fi_freq_psd, obs_array, label='obs')
ax[2].plot(fi_freq_psd, full_array, label='full')
for f in f0_array:
    ax[2].axvline(f, ls=':', color='k')
ax[2].set_xscale('log')
ax[2].set_yscale('log')
ax[2].grid(True)
ax[2].legend(title=f0_msg)
ax[2].set_title('PSD')
ax[2].set_xlabel('Frequency [Hz]')
ax[2].set_ylabel('PSD [$K^2/Hz$]')

fig.tight_layout()
