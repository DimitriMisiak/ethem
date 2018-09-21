#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Test file for the et_scheme.py script

@author: misiak
"""

import matplotlib.pyplot as plt
import numpy as np
import sympy as sy

from config import eth
from config import evad, per, time, freq

t=time
f=freq


### closing previous plot
plt.close('all')

#==============================================================================
# STEADY STATE RESOLUTION
#==============================================================================
bath_list = eth.System.bath_list
num_bath = len(bath_list)

sol_ss = eth.solve_sse(evad, x0=[0.018, 0.018, 0.018, 0.])
# updating the evaluation dictionnary
ss_dict = {b : v for b,v in zip(eth.System.phi_vect, sol_ss)}

# new evaluation dictionnary taking updated with the steady state
evad_ss = evad.copy()
evad_ss.update(ss_dict)

#==============================================================================
# DISCRETIZATION VARIABLES
#==============================================================================
fs = 1e3
L = 1.
N = int(L * fs)

print 'fs :', fs
print 'L :', L
print 'N :', N

fny = fs/2
fres = fs/N
# array for frequency plot
freq = np.flip(np.arange(fs/2, 0., -fres), axis=0)
# array for numpy.fft calculation
freqnp = np.fft.fftfreq(N, fs**-1)
#time array for plot
time = np.arange(0., L, fs**-1)

#==============================================================================
# GENERAL SYSTEM RESPONSE
#==============================================================================

sol_num = eth.num_int(per, evad, sol_ss)[1:]

### first order
# sensitivity calculation
sens = eth.response_event(per, evad_ss, fs)
sv_arraynp = sens(freqnp)
# temporal pulse
pulse = np.real(np.fft.ifft(sv_arraynp, axis=0))
pulse = pulse.T
# psd sensitivity
#sv_mod_array = [eth.psd(sv_arraynp[:,k], fs)[1] for k in range(num_bath)]
# HOTFIX
sv_mod_array = [np.abs(sv_arraynp[1:N/2+1,k]/fs)**2 for k in range(num_bath)]

### noise in first order
tot_lab = 'Total'
psd_fun_dict = eth.response_noise(evad_ss)
psd_full = {k:fun(freq) for k, fun in psd_fun_dict.iteritems()}
psd_full[tot_lab] = np.sum(psd_full.values(),axis=0)

### SYSTEM PLOT
fig = plt.figure('system_plot', figsize=(15, 10))
ax = fig.get_axes()
try:
    ax = np.reshape(ax, (4,3))
except:
    fig, ax = plt.subplots(nrows=4, ncols=3,
                           sharex='col', num='system_plot')

for k in range(num_bath):
    sv_bath = sv_arraynp[:, k]
    psd_bath = sv_mod_array[k]

    freqy = np.fft.fftshift(freqnp)
    svy = np.fft.fftshift(sv_bath)

    ax[k][0].plot(freqy, np.real(svy), label='R(fft) perthe')
    ax[k][0].plot(freqy, np.imag(svy), label='I(fft) perthe')
    ax[k][1].plot(freq, psd_bath, label='psd perthe', color='k', lw=2)

    ax[k][2].plot(time, pulse[k], "-", label='PerThe')
    ax[k][2].plot(time, sol_num[k], ":", label='NumRes')

    try :
        for key in psd_full:
            if key == tot_lab:
                ax[k][1].plot(freq, psd_full[key][:,k],label=key,
                              ls='-', lw=2,  color='red')
            else:
                ax[k][1].plot(freq, psd_full[key][:,k],label=key)

    except Exception as e:
        print e

for row in ax:
    for k in range(len(row)):
        row[k].grid(b=True)
        row[k].legend()
        if k in (0,):
            row[k].set_xscale('symlog')
        if k in (1,):
            row[k].set_xscale('log')
        if k in (1,):
            row[k].set_yscale('log')
        if k in (2,):
            row[k].set_xlim(-0.4, 1.0)

ax[0][0].set_title('FFT')
ax[-1][0].set_xlabel('Frequency [Hz]')
ax[0][1].set_title('PSD')
ax[-1][1].set_xlabel('Frequency [Hz]')
ax[0][2].set_title('Temporal')
ax[-1][2].set_xlabel('Time [s]')

fig.tight_layout()
plt.subplots_adjust(hspace=0.0)

#%%
#==============================================================================
# RESPONSE IN REFERENCE BATH
#==============================================================================
### Measured bath / Reference Bath
ref_bath = bath_list[-1]
ref_ind = bath_list.index(ref_bath)

### numerical integration
numres_pulse = sol_num[ref_ind]
numres_pulse *= -1

fft_numres = np.fft.fft(numres_pulse)
#psd_numres = eth.psd(fft_numres, fs)[1]
psd_numres = np.abs(fft_numres[1:N/2+1])**2

### first order
pulse_ref = pulse[ref_ind]
pulse_ref *= -1

psd_ref = sv_mod_array[ref_ind]

print 'Approx. Sensitivity [nV/keV]: ', max(pulse_ref) * 1e9
print 'Approx. Sensitivity [nV/keV]: ', max(numres_pulse) * 1e9
print 'Sensitivity in first bin [V²/Hz]: ', psd_ref[0]

### noise in first order
psd_full_ref = {k:v[:, ref_ind] for k,v in psd_full.iteritems()}

psd_obs_fun = eth.measure_noise(ref_bath, evad_ss)
psd_obs = {k:fun(freq) for k,fun in psd_obs_fun.iteritems()}

psd_full_ref.update(psd_obs)
psd_list_without_tot = [v for k,v in psd_full_ref.iteritems() if k not in (tot_lab, 'Test')]
psd_full_ref[tot_lab] = np.sum(psd_list_without_tot, axis=0)

### REFERENCE BATH PLOT
fig = plt.figure('ref_plot', figsize=(8,8))
ax = fig.get_axes()
try:
    ax = np.reshape(ax, (3,))
except:
    fig, ax = plt.subplots(nrows=3, num='ref_plot')

ax[0].plot(freq, psd_ref, label='PerThe')
ax[0].plot(freq, psd_numres, ls=':', label='PerThe')

for key in psd_full_ref:
    if key == tot_lab:
        ax[0].plot(freq, psd_full_ref[key],label=key,
                      ls='-', lw=2,  color='red')
    else:
        ax[0].plot(freq, psd_full_ref[key],label=key)

for k in (1,2):
    ax[k].plot(time, pulse_ref, label='PerThe')
    ax[k].plot(time, numres_pulse, ls=':', label='NumRes')
    ax[k].set_xlim(-0.3, 1.0)

ax[0].set_xscale('log')

for k in (0,2):
    ax[k].set_yscale('log')

for a in ax:
    a.grid(True)
    a.legend()

fig.tight_layout()

#==============================================================================
# NEP AND RESOLUTION
#==============================================================================
# dictionnary of nep array
nep_dict = {k: v/psd_ref for k,v in psd_full_ref.iteritems()}

# dictionnary of 4/nep array
invres_dict = {k: 4./v for k,v in nep_dict.iteritems()}

inf = list(freq).index(1)
sup = list(freq).index(500)
invres_int = np.trapz(invres_dict[tot_lab][inf:sup], freq[inf:sup])
res = (invres_int)**-0.5

#==============================================================================
# MONITORING PLOT
#==============================================================================

plt.figure('Noise PSD')
for key, psd_array in psd_full_ref.iteritems():
    plt.plot(freq, psd_array, label=key)

# SV
plt.plot(freq, psd_ref, label='SV', lw=3, ls='--', color='k')

plt.legend()
plt.xscale('log')
plt.yscale('log')
plt.grid(True)

plt.figure('NEP^2')
for key in nep_dict:
    nep_array = nep_dict[key]
    plt.plot(freq, nep_array, label=key)
plt.legend()
plt.xscale('log')
plt.yscale('log')
plt.grid(True)

plt.figure('4/NEP^2')
for key in ['Total']:
    invres_array = invres_dict[key]
    plt.plot(freq[inf:sup], invres_array[inf:sup], label=key,
             color='slateblue', lw=2.0)
plt.legend(title='RES= {:.3f} keV'.format(res))
plt.xscale('log')
plt.yscale('log')
plt.grid(True)
