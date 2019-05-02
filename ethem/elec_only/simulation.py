#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Sum up all the possibility of the ethem package first applied to the
nbsi_solo and nbsi_duo detectors.

@author: misiak
"""

# adding ethem module path to the pythonpath
import sys
import tqdm
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as LA
from os.path import dirname, abspath

import ethem as eth

from config import evad

plt.close('all')

#==============================================================================
# PARAMETERS
#==============================================================================
L = 10.
fs = 1e4

#==============================================================================
# STEADY STATE SOLUTION
#==============================================================================
# checking the quantities at steady state
edict = eth.dict_sse(evad)

sol_ss = (eth.solve_sse(evad)).x
# sol_ss is OK
R = eth.System.Resistor_ntd.resistivity.subs(edict)
# R is OK

num = len(eth.System.bath_list)
#==============================================================================
# NOISE RESPONSE
#==============================================================================
ref_bath = eth.System.Capacitor_f
ref_ind = eth.System.bath_list.index(ref_bath)

inf = 1.
sup = 100.

freq_array = np.flip(np.arange(fs/2., 0., -L**-1), axis=0)


psd_fun_dict = eth.response_noise(edict)
psd_eval_dict = {k:v(freq_array) for k,v in psd_fun_dict.iteritems()}

obs_fun_dict = eth.measure_noise(ref_bath, edict)
obs_eval_dict = {k:v(freq_array) for k,v in obs_fun_dict.iteritems()}

full_array = eth.noise_tot_fun(ref_bath, edict)(freq_array)

ref_10_ind = 0
#signal_level = ref_pulse_psd[-1, ref_10_ind]
noise_level = full_array[ref_10_ind]

#print 'Signal Level =', signal_level
print 'Noise_Level =', noise_level
#==============================================================================
# NOISE PSD PLOT
#==============================================================================
fig, ax = plt.subplots(nrows=num, num='NOISE PSD PLOT', figsize=(9,9), squeeze=False)
ax = np.ravel(ax)
for i in range(num):

    for k,v in psd_eval_dict.iteritems():
        ax[i].plot(freq_array, v[i], label=k)

    if i == ref_ind:

        for k,v in obs_eval_dict.iteritems():
            ax[i].plot(freq_array, v, label=k)

        ax[i].plot(freq_array, full_array, label='full', color='k', lw=2.)

    ax[i].set_xscale('log')
    ax[i].set_yscale('log')
    ax[i].grid(True)
    ax[i].set_xlabel('Frequency [Hz]')
    ax[i].set_ylabel('PSD [$K^2/Hz$]')

handles, labels = fig.gca().get_legend_handles_labels()
fig.legend(handles, labels, loc='right')

fig.tight_layout(rect=(0.,0.,0.8,1.))
fig.show()

### PARAM

param_sym = (eth.System.Resistor_ntd.resistivity,)

param_eval = (3e6,)

ref_bath = eth.System.Capacitor_f

#A = eth.measure_noise_param(param_sym, evad, ref_bath)
#A = eth.response_noise_param(param_sym, evad)
A = eth.noise_tot_param(param_sym, evad, ref_bath)

#AA = A(param_eval)
#
#noise_dict = dict()
#for key, nfun in AA.iteritems():
#    noise_dict[key] = nfun(freq_array)

plt.figure()

for r in 10**np.linspace(5, 7, 10):
    param_eval = (r,)
    AAA = A(param_eval)(freq_array)

    plt.loglog(freq_array, AAA, label=r)

plt.legend()
plt.grid(True)