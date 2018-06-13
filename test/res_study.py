#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Test file for the et_scheme.py script

@author: misiak
"""

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


### importing ethem module
#import sys
#sys.path.append(u'/home/misiak/Scripts/ETHEM project/')
import ethem as eth

#from config_ethem import evad, per, t, f
import config_ethem as CE

### closing previous plot
plt.close('all')

bath_list = eth.System.bath_list
num_bath = len(bath_list)
### Measured bath / Reference Bath
ref_bath = bath_list[-1]
ref_ind = bath_list.index(ref_bath)

def resolution(per, eval_dict, v_bias, t_cryo, fs=1e3, L=1., plot=False):
    """ Compute the resolution of the detector for the given event
    perturbation, evluation dictionnary and polarisation point.

    Parameters
    ==========
    per : Sympy matrix
        Power perturbation of the system. Its shape must matches the one
    eval_dict : dict
        Evaluation dictionnary containing the numerical values of all the
        symbols of eth.System.
    v_bias : float
        Evaluation value for the bias voltage.
    t_cryo : float
        Evaluation temperature for the cryostat temperature.
    fs : float, optional
        Sampling frequency, by default set to 1e3 Hz.
    L : float, optional
        Lenth of the time window, by default set to 1 second.
    plot : boolean, optional
        If True, plot the integrand for the resolution calculation.
    Return
    ======
    res : float
        Energy Resolution in keV.
    """
    x0 = [t_cryo, t_cryo, t_cryo, 0.]
    eval_dict.update({'V_b':v_bias, 'T_b':t_cryo})
    sol_ss = eth.solve_sse(eval_dict, x0)

    # updating the evaluation dictionnary
    ss_dict = {b : v for b,v in zip(eth.System.phi_vect, sol_ss)}

    # new evaluation dictionnary taking updated with the steady state
    evad_ss = eval_dict.copy()
    evad_ss.update(ss_dict)

    N = int(L * fs)

    fres = fs/N
    # array for frequency plot
    freq = np.flip(np.arange(fs/2, 0., -fres), axis=0)
    # array for numpy.fft calculation
    freqnp = np.fft.fftfreq(N, fs**-1)

    #==============================================================================
    # RESPONSE TO EVENT
    #==============================================================================
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
    psd_ref = sv_mod_array[ref_ind]


    #%%
    #==============================================================================
    # RESPONSE TO NOISE
    #==============================================================================

    # noise system in ref bath
    psd_fun_dict = eth.response_noise(evad_ss)
    psd_full = {k:fun(freq) for k, fun in psd_fun_dict.iteritems()}
    psd_full_ref = {k:v[:, ref_ind] for k,v in psd_full.iteritems()}

    # noise measure (directly in ref bath)
    psd_obs_fun = eth.measure_noise(ref_bath, evad_ss)
    psd_obs = {k:fun(freq) for k,fun in psd_obs_fun.iteritems()}

    # calculating total noise psd
    psd_full_ref.update(psd_obs)
    psd_noise_tot = np.sum(psd_full_ref.values(), axis=0)

    #==============================================================================
    # NEP AND RESOLUTION
    #==============================================================================
    # dictionnary of nep array
    nep_tot = psd_noise_tot/psd_ref

    # dictionnary of 4/nep array
    invres = 4/nep_tot

    bounds = None

    inf, sup = (0, -1)
    if bounds is not None:
        inf, sup = bounds

    invres_int = np.trapz(invres[inf:sup], freq[inf:sup])
    res = (invres_int)**-0.5

    return res

    ##==============================================================================
    ## MONITORING PLOT
    ##==============================================================================
    if plot == True:
        plt.figure('4/NEP^2')
        plt.plot(freq, invres,
                 label='V_bias = {:.2e} V, T_cryo = {:.3f} K'.format(v_bias, t_cryo),
                 color='slateblue', lw=2.0)
        plt.legend(title='RES= {:.3f} keV'.format(res))
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Integrand [keV/Hz]')
        plt.grid(True)

#%%
t_range = [0.012, 0.014, 0.016, 0.018, 0.020]
v_range = 10**np.linspace(-3, np.log10(10), 20)

res_array = list()
for t in tqdm(t_range):
    aux_list = list()
    for v in v_range:
        aux_list.append(resolution(CE.per, CE.evad, v, t))
    res_array.append(aux_list)

res_array = np.array(res_array)

plt.figure()
for i,t in enumerate(t_range):
    plt.plot(v_range, res_array[i], label=t)
plt.xscale('log')

