#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Sum up all the possibility of the ethem package first applied to the
nbsi_solo and nbsi_duo detectors.

@author: misiak
"""

# adding ethem module path to the pythonpath
import tqdm
import matplotlib.pyplot as plt
import numpy as np

import ethem as eth

import config_red_toy as config
from config_red_toy import syst, evad

plt.close('all')

#==============================================================================
# PARAMETERS
#==============================================================================
L = 1
fs = 1e4

#%%
#==============================================================================
# SOLVE SSE PARAM
#==============================================================================
def test_solve_sse_param():
    """ Scripts testing the solve_sse_param function.
    Produce 4 plots similar to a crude search for an optimization point.
    """
    v_array = 10**np.linspace(np.log10(0.02), np.log10(50), 100)

    param_plot = (
            syst.Thermostat_b.temperature,
            syst.ThermalLink_ep.cond_alpha,
            syst.ThermalLink_leak.cond_alpha,
            config.R0
    )

    param_arrays = (
            np.linspace(0.015, 0.050, 10),
            10**np.linspace(1, 3, 10),
            10**np.linspace(-3, -1, 10),
            np.linspace(0.5, 20, 10)
    )

    ### PLOT
    fig_iv, ax_iv = plt.subplots(ncols=2, nrows=2,
                                 num='pseudo IV curves', figsize=(11, 11))
    ax_iv = ax_iv.ravel()

    for ind, ax in enumerate(ax_iv):

        param_sym = (
                syst.Voltstat_b.voltage,
                param_plot[ind]
        )

        ss_point = eth.solve_sse_param(syst, param_sym, evad)

        for p in tqdm.tqdm(param_arrays[ind]):

            iv_list = list()
            for volt in v_array:

                sol = ss_point((volt, p))
                iv_list.append(sol.x[-1])

            ax.loglog(v_array, iv_list, label='{0:.4f}'.format(p))

        ax.grid(True)
        ax.set_title(str(param_plot[ind]))
        ax.set_ylabel('Capa voltage [V]')
        ax.set_xlabel('Bias Voltage [V]')
        ax.legend(loc='right')

    fig_iv.tight_layout()
    fig_iv.savefig('output/solve_sse_param.png')

#%%
#==============================================================================
# IMPEDANCE MATRIX PARAM
#==============================================================================
def test_impedance_matrix_param():
    """ Scripts testing the impedance_matrix_param function.
    Produce 4 plots similar to a crude search for an optimization point.
    """
    freq_array = np.arange(L**-1, fs/2.+L**-1, L**-1)

    param_plot = (
            syst.Voltstat_b.voltage,
            syst.Thermostat_b.temperature,
            syst.ThermalLink_ep.cond_alpha,
            syst.ThermalLink_leak.cond_alpha,
    )

    param_arrays = (
            10**np.linspace(np.log10(0.02), np.log10(20), 10),
            np.linspace(0.015, 0.050, 10),
            10**np.linspace(1, 3, 10),
            10**np.linspace(-3, -1, 10),
    )

    ### PLOT
    fig_iv, ax_iv = plt.subplots(ncols=2, nrows=2,
                                 num='sensitivity curves', figsize=(11, 11))
    ax_iv = ax_iv.ravel()

    for ind, ax in enumerate(ax_iv):

        param_sym = (
                param_plot[ind],
        )

        jac_param_fun = eth.impedance_matrix_param(syst, param_sym, evad,
                                                   auto_ss=True)

        for p in tqdm.tqdm(param_arrays[ind]):

            jac_fun = jac_param_fun((p,))
            jac_array = jac_fun(freq_array)
            sen_array = np.abs(jac_array[:,-1,0])

            ax.loglog(freq_array, sen_array, label='{0:.4f}'.format(p))

        ax.grid(True)
        ax.set_title(str(param_plot[ind]))
        ax.set_ylabel('Sensitivity [V/W]')
        ax.set_xlabel('Frequency [Hz]')
        ax.legend(loc='right')

    fig_iv.tight_layout()
    fig_iv.savefig('output/impedance_matrix_param.png')

#%%
#==============================================================================
# EIGEN PARAM
#==============================================================================
def test_eigen_param():
    """ Scripts testing the eigen_param function.
    Produce 4 plots similar to a crude search for an optimization point.
    """
    time_array = np.arange(0., L, fs**-1)

    param_plot = (
            syst.Voltstat_b.voltage,
            syst.Thermostat_b.temperature,
            syst.ThermalLink_ep.cond_alpha,
            syst.ThermalLink_leak.cond_alpha,
    )

    param_arrays = (
            10**np.linspace(np.log10(0.02), np.log10(20), 10),
            np.linspace(0.015, 0.050, 10),
            10**np.linspace(1, 3, 10),
            10**np.linspace(-3, -1, 10),
    )

    ### PLOT
    fig_iv, ax_iv = plt.subplots(ncols=2, nrows=2,
                                 num='temporal pulses', figsize=(11, 11))
    ax_iv = ax_iv.ravel()

    for ind, ax in enumerate(ax_iv):

        param_sym = (
                param_plot[ind],
        )

        eigen_param_fun = eth.eigen_param(syst, param_sym, evad, auto_ss=True)

        for p in tqdm.tqdm(param_arrays[ind]):

            tau_array, amp_array, pulse_fun = eigen_param_fun((p,))
            pulse_array = pulse_fun(time_array)[-1]

            ax.plot(time_array, pulse_array, label='{0:.4f}'.format(p))

        ax.grid(True)
        ax.set_title(str(param_plot[ind]))
        ax.set_ylabel('Voltage [V]')
        ax.set_xlabel('Time [s]')
        ax.legend(loc='right')

    fig_iv.tight_layout()
    fig_iv.savefig('output/eigen_param.png')


#%%
#==============================================================================
# PER PARAM
#==============================================================================
def test_per_param():
    """ Scripts testing the impedance_per_param function.
    Produce 2 plots similar to a crude search for an optimization point.
    """
    time_array = np.arange(0., L, fs**-1)

    param_plot = (
            config.tau_therm,
            config.eps
    )

    param_arrays = (
            10**np.linspace(-3,-1, 10),
            np.linspace(0., 0.9, 10),
    )

    ### PLOT
    fig_iv, ax_iv = plt.subplots(ncols=2, nrows=1,
                                 num='perturbation curves', figsize=(11, 5))
    ax_iv = ax_iv.ravel()

    for ind, ax in enumerate(ax_iv):

        param_sym = (
                param_plot[ind],
        )

        per_param_fun = eth.per_param(syst, param_sym, evad)

        for p in tqdm.tqdm(param_arrays[ind]):

            per_fun = per_param_fun((p,))
            per_array = per_fun(time_array)
            per_abso_array = per_array[:,0]

            ax.plot(time_array, per_abso_array, label='{0:.4f}'.format(p))

        ax.grid(True)
        ax.set_title(str(param_plot[ind]))
        ax.set_ylabel('Power [W]')
        ax.set_xlabel('Time [s]')
        ax.legend(loc='right')

    fig_iv.tight_layout()
    fig_iv.savefig('output/per_param.png')

#%%
#==============================================================================
# PERF PARAM
#==============================================================================
def test_perf_param():
    """ Scripts testing the impedance_perf_param function.
    Produce 2 plots similar to a crude search for an optimization point.
    """
    freq_array = np.arange(L**-1, fs/2.+L**-1, L**-1)

    param_plot = (
            config.tau_therm,
            config.eps
    )

    param_arrays = (
            10**np.linspace(-3,-1, 10),
            np.linspace(0., 0.9, 10),
    )

    ### PLOT
    fig_iv, ax_iv = plt.subplots(ncols=2, nrows=1,
                                 num='perturbation fft curves', figsize=(11, 5))
    ax_iv = ax_iv.ravel()

    for ind, ax in enumerate(ax_iv):

        param_sym = (
                param_plot[ind],
        )

        perf_param_fun = eth.perf_param(syst, param_sym, evad)

        for p in tqdm.tqdm(param_arrays[ind]):

            perf_fun = perf_param_fun((p,))
            perf_array = perf_fun(freq_array)
            perf_abso_array = np.abs(perf_array[:,0])

            ax.loglog(freq_array, perf_abso_array, label='{0:.4f}'.format(p))

        ax.grid(True)
        ax.set_title(str(param_plot[ind]))
        ax.set_ylabel('Power [W]')
        ax.set_xlabel('Frequency [Hz]')
        ax.legend(loc='right')

    fig_iv.tight_layout()
    fig_iv.savefig('output/perf_param.png')

#%%
#==============================================================================
# PERF PARAM
#==============================================================================
def test_response_event_param():
    """ Scripts testing the impedance_matrix_param function.
    Produce 6 plots similar to a crude search for an optimization point.
    """
    freq_array = np.arange(L**-1, fs/2.+L**-1, L**-1)

    param_plot = (
            syst.Voltstat_b.voltage,
            syst.Thermostat_b.temperature,
            syst.ThermalLink_ep.cond_alpha,
            syst.ThermalLink_leak.cond_alpha,
            config.tau_therm,
            config.eps,
    )

    param_arrays = (
            10**np.linspace(np.log10(0.02), np.log10(20), 10),
            np.linspace(0.015, 0.050, 10),
            10**np.linspace(1, 3, 10),
            10**np.linspace(-3, -1, 10),
            10**np.linspace(-3,-1, 10),
            np.linspace(0., 0.9, 10),
    )

    ### PLOT
    fig_iv, ax_iv = plt.subplots(ncols=3, nrows=2,
                                 num='response event curves', figsize=(17, 11))
    ax_iv = ax_iv.ravel()

    for ind, ax in enumerate(ax_iv):

        param_sym = (
                param_plot[ind],
        )

        response_param_fun = eth.response_event_param(syst, param_sym, evad)

        for p in tqdm.tqdm(param_arrays[ind]):

            response_fun = response_param_fun((p,))
            response_array = response_fun(freq_array)
            response_capa_array = np.abs(response_array[-1,:])

            ax.loglog(freq_array, response_capa_array, label='{0:.4f}'.format(p))

        ax.grid(True)
        ax.set_title(str(param_plot[ind]))
        ax.set_ylabel('Capa Voltage [V]')
        ax.set_xlabel('Frequency [Hz]')
        ax.legend(loc='right')

    fig_iv.tight_layout()
    fig_iv.savefig('output/response_event_param.png')

#%%
#==============================================================================
# NOISE OBS PARAM AND MEASURE NOISE PARAM
#==============================================================================
def testy_measure_noise_param():
    freq_array = np.arange(L**-1, fs/2.+L**-1, L**-1)

    param_sym = (syst.Voltstat_b.voltage, syst.Thermostat_b.temperature)

    param_eval = (2., 0.018)

    ref_bath = syst.Capacitor_f

#    A = eth.noise_obs_param(syst, param_sym, evad, ref_bath)
    A = eth.measure_noise_param(syst, param_sym, evad, ref_bath)

    AA = A(param_eval)

    noise_dict = dict()
    for key, nfun in AA.items():
        noise_dict[key] = nfun(freq_array)

    return noise_dict

#%%
#==============================================================================
# NOISE FLUX FUN PARAM AND RESPONSE NOISE PARAM
#==============================================================================
def testy_response_noise_param():
    freq_array = np.arange(L**-1, fs/2.+L**-1, L**-1)

    param_sym = (syst.Voltstat_b.voltage, syst.Thermostat_b.temperature)

    param_eval = (2., 0.018)

#    A = eth.noise_flux_fun_param(syst, param_sym, evad, ref_bath)
    A = eth.response_noise_param(syst, param_sym, evad)

    AA = A(param_eval)

    noise_dict = dict()
    for key, nfun in AA.items():
        noise_dict[key] = nfun(freq_array)

    return noise_dict

#%%
#==============================================================================
# RESPONSE NOISE PARAM PLOT
#==============================================================================
def test_response_noise_param():
    """ Scripts testing the response_noise_param function.
    Produce 4 plots similar to a crude search for an optimization point.
    """
    freq_array = np.arange(L**-1, fs/2.+L**-1, L**-1)

    param_plot = (
            syst.Voltstat_b.voltage,
            syst.Thermostat_b.temperature,
            syst.ThermalLink_ep.cond_alpha,
            syst.ThermalLink_leak.cond_alpha,
    )

    param_arrays = (
            10**np.linspace(np.log10(0.02), np.log10(20), 10),
            np.linspace(0.015, 0.050, 10),
            10**np.linspace(1, 3, 10),
            10**np.linspace(-3, -1, 10),
    )

    ### PLOT
    fig_iv, ax_iv = plt.subplots(ncols=2, nrows=2,
                                 num='response noise curves', figsize=(17, 11))
    ax_iv = ax_iv.ravel()

    for ind, ax in enumerate(ax_iv):

        param_sym = (
                param_plot[ind],
        )

        noise_param_fun = eth.response_noise_param(syst, param_sym, evad)

        ref_bath = syst.Capacitor_f
        noise_tot_param = eth.noise_tot_param(syst, param_sym, evad, ref_bath)

        for p in tqdm.tqdm(param_arrays[ind]):

            noise_fun_dict = noise_param_fun((p,))

            for k, nfun in noise_fun_dict.items():

                noise_array = nfun(freq_array)[-1, :]
                ax.loglog(freq_array, noise_array)

            noise_tot_fun = noise_tot_param((p,))
            noise_tot_array = noise_tot_fun(freq_array)
            ax.loglog(freq_array, noise_tot_array, label='{0:.4f}'.format(p),
                      color='k')


        ax.grid(True)
        ax.set_title(str(param_plot[ind]))
        ax.set_ylabel('Noise PSD [V**2/Hz]')
        ax.set_xlabel('Frequency [Hz]')
        ax.legend(loc='right')

    fig_iv.tight_layout()
    fig_iv.savefig('output/response_noise_param.png')

#%%
#==============================================================================
# NOISE TOT PARAM
#==============================================================================
def test_noise_tot_param():
    """ Scripts testing the noise_tot_param function.
    Produce 4 plots similar to a crude search for an optimization point.
    """
    freq_array = np.arange(L**-1, fs/2.+L**-1, L**-1)

    ref_bath = syst.Capacitor_f

    param_plot = (
            syst.Voltstat_b.voltage,
            syst.Thermostat_b.temperature,
            syst.ThermalLink_ep.cond_alpha,
            syst.ThermalLink_leak.cond_alpha,
    )

    param_arrays = (
            10**np.linspace(np.log10(0.02), np.log10(20), 10),
            np.linspace(0.015, 0.050, 10),
            10**np.linspace(1, 3, 10),
            10**np.linspace(-3, -1, 10),
    )

    ### PLOT
    fig_iv, ax_iv = plt.subplots(ncols=2, nrows=2,
                                 num='noise tot curves', figsize=(17, 11))
    ax_iv = ax_iv.ravel()

    for ind, ax in enumerate(ax_iv):

        param_sym = (
                param_plot[ind],
        )

        noise_param_fun = eth.noise_tot_param(syst, param_sym, evad, ref_bath)

        for p in tqdm.tqdm(param_arrays[ind]):

            nfun = noise_param_fun((p,))

            noise_array = nfun(freq_array)
            ax.loglog(freq_array, noise_array, label='{0:.4f}'.format(p))

        ax.grid(True)
        ax.set_title(str(param_plot[ind]))
        ax.set_ylabel('Noise PSD [V**2/Hz]')
        ax.set_xlabel('Frequency [Hz]')
        ax.legend(loc='right')

    fig_iv.tight_layout()
    fig_iv.savefig('output/noise_tot_param.png')

#%%
#==============================================================================
# NEP REF PARAM
#==============================================================================
def test_nep_ref_param():
    """ Scripts testing the nep_ref_param function.
    Produce 4 plots similar to a crude search for an optimization point.
    """
    freq_array = np.arange(L**-1, fs/2.+L**-1, L**-1)

    ref_bath = syst.Capacitor_f

    param_plot = (
            syst.Voltstat_b.voltage,
            syst.Thermostat_b.temperature,
            syst.ThermalLink_ep.cond_alpha,
            syst.ThermalLink_leak.cond_alpha,
    )

    param_arrays = (
            10**np.linspace(np.log10(0.02), np.log10(20), 10),
            np.linspace(0.015, 0.050, 10),
            10**np.linspace(1, 3, 10),
            10**np.linspace(-3, -1, 10),
    )

    ### PLOT
    fig_iv, ax_iv = plt.subplots(ncols=2, nrows=2,
                                 num='nep curves', figsize=(17, 11))
    ax_iv = ax_iv.ravel()

    for ind, ax in enumerate(ax_iv):

        param_sym = (
                param_plot[ind],
        )

        nep_param = eth.nep_ref_param(syst, param_sym, evad, ref_bath)
        res_param = eth.res_ref_param(syst, param_sym, evad, ref_bath, fs, L)

        for p in tqdm.tqdm(param_arrays[ind]):

            nfun = nep_param((p,))
            nep_array = nfun(freq_array)

#            res = eth.nep_to_res(freq_array, nep_array)
            res = res_param((p,))

            res_ev = res * config.energy.subs(evad) / 1.6e-19
            ax.loglog(freq_array, nep_array, label='{0:.4f}, {1:.3f} eV'.format(p, res_ev))

        ax.grid(True)
        ax.set_title(str(param_plot[ind]))
        ax.set_ylabel('NEP [W**2/Hz]')
        ax.set_xlabel('Frequency [Hz]')
        ax.legend(loc='right')

    fig_iv.tight_layout()
    fig_iv.savefig('output/nep_param_param.png')

    return freq_array, nep_array


#%%
if __name__ == '__main__':

#    test_solve_sse_param()
#
#    test_impedance_matrix_param()
#
#    test_eigen_param()
#
#    test_per_param()
#
#    test_perf_param()
#
#    test_response_event_param()
#
#    print(testy_measure_noise_param())
#
#    AA = testy_response_noise_param()
#
#    AAA = dict()
#    for k,v in AA.items():
#        print(v.shape)
#        AAA[k] = v[-1, :]

#    test_response_noise_param()
#
#    test_noise_tot_param()
#
#    fa, na = test_nep_ref_param()

    print('Done.')

