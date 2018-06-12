#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Group of function manipulating the bath_list in order to plot graphical
results for convenience.

@author: misiak
"""

import matplotlib.pyplot as plt
import numpy as np
import sympy as sy
import tqdm
from .system_eq import phi_vect, ete
from .steady_state import solve_sse
from .et_scheme import ThermalBath, Capacitor
from scipy.integrate import odeint

def plot_steady_state(bath_list, eval_dict, free_param, param_range,
                      quantities=[], label=[],
                      xscale='log', yscale='log'):
    """
    Solve the steady state equations on the range of a given free parameter,
    and plot the main physical quantities as well as bonus quantities.

    Parameters:
    ===========
    bath_list : list of ethem.ThermalBath or ethem.Capacitor
        List of the baths composing the electro-thermal system.
    eval_dict : dict
        Dictionnary used for the evaluation of the sympy symbols.
    free_param : sympy.core.symbol.Symbol
        Sympy symbol being the free parameter.
    param_range : array_like
        Range of the free parameter.
    quantities : list of list of symbols, optionnal
        Each sublist contains the additional physical quantities derived
        from the steady-state resolution which are plotted in the same
        subplots. Should be the same physical units to make sense.
    label : list of list of str, optionnal
        Label relative to the additionnal quantities. List structure must
        match with quantities. Else, the default label are set to be
        the string of the symbol (rarely readable).
    xscale, yscale : 'linear' or 'log'
        Set the x and y scales of the plots.

    Returns:
    ========
    fig, ax : matplotlib.pyplot Figure and Axes

    # TODO return the array of the quantities, might come in handy to fit
    the array, or further process it.
    """
    # copying the evaluation directory to keep the changes local
    eval_copy = eval_dict.copy()

    # phi vector
    phi = phi_vect(bath_list)

    # empty list to retrieve temperatures, ntd voltage,
    phi_array = list()

    # empty lists to retreive bonus quantities
    qq_array = list()
    for q_list in quantities:
        qq_list = []
        for q in q_list:
            qq_list.append([])
        qq_array.append(qq_list)

    # Solving the steady state on the bias voltage range
    for x in tqdm.tqdm(param_range):

        param = {free_param : x}
        eval_copy.update(param)

        solx = solve_sse(bath_list, eval_copy)

        # retrieving the temperatures of the electro-thermal baths
        phi_array.append(solx)

        dict_sol = {k : v for k, v in zip(phi, solx)}
        evad_ss = dict(eval_copy)
        evad_ss.update(dict_sol)

        # retrieving bonus quantities
        for k, q_list in enumerate(quantities):
            for j, q in enumerate(q_list):
                q_num = q.subs(evad_ss)
                qq_array[k][j].append(q_num)

    # handy conversion to numpy.ndarray
    phi_array = np.array(phi_array)

    # index corresponding to thermal bath or capacitor
    ind_thermal = []
    ind_capacitor = []
    for k,b in enumerate(bath_list):
        if isinstance(b, ThermalBath):
            ind_thermal.append(k)
        if isinstance(b, Capacitor):
            ind_capacitor.append(k)

    # number of subplots, and auxiliary variables for ordering subplots
    th_off = int(bool(ind_thermal))
    ca_off = int(bool(ind_capacitor))
    ndim_plot = th_off + ca_off + len(quantities)

    # creating the figure
    fig = plt.figure('plot_steady_state', figsize=(8, 7))
    ax = fig.get_axes()
    if len(ax) != ndim_plot:
        fig, ax = plt.subplots(ndim_plot, sharex=True,
                               figsize=(10, 10), num='plot_steady_state')

   # plotting thermal bath temperature
    for k in ind_thermal:
        ax[0].plot(param_range, phi_array[:, k], label=str(phi[k]))
        ax[0].set_ylabel('Temperature [K]')

    # plotting capacitor voltage
    for k in ind_capacitor:
        ax[th_off].plot(param_range, phi_array[:,k], label=str(phi[k]))
        ax[th_off].set_ylabel('NTD Voltage [V]')

    for k, q_list in enumerate(qq_array):
        for j, q in enumerate(q_list):
            try :
                ax[th_off+ca_off+k].plot(param_range, q,
                                         label=label[k][j])
            except :
                print 'FAIL'
                ax[th_off+ca_off+k].plot(param_range, q,
                                         label=str(quantities[k][j]))

    for a in ax:
        a.set_xscale(xscale)
        a.set_yscale(yscale)
        a.grid(True)
        a.legend()

    ax[-1].set_xlabel('Free Parameter')

    # COMMENT IF ANYTHING IS WRONG WITH THIS FUNCTION
    # show the maximum of the approximate sensitivity
    ind_max = np.argmax(qq_array[-1][0])
    v_opt = param_range[ind_max]
    q_max = qq_array[-1][0][ind_max]
    ax[-1].text(v_opt, q_max, "Vb opt. = %r" % round(v_opt,3))
    ax[-1].axvline(x=v_opt, ymin=0, ymax=1, ls='--', c="k")


    fig.tight_layout()
    plt.subplots_adjust(hspace=0.0)

    return fig, ax


def plot_odeint(bath_list, per, eval_dict, tsym, fs, time_length=1., plot=True, **kwargs):

    ### temperature vector
    phi = phi_vect(bath_list)

    assert phi.shape == per.shape

    sol_ss = solve_sse(eval_dict, x0, **kwargs)

    eteq = ete(bath_list)
    eteq_num = eteq.subs(eval_dict)
    eteq_f = sy.lambdify([tsym]+list(phi), eteq_num, modules="numpy")

    per_num = per.subs(eval_dict)
    per_lambda = sy.lambdify([tsym]+list(phi), per_num, modules="numpy")

    funky = lambda x, t: eteq_f(t, *x).flatten() + per_lambda(t, *x).flatten()

    x0 = sol_ss
    trange = np.arange(0, time_length, fs**-1)
    time_plot = np.hstack((np.arange(-time_length, 0, fs**-1), trange))
    sol = odeint(funky, x0, trange, printmessg=True,
                 rtol=1e-15, atol=1e-15, hmax=fs**-1)

    sol_per = sol-x0
    per_plot = np.array([per_lambda(tt, *xx).flatten()
                         for tt, xx in zip(trange, sol)])

    if plot == True:
        fig = plt.figure('plot_odeint')
        ax = fig.get_axes()
        if len(ax) == 0:
            fig, ax = plt.subplots(4, sharex=True, num='plot_odeint')

        for s, a, p in zip(sol_per.T, ax, per_plot.T):
            s_plot = np.pad(s, (int(fs*time_length), 0), 'constant', constant_values=0)
            p_plot = np.pad(p, (int(fs*time_length), 0), 'constant', constant_values=0)*1e-2
            a.plot(time_plot, s_plot, label='signal')
            a.plot(time_plot, p_plot, label='perturbation')
            a.legend()
            a.grid(True)

    return sol_per.T

