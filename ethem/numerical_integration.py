#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: misiak

Functions to resolve the temporal response of the system with
 numerical integration.
"""

import numpy as np
import sympy as sy
from scipy.integrate import odeint, solve_ivp
from .core_classes import System


def num_int(per, eval_dict, x0, fs=1e3, L=1.):
    """ DEPRECATED.
    Numerical integration of the electro-thermal equation with
    power perturbation for the given evaluation dictionnary.

    Parameters:
    ===========
    per : Sympy matrix
        Power perturbation of the system. Its shape must matches the one
        of System.eteq
    eval_dict : dict
        Contains the evaluation values for the system characteristics symbols.
    x0 : array_like
        Initial vector for the integration. Should be the solution of
        the steady state, if not this is equivalent to add a Dirac perturbation
        to the system.
    fs : float, optionnal
        Sampling frequency. By default, 1e3 Hz.
    L : float, optionnal
        Time length of the window in second. By default, 1 second.

    Returns:
    ========
    sol_array : numpy.ndarray
        2d array containing the time and the pulses in the electro-thermal
        baths.
        sol_array[0] is the time array.
        sol_array[1:] are the pulses in the baths.

    See also:
    =========
    scipy.integrate.odeint
    """
    #time array for plot
    time = np.arange(0., L, fs**-1)

    t = System.t

    capa_matrix = System.capacity_matrix
    per_arg = capa_matrix**-1 * per / sy.Heaviside(t)

    phi = System.phi_vect

    eteq = System.eteq

    eteq_num = eteq.subs(eval_dict)
    per_num = per_arg.subs(eval_dict)

    eteq_f = sy.lambdify([t]+list(phi), eteq_num, modules="numpy")
    per_lambda = sy.lambdify([t]+list(phi), per_num, modules="numpy")
    funky = lambda x, t: eteq_f(t, *x).flatten() + per_lambda(t, *x).flatten()

    trange = np.arange(0, L, fs**-1)

    sol = odeint(funky, x0, trange, printmessg=True,
                 rtol=1e-15, atol=1e-15, hmax=fs**-1)

    # substracting the initial vector
    sol_per = sol-x0

    sol_array = np.insert(sol_per.T, 0, time, axis=0)

    return sol_array


def num_int_proto(per, eval_dict, x0, fs=1e3, L=1., max_step_coef=10.):

    x0 = np.array(x0)

    time = np.arange(0., L, fs**-1)

    phi = System.phi_vect
    param = [System.t] + list(phi)

    # perturbation
    capa_matrix = System.capacity_matrix
    per_arg = capa_matrix**-1 * per / sy.Heaviside(System.t)

    eteq = System.eteq

    eq = per_arg + eteq
    eq_num = list(eq.subs(eval_dict))
    #eq_fun = sy.lambdify(param, eq_num, modules='numpy')

    eq_fun = sy.lambdify(param, eq_num)

    def eq_aux(t, y):
        list_y = list(y)
        param = [t] + list_y
        vec = eq_fun(*param)
        return vec

    sol = solve_ivp(eq_aux, [0, L], x0, t_eval=time,
                    max_step=max_step_coef*fs**-1)

    sol_array = np.vstack((sol.t, sol.y))

    return sol_array
