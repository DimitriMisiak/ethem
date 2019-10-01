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


def num_int(system, eval_dict, x0, fs=1e3, L=1.):
    """ Numerical integration of the electro-thermal equation with
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

    t = system.time

    capa_matrix = system.capacity_matrix
    
    ### TEST 2019/07/01 to fix Heaviside computation
#    per_arg = capa_matrix**-1 * per / sy.Heaviside(t)
    per_arg = capa_matrix**-1 * system.perturbation.matrix

    def my_heaviside(t):
        return np.heaviside(t, 1.)
        
    phi = system.phi_vect

    eteq = system.eteq

    ###
    print(per_arg)

    eteq_num = eteq.subs(eval_dict)
    per_num = per_arg.subs(eval_dict)

    print(per_num)

    eteq_f = sy.lambdify([t]+list(phi), eteq_num, modules="numpy")
    
    ### TEST
#    per_lambda = sy.lambdify([t]+list(phi), per_num, modules="numpy")   
    per_lambda = sy.lambdify([t]+list(phi), per_num, modules=[{'Heaviside':my_heaviside}, 'numpy'])
    
    funky = lambda x, t: eteq_f(t, *x).flatten() + per_lambda(t, *x).flatten()

    trange = np.arange(0, L, fs**-1)

    sol = odeint(funky, x0, trange,
                 rtol=1e-15, atol=1e-15, hmax=fs**-1)

    print(sol)

    # substracting the initial vector
    sol_per = sol-x0

# old version where the return was formatted in an other way
#    sol_array = np.insert(sol_per.T, 0, time, axis=0)

    return (time, sol_per.T,)


def num_int_param(system, param,  eval_dict, fs, L):
#(per, eval_dict, x0, fs=1e3, L=1.):

    """ Return an auxiliary function numerically integrating the equations
    of eth.System for a configuration of the given parameters.
    This is efficient to compare different configurations of parameters.

    Parameters
    ----------
    param : tuple of sympy.symbols
        Tuple of symbols associated to the parameters of the returned function.
        In orde to quickly compute a IV curve, this param should be:
        (cryo_temp, bias_voltage).
    eval_dict : dict
        Evaluation dictionnary. Contains the evaluation values
        for the system characteristics symbols.
    fs : float, optionnal
        Sampling frequency. By default, 1e3 Hz.
    L : float, optionnal
        Time length of the window in second. By default, 1 second.

    Others settings of the steady-state resolution are passed as parameters
    for the returned function. See solve_sse_fun doc.

    Return
    ------
    sol_see_fun : function
        Actual function solving the steady state with the given parameters
        values. See its doc for more info.

    See also
    --------
    Doc of the auxiliary returned function solve_sse_fun
    eth.phi_init, eth.solve_sse_perf
    """
    npar = len(param)

    char_dict = eval_dict.copy()

    for p in param:
        try:
            char_dict.pop(p)
        except:
            pass

    trange = np.arange(0, L, fs**-1)

    t = system.time

    capa_matrix = system.capacity_matrix

    per = system.perturbation
    per_arg = capa_matrix**-1 * per.matrix / sy.Heaviside(t)

    phi = tuple(system.phi_vect)
    nphi = len(phi)

    eteq = system.eteq

    eteq_num = eteq.subs(char_dict)
    per_num = per_arg.subs(char_dict)

    eq_list = list(eteq_num + per_num)
    eq_lambda = sy.lambdify(phi+(t,)+param, eq_list, 'math')

    def num_int_fun(p, x0):
        """ Numerical integration of the electro-thermal equation with
        power perturbation for the given evaluation dictionnary.

        Parameters
        ----------
        p : tuple of floats
            Values for the parameters param.
        x0 : array_like
            Initial vector for the integration. Should be the solution of
            the steady state, if not this is equivalent to add a Dirac perturbation
            to the system.

        """
        assert len(p) == npar
        assert len(x0) == nphi

        def aux(phi, t):
            args = tuple(phi) + (t,) + p
            return eq_lambda(*args)

        sol = odeint(aux, x0, trange,
                     rtol=1e-15, atol=1e-15, hmax=fs**-1)

        # substracting the initial vector
        sol_per = sol-x0

        return sol_per.T

    return num_int_fun
