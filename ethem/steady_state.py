#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: misiak

Functions to access the steady state.
"""

import numpy as np
import sympy as sy
from scipy.optimize import root
from scipy.integrate import odeint
from .core_classes import System
from .et_classes import ThermalBath, Thermostat, Capacitor


def solve_sse_old(eval_dict, x0, method=None, printsuccess=False):
    """ Solve the steady-state system for the given system characteristics.

    Parameters:
    ===========
    eval_dict :dict
        Contains the evaluation values for the system characteristics symbols.
    x0 : array_like
        Initial vector for the resolution.
    method : str, optional
        Type of solver. See scipy.optimize.root for more precision. By default,
        set to None which uses 'lm' method.

    Returns:
    ========
    sol.x : numpy.ndarray
        Solution vector, returned in the same order as bath_list.

    See also:
    =========
    scipy.optimize.root
    """
    # Quantities to be evaluated by the resolution
    phi = System.phi_vect
    # checking that the initial vector is adapted in length
    assert len(phi) == len(x0)

    # Steady state equations
    sseq = System.sseq.subs(eval_dict)

    # checking that all symbols the desired symbols are evaluated
    # if an error is raised, a term is missing from the given dictionnary
    assert set(phi) == set(sseq.free_symbols)

    # process the sympy equations into a function adaptated to scipy root
    funk = sy.lambdify(phi, sseq, 'numpy')
    system_eq = lambda x: np.squeeze(funk(*x))

    if method is None:
        method = 'lm'

    # Resolution with scipy.optimize.root
    sol = root(system_eq, x0, method=method,
               options={'ftol':1e-15, 'xtol':1e-15, 'maxiter':1000})

    if printsuccess == True:
        print sol.success

    return sol.x


def solve_sse(eval_dict, x0=None, twin=10., method=None, printsuccess=False):
    """ Solve the steady-state system for the given system characteristics.
    Use the odeint function to emulate the convergence of the physical system.
    Then, the root finder is used to obtain a precise estimation of the
    steady-state solution.

    Parameters:
    ===========
    eval_dict :dict
        Contains the evaluation values for the system characteristics symbols.
    x0 : array_like, optional
        Initial vector for the resolution. By default, set to None. For a x0
        set to None, the initial vector is 0 for the voltage, and the minimal
        thermostat temperature of the system if such a ThermalBath exist, else
        it is 0 for the temperature.
    twin : float, optional
        Time window for the odeint integration. By default, 10 seconds.
    method : str, optional
        Type of solver. See scipy.optimize.root for more precision. By default,
        set to None which uses 'lm' method.

    Returns:
    ========
    sol.x : numpy.ndarray
        Solution vector, returned in the same order as bath_list.

    See also:
    =========
    scipy.optimize.root, eth.phi_init
    """
    # Quantities to be evaluated by the resolution
    phi = System.phi_vect

    if x0 is None:
        x0 = phi_init(eval_dict)

    # checking that the initial vector is adapted in length
    assert len(phi) == len(x0)

    # Steady state equations
    eteq = System.eteq.subs(eval_dict)

    # checking that all symbols the desired symbols are evaluated
    # if an error is raised, a term is missing from the given dictionnary
    assert set(phi) == set(eteq.free_symbols)

    # process the sympy equations into a function adaptated to scipy root
    funk = sy.lambdify(phi, eteq, 'numpy')
    system_eq = lambda x: np.squeeze(funk(*x))

    system_eq_odeint = lambda x,t: system_eq(x)

    time_odeint = np.linspace(0., twin, 10)

    inte = odeint(system_eq_odeint, x0, time_odeint)

    if method is None:
        method = 'lm'

    # Resolution with scipy.optimize.root
    sol = root(system_eq, inte[-1], method=method,
               options={'ftol':1e-15, 'xtol':1e-15, 'maxiter':1000})

    if printsuccess == True:
        print sol.success

    return sol.x


def solve_sse_manual(funk, x0, twin=10., method=None, printsuccess=False,
                     no_odeint=False):

    system_eq = lambda x: np.squeeze(funk(*x))

    if no_odeint:

        x0_root = x0

    if not no_odeint:

        system_eq_odeint = lambda x,t: system_eq(x)

        time_odeint = np.linspace(0., twin, 10)

        inte = odeint(system_eq_odeint, x0, time_odeint)

        x0_root = inte[-1]

    if method is None:
        method = 'lm'

    # Resolution with scipy.optimize.root
    sol = root(system_eq, x0_root, method=method,
               options={'ftol':1e-15, 'xtol':1e-15, 'maxiter':1000})

    if printsuccess == True:
        print sol.success

    return sol.x


def solve_sse_perf(funk, x0, safe_odeint=True, **kwargs):

    if not safe_odeint:

        x0_root = x0

    if safe_odeint:

        system_eq_odeint = lambda x,t: funk(x)

        time_odeint = np.linspace(0., 10., 10)

        inte = odeint(system_eq_odeint, x0, time_odeint)

        x0_root = inte[-1]

    # Resolution with scipy.optimize.root
    sol = root(funk, x0_root, **kwargs)

    return sol


#def solve_sse_perf2(funk, x0, safe_odeint=True, **kwargs):
#
#    if not safe_odeint:
#
#        x0_root = x0
#
#    if safe_odeint:
#
#        system_eq_odeint = lambda x,t: funk(x)
#
#        time_odeint = np.linspace(0., 10., 10)
#
#        inte = odeint(system_eq_odeint, x0, time_odeint)
#
#        x0_root = inte[-1]
#
#    # Resolution with scipy.optimize.root
#    sol = root(funk, x0_root, **kwargs)
#
#    return sol


def phi_init(eval_dict):
    """ Return the intial vector used for the search of the steady-state
    solution. For the capacitor, the initial voltage is set to 0. For the
    thermal bath, the initial temperature is set to the lowest thermostat
    temperature of the system. If none exists, the intial temperature is
    set to 0.

    Parameters
    ==========
    eval_dict : dict
        Evaluation dictionnary.

    Return
    ======
    init : list of float
        Initial vector.
    """
    bath_list = System.bath_list

    v0 = 0.0 #V
    t0 = 0.0 #K

    thermo_list = System.subclass_list(Thermostat)
    pure_list = list(set(thermo_list) - set(bath_list))
    if len(pure_list):
        temp_list = [(b.temperature).subs(eval_dict) for b in pure_list]
        t0 = float(min(temp_list))

    init = []
    for b in bath_list:
        if isinstance(b, Capacitor):
            init.append(v0)
        elif isinstance(b, ThermalBath):
            init.append(t0)
        else:
            raise Exception('Invalid class in System.bath_list'
                            '(only valid: Capacitor and ThermalBath).'
                            'Class is '.format(type(b)))

    return init


def dict_sse(eval_dict, **kwargs):
    """ Return an evaluation dictionnary with the steady-state solution.

    Parameters
    ==========
    eval_dict : dict
        Evaluation dictionnary.

    **kwargs parameters are passed to eth.solve_sse function.

    Return
    ======
    eval_dict_sse : dict
        Updated copy of the evaluation dictionnary.

    See also
    ========
    eth.solve_sse
    """
    sol_ss = solve_sse(eval_dict, **kwargs)

    sse_dict = {bath.main_quant:sol for bath,sol in zip(System.bath_list, sol_ss)}

    eval_dict_sse = eval_dict.copy()
    eval_dict_sse.update(sse_dict)

    return eval_dict_sse
