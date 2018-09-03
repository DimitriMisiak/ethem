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


def solve_sse(eval_dict, x0, twin=10., method=None, printsuccess=False):
    """ Solve the steady-state system for the given system characteristics.
    Use the odeint function to emulate the convergence of the physical system.
    Then, the root finder is used to obtain a precise estimation of the
    steady-state solution.

    Parameters:
    ===========
    eval_dict :dict
        Contains the evaluation values for the system characteristics symbols.
    x0 : array_like
        Initial vector for the resolution.
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
    scipy.optimize.root
    """
    # Quantities to be evaluated by the resolution
    phi = System.phi_vect
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
