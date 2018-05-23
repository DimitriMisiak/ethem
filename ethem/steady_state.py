#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: misiak

Functions to access the steady state.
"""

import numpy as np
import sympy as sy
from scipy.optimize import root
from .system_eq import phi_vect, sse


def solve_sse(bath_list, eval_dict, x0=[0.018, 0.018, 0.018, 2e-3]):
    """ Solve the steady-state system for the given system characteristics.

    Parameters:
    ===========
    bath_list : list
        List containing all the electro-thermal baths of the system.
    eval_dict :dict
        Contains the evaluation values for the system characteristics symbols.
    x0 : array_like
        Initial vector for the resolution.

    Returns:
    ========
    sol.x : numpy.ndarray
        Solution vector, returned in the same order as bath_list.

    See also:
    =========
    scipy.optimize.root
    """
    # Quantities to be evaluated by the resolution
    phi = phi_vect(bath_list)

    # checking that the initial vector is adapted in length
    assert len(phi) == len(x0)

    # Steady state equations
    sseq = sse(bath_list).subs(eval_dict)

    # checking that all symbols the desired symbols are evaluated
    # if an error is raised, a term is missing from the given dictionnary
    assert set(phi) == set(sseq.free_symbols)

    # process the sympy equations into a function adaptated to scipy root
    funk = sy.lambdify(phi, sseq, 'numpy')
    system_eq = lambda x: np.squeeze(funk(*x))

    # Resolution with scipy.optimize.root
    sol = root(system_eq, x0)#, method= 'lm')

    return sol.x
