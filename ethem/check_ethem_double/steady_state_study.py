#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Test file for the et_scheme.py script

@author: misiak
"""

import matplotlib.pyplot as plt
import numpy as np
import tqdm

from config_ethem import eth
from config_ethem import evad

### closing previous plot
plt.close('all')

def solve_ss(eval_dict, t_cryo, x0=None, method=None):
    """ Solve the steady state for the given polarization point.

    Parameters
    ==========
    eval_dict : dict
        Evaluation dictionnary containing the numerical values of all the
        symbols of eth.System.
    v_bias : float
        Evaluation value for the bias voltage.
    t_cryo : float
        Evaluation temperature for the cryostat temperature.
    x0 : list of float, optional
        Initial guess for the solver. Set to None by default for a
        standard initial guess.
    method : str, optional
        Type of solver. See scipy.optimize.root for more precision. By default,
        set to None which uses 'lm' method.

    Return
    ======
    sol : list of float
        Solution vector of the OptimizeResult of scipy.optimize.root function.
    """
    if x0 is None:
        x0 = [t_cryo]

    eval_dict.update({'T_cryo':t_cryo})
    sol = eth.solve_sse(eval_dict, x0, method=method)

    return sol

#==============================================================================
# BUILDING ARRAY OF SOLUTION OF STEADY STATE
#==============================================================================
t_range = np.linspace(1, 30, 10)

sol_list = list()
for tb in t_range:
    sol = solve_ss(evad, tb)
    sol_list.append(*sol)

#==============================================================================
# PLOTTING EVOLUTION
#==============================================================================

plt.figure()
plt.plot(t_range, sol_list)





