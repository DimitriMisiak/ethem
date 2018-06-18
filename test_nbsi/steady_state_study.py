#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Test file for the et_scheme.py script

@author: misiak
"""

import matplotlib.pyplot as plt
import numpy as np
import tqdm

import ethem as eth
from config_ethem import evad, elntd

### closing previous plot
plt.close('all')

def solve_ss(eval_dict, v_bias, t_cryo, x0=None, method=None):
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
        x0 = [t_cryo, t_cryo, t_cryo, 0.]

    eval_dict.update({'V_b':v_bias, 'T_b':t_cryo})
    sol = eth.solve_sse(eval_dict, x0, method=method)

    return sol

#==============================================================================
# BUILDING ARRAY OF SOLUTION OF STEADY STATE
#==============================================================================
#t_range = [0.012, 0.014, 0.016, 0.018, 0.020]
#t_range = [0.018, 0.019, 0.020, 0.021, 0.022]
t_range = np.linspace(0.018, 0.022, 10)
v_range = 10**np.linspace(-1, np.log10(100), 50)

sol_dict = dict()
r_dict = dict()

for tb in t_range:
    sol_list = []
    r_list = []
    sol = [tb, tb, tb, 0.]
    for v in tqdm.tqdm(v_range):
        sol = solve_ss(evad, v, tb, x0=sol)
        sol_list.append(sol)

        # updating the evaluation dictionnary
        ss_dict = {b : v for b,v in zip(eth.System.phi_vect, sol)}
        r_nbsi = elntd.resistivity.subs(evad).subs(ss_dict)
        r_list.append(r_nbsi)

    r_dict[tb] = np.array(r_list)
    sol_array = np.vstack(sol_list)
    sol_dict[tb] = sol_array
#==============================================================================
# STEADY STATE PLOT
#==============================================================================
fig, ax = plt.subplots(3, sharex=True)

for a in ax:
    a.set_xscale('log')
    a.set_xlabel('Bias Voltage [V]')
    a.grid(True)

ax[0].set_ylabel('Temperature [K]')
ax[1].set_ylabel('Voltage [V]')
ax[2].set_ylabel('Resistance [$\Omega$]')
ax[1].set_yscale('log')


for k,v in sol_dict.iteritems():

    for i in range(4):
        if i == 3:
            ax[1].plot(v_range, v[:, i], label=str(eth.System.phi_vect[i]))
            continue
        ax[0].plot(v_range, v[:, i], label=str(eth.System.phi_vect[i]))

    ax[2].plot(v_range, r_dict[k], label=k)

