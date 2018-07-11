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
from config_ethem import evad, nbsi, cryo, bias, load
import multiprocessing as mp
import time

### closing previous plot
plt.close('all')

def solve_ss(eval_dict, v_bias, t_cryo, x0=None,
             method=None, printsuccess=False):
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

#    eval_dict.update({'V_b':v_bias, 'T_b':t_cryo})
    eval_dict.update({'V_b':v_bias, cryo.temperature:t_cryo})
    sol = eth.solve_sse(eval_dict, x0, method=method, printsuccess=printsuccess)
#    print cryo.temperature.subs(eval_dict), bias.voltage.subs(eval_dict)
    return sol

#==============================================================================
# BUILDING ARRAY OF SOLUTION OF STEADY STATE
#==============================================================================
#t_range = np.linspace(0.016, 0.019, 20)
#t_range = [0.0189]
#t_range = [0.019]
t_range = [0.017]

rt_list = [nbsi.resistivity.subs(evad).subs({'T_nbsi':t}) for t in t_range]
plt.figure('R(T)')
plt.plot(t_range, rt_list, marker='+')
plt.grid()

#%%
v_range = 10**np.linspace(-2, 0, 100)
#v_range = 10**np.linspace(-1, 0, 100)
#v_range = np.linspace(0.1975, 0.2025, 100)
#v_range = np.linspace(0.2025, 0.1975, 100)
#v_range = 10**np.linspace(1, -2, 10)

#sol_dict = dict()
#r_dict = dict()

def worker(tb):
    sol_list = []
    r_list = []
    sens_list = []
#    i0 = v_range[0] /load.resistivity.subs(evad)
#    r_norm = load.resistivity.subs(evad).subs({nbsi.temperature:30e-3})
#    sol = [tb, tb, tb, float(i0/r_norm)]

    sol = [tb, tb, tb, 0.]
#    sol = [19.5e-3, 19.5e-3, 20.3e-3, 20.2e-4]
    XO = [tb, tb, tb, 0.]

    for v in tqdm.tqdm(v_range):
        sol = solve_ss(evad, v, tb, x0=sol, printsuccess=False)
#        sol = solve_ss(evad, v, tb, x0=XO, printsuccess=False)
        sol_list.append(sol)

        # updating the evaluation dictionnary
        ss_dict = {b : v for b,v in zip(eth.System.phi_vect, sol)}

        r_nbsi = nbsi.resistivity.subs(ss_dict).subs(evad)
        r_list.append(r_nbsi)

        sens_nbsi_sy = (nbsi.current * nbsi.resistivity.diff(nbsi.temperature))
        sens_nbsi = sens_nbsi_sy.subs(ss_dict).subs(evad)
        sens_list.append(sens_nbsi)

    sol_array = np.vstack(sol_list)
    r_array = np.array(r_list)
    sens_array = np.array(sens_list)

    return r_array, sol_array, sens_array

#    sol_dict[tb] = sol_array
#    r_dict[tb] = r_array

#pool = mp.Pool()
#pool_res = pool.map(worker, t_range)

pool_res = map(worker, t_range)

# should prevent an unwanted crash by giving some time to the cpus
time.sleep(1)
print 'Multiprocessing Done.'

r_dict = {tb:pool_res[i][0] for i,tb in enumerate(t_range)}
sol_dict = {tb:pool_res[i][1] for i,tb in enumerate(t_range)}
sens_dict = {tb:pool_res[i][2] for i,tb in enumerate(t_range)}

#r_dict[tb] = np.array(r_list)
#sol_dict[tb] = sol_array

#==============================================================================
# STEADY STATE PLOT
#==============================================================================

fig = plt.figure('plot_ss')
ax = fig.get_axes()
if len(ax) == 0:
    fig, ax = plt.subplots(4, sharex=True, num='plot_ss')

for a in ax:
    a.set_xscale('log')
    a.set_xlabel('Bias Voltage [V]')
    a.grid(True)

ax[0].set_ylabel('Temperature [K]')
ax[1].set_ylabel('Voltage [V]')
ax[2].set_ylabel('Resistance [$\Omega$]')
ax[3].set_ylabel('Sens. [$V/K$]')
ax[1].set_yscale('log')


for k,v in sol_dict.iteritems():

    for i in range(4):
        if i == 3:
            ax[1].plot(v_range, v[:, i], label=str(eth.System.phi_vect[i]))
            continue
        ax[0].plot(v_range, v[:, i], label=str(eth.System.phi_vect[i]))

    ax[2].plot(v_range, r_dict[k], label='{:.6f} K'.format(k))
    ax[3].plot(v_range, sens_dict[k], label='{:.6f} K'.format(k))

for a in ax:
    a.grid(True)
    a.legend()
