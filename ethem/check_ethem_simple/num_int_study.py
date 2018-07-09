#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Test file for the et_scheme.py script

@author: misiak
"""

import matplotlib.pyplot as plt
import numpy as np
import sympy as sy
from scipy.integrate import odeint, solve_ivp

from config_ethem import eth
from config_ethem import evad, per


### closing previous plot
plt.close('all')

#==============================================================================
# STEADY STATE RESOLUTION
#==============================================================================
bath_list = eth.System.bath_list
num_bath = len(bath_list)

sol_ss = eth.solve_sse(evad, x0=[1.])
# updating the evaluation dictionnary
ss_dict = {b : v for b,v in zip(eth.System.phi_vect, sol_ss)}

# new evaluation dictionnary taking updated with the steady state
evad_ss = evad.copy()
evad_ss.update(ss_dict)

sol_int = eth.num_int(per, evad, sol_ss)
time, pulse = sol_int[0], sol_int[1]

#################################
fs = 1e3
L = 10.
System = eth.System
eval_dict = evad

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
funky_origin = lambda x, t: eteq_f(t, x).flatten() + per_lambda(t, x).flatten()
x0 = sol_ss

#funky = lambda T, t: -(T-10.)
#x0 = sol_ss[0]

trange = np.arange(0, L, fs**-1)

event = [funky_origin(x0, t) for t in trange]
plt.plot(trange, event)

#sol = odeint(funky, x0, trange, printmessg=True,
#             rtol=1e-15, atol=1e-15, hmax=fs**-1)

#sol = odeint(funky_origin, x0, trange, printmessg=True, rtol=1e-15, hmax=fs**-1)
sol = solve_ivp(funky_origin,[0,10], x0, max_step=10* fs**-1)

# substracting the initial vector
sol_per = sol.y-x0

plt.figure()
plt.plot(sol.t, sol_per[0])
#plt.plot(trange, sol)

#sol_array = np.insert(sol_per.T, 0, time, axis=0)






















#sens = max(abs(pulse[-1]))
#
#fig = plt.figure('plot_odeint')
#ax = fig.get_axes()
#if len(ax) == 0:
#    fig, ax = plt.subplots(num_bath,
#                           sharex=True,
#                           num='plot_odeint',
#                           squeeze=False)
#
#for i,a in enumerate(ax[0]):
#    a.plot(time, pulse[i])
#    a.grid(True)
#
#ax[0][0].set_title('Sensitivity : {:.2f} nV/keV'.format(sens*1e9))
#fig.tight_layout()
