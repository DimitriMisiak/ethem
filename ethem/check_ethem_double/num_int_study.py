#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Test file for the et_scheme.py script

@author: misiak
"""

import matplotlib.pyplot as plt
import numpy as np
import sympy as sy
from scipy.integrate import solve_ivp

### closing previous plot
plt.close('all')




from config_ethem import eth
System = eth.System
from config_ethem import evad, per



x0_ss = [5., 5.]
x0 = eth.solve_sse(evad, x0_ss)

sol_int = eth.num_int(per, evad, x0)
#sol_int = eth.num_int_dep(per, evad, x0)

time, pulse = sol_int[0], sol_int[1:]

sens = max(abs(pulse[-1]))

fig = plt.figure('plot_odeint')
ax = fig.get_axes()
if len(ax) == 0:
    fig, ax = plt.subplots(2, sharex=True, num='plot_odeint')

for i,a in enumerate(ax):
    a.plot(time, pulse[i])
    a.grid(True)

ax[0].set_title('Sensitivity : {:.2f} nV/keV'.format(sens*1e9))
fig.tight_layout()



#
#
#
#L = 10.
#fs = 1e3
#
#time = np.arange(0., L, fs**-1)
#
#phi = System.phi_vect
#param = [System.t] + list(phi)
#
## perturbation
#capa_matrix = System.capacity_matrix
#per_arg = capa_matrix**-1 * per / sy.Heaviside(System.t)
#
#eteq = System.eteq
#
#eq = per_arg + eteq
#eq_num = list(eq.subs(evad))
##eq_fun = sy.lambdify(param, eq_num, modules='numpy')
#
#eq_fun = sy.lambdify(param, eq_num)
#
#def eq_aux(t, y):
#    list_y = list(y)
#    param = [t] + list_y
#    vec = eq_fun(*param)
#    return vec
#
#
#sol = solve_ivp(eq_aux, [0, L], x0, t_eval=time, max_step=10*fs**-1)
##sol = solve_ivp(eq_aux, [0, 1.], x0)
#
## substracting the initial vector
#sol.yy = np.array([s-x for s,x in zip(sol.y, x0)])
#
#plt.figure()
#for y in sol.yy:
#    plt.plot(sol.t, y)
#
#power = [eq_aux(t,x) for t,x in zip(sol.t, sol.y.T)]
#power = np.vstack(power).T
#plt.figure()
#for p in power:
#    plt.plot(sol.t, p)
#







