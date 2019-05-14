#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Test file for the et_scheme.py script

@author: misiak
"""

import matplotlib.pyplot as plt
import numpy as np
import sympy as sy
from scipy.optimize import minimize

from tqdm import tqdm
from config_ethem import eth
from config_ethem import evad, per

### closing previous plot
plt.close('all')

#t_range = [0.016, 0.018, 0.020]
t_range = np.linspace(0.010, 0.040, 10)

#cmap = plt.get_cmap('jet')
#
#cmap_range = [cmap(i) for i in np.linspace(0, 1, len(t_range))]

def sens(value, col='slateblue'):

    evad.update({sy.symbols('H_ntd'):value[0]})

    tb = 20e-3
    evad.update({eth.System.Thermostat_b.temperature:tb})

#    evad.update({eth.System.Voltstat_b.voltage:0.})

#    sol_b4 = eth.solve_sse(evad, x0=[tb, tb, tb, 0.])
#    b4_dict = {b : v for b,v in zip(eth.System.phi_vect, sol_b4)}
#    evad_b4 = evad.copy()
#    evad_b4.update(b4_dict)
#
#    g_b4 = eth.System.ThermalLink_ep.conductance.subs(evad_b4)
#
#    sens_b4 = eth.System.Resistor_nbsi.resistivity.diff(
#            eth.System.Resistor_nbsi.temperature
#    ).subs(evad_b4)
#
#    ib = (g_b4/sens_b4)**0.5
#
##    print 'Imax = ', ib
#
#    vmax = eth.System.Resistor_L.resistivity.subs(evad_b4)*ib
#
##    print 'Vmax = ', vmax
#
#    vb = vmax/3
#
##    print 'Vb = ', vb

#    evad.update({eth.System.Voltstat_b.voltage:value[0]})

    bath_list = eth.System.bath_list
    num_bath = len(bath_list)

    sol_ss = eth.solve_sse(evad, x0=[tb, tb, tb, 0.])
    # updating the evaluation dictionnary
    ss_dict = {b : v for b,v in zip(eth.System.phi_vect, sol_ss)}

    # new evaluation dictionnary taking updated with the steady state
    evad_ss = evad.copy()
    evad_ss.update(ss_dict)

#    eth.dynamic_check(evad_ss)

    sol_int = eth.num_int(per, evad, sol_ss, L=1.)
    time, pulse = sol_int[0], sol_int[1:]

    sens = max(abs(pulse[-1]))


    fig = plt.figure('plot_odeint')
    ax = fig.get_axes()
    if len(ax) == 0:
        fig, ax = plt.subplots(num_bath, sharex=True, num='plot_odeint')

    for i,a in enumerate(ax):
        a.plot(time, pulse[i], label='value = {}'.format(value[0]),
               color=col)
        a.grid(True)

#        a.set_yscale('log')
#        a.legend()

    ax[-1].legend()

    ax[0].set_title('Sensitivity : {:.2f} nV/keV'.format(sens*1e9))
    fig.tight_layout()

    print sens*1e9
    return sens*1e9

#### MINIMIZATION
#aux = lambda p: -sens(p)
#res = minimize(aux, [1.], method='nelder-mead', options={'maxiter':10})

### PLOT CHECK
t_range = 10**np.linspace(-2, 1.3, 10)
cmap = plt.get_cmap('jet')
cmap_range = [cmap(i) for i in np.linspace(0, 1, len(t_range))]

slist = []
for t,c in tqdm(zip(t_range, cmap_range)):
    slist.append(sens([t], col=c))

plt.figure()
plt.plot(t_range, slist)
plt.xscale('log')
plt.yscale('log')


