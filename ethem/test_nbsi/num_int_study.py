#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Test file for the et_scheme.py script

@author: misiak
"""

import matplotlib.pyplot as plt
import numpy as np
import sympy as sy

from config_ethem import eth
from config_ethem import evad, per

### closing previous plot
plt.close('all')

##==============================================================================
## EXPLORE BIAS VOLTAGE AT ONE TEMP.
##==============================================================================
##v_range = [1., 0.1, 0.01, 0.001, 0.0001]
#v_range = np.linspace(0., 0.2, 10)
##v_range = [0.16]
#for v in v_range:
#    evad.update({eth.System.Voltstat_b.voltage:v})
#
#    bath_list = eth.System.bath_list
#    num_bath = len(bath_list)
#
#    sol_ss = eth.solve_sse(evad, x0=[0.018, 0.018, 0.018, 0.])
#    # updating the evaluation dictionnary
#    ss_dict = {b : v for b,v in zip(eth.System.phi_vect, sol_ss)}
#
#    # new evaluation dictionnary taking updated with the steady state
#    evad_ss = evad.copy()
#    evad_ss.update(ss_dict)
#
#    eth.dynamic_check(evad_ss)
#
#    sol_int = eth.num_int(per, evad, sol_ss, L=1.)
#    time, pulse = sol_int[0], sol_int[1:]
#
#    sens = max(abs(pulse[-1]))
#
#    fig = plt.figure('plot_odeint')
#    ax = fig.get_axes()
#    if len(ax) == 0:
#        fig, ax = plt.subplots(num_bath, sharex=True, num='plot_odeint')
#
#    for i,a in enumerate(ax):
#        a.plot(time, pulse[i], label='{0:.3f} V'.format(v))
#        a.grid(True)
#
#        a.set_yscale('log')
#        a.legend()
#
#    ax[0].set_title('Sensitivity : {:.2f} nV/keV'.format(sens*1e9))
#    fig.tight_layout()


##==============================================================================
## EXPLORE TEMP. AT SAFETY OPTIMAL BIAS VOLTAGE
##==============================================================================
##t_range = [0.016, 0.018, 0.020]
#t_range = np.linspace(0.016, 0.020, 10)
#
#cmap = plt.get_cmap('jet')
#
#cmap_range = [cmap(i) for i in np.linspace(0, 1, len(t_range))]
#
#for tb,col in zip(t_range, cmap_range):
#    evad.update({eth.System.Thermostat_b.temperature:tb})
#
#    evad.update({eth.System.Voltstat_b.voltage:0.})
#
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
#    print 'Imax = ', ib
#
#    vmax = eth.System.Resistor_L.resistivity.subs(evad_b4)*ib
#
#    print 'Vmax = ', vmax
#
#    vb = vmax/3
#
#    print 'Vb = ', vb
#
#    evad.update({eth.System.Voltstat_b.voltage:vb})
#
#    bath_list = eth.System.bath_list
#    num_bath = len(bath_list)
#
#    sol_ss = eth.solve_sse(evad, x0=[tb, tb, tb, 0.])
#    # updating the evaluation dictionnary
#    ss_dict = {b : v for b,v in zip(eth.System.phi_vect, sol_ss)}
#
#    # new evaluation dictionnary taking updated with the steady state
#    evad_ss = evad.copy()
#    evad_ss.update(ss_dict)
#
#    eth.dynamic_check(evad_ss)
#
#    sol_int = eth.num_int(per, evad, sol_ss, L=10.)
#    time, pulse = sol_int[0], sol_int[1:]
#
#    sens = max(abs(pulse[-1]))
#
#    fig = plt.figure('plot_odeint')
#    ax = fig.get_axes()
#    if len(ax) == 0:
#        fig, ax = plt.subplots(num_bath, sharex=True, num='plot_odeint')
#
#    for i,a in enumerate(ax):
#        a.plot(time, pulse[i], label='at {0:.4f} : {1:.3f} V'.format(tb, vb),
#               color=col)
#        a.grid(True)
#
##        a.set_yscale('log')
##        a.legend()
#
#    ax[-1].legend()
#
#    ax[0].set_title('Sensitivity : {:.2f} nV/keV'.format(sens*1e9))
#    fig.tight_layout()


#==============================================================================
# EXPLORE TRANSITION TEMP. AT SAFETY OPTIMAL BIAS VOLTAGE
#==============================================================================
#t_range = [0.016, 0.018, 0.020]
t_range = np.linspace(0.010, 0.040, 10)

cmap = plt.get_cmap('jet')

cmap_range = [cmap(i) for i in np.linspace(0, 1, len(t_range))]

for tb,col in zip(t_range, cmap_range):
    evad.update({sy.symbols('Tc'):tb})
    evad.update({eth.System.Thermostat_b.temperature:tb})

    evad.update({eth.System.Voltstat_b.voltage:0.})

    sol_b4 = eth.solve_sse(evad, x0=[tb, tb, tb, 0.])
    b4_dict = {b : v for b,v in zip(eth.System.phi_vect, sol_b4)}
    evad_b4 = evad.copy()
    evad_b4.update(b4_dict)

    g_b4 = eth.System.ThermalLink_ep.conductance.subs(evad_b4)

    sens_b4 = eth.System.Resistor_nbsi.resistivity.diff(
            eth.System.Resistor_nbsi.temperature
    ).subs(evad_b4)

    ib = (g_b4/sens_b4)**0.5

    print 'Imax = ', ib

    vmax = eth.System.Resistor_L.resistivity.subs(evad_b4)*ib

    print 'Vmax = ', vmax

    vb = vmax/3

    print 'Vb = ', vb

    evad.update({eth.System.Voltstat_b.voltage:vb})

    bath_list = eth.System.bath_list
    num_bath = len(bath_list)

    sol_ss = eth.solve_sse(evad, x0=[tb, tb, tb, 0.])
    # updating the evaluation dictionnary
    ss_dict = {b : v for b,v in zip(eth.System.phi_vect, sol_ss)}

    # new evaluation dictionnary taking updated with the steady state
    evad_ss = evad.copy()
    evad_ss.update(ss_dict)

    eth.dynamic_check(evad_ss)

    sol_int = eth.num_int(per, evad, sol_ss, L=10.)
    time, pulse = sol_int[0], sol_int[1:]

    sens = max(abs(pulse[-1]))

    fig = plt.figure('plot_odeint')
    ax = fig.get_axes()
    if len(ax) == 0:
        fig, ax = plt.subplots(num_bath, sharex=True, num='plot_odeint')

    for i,a in enumerate(ax):
        a.plot(time, pulse[i], label='at {0:.4f} : {1:.3f} V'.format(tb, vb),
               color=col)
        a.grid(True)

#        a.set_yscale('log')
#        a.legend()

    ax[-1].legend()

    ax[0].set_title('Sensitivity : {:.2f} nV/keV'.format(sens*1e9))
    fig.tight_layout()