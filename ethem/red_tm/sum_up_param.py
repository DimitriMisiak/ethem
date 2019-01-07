#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Sum up all the possibility of the ethem package first applied to the
nbsi_solo and nbsi_duo detectors.

@author: misiak
"""

# adding ethem module path to the pythonpath
import sys
import tqdm
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as LA
from os.path import dirname, abspath

import ethem as eth

import config_red_tm as config
from config_red_tm import evad

plt.close('all')

#==============================================================================
# PARAMETERS
#==============================================================================
L = 1
fs = 1e3

#==============================================================================
# STEADY STATE SOLUTION
#==============================================================================
# checking the quantities at steady state
edict = eth.dict_sse(evad)

sol_ss = (eth.solve_sse(evad)).x
CA = eth.System.ThermalBath_a.th_capacity.subs(edict)
CP = eth.System.ThermalBath_p.th_capacity.subs(edict)
CE = eth.System.ThermalBath_ntd.th_capacity.subs(edict)
Gep = eth.System.ThermalLink_ep.conductance.subs(edict)
Gglue = eth.System.ThermalLink_glue.conductance.subs(edict)
Gleak = eth.System.ThermalLink_leak.conductance.subs(edict)
R = eth.System.Resistor_ntd.resistivity.subs(edict)

#==============================================================================
# IV CURVES
#==============================================================================
v_array = 10**np.linspace(np.log10(0.02), np.log10(50), 100)

param_plot = (
        eth.System.Thermostat_b.temperature,
        eth.System.ThermalLink_ep.cond_alpha,
        eth.System.ThermalLink_leak.cond_alpha,
        config.R0
)

param_arrays = (
        np.linspace(0.015, 0.050, 10),
        10**np.linspace(1, 3, 10),
        10**np.linspace(-3, -1, 10),
        np.linspace(0.5, 20, 10)
)

### PLOT
nplt = 4
fig_iv, ax_iv = plt.subplots(ncols=2, nrows=2,
                             num='pseudo IV curves', figsize=(11, 11))
ax_iv = ax_iv.ravel()

for ind, ax in enumerate(ax_iv):

    param_sym = (
            eth.System.Voltstat_b.voltage,
            param_plot[ind]
    )

    ss_point = eth.solve_sse_param(param_sym, evad)

    for p in tqdm.tqdm(param_arrays[ind]):

        iv_list = list()
        for volt in v_array:

            sol = ss_point((volt, p))
            iv_list.append(sol.x[-1])

        ax.loglog(v_array, iv_list, label='{0:.4f}'.format(p))

    ax.grid(True)
    ax.set_title(str(param_plot[ind]))
    ax.set_ylabel('Capa voltage [V]')
    ax.set_xlabel('Bias Voltage [V]')
    ax.legend(loc='right')

fig_iv.tight_layout()

### TODO:
    # evidence of instability in the steady-state resolution
    # at low temperature and high currents
    # in-depth study needed
    # a renormalization of the equation might do the trick

