#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Test file for the et_scheme.py script

@author: misiak
"""

import matplotlib.pyplot as plt

from config_ethem import eth
from config_ethem import evad, per

### closing previous plot
plt.close('all')

#==============================================================================
# STEADY STATE RESOLUTION
#==============================================================================
bath_list = eth.System.bath_list
num_bath = len(bath_list)

sol_ss = eth.solve_sse(evad, x0=[0.018, 0.018, 0.018, 0.])
# updating the evaluation dictionnary
ss_dict = {b : v for b,v in zip(eth.System.phi_vect, sol_ss)}

# new evaluation dictionnary taking updated with the steady state
evad_ss = evad.copy()
evad_ss.update(ss_dict)

sol_int = eth.num_int(per, evad, sol_ss)
time, pulse = sol_int[0], sol_int[1:]

sens = max(abs(pulse[-1]))

fig = plt.figure('plot_odeint')
ax = fig.get_axes()
if len(ax) == 0:
    fig, ax = plt.subplots(num_bath, sharex=True, num='plot_odeint')

for i,a in enumerate(ax):
    a.plot(time, pulse[i])
    a.grid(True)

ax[0].set_title('Sensitivity : {:.2f} nV/keV'.format(sens*1e9/2))
fig.tight_layout()

