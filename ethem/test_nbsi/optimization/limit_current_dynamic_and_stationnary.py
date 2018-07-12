#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Test file for the et_scheme.py script

@author: misiak
"""

import matplotlib.pyplot as plt
import numpy as np
import sympy as sy
from scipy.optimize import minimize, root

from tqdm import tqdm
from config_ethem import eth
from config_ethem import evad, per, i2u

## closing previous plot
plt.close('all')

capa = eth.System.Capacitor_f
th_nbsi = eth.System.NBSI_nbsi
cryo = eth.System.Thermostat_b
waffer = eth.System.ThermalBath_w
abso = eth.System.ThermalBath_a

nbsi = eth.System.Resistor_nbsi
epcoup = eth.System.ThermalLink_ep
leak = eth.System.ThermalLink_leak

t_cryo = cryo.temperature
t_nbsi = th_nbsi.temperature

i_bias = sy.symbols('i_bias')
capa.voltage = nbsi.resistivity * i_bias

fig, ax = plt.subplots(2, sharex=True, num='Power')
ax[0].set_ylabel('Power Term[W]')
ax[1].set_ylabel('Total Power in NbSi[W]')
ax[1].set_xlabel('NbSi Temperature [K]')

figg, axx = plt.subplots(2, sharex=True, num='Conductance')
axx[0].set_ylabel('Conductance Term [W/K]')
axx[1].set_ylabel('Total Conductance [W/K]')
axx[1].set_xlabel('NbSi Temperature [K]')


t_array = np.linspace(10e-3, 50e-3, 1000)

evad_vf = evad.copy()
#evad_vf.update({t_cryo:0.016})
#evad_vf.update({t_cryo:0.018})
evad_vf.update({t_cryo:0.017})

### STAIONNARY
epow = -epcoup.power.subs({waffer.temperature: cryo.temperature})
epow_num = epow.subs(evad_vf)
epow_fun = sy.lambdify(t_nbsi, epow_num, modules='numpy')
epow_array = epow_fun(t_array)
ax[0].plot(t_array, epow_array, label='EP coupling', color='k')

leakpow = leak.power.subs({waffer.temperature: th_nbsi.temperature})
leakpow_num = leakpow.subs(evad_vf)
leakpow_fun = sy.lambdify(t_nbsi, leakpow_num, modules='numpy')
leakpow_array = leakpow_fun(t_array)
ax[0].plot(t_array, leakpow_array, label='Leak', color='k', ls='--')

### DYNAMIC
cond_ep = epow.diff(t_nbsi)
cond_ep_num = cond_ep.subs(evad_vf)
cond_ep_fun = sy.lambdify(t_nbsi, cond_ep_num, modules='numpy')
cond_ep_array = cond_ep_fun(t_array)
axx[0].plot(t_array, cond_ep_array, label='EP coupling', color='k')

cond_leak = leakpow.diff(t_nbsi)
cond_leak_num = cond_leak.subs(evad_vf)
cond_leak_fun = sy.lambdify(t_nbsi, cond_leak_num, modules='numpy')
cond_leak_array = cond_leak_fun(t_array)
axx[0].plot(t_array, cond_leak_array, label='Leak', color='k', ls='--')

i_range = 10**np.linspace(-11, -9, 20)
#i_range = np.linspace(0.8e-10, 3e-10, 20)
#i_range = [2e-10]

cmap = plt.get_cmap('jet')
c_range = [cmap(i) for i in np.linspace(0.1, 0.9, len(i_range))]

for i,c in tqdm(zip(i_range, c_range)):

    evad_vf.update({i_bias : i})

    ### STATIONNARY
    joule = th_nbsi.power
    joule_num = joule.subs(evad_vf)
    joule_fun = sy.lambdify(t_nbsi, joule_num, modules='numpy')

    eq = th_nbsi.eq().args[1].subs({waffer.temperature: cryo.temperature})
    eq_num = eq.subs(evad_vf)
    eq_fun = sy.lambdify(t_nbsi, eq_num, modules='numpy')

    sol = root(eq_fun, [0.016])

#    if sol.success is False:
#        print sol

    joule_array = joule_fun(t_array)
    eq_array = eq_fun(t_array)

    ax[0].plot(t_array, joule_array, label='Joule I={:.2e}'.format(i), color=c)
    ax[1].plot(t_array, eq_array, label='I={:.2e}'.format(i), color=c)
    ax[1].scatter(sol.x, sol.fun, marker='*', color=c, edgecolors='k')
    ax[0].scatter(sol.x, joule_fun(sol.x), marker='*', color=c, edgecolors='k')

    ### DYNAMIC
    cond_joule = joule.diff(t_nbsi)
    cond_joule_num = cond_joule.subs(evad_vf)
    cond_joule_fun = sy.lambdify(t_nbsi, cond_joule_num, modules='numpy')

    eq_dyn = cond_joule - cond_ep
    eq_dyn_num = eq_dyn.subs(evad_vf)
    eq_dyn_fun = sy.lambdify(t_nbsi, eq_dyn_num, modules='numpy')

    cond_joule_array = cond_joule_fun(t_array)
    eq_dyn_array = eq_dyn_fun(t_array)

    axx[0].plot(t_array, cond_joule_array,
               label='Joule I={:.2e}'.format(i), color=c)
    axx[1].plot(t_array, eq_dyn_array, label='I={:.2e}'.format(i), color=c)
    axx[1].scatter(sol.x, eq_dyn_fun(sol.x), marker='*', color=c, edgecolors='k')
    axx[0].scatter(sol.x, cond_joule_fun(sol.x), marker='*', color=c, edgecolors='k')
for a in ax:
    a.grid()
    a.legend(fontsize='xx-small')

ax[0].set_yscale('log')

for a in axx:
    a.grid()
    a.legend(fontsize='xx-small')




