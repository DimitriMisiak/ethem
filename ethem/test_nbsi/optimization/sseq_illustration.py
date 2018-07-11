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

### closing previous plot
#plt.close('all')

t_nbsi = eth.System.ThermalBath_nbsi.temperature
v_f = eth.System.Capacitor_f.voltage
v_b = eth.System.Voltstat_b.voltage

evad_tn = evad.copy()
evad_tn.update({t_nbsi : 0.018, v_b: i2u(1e-9,evad)})

capa = eth.System.Capacitor_f

eq_ss = capa.eq().args[1].subs(evad_tn)
eq_ss_fun = sy.lambdify(v_f, eq_ss, modules='numpy')

sol = root(eq_ss_fun, x0=[1e-3])

v_array = 10**np.linspace(-6, -1, 100)

plt.figure('capacitor main_flux')

for l in capa.link_in:
    flux = l.main_flux.subs(evad_tn)
    flux_fun = sy.lambdify(v_f, flux, modules='numpy')
    flux_array = flux_fun(v_array)
    plt.plot(v_array, flux_array, label=l.name, color='red')

for l in capa.link_out:
    flux = l.main_flux.subs(evad_tn)
    flux_fun = sy.lambdify(v_f, flux, modules='numpy')
    flux_array = flux_fun(v_array)
    plt.plot(v_array, flux_array, label=l.name, color='blue')

plt.axvline(sol.x,color='k', ls=':')
plt.xscale('log')
plt.grid()
plt.legend()
#
#
#evad_vf = evad.copy()
#evad_vf.update({v_f : 2e-9 * 2e6})
#
#t_nbsi = eth.System.ThermalBath_nbsi.temperature
#
#joule = eth.System.ThermalBath_nbsi.power
#joule_num = joule.subs(evad_vf)
#joule_fun = sy.lambdify(t_nbsi, joule_num, modules='numpy')
#
#res = eth.System.Resistor_nbsi.resistivity
#res_num = res.subs(evad_vf)
#res_fun = sy.lambdify(t_nbsi, res_num, modules='numpy')
#
#current = eth.System.Resistor_nbsi.current
#current_num = current.subs(evad_vf)
#current_fun = sy.lambdify(t_nbsi, current_num, modules='numpy')
#
#t_array = np.linspace(15e-3, 25e-3, 100)
#joule_array = joule_fun(t_array)
#res_array = res_fun(t_array)
#current_array = current_fun(t_array)
#
#plt.figure('Power')
#plt.plot(t_array, joule_array, label='Joule')
#plt.plot()
#plt.grid()
#plt.legend()
#
#plt.figure('Resistance')
#plt.plot(t_array, res_array, label='nbsi')
#plt.grid()
#plt.legend()
#
#plt.figure('Current')
#plt.plot(t_array, current_array, label='nbsi')
#plt.grid()
#plt.legend()
#





