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
nbsi = eth.System.Resistor_nbsi
th_nbsi = eth.System.NBSI_nbsi

t_nbsi = th_nbsi.temperature

i_bias = sy.symbols('i_bias')
capa.voltage = nbsi.resistivity * i_bias

evad_vf = evad.copy()
evad_vf.update({i_bias : 2e-9})

joule = th_nbsi.power
joule_num = joule.subs(evad_vf)
joule_fun = sy.lambdify(t_nbsi, joule_num, modules='numpy')

res = eth.System.Resistor_nbsi.resistivity
res_num = res.subs(evad_vf)
res_fun = sy.lambdify(t_nbsi, res_num, modules='numpy')

current = eth.System.Resistor_nbsi.current
current_num = current.subs(evad_vf)
current_fun = sy.lambdify(t_nbsi, current_num, modules='numpy')

t_array = np.linspace(15e-3, 25e-3, 100)
joule_array = joule_fun(t_array)
res_array = res_fun(t_array)
current_array = current_fun(t_array)

plt.figure('Power')
plt.plot(t_array, joule_array, label='Joule')
plt.plot()
plt.grid()
plt.legend()

plt.figure('Resistance')
plt.plot(t_array, res_array, label='nbsi')
plt.grid()
plt.legend()

plt.figure('Current')
plt.plot(t_array, current_array, label='nbsi')
plt.grid()
plt.legend()






