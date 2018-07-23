#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Test file for the et_scheme.py script

@author: misiak
"""
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import numpy as np
import sympy as sy
from scipy.optimize import minimize, root

from tqdm import tqdm
from config_ethem import eth, Tc
from config_ethem import evad, per, i2u

from tqdm import tqdm

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

param = (i_bias,)
edict = evad.copy()
edict.update({
    t_nbsi:0.018,
    t_cryo: 0.017,
    waffer.temperature: 0.0175
})

for p in param:
    try:
        print edict.pop(p)
    except:
        pass

key_list = list()
value_list = list()
for k,v in edict.iteritems():
    key_list.append(k)
    value_list.append(v)
theta = list(param) + key_list
theta_val = [1e-10] + value_list

eq1 = th_nbsi.eq().args[1]
eq1_fun = sy.lambdify(theta, eq1, modules='numpy')
i_array = 10**np.linspace(-12, -9, 100)

#==============================================================================
# F9 TO LAuNCH in console magic command %%timeit
#==============================================================================
#%%timeit -n 100 -r 3
eq_fun = lambda p: eq1_fun(p, *theta_val[1:])
eq_fun(i_array)

#%%timeit -n 100 -r 3
eq1_aux = eq1.subs(edict)
eq1_aux_fun = sy.lambdify(i_bias, eq1_aux, modules='numpy')
eq1_aux_fun(1e-10)



#eq2 = eq1.diff(t_nbsi)
#eq3 = eq2.diff(t_nbsi)
#
#eqs = [eq1, eq2]
##eqs = [eq1, eq2, eq3]

