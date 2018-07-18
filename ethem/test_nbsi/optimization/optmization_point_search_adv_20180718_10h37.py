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
from config_ethem import evad, per, i2u, t

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
v_bias = eth.System.Voltstat_b.voltage

#==============================================================================
# PARAM AND EVALUATION DICT
#==============================================================================
edict = evad.copy()
#key_tuple, value_tuple = zip( *list( edict.iteritems() ) )
phi_vect = eth.System.phi_vect
#param = list(key_tuple) + list(phi_vect)
polar = [v_bias, t_cryo]
for p in polar:
    edict.pop(p)

param = polar + list(phi_vect)

#==============================================================================
# SSEQ
#==============================================================================
sseq = eth.System.sseq.subs(edict)
sseq_list = list(sseq)
sseq_gfun = sy.lambdify(param, sseq_list, modules="numpy")

#==============================================================================
# ETEQ + PER
#==============================================================================
eteq = eth.System.eteq
capa_matrix = eth.System.capacity_matrix
per_arg = capa_matrix**-1 * per / sy.Heaviside(t)
numeq = (eteq + per_arg).subs(edict)
numeq_list = list(numeq)
numeq_gfun = sy.lambdify(param+[t], numeq_list, modules="numpy")

#==============================================================================
# TEST
#==============================================================================
x0 = [0.018, 0.018, 0.018, 0.]
theta = [0.05, 0.017] + x0
theta_num = theta + [0.]

print 'sseq :', sseq_gfun(*theta)
print 'numeq :', numeq_gfun(*theta_num)

i_array = 10**np.linspace(-3, 0, 100)
t_array = np.linspace(0.017, 0.030, 100)
#i_array = [1e-6]
#t_array = [0.017]

for i,t in zip(i_array, t_array):

    x0 = [t, t, t, 0.]

    def aux(param_ss):
        p = [i,t] + list(param_ss)
        return sseq_gfun(*p)

    sol = root(aux, x0)
    print sol.success

#i_mesh, t_mesh = np.meshgrid(i_array, t_mesh)
#sseq_mesh

