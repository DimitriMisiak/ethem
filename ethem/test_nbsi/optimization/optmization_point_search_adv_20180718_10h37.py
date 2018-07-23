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

sseq_jac = sseq.jacobian(phi_vect)
sseq_jac_gfun = sy.lambdify(param, sseq_jac, modules="numpy")

cond = sseq[2]
cond_gfun = sy.lambdify(param, cond, modules="numpy")

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
print 'sseq_jac :', sseq_jac_gfun(*theta)
print 'numeq :', numeq_gfun(*theta_num)

v_array = 10**np.linspace(-3, 0, 50)
t_array = np.linspace(0.010, 0.019, 50)


def sseq_solve(vbias, tcryo):

    x0 = [tcryo, tcryo, tcryo, 0.]

    def aux(param_ss):
        p = [vbias, tcryo] + list(param_ss)
        return sseq_gfun(*p), sseq_jac_gfun(*p)

    sol = root(aux, x0, jac=True)

#    x_return = np.append(sol.x, sol.fjac[2,2])
    x_return = sol.x

    if sol.success is False:
        x_return = np.ones( len(x_return) ) * np.nan

    return tuple(x_return)

sseq_solve = np.vectorize(sseq_solve)

v_mesh, t_mesh = np.meshgrid(v_array, t_array)

sseq_mesh = sseq_solve(v_mesh, t_mesh)
cond_mesh = cond_gfun(v_mesh, t_mesh, *sseq_mesh)

fig_t = plt.figure('Temperature')
ax_t = plt.axes(projection='3d')
for i,c in enumerate( ('orange', 'green', 'blue') ):
    ax_t.plot_wireframe(np.log10(v_mesh), t_mesh, sseq_mesh[i]-t_mesh,
                        color=c, alpha=0.3)

fig_v = plt.figure('Voltage')
ax_v = plt.axes(projection='3d')
ax_v.plot_wireframe(np.log10(v_mesh), t_mesh, sseq_mesh[-1])

fig = plt.figure('Conductance')
ax = plt.axes(projection='3d')
ax.plot_wireframe(np.log10(v_mesh), t_mesh, cond_mesh)

current = nbsi.current.subs(edict)
current_gfun = sy.lambdify(param, current, "numpy")
current_mesh = current_gfun(v_mesh, t_mesh, *sseq_mesh)

fig = plt.figure('Current')
ax = plt.axes(projection='3d')
ax.plot_wireframe(np.log10(v_mesh), t_mesh, current_mesh-v_mesh/2.e9)






