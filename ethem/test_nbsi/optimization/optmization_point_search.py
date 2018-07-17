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

eq1 = th_nbsi.eq().args[1].subs({waffer.temperature: cryo.temperature})
eq2 = eq1.diff(t_nbsi)
eq3 = eq2.diff(t_nbsi)

eqs = [eq1, eq2]
#eqs = [eq1, eq2, eq3]

evad_opt = evad.copy()
evad_opt.pop(t_cryo)

eqs_aux = list()
for eq in eqs:
    eqs_aux.append(eq.subs({t_nbsi:Tc}))

eqs_num = list()
for eq in eqs_aux:
    eqs_num.append(eq.subs(evad_opt))

param = (i_bias, t_cryo)
eqs_fun = sy.lambdify(param, eqs_num, modules="numpy")

t_array = np.linspace(0.e-3, 20e-3, 20)
i_array = 10**np.linspace(-11, -9.5, 20)

i_mesh, t_mesh = np.meshgrid(i_array, t_array)
eqs_mesh = eqs_fun(i_mesh, t_mesh)

label = ('power', 'conductance', 'ultra')

#for i, mesh in enumerate(eqs_mesh):
#    lab = label[i]

fig = plt.figure('power')
ax = plt.axes(projection='3d')
#ax.contour3D(np.log10(i_mesh), t_mesh, eqs_mesh[0], 100, cmap='jet')
ax.plot_wireframe(np.log10(i_mesh), t_mesh, eqs_mesh[0], cmap='jet', alpha=0.2)
ax.set_xlabel('Current I')
ax.set_ylabel('Temperature T')
ax.set_zlabel('power')

fig2 = plt.figure('conductance')
ax2 = plt.axes(projection='3d')
#ax2.contour3D(np.log10(i_mesh), t_mesh, eqs_mesh[1], 100, cmap='jet')
#ax2.plot_wireframe(np.log10(i_mesh), t_mesh, eqs_mesh[1], cmap='jet', alpha=0.2)
ax2.scatter(np.log10(i_mesh), t_mesh, eqs_mesh[1], cmap='jet', alpha=0.2)
ax2.set_xlabel('Current I')
ax2.set_ylabel('Temperature T')
ax2.set_zlabel('conductance')

t_sol1 = list()
i_sol1 = list()
for i in i_array[i_array<1e-10]:
    aux = lambda t: eqs_fun(i, t)[0]
    sol = root(aux, 0.020)
    if sol.success is True:
        t_sol1.append(*sol.x)
        i_sol1.append(i)

#i_sol1 = np.array(i_sol1)
#t_sol1 = np.array(t_sol1).reshape(i_sol1.shape)
#eq_sol = eqs_fun(i_sol, t_sol)


t_sol2 = list()
i_sol2 = list()
for t in t_array:
    aux = lambda i: eqs_fun(i, t)[0]
    sol = root(aux, 1e-10)
    if sol.success is True:
        t_sol2.append(t)
        i_sol2.append(*sol.x)
#
#t_sol2 = np.array(t_sol2)
#i_sol2 = np.array(i_sol2).reshape(t_sol2.shape)
#eq_sol2 = eqs_fun(i_sol2, t_sol2)

t_sol = t_sol1 + t_sol2
i_sol = i_sol1 + i_sol2

t_sol = np.array(t_sol)
i_sol = np.array(i_sol).reshape(t_sol.shape)
eq_sol = eqs_fun(i_sol, t_sol)
eq_sol = np.array(eq_sol)


stable = eq_sol[1]<=0
i_stable = i_sol[stable]
t_stable = t_sol[stable]
eq_sol_stable = eq_sol[:,stable]

over = eq_sol[1]>0
i_over = i_sol[over]
t_over = t_sol[over]
eq_sol_over = eq_sol[:,over]

ax.scatter(np.log10(i_stable), t_stable, eq_sol_stable[0], color='k')
ax2.scatter(np.log10(i_stable), t_stable, eq_sol_stable[1], color='k')

ax.scatter(np.log10(i_over), t_over, eq_sol_over[0], color='r')
ax2.scatter(np.log10(i_over), t_over, eq_sol_over[1], color='r')




