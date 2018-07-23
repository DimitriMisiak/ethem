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
from scipy.integrate import solve_ivp, odeint

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

v_array = 10**np.linspace(-3, 0, 100)
t_array = np.linspace(0.017, 0.030, 100)
#v_array = [1e-6]
#t_array = [0.017]

def aux(v,t):
    A = lambda param_ss: sseq_gfun(v,t, *param_ss)
    return A

def aux_solve(v,t):
    A = lambda y,t : sseq_gfun(v,t, *y)
    return A

X0 = [0.018, 0.018, 0.018, 0.]

sol_root = root(aux(0.05,0.018), X0)
print 'root :', sol_root.x

#sol_solve = solve_ivp(aux_solve(0.5, 0.018), [0, 10.], [0.020, 0.018, 0.018, 0.],
#                      t_eval=np.linspace(0, 10, 100),
#                      method='LSODA')

time = np.arange(0, 100, 0.1)
sol_odeint = odeint(aux_solve(0.05, 0.018), X0, time)

fig, ax = plt.subplots(nrows=4, num='check solve_ivp')

for i,a in enumerate(ax):
    a.plot(time, sol_odeint.T[i])

fig, ax = plt.subplots(nrows=4, num='check power')
powa = np.array([aux_solve(0.05, 0.018)(x, 0.) for x in sol_odeint])
for i,a in enumerate(ax):
    a.plot(time, powa.T[i])


print 'odeint :', sol_odeint[-1]


#v_sol = list()
#suc_sol = list()
#
#for v,t in tqdm(zip(v_array, t_array)):
#
#    t = 0.016
#    x0 = [t, t, t, 0.]
#
##    def aux(param_ss):
##        p = [i,t] + list(param_ss)
##        return sseq_gfun(*p)
#
#    sol = root(aux(v,t), x0)
#
#    v_sol.append(sol.x[-1])
#    suc_sol.append(sol.success)
#
#v_sol = np.array(v_sol)
#suc_sol = np.array(suc_sol)
#
#plt.figure()
#plt.loglog(v_array[suc_sol], v_sol[suc_sol], color='slateblue')
#plt.loglog(v_array[np.logical_not(suc_sol)],
#                   v_sol[np.logical_not(suc_sol)],
#                   color='red')


