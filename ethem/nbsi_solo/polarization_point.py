#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Config file for the test detector using NbSi technology.

@author: misiak
"""

import sympy as sy

# adding ethem module path to the pythonpath
import sys
from os.path import dirname
sys.path.append( dirname(dirname(dirname(__file__))) )

import ethem as eth

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from scipy.integrate import odeint
from scipy.optimize import root
import numpy as np

from config_nbsi_solo import evad, nbsi, cryo

plt.close('all')


ib = 1.5e-10
tc = 0.016

#evad.update({nbsi.current : ib,
#             cryo.temperature : tc})


edict = evad.copy()
edict.pop(nbsi.current)
edict.pop(cryo.temperature)

phi_vect = eth.System.phi_vect
param = [nbsi.current, cryo.temperature] + list(phi_vect)

eteq = eth.System.eteq
eteq_num = eteq.subs(edict)
eteq_list = list(eteq_num)
eteq_fun = sy.lambdify(param, eteq_list, 'numpy')

t0 = np.random.uniform(0.010, 0.030)
#t0 = 0.0

time_array = np.linspace(0., 10., 10)

@np.vectorize
def ss_solve(current, temp, t0=0.0):
    eteq_aux0 = lambda y: eteq_fun(current, temp, *y)
    eteq_aux1 = lambda y,t: eteq_aux0(y)

    inte = odeint(eteq_aux1, [t0], time_array)

    t_conv = inte[-1]

    sol = root(eteq_aux0, t_conv)

#    if not sol.success:
#        print ('Success: {}, Current: {:.2e}, Temperature: {:.4e}, '
#               'Init: {:.3f}').format(
#                sol.success, current, temp, t0
#        )

    return sol.x


i_array = 10**np.linspace(-11, -9, 50)
t_array = np.linspace(0.015, 0.020, 50)

sol_t_R = ss_solve(i_array, tc, t0=0.0)
sol_t_L = ss_solve(i_array, tc, t0=0.1)


plt.figure('nbsi temperature')
plt.plot(i_array, sol_t_R, color='red', label='to right')
plt.plot(i_array, sol_t_L, color='blue', label='to left')
plt.xscale('log')
plt.legend()
plt.grid(True, which='both')

sol_t_R = ss_solve(ib, t_array, t0=0.0)
sol_t_L = ss_solve(ib, t_array, t0=0.1)
plt.figure('temp check')
plt.plot(t_array, sol_t_R, color='red', label='to right')
plt.plot(t_array, sol_t_L, color='blue', label='to left')
plt.grid(True, which='both')
plt.legend()


i_mesh, t_mesh = np.meshgrid(i_array, t_array)

sol_mesh_R = ss_solve(i_mesh, t_mesh, t0=0.0)
sol_mesh_L = ss_solve(i_mesh, t_mesh, t0=0.1)

#fig = plt.figure('nbsi temperature meshplot')
#ax = plt.subplot(projection='3d')
#ax.plot_wireframe(np.log10(i_mesh), t_mesh, sol_mesh_R, color='red', alpha=0.3)
#ax.plot_wireframe(np.log10(i_mesh), t_mesh, sol_mesh_L, color='blue', alpha=0.3)


#### NBSI RESISTANCE
#res = nbsi.resistivity
#res_num = res.subs(evad)
#res_fun = sy.lambdify(param, res_num, 'numpy')
#
#res_R = res_fun(i_array, tc, sol_t_R)
#res_L = res_fun(i_array, tc, sol_t_L)
#
#plt.figure('nbsi resistance')
#plt.plot(i_array, res_R, color='red', label='to right')
#plt.plot(i_array, res_L, color='blue', label='to left')
#plt.xscale('log')
#plt.legend()
#plt.grid(True, which='both')
#
#res_mesh_R = res_fun(i_mesh, t_mesh, sol_mesh_R)
#res_mesh_L = res_fun(i_mesh, t_mesh, sol_mesh_L)
#
#fig = plt.figure('nbsi resistance meshplot')
#ax = plt.subplot(projection='3d')
#ax.plot_wireframe(np.log10(i_mesh), t_mesh, res_mesh_R, color='red', alpha=0.3)
#ax.plot_wireframe(np.log10(i_mesh), t_mesh, res_mesh_L, color='blue', alpha=0.3)
#
#### NBSI VOLTAGE
#volt = nbsi.resistivity * nbsi.current
#volt_num = volt.subs(edict)
#volt_fun = sy.lambdify(param, volt_num, 'numpy')
#
#volt_R = volt_fun(i_array, tc, sol_t_R)
#volt_L = volt_fun(i_array, tc, sol_t_L)
#
#plt.figure('nbsi voltage')
#plt.plot(i_array, volt_R, color='red', label='to right')
#plt.plot(i_array, volt_L, color='blue', label='to left')
#plt.xscale('log')
#plt.legend()
#plt.grid(True, which='both')
#
#volt_mesh_R = volt_fun(i_mesh, t_mesh, sol_mesh_R)
#volt_mesh_L = volt_fun(i_mesh, t_mesh, sol_mesh_L)
#
#fig = plt.figure('nbsi voltage meshplot')
#ax = plt.subplot(projection='3d')
#ax.plot_wireframe(np.log10(i_mesh), t_mesh, np.log10(volt_mesh_R), color='red', alpha=0.3)
#ax.plot_wireframe(np.log10(i_mesh), t_mesh, np.log10(volt_mesh_L), color='blue', alpha=0.3)
#

### TOT CONDUCTANCE
cond = eteq.diff(nbsi.temperature)[0]
cond_num = cond.subs(edict)
cond_fun = sy.lambdify(param, cond_num, 'numpy')

cond_R = cond_fun(i_array, tc, sol_t_R)
cond_L = cond_fun(i_array, tc, sol_t_L)

plt.figure('tot conductance')
plt.plot(i_array, cond_R, color='red', label='to right')
plt.plot(i_array, cond_L, color='blue', label='to left')
plt.xscale('log')
plt.legend()
plt.grid(True, which='both')

cond_mesh_R = cond_fun(i_mesh, t_mesh, sol_mesh_R)
cond_mesh_L = cond_fun(i_mesh, t_mesh, sol_mesh_L)

fig = plt.figure('tot conductance meshplot')
ax = plt.subplot(projection='3d')
ax.plot_wireframe(np.log10(i_mesh), t_mesh, cond_mesh_R, color='red', alpha=0.3)
ax.plot_wireframe(np.log10(i_mesh), t_mesh, cond_mesh_L, color='blue', alpha=0.3)

#
#### TOT ULTRA
#ultra = cond.diff(nbsi.temperature)
#ultra_num = ultra.subs(edict)
#ultra_fun = sy.lambdify(param, ultra_num, 'numpy')
#
#ultra_R = ultra_fun(i_array, tc, sol_t_R)
#ultra_L = ultra_fun(i_array, tc, sol_t_L)
#
#plt.figure('tot ultra')
#plt.plot(i_array, ultra_R, color='red', label='to right')
#plt.plot(i_array, ultra_L, color='blue', label='to left')
#plt.xscale('log')
#plt.legend()
#plt.grid(True, which='both')
#
#ultra_mesh_R = ultra_fun(i_mesh, t_mesh, sol_mesh_R)
#ultra_mesh_L = ultra_fun(i_mesh, t_mesh, sol_mesh_L)
#
#fig = plt.figure('tot ultra meshplot')
#ax = plt.subplot(projection='3d')
#ax.plot_wireframe(np.log10(i_mesh), t_mesh, ultra_mesh_R, color='red', alpha=0.3)
#ax.plot_wireframe(np.log10(i_mesh), t_mesh, ultra_mesh_L, color='blue', alpha=0.3)


### TIME CONSTANT
import scipy.linalg as LA

coup_mat = eth.System.coupling_matrix
coup_mat_num = coup_mat.subs(edict)
coup_mat_fun = sy.lambdify(param, coup_mat_num, 'numpy')

@np.vectorize
def tau_fun(current, temp1, temp2):
    coup_mat_eval = coup_mat_fun(current, temp1, temp2)

    eig, P = LA.eig(coup_mat_eval)

    return np.real(1./eig)

tau_R = tau_fun(i_array, tc, sol_t_R)
tau_L = tau_fun(i_array, tc, sol_t_L)

plt.figure('tau')
plt.plot(i_array, tau_R, color='red', label='to right')
plt.plot(i_array, tau_L, color='blue', label='to left')
plt.xscale('log')
plt.legend()
plt.grid(True, which='both')

tau_mesh_R = tau_fun(i_mesh, t_mesh, sol_mesh_R)
tau_mesh_L = tau_fun(i_mesh, t_mesh, sol_mesh_L)

fig = plt.figure('tau meshplot')
ax = plt.subplot(projection='3d')
ax.plot_wireframe(np.log10(i_mesh), t_mesh, np.log10(tau_mesh_R), color='red', alpha=0.3)
ax.plot_wireframe(np.log10(i_mesh), t_mesh, np.log10(tau_mesh_L), color='blue', alpha=0.3)



from corner import corner

tau_fix = tau_mesh_R
tau_fix[tau_fix<0] = 10000.

AAA = np.log10(tau_fix).ravel()
NNN = np.random.normal(size=(900,))

ravel_array = (
        np.log10(i_mesh).ravel(),
        t_mesh.ravel(),
        sol_mesh_R.ravel(),
        AAA,
        cond_mesh_R.ravel(),
)

labels = ('ibias', 'tcryo', 'tnbsi', 'tau', 'cond')

samples = np.vstack(ravel_array)

fig_corner = corner(samples.T, bins=50, smooth=1,
                                labels=['{}'.format(l) for l in labels],
                                quantiles=[0.16, 0.5, 0.84], show_titles=True,
                                title_kwargs={"fontsize": 12})



