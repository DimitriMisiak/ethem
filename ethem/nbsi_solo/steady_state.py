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
from scipy.integrate import odeint
from scipy.optimize import root
import numpy as np

from config_nbsi_solo import evad

plt.close('all')

phi_vect = eth.System.phi_vect

eteq = eth.System.eteq
eteq_num = eteq.subs(evad)
eteq_list = list(eteq_num)
eteq_fun = sy.lambdify(phi_vect, eteq_list, 'numpy')

eq0 = eteq[0].args[-1].args[0] * sy.Mul(*eteq[0].args[:-1])
eq1 = eteq[0].args[-1].args[1] * sy.Mul(*eteq[0].args[:-1])
eq0_list = [eq0.subs(evad)]
eq1_list = [eq1.subs(evad)]
eq0_fun = sy.lambdify(phi_vect, eq0_list, 'numpy')
eq1_fun = sy.lambdify(phi_vect, eq1_list, 'numpy')


t_array = np.linspace(0.010, 0.030, 1000)

plt.figure('check steady-state odeint')
plt.plot(t_array, eteq_fun(t_array)[0], label='eteq')
plt.plot(t_array, eq0_fun(t_array)[0], label='eq0')
plt.plot(t_array, eq1_fun(t_array)[0], label='eq1')
plt.legend()
plt.grid(True)


time_array = np.linspace(0., 10., 100)
eteq_aux = lambda y,t: eteq_fun(*y)
eteq_aux_root = lambda y: eteq_fun(*y)

for i in xrange(100):
    t0 = np.random.uniform(0.010, 0.030)
    inte = odeint(eteq_aux, [t0], time_array)

    plt.figure('check integration')
    plt.plot(time_array, inte, lw=1.0)
    plt.grid(True)

    t_conv = inte[-1]
    plt.figure('check steady-state odeint')
    plt.scatter(t_conv, 0., marker='*', color='grey', alpha=0.2)

    sol = root(eteq_aux_root, t_conv)
    if not sol.success:
        print sol.success

    plt.figure('check steady-state odeint')
    plt.scatter(sol.x, 0., marker='*', color='red')

