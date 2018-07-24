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

from config_nbsi_solo import evad, nbsi, cryo

plt.close('all')


ib = 1e-10
tc = 0.017

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

print eteq_fun(ib, tc, t0)

#time_array = np.linspace(0., 10., 10)
#eteq_aux = lambda y,t: eteq_fun(*y)
#eteq_aux_root = lambda y: eteq_fun(*y)
#
#t0 = np.random.uniform(0.010, 0.030)
#inte = odeint(eteq_aux, [t0], time_array)
#
#t_conv = inte[-1]
#
#sol = root(eteq_aux_root, t_conv)
#
#print sol