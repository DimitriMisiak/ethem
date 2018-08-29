9#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Sum up all the possibility of the ethem package applied to the nbsi_solo simulation.

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
import scipy.signal as sgl
import scipy.linalg as LA

from config_nbsi_solo import evad, nbsi, cryo, per, time, freq, E

from scipy.optimize import minimize





sol = eth.solve_sse(evad, [0.0175])

print sol

#plt.close('all')
#
#
#ib = 2.e-10
#tc = 0.018
#
#edict = evad.copy()
#edict.pop(nbsi.current)
#edict.pop(cryo.temperature)
#
#phi_vect = eth.System.phi_vect
#param = [nbsi.current, cryo.temperature] + list(phi_vect)
#
#eteq = eth.System.eteq
#eteq_num = eteq.subs(edict)
#eteq_list = list(eteq_num)
#eteq_fun = sy.lambdify(param, eteq_list, 'numpy')
#
#
#
#def ss_solve(current, temp, t0=0.0):
#    """ Solve the steady-state.
#
#    Parameters
#    ==========
#    current : float
#        Bias current.
#    temp : float
#        Temperature of the cryostat.
#    t0 : float, optional
#        Starting point for the nbsi temperature search (by default 0K,
#        this is working great)
#
#    Return
#    ======
#    sol.x : float
#        Nbsi temperature solution of the steady-state.
#    """
#    eteq_aux0 = lambda y: eteq_fun(current, temp, *y)
#    eteq_aux1 = lambda y,t: eteq_aux0(y)
#
#    time_ss_array = np.linspace(0., 10., 10)
#    inte = odeint(eteq_aux1, [t0], time_ss_array)
#
#    t_conv = inte[-1]
#
#    sol = root(eteq_aux0, t_conv)
#
#    return sol.x


