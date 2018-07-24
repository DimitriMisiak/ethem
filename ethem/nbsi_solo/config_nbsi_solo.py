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

#==============================================================================
# SYSTEM
#==============================================================================
### Defining time and frequency variables
time, freq = eth.System.time, eth.System.freq

### Defining the thermal system
### cryostat
cryo = eth.Thermostat('b')
### nbsi thermal bath
nbsi = eth.ThermalBath('nbsi')
### ep coupling
epcoup = eth.ThermalLink(cryo, nbsi, 'ep')

#==============================================================================
# PHYSICAL RELATIONS AND ADDITIONNAL SYMBOLS
#==============================================================================
# Volume of nbsi from its dimensions, same volume for p_bath, e_bath and nbsi
nbsi.height, nbsi.length, nbsi.width = sy.symbols('H_nbsi, L_nbsi, W_nbsi')
nbsi.section = nbsi.height * nbsi.width
nbsi.volume = nbsi.height * nbsi.length * nbsi.width

# nbsi characteristics
Tc, rho, sig = sy.symbols('Tc, rho, sig')
R_norm = rho * nbsi.length / nbsi.section
nbsi.resistivity = R_norm / (1 + sy.exp(-((nbsi.temperature-Tc)/sig)))

# Thermal Capacity expression in NbSi
ce_nbsi = sy.symbols('ce_nbsi')
nbsi.th_capacity = nbsi.volume * ce_nbsi

# Joule Power from nbsi resistor to nbsi electron bath
nbsi.current = sy.symbols('i_bias')
nbsi.power = nbsi.resistivity * nbsi.current**2

# Power expression in epcoup link
epcoup.cond_alpha, epcoup.cond_expo = sy.symbols('g_ep, n_ep')
epcoup.power = eth.kapitsa_power(nbsi.volume*epcoup.cond_alpha,
                                 epcoup.cond_expo,
                                 epcoup.from_bath.temperature,
                                 epcoup.to_bath.temperature)


#==============================================================================
# UPDATING THE SYSTEM
#==============================================================================
eth.System.build_sym()

#==============================================================================
# EVALUATION DICT
#==============================================================================
evad_const = {
        'kB' : 1.3806485e-23, #J/K
        ce_nbsi : 35, #J/K/m**3
}

evad_sys = {
        epcoup.cond_alpha : 200.e6, #W/K**5/m**3
        #            epcoup.cond_alpha : 200.e7, #W/K**5/m**3
        epcoup.cond_expo : 5.,
        nbsi.length :15e-2, #m
        nbsi.width :20e-6, #m
        nbsi.height :50e-9, #m
        rho : 20e-6, #Ohms/m
        Tc : 0.018, #K
        sig : 0.0005, #K
        cryo.temperature : 16e-3, #K
        nbsi.current : 1.5e-10 #A
}

evad = dict()
evad.update(evad_const)
evad.update(evad_sys)
