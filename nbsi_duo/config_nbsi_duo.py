#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Config file for the test detector using NbSi technology.
Basically set up the simulation of the detector.

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
### absorber
abso = eth.ThermalBath('abso')
### nbsi thermal bath
nbsi = eth.ThermalBath('nbsi')
### ep coupling
epcoup = eth.ThermalLink(abso, nbsi, 'ep')
### leak
leak = eth.ThermalLink(cryo, abso, 'leak')

#==============================================================================
# PHYSICAL RELATIONS AND ADDITIONNAL SYMBOLS
#==============================================================================
# Volume of absorber from its mass
D_Ge, abso.mass = sy.symbols('D_Ge, M_a')
abso.volume = abso.mass / D_Ge

# Volume of nbsi from its dimensions, same volume for p_bath, e_bath and nbsi
nbsi.height, nbsi.length, nbsi.width = sy.symbols('H_nbsi, L_nbsi, W_nbsi')
nbsi.section = nbsi.height * nbsi.width
nbsi.volume = nbsi.height * nbsi.length * nbsi.width

# nbsi characteristics
Tc, rho, sig = sy.symbols('Tc, rho, sig')
R_norm = rho * nbsi.length / nbsi.section
nbsi.resistivity = R_norm / (1 + sy.exp(-((nbsi.temperature-Tc)/sig)))

# Thermal Capacity expression in germanium
ce_Ge, cp_Ge = sy.symbols('ce_Ge, cp_Ge')
abso.th_capacity = abso.volume * cp_Ge * abso.temperature**3

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

# Power expression in gold link
leak.surface, leak.cond_alpha, leak.cond_expo = sy.symbols('S_Au, g_Au, n_Au')
leak.power = eth.kapitsa_power(leak.surface*leak.cond_alpha,
                               leak.cond_expo,
                               leak.from_bath.temperature,
                               leak.to_bath.temperature)
# Noise

# TFN noise for each link
for link in eth.System.subclass_list(eth.ThermalLink):
    tfn = eth.tfn_noise(link.conductance,
                        link.from_bath.temperature,
                        link.to_bath.temperature)
    tfn = tfn**0.5 # to obtain the LPSD
    link.noise_flux['TFN '+link.label] = tfn

# amplifier voltage noise (impact the observer only)
e_amp = sy.symbols('e_amp')
noise_voltage = e_amp
nbsi.noise_obs['flat'] = noise_voltage

#==============================================================================
# UPDATING THE SYSTEM
#==============================================================================
eth.System.build_sym()

#==============================================================================
# EVENT PERTURBATION
#==============================================================================
energy, tau_therm, t0 = sy.symbols('E, tau_th, t0')
per = sy.zeros(len(eth.System.bath_list), 1)
per[0] = eth.event_power(energy, tau_therm, time)

#==============================================================================
# EVALUATION DICT
#==============================================================================
evad_const = {
        'kB' : 1.3806485e-23, #J/K
        D_Ge : 5.32e6, #g/m**3
        ce_Ge : 1.03, #J/K**2/m**3
        cp_Ge : 2.66, #J/K**4/m**3
        ce_nbsi : 35, #J/K/m**3
}

evad_sys = {
        epcoup.cond_alpha : 200.e6, #W/K**5/m**3
        epcoup.cond_expo : 5.,
        leak.surface : 1e-7, #m**2
        leak.cond_alpha : 125., #W/K**4/m**2
        nbsi.length :15e-2, #m
        nbsi.width :20e-6, #m
        nbsi.height :50e-9, #m
        leak.cond_expo : 4.,
        abso.mass : 255.36,
        rho : 20e-6, #Ohms/m
        Tc : 0.018, #K
        sig : 0.0005, #K
        cryo.temperature : 16e-3, #K
        nbsi.current : 1.5e-10 #A
}

evad_per = {tau_therm : 4.03e-3, #s
            energy : 1e3 * 1.6e-19, #J
            t0 : 0.0}

evad_noise = {
        e_amp :1e-7
}

evad = dict()
evad.update(evad_const)
evad.update(evad_sys)
evad.update(evad_per)
evad.update(evad_noise)
