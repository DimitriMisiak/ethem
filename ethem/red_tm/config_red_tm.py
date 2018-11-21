#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Config file for the test detector using ntd technology.
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
### absorber thermal bath
abso = eth.ThermalBath('a')
### ntd phonon bath
phntd = eth.ThermalBath('p')
### ntd thermal bath
thntd = eth.ThermalBath('ntd')
### thermal leak
leak = eth.ThermalLink(abso, cryo, 'leak')
### glue between absorber and ntd
glue = eth.ThermalLink(abso, phntd, 'glue')
### ep coupling
epcoup = eth.ThermalLink(phntd, thntd, 'ep')

### Chassis ground
ground = eth.Voltstat('ground')
ground.voltage = 0
### Bias voltage
bias = eth.Voltstat('b')
### Wire capacitance
capa = eth.Capacitor('f')
### Load resistance
load = eth.Resistor(bias, capa, 'L')
### NTD resistance
elntd = eth.Resistor(capa, ground, 'ntd')

#==============================================================================
# PHYSICAL RELATIONS AND ADDITIONNAL SYMBOLS
#==============================================================================
# temperature of NTD resistor is temperature of the NTD electron bath
elntd.temperature = thntd.temperature

# NTD characteristics
R0, T0 = sy.symbols('R0, T0')
elntd.resistivity = eth.ntd_char(R0, T0, thntd.temperature)

# Joule Power from NTD resistor to NTD electron bath
thntd.power = eth.joule_power(capa.voltage, elntd.resistivity)

# Volume of absorber from its mass
D_Ge, abso.mass = sy.symbols('D_Ge, M_a')
abso.volume = abso.mass / D_Ge

# Volume of NTD from its dimensions, same volume for phntd, thntd and elntd
elntd.height, elntd.length, elntd.width = sy.symbols('H_ntd, L_ntd, W_ntd')
elntd.volume = elntd.height * elntd.length * elntd.width
phntd.volume = thntd.volume = elntd.volume

# Thermal Capacity expression in germanium
ce_Ge, cp_Ge = sy.symbols('ce_Ge, cp_Ge')
abso.th_capacity = abso.volume * cp_Ge * abso.temperature**3
phntd.th_capacity = phntd.volume * cp_Ge * phntd.temperature**3
thntd.th_capacity = thntd.volume * ce_Ge * thntd.temperature


# Power expression in gold link
leak.surface, leak.cond_alpha, leak.cond_expo = sy.symbols('S_Au, g_Au, n_Au')
leak.power = eth.kapitsa_power(leak.surface*leak.cond_alpha,
                               leak.cond_expo,
                               leak.from_bath.temperature,
                               leak.to_bath.temperature)

# Power expression in glue link
glue.cond_alpha, glue.cond_expo = sy.symbols('g_glue, n_glue')
glue.power = eth.kapitsa_power(glue.cond_alpha,
                               glue.cond_expo,
                               glue.from_bath.temperature,
                               glue.to_bath.temperature)

# Power expression in epcoup link
epcoup.cond_alpha, epcoup.cond_expo = sy.symbols('g_ep, n_ep')
epcoup.power = eth.kapitsa_power(phntd.volume*epcoup.cond_alpha,
                                 epcoup.cond_expo,
                                 epcoup.from_bath.temperature,
                                 epcoup.to_bath.temperature)

#==============================================================================
# NOISE POWER
#==============================================================================
# TFN noise for each link
for link in [glue, leak, epcoup]:
    tfn = eth.tfn_noise(link.conductance,
                        link.from_bath.temperature,
                        link.to_bath.temperature)
    tfn = tfn**0.5 # to obtain the LPSD
    link.noise_flux['TFN '+link.label] = tfn

# Johnson noise for each
for resi in [load, elntd]:
    john = eth.johnson_noise(resi.resistivity, resi.temperature)
    john = john**0.5 # to obtain the LPSD
    john /= resi.resistivity # to obtain the noise current
    resi.noise_flux['Johnson '+resi.label] = john

# amplifier current noise (impact the system, and so the observer)
i_a1, i_a2, i_a3 = sy.symbols('i_a1, i_a2, i_a3')
noise_current = (i_a1**2 + i_a2**2 *freq + i_a3**2 *freq**2)**0.5
capa.noise_sys['Ampli. Current'] = noise_current

# amplifier voltage noise (impact the observer only)
e_amp = sy.symbols('e_amp')
noise_voltage = e_amp
capa.noise_obs['Ampli. voltage'] = noise_voltage

# low-frequency noise (impact the observer only)
A_LF, B_LF = sy.symbols('A_LF, B_LF')
noise_lf =  ((A_LF/freq)**2 + (B_LF/freq**0.5)**2)**0.5
capa.noise_obs['Low Freq.'] = noise_lf

# Bias voltage noise in load resistor
e_bias = sy.symbols('e_bias')
bias_noise = e_bias / load.resistivity
load.noise_flux['Bias Voltage'] = bias_noise

# Test noise
test = sy.symbols('test')
test_noise = test**0.5
#capa.noise_obs['Test'] = test_noise

#==============================================================================
# UPDATING THE SYSTEM
#==============================================================================
eth.System.build_sym(savepath='output/build_sym')

#==============================================================================
# EVENT PERTURBATION
#==============================================================================
energy, tau_therm, eps = sy.symbols('E, tau_th, eps')

#per = sy.zeros(len(eth.System.bath_list), 1)
#per[0] = eth.event_power(energy*0.01, tau_therm, time)
#per[1] = eth.event_power(energy, tau_therm, time)

#abso.perturbation = eth.event_power(0.01*energy, tau_therm, time)
#ntd.perturbation = eth.event_power(energy, tau_therm, time)
per = eth.Perturbation(energy,
                       [1-eps, 0., eps, 0.],
                       [tau_therm, tau_therm, tau_therm, tau_therm])

#==============================================================================
# EVALUATION DICT
#==============================================================================
evad_const = {'kB' : 1.3806485e-23, #J.K-1
              D_Ge : 5.32, # g.cm-3
              ce_Ge : 1.03e-6, # J.K-2.cm-3
              cp_Ge : 2.66e-6, # J.K-4.cm-3
              test :1e-20}

evad_sys = {load.resistivity : 2e9, # Ohms
            load.temperature :0.02, # K
            leak.surface : 0.25, # cm2
            glue.cond_alpha : 1.66e-7, # W.K-1
            glue.cond_expo : 1., # 1
            epcoup.cond_alpha : 100., # W.K-6.cm-3
            epcoup.cond_expo : 6., # 1
            leak.cond_alpha : 5e-3, # W.K-4.cm-2
            leak.cond_expo : 4., # 1
            capa.capacity : 1e-20, # F
            abso.mass : 820, # g
            elntd.length :0.4, # cm
            elntd.width :0.4, # cm
            elntd.height :0.1, # cm
            R0 : 7.2, # Ohms
            T0 : 3.3, # K
            cryo.temperature : 18e-3, # K
            bias.voltage : 4.54, #V
}

evad_per = {tau_therm : 1e-20, # s
            energy : 1e3 * 1.6e-19, # J
            eps : 0., #fraction
}

evad_noise = {e_amp :3.27e-9,
              A_LF: 2.99e-8,
              B_LF: 1.15e-8,
              i_a1: 1.94e-15,
              i_a2: 6.12e-16,
              i_a3: 1.16e-17,
              e_bias: 2.02e-9}

evad = dict()
evad.update(evad_const)
evad.update(evad_sys)
evad.update(evad_per)
evad.update(evad_noise)

### checking the completeness of the evaluation dictionnary
# free symbols without evaluation
free_set = set(eth.System.phi_vect)|{time,freq}

# checking the electro-thermal equations
ete_free = eth.System.eteq.subs(evad).free_symbols
assert ete_free.issubset(free_set)

# checking the event perturbation
per_free = per.matrix.subs(evad).free_symbols
assert per_free.issubset(free_set)

# checking the noise power
for e in eth.System.elements_list:

    if isinstance(e, eth.RealBath):

        for v in e.noise_obs.values():
            noise_free = v.subs(evad).free_symbols
            assert noise_free.issubset(free_set)

        for v in e.noise_sys.values():
            noise_free = v.subs(evad).free_symbols
            assert noise_free.issubset(free_set)

    if isinstance(e, eth.Link):

        for v in e.noise_flux.values():
            noise_free = v.subs(evad).free_symbols
            assert noise_free.issubset(free_set)

