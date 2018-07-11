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
t, f = eth.System.t, eth.System.f

### Defining the thermal system
### cryostat
cryo = eth.Thermostat('b')
### absorber thermal bath
abso = eth.ThermalBath('a')
### waffer thermal bath
waff = eth.ThermalBath('w')
### nbsi thermal bath
class NBSI(eth.ThermalBath):
    pass
e_bath = NBSI('nbsi')
### Au thermal leak
leak = eth.ThermalLink(waff, cryo, 'leak')
### glue between absorber and waffer
glue = eth.ThermalLink(abso, waff, 'glue')
### ep coupling
epcoup = eth.ThermalLink(waff, e_bath, 'ep')

### Chassis ground
ground = eth.Voltstat('ground')
ground.voltage = 0
### Bias voltage
bias = eth.Voltstat('b')
### Wire capacitance
capa = eth.Capacitor('f')
### Load resistance
load = eth.Resistor(bias, capa, 'L')
### NbSi thermistance
nbsi = eth.Resistor(capa, ground, 'nbsi')

#==============================================================================
# EARLY BUILD FOR DEBUG PURPOSE
#==============================================================================
eth.System.build_sym(savepath='results/check_early')

#==============================================================================
# PHYSICAL RELATIONS AND ADDITIONNAL SYMBOLS
#==============================================================================

# temperature of nbsi resistor is temperature of the nbsi electron bath
nbsi.temperature = e_bath.temperature

# Volume of absorber from its mass
D_Ge, abso.mass = sy.symbols('D_Ge, M_a')
abso.volume = abso.mass / D_Ge

# volume of waffer from its dimension
waff.height, waff.radius = sy.symbols('H_w, R_w')
waff.surface = sy.pi * waff.radius**2
waff.volume = waff.height * waff.surface

# Volume of nbsi from its dimensions, same volume for p_bath, e_bath and nbsi
nbsi.height, nbsi.length, nbsi.width = sy.symbols('H_nbsi, L_nbsi, W_nbsi')
nbsi.section = nbsi.height * nbsi.width
nbsi.volume = nbsi.height * nbsi.length * nbsi.width
e_bath.volume = nbsi.volume

# nbsi characteristics
Tc, rho, sig = sy.symbols('Tc, rho, sig')
R_norm = rho * nbsi.length / nbsi.section
nbsi.resistivity = R_norm / (1 + sy.exp(-((nbsi.temperature-Tc)/sig)))

# Thermal Capacity expression in germanium
ce_Ge, cp_Ge = sy.symbols('ce_Ge, cp_Ge')
abso.th_capacity = abso.volume * cp_Ge * abso.temperature**3
waff.th_capacity = waff.volume * cp_Ge * waff.temperature**3

# Thermal Capacity expression in NbSi
ce_nbsi = sy.symbols('ce_nbsi')
e_bath.th_capacity = e_bath.volume * ce_nbsi

## Joule Power from nbsi resistor to nbsi electron bath
NBSI.power = property(lambda self: eth.joule_power(capa.voltage, nbsi.resistivity))
#e_bath.power = eth.joule_power(capa.voltage, nbsi.resistivity)
print e_bath.power

# Power expression in gold link
leak.surface, leak.cond_alpha, leak.cond_expo = sy.symbols('S_Au, g_Au, n_Au')
leak.power = eth.kapitsa_power(leak.surface*leak.cond_alpha,
                               leak.cond_expo,
                               leak.from_bath.temperature,
                               leak.to_bath.temperature)

# Power expression in glue link
glue.cond_alpha, glue.cond_expo = sy.symbols('g_glue, n_glue')
glue.surface = nbsi.width * nbsi.length
glue.power = eth.kapitsa_power(glue.surface*glue.cond_alpha,
                               glue.cond_expo,
                               glue.from_bath.temperature,
                               glue.to_bath.temperature)

# Power expression in epcoup link
epcoup.cond_alpha, epcoup.cond_expo = sy.symbols('g_ep, n_ep')
epcoup.power = eth.kapitsa_power(e_bath.volume*epcoup.cond_alpha,
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
for resi in [load, nbsi]:
    john = eth.johnson_noise(resi.resistivity, resi.temperature)
    john = john**0.5 # to obtain the LPSD
    john /= resi.resistivity # to obtain the noise current
    resi.noise_flux['Johnson '+resi.label] = john

# amplifier current noise (impact the system, and so the observer)
i_a1, i_a2, i_a3 = sy.symbols('i_a1, i_a2, i_a3')
noise_current = (i_a1**2 + i_a2**2 *f + i_a3**2 *f**2)**0.5
capa.noise_sys['Ampli. Current'] = noise_current

# amplifier voltage noise (impact the observer only)
e_amp = sy.symbols('e_amp')
noise_voltage = e_amp
capa.noise_obs['Ampli. voltage'] = noise_voltage

# low-frequency noise (impact the observer only)
A_LF, B_LF = sy.symbols('A_LF, B_LF')
noise_lf =  ((A_LF/f)**2 + (B_LF/f**0.5)**2)**0.5
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
eth.System.build_sym(savepath='results/check_full')

#==============================================================================
# EVENT PERTURBATION
#==============================================================================
E, sth, epsa, epse, t0 = sy.symbols('E, sth, epsa, epse, t0')
per = sy.zeros(len(eth.System.bath_list), 1)
per[0] = epsa * eth.event_power(E, sth, t)
per[2] = epse * eth.event_power(E, sth, t)

#==============================================================================
# EVALUATION DICT
#==============================================================================
evad_const = {'kB' : 1.3806485e-23, #J/K
              D_Ge : 5.32e6, #g/m**3
              ce_Ge : 1.03, #J/K**2/m**3
              cp_Ge : 2.66, #J/K**4/m**3
              ce_nbsi : 35, #J/K/m**3
              test :1e-20}

evad_sys = {load.resistivity : 2e9, #Ohms
            load.temperature :0.02, #K
            glue.cond_alpha : 1.e2, #W/K**3.5/m**2
            glue.cond_expo : 3.5,
            epcoup.cond_alpha : 200.e6, #W/K**5/m**3
            epcoup.cond_expo : 5.,
            leak.surface : 1e-7, #m**2
            leak.cond_alpha : 125., #W/K**4/m**2
            leak.cond_expo : 4.,
            capa.capacity : 2.94e-10,
            abso.mass : 255.36,
            nbsi.length :15e-2, #m
            nbsi.width : 3.15e-05, #m
            nbsi.height : 2.750e-08, #m
            waff.height : 150e-6, #m
            waff.radius : 22e-3, #m
            rho : 20e-6, #Ohms/m
            Tc : 0.018, #K
            sig : 0.0005, #K
            cryo.temperature : 18e-3, #K
            bias.voltage : 0.1 #V
            }

evad_per = {sth : 4.03e-3, #s
            E : 1e3 * 1.6e-19, #J
            epsa : 1.0,
            epse : 0.,
#            epsa : 0.0001,
#            epse : 1.0,
#            epsa : 1.0-2.02e-1,
#            epse : 2.02e-1,
            t0 : 0.0}

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
free_set = set(eth.System.phi_vect)|{t,f}

# checking the electro-thermal equations
ete_free = eth.System.eteq.subs(evad).free_symbols
assert ete_free.issubset(free_set)

# checking the event perturbation
per_free = per.subs(evad).free_symbols
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

#==============================================================================
# NBSI SPECIFIC FUNCTION
#==============================================================================
def i2u(i, eval_dict):
    """ Return the approximate voltage bias needed to achieve the current i
    into the nbsi.
    """
    rload = eth.System.Resistor_L.resistivity.subs(eval_dict)
    u = i * rload
    return u

def opti(eval_ss):
    """ Determine the optimal safety current for the nbsi.
    """

    pass







