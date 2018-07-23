#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Test file for the et_scheme.py script

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
cryo = eth.Thermostat('cryo')
### cryostat
cryo2 = eth.Thermostat('cryo2')
### absorber thermal bath
abso = eth.ThermalBath('abso')

### thermal leak
leak = eth.ThermalLink(abso, cryo, 'leak')
### thermal leak
leak2 = eth.ThermalLink(abso, cryo2, 'leak2')

# DISPLAY SYSTEM
#savepath = 'results/system_scheme.png'
#eth.display_scheme(savepath)

for e in eth.System.elements_list:
    if isinstance(e, eth.RealBath):
        sy.pprint(e.eq(), wrap_line=False)

#==============================================================================
# PHYSICAL RELATIONS AND ADDITIONNAL SYMBOLS
#==============================================================================

## Power expression in gold link
#leak.surface, leak.cond_alpha, leak.cond_expo = sy.symbols('S_Au, g_Au, n_Au')
#leak.power = eth.kapitsa_power(leak.surface*leak.cond_alpha,
#                               leak.cond_expo,
#                               leak.from_bath.temperature,
#                               leak.to_bath.temperature)

leak.cond = sy.symbols('G')
leak.power = leak.cond * (leak.from_bath.temperature - leak.to_bath.temperature)


leak2.cond = sy.symbols('G_2')
leak2.power = leak2.cond * (leak2.from_bath.temperature - leak2.to_bath.temperature)

##==============================================================================
## NOISE POWER
##==============================================================================
#
## TFN noise for each link
#for link in [glue, leak, epcoup]:
#    tfn = eth.tfn_noise(link.conductance,
#                        link.from_bath.temperature,
#                        link.to_bath.temperature)
#    tfn = tfn**0.5 # to obtain the LPSD
#    link.noise_flux['TFN '+link.label] = tfn
#
## Johnson noise for each
#for resi in [load, elntd]:
#    john = eth.johnson_noise(resi.resistivity, resi.temperature)
#    john = john**0.5 # to obtain the LPSD
#    john /= resi.resistivity # to obtain the noise current
#    resi.noise_flux['Johnson '+resi.label] = john
#
## amplifier current noise (impact the system, and so the observer)
#i_a1, i_a2, i_a3 = sy.symbols('i_a1, i_a2, i_a3')
#noise_current = (i_a1**2 + i_a2**2 *f + i_a3**2 *f**2)**0.5
#capa.noise_sys['Ampli. Current'] = noise_current
#
## amplifier voltage noise (impact the observer only)
#e_amp = sy.symbols('e_amp')
#noise_voltage = e_amp
#capa.noise_obs['Ampli. voltage'] = noise_voltage
#
## low-frequency noise (impact the observer only)
#A_LF, B_LF = sy.symbols('A_LF, B_LF')
#noise_lf =  ((A_LF/f)**2 + (B_LF/f**0.5)**2)**0.5
#capa.noise_obs['Low Freq.'] = noise_lf
#
## Bias voltage noise in load resistor
#e_bias = sy.symbols('e_bias')
#bias_noise = e_bias / load.resistivity
#load.noise_flux['Bias Voltage'] = bias_noise
#
## Test noise
#test = sy.symbols('test')
#test_noise = test**0.5
##capa.noise_obs['Test'] = test_noise
#
#
#==============================================================================
# UPDATING THE SYSTEM
#==============================================================================
eth.System.build_sym()

#==============================================================================
# EVENT PERTURBATION
#==============================================================================
E, sth, t0 = sy.symbols('E, sth, t0')
per = sy.zeros(len(eth.System.bath_list), 1)
per[0] = eth.event_power(E, sth, t)

#==============================================================================
# EVALUATION DICT
#==============================================================================
evad_const = {'kB' : 1.3806485e-23}

evad_sys = {leak.cond : 0.2,
            leak2.cond : 0.1,
            abso.capacity : 0.1,
            cryo.temperature : 5.,
            cryo2.temperature : 10.
            }

evad_per = {sth : 4.03e-2,
            E : 1e-5,
            t0 : 0.0
            }

evad = dict()
evad.update(evad_const)
evad.update(evad_sys)
evad.update(evad_per)

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
