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
### absorber thermal bath
abso = eth.ThermalBath('abso')
### ntd phonon bath
phntd = eth.ThermalBath('phonon')
### ntd thermal bath
thntd = eth.ThermalBath('ntd')
### thermal leak
leak = eth.ThermalLink(phntd, cryo, 'leak')
### glue between absorber and ntd
glue = eth.ThermalLink(abso, phntd, 'glue')
### ep coupling
epcoup = eth.ThermalLink(phntd, thntd, 'epcoup')

### Chassis ground
ground = eth.Voltstat('ground')
#ground.voltage = 0
### Bias voltage
bias = eth.Voltstat('bias')
### Wire capacitance
capa = eth.Capacitor('wire')
### Load resistance
load = eth.Resistor(bias, capa, 'load')
### NTD resistance
elntd = eth.Resistor(capa, ground, 'ntd')

savepath = 'results/system_scheme.png'
eth.display_scheme(savepath)

for e in eth.System.elements_list:
    if isinstance(e, eth.RealBath):
        sy.pprint(e.eq(), wrap_line=False)

##==============================================================================
## PHYSICAL RELATIONS AND ADDITIONNAL SYMBOLS
##==============================================================================
#
## temperature of NTD resistor is temperature of the NTD electron bath
#elntd.temperature = thntd.temperature
#
## NTD characteristics
#R0, T0 = sy.symbols('R0, T0')
#elntd.resistivity = eth.ntd_char(R0, T0, thntd.temperature)
#
## Joule Power from NTD resistor to NTD electron bath
#thntd.power = eth.joule_power(capa.voltage, elntd.resistivity)
#
## Volume of absorber from its mass
#D_Ge, abso.mass = sy.symbols('D_Ge, M_a')
#abso.volume = abso.mass / D_Ge
#
## Volume of NTD from its dimensions, same volume for phntd, thntd and elntd
#elntd.height, elntd.length, elntd.width = sy.symbols('H_ntd, L_ntd, W_ntd')
#elntd.volume = elntd.height * elntd.length * elntd.width
#phntd.volume = thntd.volume = elntd.volume
#
## Thermal Capacity expression in germanium
#ce_Ge, cp_Ge = sy.symbols('ce_Ge, cp_Ge')
#abso.th_capacity = abso.volume * cp_Ge * abso.temperature**3
#phntd.th_capacity = phntd.volume * cp_Ge * phntd.temperature**3
#thntd.th_capacity = thntd.volume * ce_Ge * thntd.temperature
#
#
## Power expression in gold link
#leak.surface, leak.cond_alpha, leak.cond_expo = sy.symbols('S_Au, g_Au, n_Au')
#leak.power = eth.kapitsa_power(leak.surface*leak.cond_alpha,
#                               leak.cond_expo,
#                               leak.from_bath.temperature,
#                               leak.to_bath.temperature)
#
## Power expression in glue link
#glue.cond_alpha, glue.cond_expo = sy.symbols('g_glue, n_glue')
#glue.surface = elntd.width * elntd.length
#glue.power = eth.kapitsa_power(glue.surface*glue.cond_alpha,
#                               glue.cond_expo,
#                               glue.from_bath.temperature,
#                               glue.to_bath.temperature)
#
## Power expression in epcoup link
#epcoup.cond_alpha, epcoup.cond_expo = sy.symbols('g_ep, n_ep')
#epcoup.power = eth.kapitsa_power(phntd.volume*epcoup.cond_alpha,
#                                 epcoup.cond_expo,
#                                 epcoup.from_bath.temperature,
#                                 epcoup.to_bath.temperature)
#
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
##==============================================================================
## UPDATING THE SYSTEM
##==============================================================================
#eth.System.build_sym()
#
##==============================================================================
## EVENT PERTURBATION
##==============================================================================
#E, sth, epsa, epse, t0 = sy.symbols('E, sth, epsa, epse, t0')
#per = sy.zeros(len(eth.System.bath_list), 1)
#per[0] = epsa * eth.event_power(E, sth, t)
#per[2] = epse * eth.event_power(E, sth, t)
#
##==============================================================================
## EVALUATION DICT
##==============================================================================
#evad_const = {'kB' : 1.3806485e-23,
#              D_Ge : 5.32,
#              ce_Ge : 1.03e-6,
#              cp_Ge : 2.66e-6,
#              test :1e-20}
#
#evad_sys = {load.resistivity : 2e9,
#            load.temperature :0.02,
#            leak.surface : 40.,
#            glue.cond_alpha : 1.46e-4,
#            glue.cond_expo : 3.5,
#            epcoup.cond_alpha : 45.67,
#            epcoup.cond_expo : 6.,
#            leak.cond_alpha : 7.81e-05,
#            leak.cond_expo : 4.,
#            capa.capacity : 2.94e-10,
#            abso.mass : 255.36,
#            elntd.length :0.3,
#            elntd.width :0.5,
#            elntd.height :0.1,
#            R0 : 0.5,
#            T0 : 5.29,
#            cryo.temperature : 18e-3,
#            bias.voltage : 0.5}
#
#evad_per = {sth : 4.03e-3,
#            E : 1e3 * 1.6e-19,
##            epsa : 1.0-2.02e-1,
##            epse : 2.02e-1,
#            epsa : 1.0,
#            epse : 0.,
#            t0 : 0.0}
#
#evad_noise = {e_amp :3.27e-9,
#              A_LF: 2.99e-8,
#              B_LF: 1.15e-8,
#              i_a1: 1.94e-15,
#              i_a2: 6.12e-16,
#              i_a3: 1.16e-17,
#              e_bias: 2.02e-9}
#
#evad = dict()
#evad.update(evad_const)
#evad.update(evad_sys)
#evad.update(evad_per)
#evad.update(evad_noise)
#
#### checking the completeness of the evaluation dictionnary
## free symbols without evaluation
#free_set = set(eth.System.phi_vect)|{t,f}
#
## checking the electro-thermal equations
#ete_free = eth.System.eteq.subs(evad).free_symbols
#assert ete_free.issubset(free_set)
#
## checking the event perturbation
#per_free = per.subs(evad).free_symbols
#assert per_free.issubset(free_set)
#
## checking the noise power
#for e in eth.System.elements_list:
#
#    if isinstance(e, eth.RealBath):
#
#        for v in e.noise_obs.values():
#            noise_free = v.subs(evad).free_symbols
#            assert noise_free.issubset(free_set)
#
#        for v in e.noise_sys.values():
#            noise_free = v.subs(evad).free_symbols
#            assert noise_free.issubset(free_set)
#
#    if isinstance(e, eth.Link):
#
#        for v in e.noise_flux.values():
#            noise_free = v.subs(evad).free_symbols
#            assert noise_free.issubset(free_set)
