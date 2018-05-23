#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Test file for the et_scheme.py script

@author: misiak
"""

import matplotlib.pyplot as plt
import numpy as np
import sympy as sy
import tqdm
from scipy.integrate import odeint
import scipy.signal as sgl

### importing ethem module
#import sys
#sys.path.append(u'/home/misiak/Scripts/ETHEM project/')
import ethem_v4 as eth

### closing previous plot
plt.close('all')

DEBUG = 0
def debug(x):
    """ Print if in debug mode.
    """
    if DEBUG:
        print x

#==============================================================================
# SYSTEM
#==============================================================================
### Defining time and frequency variables
t, f = eth.t, eth.f

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

### Defining the thermal system
### cryostat
cryo = eth.Thermostat('b')
### ntd thermal bath
thntd = eth.ThermalBath('ntd')
### ntd phonon bath
phntd = eth.ThermalBath('p')
### absorber thermal bath
abso = eth.ThermalBath('a')
### thermal leak
leak = eth.ThermalLink(phntd, cryo, 'leak')
### glue between absorber and ntd
glue = eth.ThermalLink(abso, phntd, 'glue')
### ep coupling
epcoup = eth.ThermalLink(phntd, thntd, 'ep')


### Bath element list where the equation are defined
bath_list = [abso, phntd, thntd, capa]
debug('bath_list =\n{}\n'.format(bath_list))

### Number of baths
num_bath = len(bath_list)

### temperature vector
phi = eth.phi_vect(bath_list)

#==============================================================================
# TEST EQUATIONS.PY
#==============================================================================
eth.sym_check(bath_list)

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
glue.surface = elntd.width * elntd.length
glue.power = eth.kapitsa_power(glue.surface*glue.cond_alpha,
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
# EVENT PERTURBATION
#==============================================================================
E, sth, epsa, epse, t0 = sy.symbols('E, sth, epsa, epse, t0')
per = sy.zeros(len(bath_list), 1)
per[0] = epsa * eth.event_power(E, sth, t)
per[2] = epse * eth.event_power(E, sth, t)

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
capa.noise_obs['Test'] = test_noise

#==============================================================================
# EVALUATION DICT
#==============================================================================
evad_const = {'kB' : 1.3806485e-23,
              D_Ge : 5.32,
              ce_Ge : 1.03e-6,
              cp_Ge : 2.66e-6,
              test :1e-20}

evad_sys = {load.resistivity : 2e9,
            load.temperature :0.02,
            leak.surface : 40.,
            glue.cond_alpha : 1.46e-4,
            glue.cond_expo : 3.5,
            epcoup.cond_alpha : 45.67,
            epcoup.cond_expo : 6.,
            leak.cond_alpha : 7.81e-05,
            leak.cond_expo : 4.,
            capa.capacity : 2.94e-10,
            abso.mass : 255.36,
            elntd.length :0.3,
            elntd.width :0.5,
            elntd.height :0.1,
            R0 : 0.5,
            T0 : 5.29,
            cryo.temperature : 18e-3,
            bias.voltage : 0.5}

evad_per = {sth : 4.03e-3,
            E : 1e3 * 1.6e-19,
            epsa : 1.0-2.02e-1,
            epse : 2.02e-1,
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
free_set = set(phi)|{t,f}

# checking the electro-thermal equations
ete_free = eth.ete(bath_list).subs(evad).free_symbols
assert ete_free.issubset(free_set)

# checking the event perturbation
per_free = per.subs(evad).free_symbols
assert per_free.issubset(free_set)

# checking the noise power

##==============================================================================
## STEADY STATE PLOT
##==============================================================================
#v_range = 10**np.linspace(-3, np.log10(25), 20)
#quantities = [[elntd.resistivity],
#              [-elntd.resistivity.diff(thntd.temperature) * elntd.current]]
#label = [['NTD Resistance'], ['NTD dV/dT']]
#fig, ax = eth.plot_steady_state(bath_list, evad, bias.voltage, v_range,
#                                quantities=quantities, label=label)
#ax[0].set_yscale('linear')
#ax[-1].set_xlabel('Bias Voltage [V]')
#ax[-2].set_ylabel('Resistance [$\Omega$]')
#ax[-1].set_ylabel('Approx. Sensitivity [V/K]')
#==============================================================================
# STEADY STATE RESOLUTION
#==============================================================================

sol_ss = eth.solve_sse(bath_list, evad)
# updating the evaluation dictionnary
ss_dict = {b.main_quant : v for b,v in zip(bath_list, sol_ss)}

# new evaluation dictionnary taking updated with the steady state
evad_ss = evad.copy()
evad_ss.update(ss_dict)


#==============================================================================
# DISCRETIZATION VARIABLES
#==============================================================================
fs = 1e3
L = 1.
N = int(L * fs)

print 'fs :', fs
print 'L :', L
print 'N :', N

fny = fs/2
fres = fs/N
# array for frequency plot
freq = np.flip(np.arange(fs/2, 0., -fres), axis=0)
# array for numpy.fft calculation
freqnp = np.fft.fftfreq(N, fs**-1)
#time array for plot
time = np.arange(0., L, fs**-1)

#==============================================================================
# GENERAL SYSTEM RESPONSE
#==============================================================================

### numerical integration
capa_matrix = eth.capacity_matrix(bath_list)
per_num = capa_matrix**-1 * per / sy.Heaviside(t)
sol_num = eth.plot_odeint(bath_list, per_num, evad, t, fs, L,
                          plot=False)

### first order
# sensitivity calculation
sens = eth.response_event(bath_list, per, evad_ss, fs)
sv_arraynp = sens(freqnp)
# temporal pulse
pulse = np.real(np.fft.ifft(sv_arraynp, axis=0))
pulse = pulse.T
# psd sensitivity
#sv_mod_array = [eth.psd(sv_arraynp[:,k], fs)[1] for k in range(num_bath)]
# HOTFIX
sv_mod_array = [np.abs(sv_arraynp[1:N/2+1,k]/fs)**2 for k in range(num_bath)]

### noise in first order
tot_lab = 'Total'
psd_fun_dict = eth.response_noise(bath_list, evad_ss)
psd_full = {k:fun(freq) for k, fun in psd_fun_dict.iteritems()}
psd_full[tot_lab] = np.sum(psd_full.values(),axis=0)

### SYSTEM PLOT
fig = plt.figure('system_plot', figsize=(15, 10))
ax = fig.get_axes()
try:
    ax = np.reshape(ax, (4,3))
except:
    fig, ax = plt.subplots(nrows=4, ncols=3,
                           sharex='col', num='system_plot')

for k in range(num_bath):
    sv_bath = sv_arraynp[:, k]
    psd_bath = sv_mod_array[k]

    freqy = np.fft.fftshift(freqnp)
    svy = np.fft.fftshift(sv_bath)

    ax[k][0].plot(freqy, np.real(svy), label='R(fft) perthe')
    ax[k][0].plot(freqy, np.imag(svy), label='I(fft) perthe')
    ax[k][1].plot(freq, psd_bath, label='psd perthe', color='k', lw=2)

    ax[k][2].plot(time, pulse[k], "-", label='PerThe')
    ax[k][2].plot(time, sol_num[k], ":", label='NumRes')

    try :
        for key in psd_full:
            if key == tot_lab:
                ax[k][1].plot(freq, psd_full[key][:,k],label=key,
                              ls='-', lw=2,  color='red')
            else:
                ax[k][1].plot(freq, psd_full[key][:,k],label=key)


    except Exception as e:
        print e

for row in ax:
    for k in range(len(row)):
        row[k].grid(b=True)
        row[k].legend()
        if k in (0,):
            row[k].set_xscale('symlog')
        if k in (1,):
            row[k].set_xscale('log')
        if k in (1,):
            row[k].set_yscale('log')
        if k in (2,):
            row[k].set_xlim(-0.4, 1.0)

ax[0][0].set_title('FFT')
ax[-1][0].set_xlabel('Frequency [Hz]')
ax[0][1].set_title('PSD')
ax[-1][1].set_xlabel('Frequency [Hz]')
ax[0][2].set_title('Temporal')
ax[-1][2].set_xlabel('Time [s]')

fig.tight_layout()
plt.subplots_adjust(hspace=0.0)

#%%
##==============================================================================
## RESPONSE IN REFERENCE BATH
##==============================================================================
### Measured bath / Reference Bath
ref_bath = capa
ref_ind = bath_list.index(ref_bath)

### numerical integration
numres_pulse = sol_num[ref_ind]
numres_pulse *= -1

fft_numres = np.fft.fft(numres_pulse)
#psd_numres = eth.psd(fft_numres, fs)[1]
psd_numres = np.abs(fft_numres[1:N/2+1])**2

### first order
pulse_ref = pulse[ref_ind]
pulse_ref *= -1

psd_ref = sv_mod_array[ref_ind]

print 'Approx. Sensitivity [nV/keV]: ', max(pulse_ref) * 1e9
print 'Approx. Sensitivity [nV/keV]: ', max(numres_pulse) * 1e9
print 'Sensitivity in first bin [V²/Hz]: ', psd_ref[0]

### noise in first order
psd_full_ref = {k:v[:, ref_ind] for k,v in psd_full.iteritems()}

psd_obs_fun = eth.measure_noise(ref_bath, evad_ss)
psd_obs = {k:fun(freq) for k,fun in psd_obs_fun.iteritems()}

psd_full_ref.update(psd_obs)
psd_list_without_tot = [v for k,v in psd_full_ref.iteritems() if k not in (tot_lab, 'Test')]
psd_full_ref[tot_lab] = np.sum(psd_list_without_tot, axis=0)

### REFERENCE BATH PLOT
fig = plt.figure('ref_plot', figsize=(8,8))
ax = fig.get_axes()
try:
    ax = np.reshape(ax, (3,))
except:
    fig, ax = plt.subplots(nrows=3, num='ref_plot')

ax[0].plot(freq, psd_ref, label='PerThe')
ax[0].plot(freq, psd_numres, ls=':', label='PerThe')

for key in psd_full_ref:
    if key == tot_lab:
        ax[0].plot(freq, psd_full_ref[key],label=key,
                      ls='-', lw=2,  color='red')
    else:
        ax[0].plot(freq, psd_full_ref[key],label=key)

for k in (1,2):
    ax[k].plot(time, pulse_ref, label='PerThe')
    ax[k].plot(time, numres_pulse, ls=':', label='NumRes')
    ax[k].set_xlim(-0.3, 1.0)

ax[0].set_xscale('log')

for k in (0,2):
    ax[k].set_yscale('log')

for a in ax:
    a.grid(True)
    a.legend()

fig.tight_layout()

##==============================================================================
## NEP AND RESOLUTION
##==============================================================================
# dictionnary of nep array
nep_dict = {k: v/psd_ref for k,v in psd_full_ref.iteritems()}

# dictionnary of 4/nep array
invres_dict = {k: 4./v for k,v in nep_dict.iteritems()}

inf = list(freq).index(1)
sup = list(freq).index(500)
invres_int = np.trapz(invres_dict[tot_lab][inf:sup], freq[inf:sup])
res = (invres_int)**-0.5

#==============================================================================
# MONITORING PLOT
#==============================================================================

plt.figure('Noise PSD')
for key, psd_array in psd_full_ref.iteritems():
    plt.plot(freq, psd_array, label=key)

# SV
plt.plot(freq, psd_ref, label='SV', lw=3, ls='--', color='k')

plt.legend()
plt.xscale('log')
plt.yscale('log')
plt.grid(True)

plt.figure('NEP^2')
for key in nep_dict:
    nep_array = nep_dict[key]
    plt.plot(freq, nep_array, label=key)
plt.legend()
plt.xscale('log')
plt.yscale('log')
plt.grid(True)

plt.figure('4/NEP^2')
for key in ['Total']:
    invres_array = invres_dict[key]
    plt.plot(freq[inf:sup], invres_array[inf:sup], label=key,
             color='slateblue', lw=2.0)
plt.legend(title='RES= {:.3f} keV'.format(res))
plt.xscale('log')
plt.yscale('log')
plt.grid(True)
