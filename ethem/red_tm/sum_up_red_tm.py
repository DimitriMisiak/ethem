#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Sum up all the possibility of the ethem package first applied to the
nbsi_solo and nbsi_duo detectors.

@author: misiak
"""

# adding ethem module path to the pythonpath
import sys
import tqdm
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as LA
from os.path import dirname, abspath

import ethem as eth

from config_red_tm import evad

plt.close('all')

#==============================================================================
# PARAMETERS
#==============================================================================
L = 1
### PROBLEM, RESOLUTION depends on the window length drastically
# 14557 eV for L=1
# 20535 eV for L=2
# 145070 eV for L=100
### hint at a L**0.5 dependance.

fs = 1e3

#==============================================================================
# STEADY STATE SOLUTION
#==============================================================================
# checking the quantities at steady state
edict = eth.dict_sse(evad)

sol_ss = (eth.solve_sse(evad)).x
# sol_ss is OK
CA = eth.System.ThermalBath_a.th_capacity.subs(edict)
# CA is OK.
CP = eth.System.ThermalBath_p.th_capacity.subs(edict)
# CP is OK.
CE = eth.System.ThermalBath_ntd.th_capacity.subs(edict)
# CE is OK
Gep = eth.System.ThermalLink_ep.conductance.subs(edict)
# Gep is OK
Gglue = eth.System.ThermalLink_glue.conductance.subs(edict)
# Gglue is OK
Gleak = eth.System.ThermalLink_leak.conductance.subs(edict)
# Gleak is OK
R = eth.System.Resistor_ntd.resistivity.subs(edict)
# R is OK

##==============================================================================
## IV CURVES
##==============================================================================
#t_array = np.linspace(0.012, 0.050, 10)
#v_array = 10**np.linspace(np.log10(0.02), np.log10(50), 100)
#
#param = (
#        eth.System.Thermostat_b.temperature,
#        eth.System.Voltstat_b.voltage,
#)
#
#ss_point = eth.solve_sse_param(param, evad)
#
#iv_dict = dict()
#for temp in tqdm.tqdm(t_array):
#    iv_list = list()
#    for volt in v_array:
#        sol = ss_point((temp, volt))
#        iv_list.append(sol.x[-1])
#    iv_dict[temp] = iv_list
#
#### PLOT
#fig_iv = plt.figure('IV curves')
#
#temp_list = iv_dict.keys()
#temp_list.sort()
#for temp in temp_list:
#    plt.loglog(v_array, iv_dict[temp], label='{0:.4f} K'.format(temp))
#
#plt.grid(True)
#fig_iv.legend(loc='right')
#fig_iv.tight_layout()
#
#### TODO:
#    # evidence of instability in the steady-state resolution
#    # at low temperature and high currents
#    # in-depth study needed
#    # a renormalization of the equation might do the trick

#==============================================================================
# SYSTEM PERTURBATION
#==============================================================================
per = eth.System.perturbation

#==============================================================================
# NUMERICAL INTEGRATION (NI)
#==============================================================================
# temporal
ni_time_array, ni_pulse_array = eth.num_int(per.matrix, evad, sol_ss,
                                            L=L, fs=fs)

# pulse amplitude
ni_amp = max(abs(ni_pulse_array[-1]))

# fft freq
ni_freq_fft = eth.temp_to_fft(ni_time_array)
ni_pulse_fft = np.fft.fft(ni_pulse_array)

ni_freq_fftshift = np.fft.fftshift(ni_freq_fft)
ni_pulse_fftshift = np.fft.fftshift(ni_pulse_fft)

# psd freq
ni_freq_psd, ni_pulse_psd = eth.psd(ni_pulse_fft, fs)
##==============================================================================
## FREQUENCY INVERSION
##==============================================================================
edict = eth.dict_sse(evad)

# time and freq array
fi_time_array = np.arange(0, L, fs**-1)
fi_freq_fft = np.fft.fftfreq(int(L*fs), fs**-1)
fi_freq_fftshift = np.fft.fftshift(fi_freq_fft)

# fft freq
fi_pulse_fft = eth.response_event(per.matrix, edict, fs)(fi_freq_fft)
fi_pulse_fftshift = np.fft.fftshift(fi_pulse_fft)

# temporal
fi_pulse_array = np.real(np.fft.ifft(fi_pulse_fft, axis=1))

# pulse amplitude
fi_amp = max(abs(fi_pulse_array[-1]))

# fft psd
fi_freq_psd, fi_pulse_psd = eth.psd(fi_pulse_fft, fs)

#==============================================================================
# TEMPORAL DIAGONALIZATION
#==============================================================================
coup_mat = eth.System.coupling_matrix
coup_mat_num = coup_mat.subs(edict)
coup_mat_eval = np.array(coup_mat_num).astype('float64')

#eigen-values and vectors
eig, proj = LA.eig(coup_mat_eval)
tau_coup = 1.0/np.real(eig)

proj_inv = LA.inv(proj)

import sympy as sy
per_td = (eth.System.capacity_matrix)**-1 * sy.Matrix(per.fraction)

phi_amp = [float((f*per.energy).subs(edict)) for f in per_td]
eig_amp = proj_inv.dot(phi_amp)

td_time_array = np.arange(0, L, fs**-1)

eig_vec = [a*np.exp(-td_time_array/tau) for a,tau in zip(eig_amp, tau_coup)]
#exp_vec = map(lambda x,y: y*np.exp(-t/x), tau, A)
td_pulse_array = proj.dot(eig_vec)

# pulse amplitude
td_amp = max(abs(td_pulse_array[-1]))

# fft freq
td_freq_fft = eth.temp_to_fft(td_time_array)
td_pulse_fft = np.fft.fft(td_pulse_array)

td_freq_fftshift = np.fft.fftshift(td_freq_fft)
td_pulse_fftshift = np.fft.fftshift(td_pulse_fft)

# psd freq
td_freq_psd, td_pulse_psd = eth.psd(td_pulse_fft, fs)

# welch
from scipy.signal import welch
freq_welch, pulse_welch = welch(td_pulse_array, fs,
                                window='boxcar', nperseg=int(L*fs))

freq_welch = freq_welch[1:]
pulse_welch = pulse_welch[:,1:]

#######
tau_therm_eval = float(per.tau_therm[0].subs(edict))

tau_array = np.sort(np.append(tau_therm_eval, tau_coup))

tau_msg = '$\\tau$ [s] = \n'
for t in tau_array:
    tau_msg += ' {:.3e}\n'.format(t)


f0_array = tau_array**-1 / (2*np.pi)
f0_msg = '$f_0$ [Hz] = \n'
for f in f0_array:
    f0_msg += ' {:.2f}\n'.format(f)


#==============================================================================
# TEMPORAL PLOT
#==============================================================================

num = len(eth.System.bath_list)
fig, ax = plt.subplots(nrows=num,
                       ncols=2,
                       sharex=True,
                       num='TEMPORAL PLOT',
                       figsize=(13,9),
                       squeeze=False)

lab = [str(b.main_quant) for b in eth.System.bath_list]

for i in range(num):
    for j in (0,1):
        ni_line, = ax[i,j].plot(ni_time_array, ni_pulse_array[i],
                     label='Numerical Integration \n {:.2e} V/keV'.format(ni_amp))
        fi_line, = ax[i,j].plot(fi_time_array, fi_pulse_array[i],
                     label='Frequency Inversion \n {:.2e} V/keV'.format(fi_amp))
        td_line, = ax[i,j].plot(fi_time_array, td_pulse_array[i],
                     label='Temporal Diagonalization \n {:.2e} V/keV'.format(td_amp))
        ax[i,j].grid(True)
    #    ax[i,0].legend(title=tau_msg)
        ax[i,j].set_xlabel('Time [s]')
        ax[i,j].set_ylabel(lab[i])

        if j==1:
            ax[i,j].set_yscale('log')

handles, labels = fig.gca().get_legend_handles_labels()
fig.legend(handles, labels, title=tau_msg, loc='right')

#fig.tight_layout(rect=(0.,0.,1.,1.))
#fig.show()

print 'Amplitudes : ', ni_amp, fi_amp, td_amp
print tau_msg

#==============================================================================
# FFT PLOT
#==============================================================================
fig, ax = plt.subplots(nrows=num, num='FFT PLOT', figsize=(9,9))

for i in range(num):

    ax[i].plot(ni_freq_fftshift, np.real(ni_pulse_fftshift)[i], label='Real NI')
    ax[i].plot(ni_freq_fftshift, np.imag(ni_pulse_fftshift)[i], label='Imag NI')
    ax[i].plot(fi_freq_fftshift, np.real(fi_pulse_fftshift)[i], label='Real FI')
    ax[i].plot(fi_freq_fftshift, np.imag(fi_pulse_fftshift)[i], label='Imag FI')
    ax[i].plot(td_freq_fftshift, np.real(td_pulse_fftshift)[i], label='Real TD')
    ax[i].plot(td_freq_fftshift, np.imag(td_pulse_fftshift)[i], label='Imag TD')
    ax[i].set_xscale('symlog')
    ax[i].grid(True)
    ax[i].set_title('FFT')
    ax[i].set_xlabel('Frequency [Hz]')
    ax[i].set_ylabel('Temperature [K]')

handles, labels = fig.gca().get_legend_handles_labels()
fig.legend(handles, labels, loc='right')

fig.tight_layout(rect=(0.,0.,0.8,1.))
fig.show()

#==============================================================================
# PSD PLOT
#==============================================================================
fig, ax = plt.subplots(nrows=num, num='PSD PLOT', figsize=(9,9))

# take level on ntd signal at 10Hz as reference
level_ind = np.where(ni_freq_psd==10)[0][0]
ni_level = ni_pulse_psd[-1, level_ind]
fi_level = fi_pulse_psd[-1, level_ind]
td_level = td_pulse_psd[-1, level_ind]
welch_level = pulse_welch[-1, level_ind]

for i in range(num):

    ax[i].plot(ni_freq_psd, ni_pulse_psd[i],
      label='Numerical Integration \n {:.2e} $V^2$/Hz'.format(ni_level),
      drawstyle='steps-mid',)
    ax[i].plot(fi_freq_psd, fi_pulse_psd[i],
      label='Frequency Inversion \n {:.2e} $V^2$/Hz'.format(fi_level),
      drawstyle='steps-mid',)
    ax[i].plot(td_freq_psd, td_pulse_psd[i],
      label='Temporal Diagonalization \n {:.2e} $V^2$/Hz'.format(td_level),
      drawstyle='steps-mid',)
    ax[i].plot(freq_welch, pulse_welch[i],
      label='TD Welch \n {:.2e} $V^2$/Hz'.format(welch_level),
      drawstyle='steps-mid',)
    for f in f0_array:
        ax[i].axvline(f, ls=':', color='k')
    ax[i].set_xscale('log')
    ax[i].set_yscale('log')
    ax[i].grid(True)
    ax[i].set_xlabel('Frequency [Hz]')
    ax[i].set_ylabel('PSD [$K^2/Hz$]')
    ax[i].set_xlim(ni_freq_psd[0]/2, ni_freq_psd[-1] *2)

handles, labels = fig.gca().get_legend_handles_labels()
fig.legend(handles, labels, loc='right', title=f0_msg)

fig.tight_layout(rect=(0.,0.,0.7,1.))
fig.show()

print 'PSD LEvel : ', ni_level, fi_level, td_level, welch_level

#==============================================================================
# NOISE RESPONSE
#==============================================================================
ref_bath = eth.System.Capacitor_f
ref_ind = eth.System.bath_list.index(ref_bath)

inf = 1.
sup = 100.

ref_freq_fft = np.fft.fftfreq(int(L*fs), fs**-1)
ref_pulse_fft = eth.response_event(per.matrix, edict, fs)(fi_freq_fft)
ref_freq_psd, ref_pulse_psd = eth.psd(ref_pulse_fft, fs)

psd_fun_dict = eth.response_noise(edict)
psd_eval_dict = {k:v(ref_freq_psd) for k,v in psd_fun_dict.iteritems()}

obs_fun_dict = eth.measure_noise(ref_bath, edict)
obs_eval_dict = {k:v(ref_freq_psd) for k,v in obs_fun_dict.iteritems()}

full_array = eth.noise_tot_fun(ref_bath, edict)(ref_freq_psd)

#nep_array = full_array / fi_pulse_psd[ref_ind]
nep_freq_array, nep_array = eth.nep_ref(per.matrix, edict, fs, L, ref_bath)

ref_pulse_ft = eth.response_event_ft(per.matrix, edict, fs)(nep_freq_array)[ref_ind]
ref_sensitivity = np.abs(ref_pulse_ft)**2

# experimental nep and sensitivity
exp_pulse_array = td_pulse_array[ref_ind]
exp_pulse_fft = np.fft.fft(exp_pulse_array)

#exp_pulse_fft = fi_pulse_fft[ref_ind]

exp_pulse_ft = exp_pulse_fft[1:nep_freq_array.shape[0]+1] / fs
exp_sensitivity = np.abs(exp_pulse_ft)**2

## true nep search
#
#freq_ps, ps_welch = welch(td_pulse_array, fs,
#                                window='boxcar', nperseg=int(L*fs),
#                                scaling='spectrum')
#
#freq_ps = freq_ps[1:]
#ps_welch = ps_welch[:,1:]

nep_true = full_array / exp_sensitivity



invres_array = 4. / nep_array

inf_index = max(np.where(fi_freq_psd<=inf)[0])
sup_index = min(np.where(fi_freq_psd>=sup)[0])

invres_trapz = invres_array[inf_index:sup_index]
freq_trapz = fi_freq_psd[inf_index:sup_index]

#invres_int = np.trapz(invres_trapz, freq_trapz)
#res = (invres_int)**-0.5

#res = eth.nep_to_res(nep_freq_array, nep_array, (inf, sup))
res = eth.res_ref(per.matrix, edict, fs, L, ref_bath, (inf, sup))

res_msg = 'Resolution : {:.0f} eV'.format(
        res * per.energy.subs(edict) / (1.6e-19)
)
print res_msg

ref_10_ind = np.where(ref_freq_psd==10)[0][0]
signal_level = ref_pulse_psd[-1, ref_10_ind]
noise_level = full_array[ref_10_ind]

print 'Signal Level =', signal_level
print 'Noise_Level =', noise_level
#==============================================================================
# NOISE PSD PLOT
#==============================================================================
fig, ax = plt.subplots(nrows=num, num='NOISE PSD PLOT', figsize=(9,9))

for i in range(num):

    for k,v in psd_eval_dict.iteritems():
        ax[i].plot(fi_freq_psd, v[i], label=k)

    if i == ref_ind:

        for k,v in obs_eval_dict.iteritems():
            ax[i].plot(fi_freq_psd, v, label=k)

        ax[i].plot(fi_freq_psd, full_array, label='full', color='k', lw=2.)

    ax[i].set_xscale('log')
    ax[i].set_yscale('log')
    ax[i].grid(True)
    ax[i].set_xlabel('Frequency [Hz]')
    ax[i].set_ylabel('PSD [$K^2/Hz$]')

handles, labels = fig.gca().get_legend_handles_labels()
fig.legend(handles, labels, loc='right', title=f0_msg)

fig.tight_layout(rect=(0.,0.,0.8,1.))
fig.show()

#==============================================================================
# REF BATH PLOT
#==============================================================================
fig_ref, ax_ref = plt.subplots(nrows=3, num='REF BATH PLOT', figsize=(9,13))

ax_ref[0].set_title('Response PSD ; NEP ; 1/NEP^2')

#ax_ref[0].plot(fi_freq_psd, fi_pulse_psd[ref_ind], label='1keV pulse',
#      color='k', lw=5.)

ax_ref[0].plot(ref_freq_psd, ref_sensitivity, label='1keV pulse',
      color='k', lw=5.)
ax_ref[0].plot(ref_freq_psd, exp_sensitivity, label='exp. 1keV pulse',
      color='k', lw=5.)

for k,v in psd_eval_dict.iteritems():
    ax_ref[0].plot(fi_freq_psd, v[ref_ind], label=k)

for k,v in obs_eval_dict.iteritems():
    ax_ref[0].plot(fi_freq_psd, v, label=k)
ax_ref[0].plot(fi_freq_psd, full_array, label='Tot Noise', color='red', lw=5.)

ax_ref[1].plot(fi_freq_psd, nep_array, label='nep')
ax_ref[1].plot(ref_freq_psd, nep_true, label='true')
ax_ref[1].set_ylabel('NEP')

ax_ref[2].plot(fi_freq_psd, invres_array, label='invres')
ax_ref[2].set_ylabel('$1/NEP^2$')
ax_ref[2].fill_between(freq_trapz, invres_trapz, color='slateblue', alpha=0.4)

for i in range(3):
    ax_ref[i].set_xlabel('Frequency [Hz]')
    ax_ref[i].set_xscale('log')
    ax_ref[i].set_yscale('log')
    ax_ref[i].grid(True)
#    ax_ref[i].legend()
    if i == 2:
        ax_ref[i].legend(title=res_msg)

fig_ref.legend()
fig_ref.tight_layout(rect=(0., 0., 0.8, 1.))
fig_ref.show()


