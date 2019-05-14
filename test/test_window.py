#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Test of the normalization of the FFT and PSD functions of ethem and numpy and
simpy.

@author: misiak
"""

import numpy as np
import scipy.signal as sgl
import sympy as sy
import matplotlib.pyplot as plt

import matplotlib.patheffects as pe

stroke = [pe.Stroke(linewidth=2., foreground='k'),
             pe.Normal()]

stroke_anti = [pe.Stroke(linewidth=2., foreground='white'),
             pe.Normal()]

#import ethem as eth

plt.close('all')

fs = 1e3
L = 100.

#scaling = 'density'
scaling = 'spectrum'

colors=('slateblue', 'crimson', 'orange')

#for i,L in enumerate((100, 10, 1)):
#
#    color = colors[i]
#
#    N = int(fs*L)
#
#    time = np.arange(0, L, fs**-1)
#
#    noise = np.random.normal(loc=0., scale=1.0, size=N)
#    freq, noise_psd = sgl.welch(noise, fs=fs, window='boxcar', nperseg=N,
#                                scaling=scaling)
#    freq = freq[1:]
#    noise_psd = noise_psd[1:]
#
##    signal = 10*np.exp(-time/1e-2)
#    signal =10*np.sin(2*np.pi*time*10)
#    freq, signal_psd = sgl.welch(signal, fs=fs, window='boxcar', nperseg=N,
#                                 scaling=scaling)
#    freq = freq[1:]
#    signal_psd = signal_psd[1:]
#
#    data = signal + noise
#    freq, data_psd = sgl.welch(data, fs=fs, window='boxcar', nperseg=N,
#                               scaling=scaling)
#    freq = freq[1:]
#    data_psd = data_psd[1:]
#
#    # TEMP
#    plt.figure('temp')
#
#
#    plt.plot(time, noise, color=color, label=L, ls='--', lw=0.1)
#    plt.plot(time, data, label=L, color=color)
#    plt.plot(time, signal, label=L, color=color, path_effects=stroke)
#
#
#    plt.legend()
#
#    # PSD
#    plt.figure('psd')
#    plt.loglog(freq, noise_psd, color=color, ls='--', lw=0.1)
#    plt.loglog(freq, signal_psd, color=color, label=L, path_effects=stroke, zorder=10)
#    plt.loglog(freq, data_psd, color=color, label=L)
#    plt.axhline(np.mean(noise_psd), color=color, lw=2, path_effects=stroke_anti)
#    plt.legend()
#
#
#
#print 'done'

N = int(fs*L)
time = np.arange(0, L, fs**-1)
signal = 10*np.sin(2*np.pi*time*101.21256484)

freq, signal_psd = sgl.welch(signal, fs=fs, window='boxcar', nperseg=N,
                             scaling=scaling)

freq, signal_psd_2 = sgl.welch(signal, fs=fs, window='hanning', nperseg=N,
                             scaling=scaling)

freq = freq[1:]
signal_psd = signal_psd[1:]
signal_psd_2 = signal_psd_2[1:]

plt.figure('psd')
plt.plot(freq, signal_psd, label='boxcar', path_effects=stroke, zorder=10)
plt.plot(freq, signal_psd_2, label='hanning', path_effects=stroke, zorder=10)
plt.yscale('log')
plt.grid()
plt.legend()

