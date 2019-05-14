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
N = int(L*fs)

a,b,c,d,e,s = sy.symbols('a,b,c,d,e,s', positive=True)

S = 1e-1
t,f = sy.symbols('t,f')

signal_sym = sy.Heaviside(t) *sy.exp(-t/s)

A = signal_fft_sym = sy.fourier_transform(signal_sym, t,f)


time = np.arange(0, L, fs**-1)
#freq = np.linspace(L**-1, fs/2., N/2)
freq = np.fft.fftshift(np.fft.fftfreq(N, fs**-1))


#signal = np.sin(2*np.pi*time*A)
signal = np.exp(-time/S)
signal_fft = np.fft.fftshift(np.fft.fft(signal))
signal_fft_ortho = np.fft.fftshift(np.fft.fft(signal, norm='ortho'))
signal_fft_2 = np.fft.fftshift(np.fft.fft(signal)) / N**0.5
signal_fft_3 = np.fft.fftshift(np.fft.fft(signal)) / N


signal_sym_eval = (signal_sym/sy.Heaviside(t)).subs(s,S)
signal_fun = sy.lambdify(t, signal_sym_eval, 'numpy')
signal_sy = signal_fun(time)

signal_fft_eval = signal_fft_sym.subs(s,S)#.doit()
signal_fft_fun = sy.lambdify(f, signal_fft_eval, 'numpy')
signal_fft_sy = signal_fft_fun(freq)

plt.figure('temp')
plt.plot(time, signal, label='signal')
plt.plot(time, signal_sy, label='signal_sy')

plt.grid(True)
plt.legend()

plt.figure('fft')

for i,SS in enumerate((signal_fft, signal_fft_ortho, signal_fft_2, signal_fft_3)):
#for i,SS in enumerate((signal_fft_3,)):
    plt.plot(freq, np.real(SS)/fs, label='signal real {}'.format(i))
#    plt.plot(freq, np.imag(SS), label='signal imag {}'.format(i))
plt.plot(freq, np.real(signal_fft_sy), label='signal_sy real')
#plt.plot(freq, np.imag(signal_fft_sy), label='signal_sy imag')

plt.yscale('log')
plt.xscale('symlog')
plt.grid(True)
plt.legend()


