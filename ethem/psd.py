#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 14:16:29 2018

Script gathering functions related to the psd and fft calculations.

@author: misiak
"""

import numpy as np

def psd(fft, fs, weight=None):
    """
    Computes the Power Spectral Density (PSD) from the Fast Fourier Transform
    (FFT given by numpy.fft.fft).

    Parameters
    ==========
    fft : nd.array
        FFT array whose frequency are ordered as the numpy.fft.fft function
        result (i.e. [0, positive, negative]).
    fs : float
        Sampling frequency.
    weight : None or array_like
        Weights of the frequencies in the psd calculation. If None, the
        weight are all 1 which correponds to the boxcar window.

    Returns
    =======
    freq : nd.array
        Frequency array containing only the positive frequencies (no 0th freq).
    psd : nd.array
        PSD array.
    """
    nfft = fft.shape[0]
    if weight == None:
        s1 = nfft
        s2 = nfft
    else :
        s1 = np.sum(weight)
        s2 = np.sum(np.array(weight)**2)

    # Nyquist frequency
    #fny = float(fs) / 2

    # Frequency resolution
    #fres = float(fs) / nfft

    # Equivalent Noise BandWidth
    enbw = float(fs) * s2 / s1**2

    freq = np.fft.fftfreq(nfft, fs**-1)

    if nfft % 2:
        num_freqs = (nfft + 1)//2
    else:
        num_freqs = nfft//2 + 1
        # Correcting the sign of the last point
        freq[num_freqs-1] *= -1

    freq = freq[1:num_freqs]
    fft = fft[..., 1:num_freqs]

    psd_array = np.abs(fft)**2 / (enbw * s1**2)

    if nfft % 2:
        psd_array[..., :] *= 2
    else:
        # Last point is unpaired Nyquist freq point, don't double
        psd_array[..., :-1] *= 2

    return freq, psd_array

