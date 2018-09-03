#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: misiak

Functions used to complete a first-oder perturbative theory simulation.
"""

import numpy as np
import sympy as sy

from .evaluation import lambda_fun_mat, lambda_fun
from .core_classes import System
from .noise import noise_flux_fun, noise_obs_fun


def impedance_matrix_fun(eval_dict):
    """ Return a function accepting an numpy.array with the broadcasting
    ability of numpy.
    The function returns the response matrix, or complex impedance matrix
    for the given frequancies.

    Parameters
    ==========
    eval_dict : dict
        Evaluation dictionnary in first oder approximation i.e. evaluated
        for the main_quant in phi_vect.
    frange : 1d numpy.ndarray
        Numpy array of the frequencies where to evaluate the complex impedance
        function.

    Returns
    =======
    cimeq_fun :Function taking an numpy.array as parameter and returnign an
        array of matrices, mimicking the broadcasting ability of numpy.
    """
    admat = System.admittance_matrix
    admat_num = admat.subs(eval_dict)
    admat_funk = sy.lambdify(System.freq, admat_num, modules="numpy")

    admat_fun = lambda f: np.linalg.inv(lambda_fun_mat(admat_funk, f))

    return admat_fun


def per_fft(per):
    """ Return the symbolic expression of the fourier transform of the event
    perturbation.

    Parameters
    ==========
    per : sympy.matrices.dense.MutableDenseMatrix
        Event perturbation, should be a function of time.

    Return
    ======
    perf : sympy.matrices.dense.MutableDenseMatrix
        Fourier transform of the event perturbation, is a function of
        frequency.
    """
    perf = sy.zeros(*per.shape)

    # apply the fourier transform on each term
    for k, p in enumerate(per):
        perf[k] = sy.fourier_transform(p, System.time, System.freq)

    return perf


def per_fft_fun(per, eval_dict, fs):
    """ Return a function accepting a numpy.array with the broadcasting
    ability of numpy.
    The function returns the event perturbation for the given frequencies.

    Parameters
    ==========
    per : sympy.matrices.dense.MutableDenseMatrix
        Event perturbation, should be a function of time.
    eval_dict : dict
        Evaluation dictionnary in first oder approximation i.e. evaluated
        for the main_quant in phi_vect.
    fs : float
        Sampling frequency.

    Return
    ======
    perf_fun_array : Function taking an numpy.array as parameter and
        returnign an array of matrices,
        mimicking the broadcasting ability of numpy.
    """
    perf = per_fft(per)
    perf_num = perf.subs(eval_dict) * fs
    perf_fun_simple = sy.lambdify(System.freq, perf_num, modules="numpy")

    perf_fun_array = lambda frange: lambda_fun(perf_fun_simple, frange)

    return perf_fun_array


def response_event(per, eval_dict, fs):
    """ Return the response function of the system to a given perturbation
    in the frequency space.

    Parameters
    ==========
    per : Sympy matrix
        Power perturbation of the system. Its shape must matches the one
        of System.admittance_matrix**-1
    eval_dict : dict
        Evaluation dictionnary in first oder approximation i.e. evaluated
        for the main_quant in phi_vect.
    fs : float
        Sampling frequency.

    Return
    ======
    sens_fun : function
        Taking the frequency array as parameter, return the response array.
    """
    cimeq_fun = impedance_matrix_fun(eval_dict)

    perf_fun = per_fft_fun(per, eval_dict, fs)

    def sens_fun(frange):
        cimeq_array = cimeq_fun(frange)

        perf_array = perf_fun(frange)

        sv_array = np.einsum('ijk, ik -> ij', cimeq_array, perf_array)

        return sv_array.T

    return sens_fun


def response_noise(eval_dict):
    """ Return the response function of the system to noise psd
    perturbation in the frequency space.

    Parameters
    ==========
    eval_dict : dict
        Evaluation dictionnary in first oder approximation i.e. evaluated
        for the main_quant in phi_vect.

    Return
    ======
    psd_fun_dict : dict of function
        Key are a string specifying the noise source, and the values are
        the corresponding response function taking as parameter the frequency
        array and returning the response array.
    """
    cimeq_fun = impedance_matrix_fun(eval_dict)

    noise_fun_dict = noise_flux_fun(eval_dict)

    psd_fun_dict = dict()

    def psd_fun_maker(n_fun):

        def psd_fun(frange):

            cimeq_array = cimeq_fun(frange)

            lpsd_array = n_fun(frange)

            impact_array = np.einsum('ijk, ik -> ij', cimeq_array, lpsd_array)

            # computing the psd from the fft
            psd_array = np.abs(impact_array)**2

            return psd_array

        return psd_fun

    for key, noise_fun in noise_fun_dict.iteritems():

        psd_fun_dict[key] = psd_fun_maker(noise_fun)

    return psd_fun_dict


def measure_noise(ref_bath, eval_dict):
    """ pass
    """
    noise_fun_dict = noise_obs_fun(ref_bath, eval_dict)

    psd_fun_dict = dict()

    def psd_fun_maker(n_fun):

        def psd_fun(frange):

            lpsd_array = n_fun(frange)

            # computing the psd from the fft
            psd_array = np.abs(lpsd_array)**2

            return psd_array

        return psd_fun

    for key, noise_fun in noise_fun_dict.iteritems():

        psd_fun_dict[key] = psd_fun_maker(noise_fun)

    return psd_fun_dict
