#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: misiak

Functions used to complete a first-oder perturbative theory simulation.
"""

import numpy as np
import sympy as sy

from .system_eq import phi_vect, capacity_matrix
from .evaluation import lambda_fun_mat, lambda_fun
from .et_scheme import f, t
from .noise import noise_flux_fun, noise_obs_fun
from .psd import psd

def cm(bath_list):
    """ Returns the coupling matrix. Such that:
    dPhi/dt = -CM*Phi + F(t)
    """
    bath_num = len(bath_list)

    coup_list = list()
    for bath in bath_list:
        flux = bath.eq().args[1]

        coup = sy.zeros(1, bath_num)

        for j, quant in enumerate(phi_vect(bath_list)):
            coup[j] = flux.diff(quant) / bath.capacity

        coup_list.append(coup)

    M = reduce(lambda x,y: x.col_join(y), coup_list)

    ### minus sign to obtain the coupling matrix as :
    ### dPhi/dt = - M * Phi
    return -M


def admittance_mat(bath_list):
    """ Returns the complex admittance matrix. It si the inverse of the
    complex impedance matrix.
    dPhi/dt = -CM*Phi + F(t) <=> A*Phi = tf[F](w)
    with A = CM + 1j*w*Id
    """
    cm_mat = cm(bath_list)

    deri = sy.eye(cm_mat.shape[0]) * sy.I * 2 * sy.pi * f

    capa_matrix = capacity_matrix(bath_list)

    admit = capa_matrix*(cm_mat + deri)

    return admit


def impedance_matrix_fun(bath_list, eval_dict):
    """ Return a function accepting an numpy.array with the broadcasting
    ability of numpy.
    The function returns the response matrix, or complex impedance matrix
    for the given frequancies.

    Parameters
    ==========
    bath_list : list of RealBath
        List of the RealBath of the system
    eval_dict : dict
        Evaluation dictionnary.
    frange : 1d numpy.ndarray
        Numpy array of the frequencies where to evaluate the complex impedance
        function.

    Returns
    =======
    cimeq_fun :Function taking an numpy.array as parameter and returnign an
        array of matrices, mimicking the broadcasting ability of numpy.
    """
    cimeq = admittance_mat(bath_list)
    cimeq_num = cimeq.subs(eval_dict)
    cimeq_funk = sy.lambdify(f, cimeq_num, modules="numpy")

    cimeq_fun = lambda f: np.linalg.inv(lambda_fun_mat(cimeq_funk, f))

    return cimeq_fun


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
        perf[k] = sy.fourier_transform(p, t, f)

    return perf


def per_fft_fun(per, eval_dict, fs):
    """ pass
    """
    perf = per_fft(per)
    perf_num = perf.subs(eval_dict) * fs
    perf_fun_simple = sy.lambdify(f, perf_num, modules="numpy")

    perf_fun_array = lambda frange: lambda_fun(perf_fun_simple, frange)

    return perf_fun_array


def response_gen(cimeq_fun, per_fun):
    """ pass
    """
    def aux_gen(frange):
        """ pass
        """
        cimeq_array = cimeq_fun(frange)

        perf_array = per_fun(frange)

        einsum_array = np.einsum('ijk, ik -> ij', cimeq_array, perf_array)

        return einsum_array

    return aux_gen


def response_event(bath_list, per, eval_dict, fs):
    """ pass
    """
    cimeq_fun = impedance_matrix_fun(bath_list, eval_dict)

    perf_fun = per_fft_fun(per, eval_dict, fs)

    def sens_fun(frange):
        cimeq_array = cimeq_fun(frange)

        perf_array = perf_fun(frange)

        sv_array = np.einsum('ijk, ik -> ij', cimeq_array, perf_array)

        return sv_array

    return sens_fun


def response_noise(bath_list, eval_dict):
    """ pass
    """
    cimeq_fun = impedance_matrix_fun(bath_list, eval_dict)

    noise_fun_dict = noise_flux_fun(bath_list, eval_dict)

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
