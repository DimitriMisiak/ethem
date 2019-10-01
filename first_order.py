#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: misiak

Functions used to complete a first-oder perturbative theory simulation.
"""

import numpy as np
import sympy as sy

from .evaluation import lambda_fun_mat, lambda_fun
from .noise import noise_flux_fun, noise_obs_fun, noise_obs_param, noise_flux_fun_param
from .psd import psd
from .steady_state import solve_sse_param

def impedance_matrix_fun(system, eval_dict):
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
    admat = system.admittance_matrix
    admat_num = admat.subs(eval_dict)
    admat_funk = sy.lambdify(system.freq, admat_num, modules="numpy")

    admat_fun = lambda f: np.linalg.inv(lambda_fun_mat(admat_funk, f))

    return admat_fun


def impedance_matrix_param(system, param, eval_dict, auto_ss=True):
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
    npar = len(param)

    char_dict = eval_dict.copy()

    for p in param:
        try:
            char_dict.pop(p)
        except:
            pass

    admat = system.admittance_matrix
    admat_num = admat.subs(char_dict)

    phi = tuple(system.phi_vect)

    admat_simple = sy.lambdify((system.freq,)+phi+param, admat_num, modules="numpy")

    if auto_ss:
        ss_fun = solve_sse_param(system, param, eval_dict)

    def impedance_matrix_fun(p, sol_ss=[]):
        """ Return a function taking in argument a frequency numpy.array and
        returning an array of impedance matrix evaluated for each frequency.
        """
        assert len(p) == npar

        if auto_ss:
            sol_ss = ss_fun(p).x
        else:
            assert len(sol_ss) == len(phi)

        args = tuple(sol_ss) + tuple(p)
        admat_complex = lambda f: admat_simple(f, *args)

        admat_fun = lambda f: np.linalg.inv(lambda_fun_mat(admat_complex, f))

        return admat_fun

    return impedance_matrix_fun


def eigen_fun(system, eval_dict):
    """ Diagonalize of the coupling matrix and return the amplitude and the
    time constants for the temporal pulses.
    Assuming a Dirac power injection to start with.

    See also
    ========

    """
    coup_mat = system.coupling_matrix
    coup_mat_num = coup_mat.subs(eval_dict)
    coup_mat_eval = np.array(coup_mat_num).astype('float64')

    #eigen-values and vectors
    eig, proj = np.linalg.eig(coup_mat_eval)
    tau_coup = 1.0/np.real(eig)

    proj_inv = np.linalg.inv(proj)

    import sympy as sy
    per = system.perturbation
    per_td = (system.capacity_matrix)**-1 * sy.Matrix(per.fraction)

    phi_amp = [float((f*per.energy).subs(eval_dict)) for f in per_td]
    eig_amp = proj_inv.dot(phi_amp)

    def pulse_fun(time_array):

        eig_vec = [a*np.exp(-time_array/tau) for a,tau in zip(eig_amp, tau_coup)]
        pulse_array = proj.dot(eig_vec)

        return pulse_array

    return tau_coup, eig_amp, pulse_fun


def eigen_param(system, param, eval_dict, auto_ss=True):

    npar = len(param)

    char_dict = eval_dict.copy()

    for p in param:
        try:
            char_dict.pop(p)
        except:
            pass

    coup_mat = system.coupling_matrix
    coup_mat_num = coup_mat.subs(char_dict)

    per = system.perturbation
    per_matrix = (system.capacity_matrix)**-1 * sy.Matrix(per.fraction) * per.energy
    per_num = per_matrix.subs(char_dict)

    phi = tuple(system.phi_vect)

    coup_mat_simple = sy.lambdify(phi+param, coup_mat_num, modules="numpy")
    per_simple = sy.lambdify(phi+param, per_num, modules="numpy")

    if auto_ss:
        ss_fun = solve_sse_param(system, param, eval_dict)

    def eigen_fun(p, sol_ss=[]):
        assert len(p) == npar

        if auto_ss:
            sol_ss = ss_fun(p).x
        else:
            assert len(sol_ss) == len(phi)

        args = tuple(sol_ss) + tuple(p)
        coup_mat_eval = coup_mat_simple(*args)
        per_eval = per_simple(*args)

        #eigen-values and vectors
        eig, proj = np.linalg.eig(coup_mat_eval)
        tau_array = 1.0/np.real(eig)

        proj_inv = np.linalg.inv(proj)

        amp_array = proj_inv.dot(per_eval)

        def pulse_fun(time_array):

            eig_vec = [a*np.exp(-time_array/tau) for a,tau in zip(amp_array, tau_array)]
            pulse_array = proj.dot(eig_vec)

            return pulse_array

        return tau_array, amp_array, pulse_fun

    return eigen_fun


def per_fft(system):
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
    per = system.perturbation.matrix
    perf = sy.zeros(*per.shape)

    # apply the fourier transform on each term
    for k, p in enumerate(per):
        perf[k] = sy.fourier_transform(p, system.time, system.freq)

    return perf


def per_fft_fun(system, eval_dict, fs):
    """ Return a FFT function, as in Discrete Fourier Transform function.
    Return a function accepting a numpy.array with the broadcasting
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
    perf = per_fft(system)

    # multiplication by the sampling frequency to respect homogeneity later...
    perf_num = perf.subs(eval_dict) * fs
#    perf_num = perf.subs(eval_dict)

    # FIXING SYMPY LAMBDIFY BROADCASTING
    perf_num[0] += 1e-40 * system.freq

    perf_fun_simple = sy.lambdify(system.freq, perf_num, modules="numpy")

    perf_fun_array = lambda frange: lambda_fun(perf_fun_simple, frange)

    return perf_fun_array


def per_ft_fun(system, eval_dict):
    """ Return a continuous fourier transform function.
    Return a function accepting a numpy.array with the broadcasting
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
    perf = per_fft(system)

    perf_num = perf.subs(eval_dict)

    # FIXING SYMPY LAMBDIFY BROADCASTING
    perf_num[0] += 1e-40 * system.freq

    perf_fun_simple = sy.lambdify(system.freq, perf_num, modules="numpy")

    perf_fun_array = lambda frange: lambda_fun(perf_fun_simple, frange)

    return perf_fun_array


def per_param(system, param, eval_dict):

    npar = len(param)

    char_dict = eval_dict.copy()

    for p in param:
        try:
            char_dict.pop(p)
        except:
            pass

    per_num = (system.perturbation.matrix).subs(char_dict)

    perf_fun_simple = sy.lambdify((system.time,)+param, per_num,
                                  ['numpy', {'Heaviside': lambda x: np.heaviside(x,1)}])

    def per_fun(p):

        assert len(p) == npar
        perf_fun_complex = lambda t: perf_fun_simple(t,*p)

        perf_fun_array = lambda trange: lambda_fun(perf_fun_complex, trange)

        return perf_fun_array


    return per_fun


def perf_param(system, param, eval_dict):
    npar = len(param)

    char_dict = eval_dict.copy()

    for p in param:
        try:
            char_dict.pop(p)
        except:
            pass

    perf = per_fft(system)

    perf_num = perf.subs(char_dict)

    # FIXING SYMPY LAMBDIFY BROADCASTING
    perf_num[0] += 1e-40 * system.freq

    perf_fun_simple = sy.lambdify((system.freq,)+param, perf_num, 'numpy')

    def per_fun(p):

        assert len(p) == npar
        perf_fun_complex = lambda f: perf_fun_simple(f,*p)

        perf_fun_array = lambda frange: lambda_fun(perf_fun_complex, frange)

        return perf_fun_array


    return per_fun


def response_event(system, eval_dict, fs):
    """ Return the response function (fft) of the system to
    a given perturbation in the frequency space.

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
    cimeq_fun = impedance_matrix_fun(system, eval_dict)

    perf_fun = per_fft_fun(system, eval_dict, fs)

    def sens_fun(frange):
        cimeq_array = cimeq_fun(frange)

        perf_array = perf_fun(frange)

        sv_array = np.einsum('ijk, ik -> ij', cimeq_array, perf_array)

        return sv_array.T

    return sens_fun


def response_event_ft(system, eval_dict):
    """ Return the response function (FT be careful !!) of the system to
    a given perturbation in the frequency space.

    Parameters
    ==========
    per : Sympy matrix
        Power perturbation of the system. Its shape must matches the one
        of System.admittance_matrix**-1
    eval_dict : dict
        Evaluation dictionnary in first oder approximation i.e. evaluated
        for the main_quant in phi_vect.

    Return
    ======
    sens_fun : function
        Taking the frequency array as parameter, return the response array.
    """
    cimeq_fun = impedance_matrix_fun(system, eval_dict)

    perf_fun = per_ft_fun(system, eval_dict)

    def sens_fun(frange):
        cimeq_array = cimeq_fun(frange)

        perf_array = perf_fun(frange)

        sv_array = np.einsum('ijk, ik -> ij', cimeq_array, perf_array)

        return sv_array.T

    return sens_fun


def response_event_param(system, param, eval_dict):
    """ might be possible to speed up the computation with a mutual
    solve_sse_param and a wise no-use of the auto-ss optionnal parameter.
    """
    cimeq_param_fun = impedance_matrix_param(system, param, eval_dict)
    perf_param_fun = perf_param(system, param, eval_dict)


    def response_event_fun(p):

        cimeq_fun = cimeq_param_fun(p)

        perf_fun = perf_param_fun(p)

        def sens_fun(frange):
            cimeq_array = cimeq_fun(frange)

            perf_array = perf_fun(frange)

            sv_array = np.einsum('ijk, ik -> ij', cimeq_array, perf_array)

            return sv_array.T

        return sens_fun

    return response_event_fun


def psd_response_event(system, eval_dict, fs, L):
    """ Return the psd arrays of the system response.

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
    L : float
        Time length of the window in second.

    Returns
    =======
    freq_psd : numpy.ndarray
        Frequency array containing only the positive frequencies (no 0th freq).
    response_psd : numpy.ndarray
        PSD arrays.

    See also
    ========
    ethem.psd, ethem.response_event, np.ftt.fftfreq,
    """
    freq_fft = np.fft.fftfreq(int(L*fs), fs**-1)

    response_fft = response_event(system, eval_dict, fs)(freq_fft)

    freq_psd, response_psd = psd(response_fft, fs)

    return freq_psd, response_psd


def psd_response_event_ref(system, eval_dict, fs, L, ref_bath):
    """ Projection of the ethem.psd_response_event function
    on the reference bath used for the measure.

    ref_bath : ethem.RealBath
        Reference bath where the measure takes place.

    See also
    ========
    ethem.psd_response_event
    """
    ref_ind = system.bath_list.index(ref_bath)

    freq_psd, response_psd = psd_response_event(system, eval_dict, fs, L)

    return freq_psd, response_psd[ref_ind]


def response_noise(system, eval_dict):
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
    cimeq_fun = impedance_matrix_fun(system, eval_dict)

    noise_fun_dict = noise_flux_fun(system, eval_dict)

    psd_fun_dict = dict()

    def psd_fun_maker(n_fun):

        def psd_fun(frange):

            cimeq_array = cimeq_fun(frange)

            lpsd_array = n_fun(frange)

            impact_array = np.einsum('ijk, ik -> ij', cimeq_array, lpsd_array)

            # computing the psd from the fft
            psd_array = np.abs(impact_array)**2

            return psd_array.T # transposition for a pretty handy return

        return psd_fun

    for key, noise_fun in noise_fun_dict.items():

        psd_fun_dict[key] = psd_fun_maker(noise_fun)

    return psd_fun_dict


def response_noise_param(system, param, eval_dict):

    cimeq_param_fun = impedance_matrix_param(system, param, eval_dict)

    noise_param_fun = noise_flux_fun_param(system, param, eval_dict)

    def response_noise_fun(p):

        cimeq_fun = cimeq_param_fun(p)
        noise_fun_dict = noise_param_fun(p)

        def psd_fun_maker(nfun):

            def psd_fun(frange):

                cimeq_array = cimeq_fun(frange)
                lpsd_array = nfun(frange)

                impact_array = np.einsum('ijk, ik -> ij', cimeq_array, lpsd_array)

                # computing the psd from the fft
                psd_array = np.abs(impact_array)**2

                return psd_array.T # transposition for a pretty handy return

            return psd_fun

        psd_fun_dict = dict()
        for key, noise_fun in noise_fun_dict.items():

            psd_fun_dict[key] = psd_fun_maker(noise_fun)

        return psd_fun_dict

    return response_noise_fun


def response_noise_ref(system, eval_dict, ref_bath):
    """ Projection of the ethem.response_noise function on the reference bath
    used for the measure.

    ref_bath : ethem.RealBath
        Reference bath where the measure takes place.

    See also
    ========
    ethem.response_noise
    """
    ref_ind = system.bath_list.index(ref_bath)

    noise_dict = response_noise(system, eval_dict)

    noise_ref_dict = dict()

    def ref_fun_maker(n_fun):

        def ref_fun(frange):

            return n_fun(frange)[ref_ind]

        return ref_fun

    for key,nfun in noise_dict.items():

        noise_ref_dict[key] = ref_fun_maker(nfun)

    return noise_ref_dict


def measure_noise(system, ref_bath, eval_dict):
    """ Return the observationnal noise psd perturbation
    in the frequency space.

    Parameters
    ==========
    ref_bath : ethem.RealBath
        Bath where the measure takes place.
    eval_dict : dict
        Evaluation dictionnary in first oder approximation i.e. evaluated
        for the main_quant in phi_vect.

    Return
    ======
    psd_fun_dict : dict of function
        Key are a string specifying the noise source, and the values are
        the corresponding noise function taking as parameter the frequency
        array and returning the noise array.
    """
    noise_fun_dict = noise_obs_fun(system, ref_bath, eval_dict)

    psd_fun_dict = dict()

    def psd_fun_maker(n_fun):

        def psd_fun(frange):

            lpsd_array = n_fun(frange)

            # computing the psd from the fft
            psd_array = np.abs(lpsd_array)**2

            return psd_array

        return psd_fun

    for key, noise_fun in noise_fun_dict.items():

        psd_fun_dict[key] = psd_fun_maker(noise_fun)

    return psd_fun_dict


def measure_noise_param(system, param, eval_dict, ref_bath):

    noise_param_fun = noise_obs_param(system, param, eval_dict, ref_bath)

    def measure_noise_fun(p):

        noise_fun_dict = noise_param_fun(p)

        def psd_fun_maker(n_fun):

            def psd_fun(frange):

                lpsd_array = n_fun(frange)

                # computing the psd from the fft
                psd_array = np.abs(lpsd_array)**2

                return psd_array

            return psd_fun

        psd_fun_dict = dict()
        for key, noise_fun in noise_fun_dict.items():

            psd_fun_dict[key] = psd_fun_maker(noise_fun)

        return psd_fun_dict

    return measure_noise_fun

def noise_tot_fun(system, ref_bath, eval_dict):
    """ Return the total noise psd perturbation function
    in the frequency space.

    Parameters
    ==========
    ref_bath : ethem.RealBath
        Bath where the measure takes place.
    eval_dict : dict
        Evaluation dictionnary in first oder approximation i.e. evaluated
        for the main_quant in phi_vect.

    Return
    ======
    noise_fun : function
        Taking the frequency array as parameter, return the noise array.
    """
    ref_ind = system.bath_list.index(ref_bath)

    obs_dict = measure_noise(system, ref_bath, eval_dict)

    sys_dict = response_noise(system, eval_dict)

    def noise_fun(frange):

        sys_eval_dict = {k:v(frange) for k,v in sys_dict.items()}
        obs_eval_dict = {k:v(frange) for k,v in obs_dict.items()}

        full_array = (
            np.sum(list(obs_eval_dict.values()), axis=0)
            + np.sum(list(sys_eval_dict.values()), axis=0)[ref_ind]
        )

        return full_array

    return noise_fun


def noise_tot_param(system, param, eval_dict, ref_bath):

    ref_ind = system.bath_list.index(ref_bath)

    obs_dict_fun = measure_noise_param(system, param, eval_dict, ref_bath)
    sys_dict_fun = response_noise_param(system, param, eval_dict)

    def noise_tot_fun(p):

        obs_dict = obs_dict_fun(p)
        sys_dict = sys_dict_fun(p)

        def noise_fun(frange):

            sys_eval_dict = {k:v(frange) for k,v in sys_dict.items()}
            obs_eval_dict = {k:v(frange) for k,v in obs_dict.items()}

            full_array = (
                np.sum(list(obs_eval_dict.values()), axis=0)
                + np.sum(list(sys_eval_dict.values()), axis=0)[ref_ind]
            )

            return full_array

        return noise_fun

    return noise_tot_fun


def nep_ref(system, eval_dict, fs, L, ref_bath):
    """ Return the nep array in the reference bath by computing
    the response psd of the system and the total noise.

    Parameters
    ==========
    per : Sympy matrix
        Power perturbation of the system. Its shape must matches the one
        of system.admittance_matrix**-1
    eval_dict : dict
        Evaluation dictionnary in first oder approximation i.e. evaluated
        for the main_quant in phi_vect.
    fs : float
        Sampling frequency.
    L : float
        Time length of the window in second.
    ref_bath : ethem.RealBath
        Reference bath where the measure takes place.noise_tot_fun_param

    Returns
    =======
    freq_array : numpy.ndarray
        Frequency array containing only the positive frequencies (no 0th freq).
    nep_array : numpy.ndarray
        NEP array.

    See also
    ========
    ethem.psd_response_event_ref, ethem.noise_tot_fun

    """
    ref_ind = system.bath_list.index(ref_bath)

    freq_array = np.linspace(1, fs/2., int(fs*L/2.))

    pulse_fun = response_event_ft(system, eval_dict)
    pulse_array = pulse_fun(freq_array)[ref_ind]

    noise_fun = noise_tot_fun(system, ref_bath, eval_dict)

    noise_array = noise_fun(freq_array)

    nep_array = noise_array / np.abs(pulse_array)**2
    return freq_array, nep_array


def nep_ref_param(system, param, eval_dict, ref_bath):

    ref_ind = system.bath_list.index(ref_bath)

    pulse_param = response_event_param(system, param, eval_dict)

    noise_param = noise_tot_param(system, param, eval_dict, ref_bath)

    def nep_ref_fun(p):

        pulse_fun = pulse_param(p)
        noise_fun = noise_param(p)

        def nep_ref_array(frange):

            pulse_array = pulse_fun(frange)[ref_ind]
            noise_array = noise_fun(frange)

            nep_array = noise_array / np.abs(pulse_array)**2
            return nep_array

        return nep_ref_array

    return nep_ref_fun


def nep_to_res(freq_array, nep_array, flim=None):
    """ Return the resolution value from the integration of 1/nep_array^2.

    Parameters
    ==========
    freq_array : array_like
        Frequency array.
    nep_array : numpy.ndarray
        NEP array.
    flim : tuple of float
        The tuple flim = (finf, fsup) contains the lower and upper
        limit of integration.

    Return
    ======
    res : float
        Resolution value. Be careful to the unit ! Should be the same as
        the energy referenced in the "perturbation object"
        affecting the system.

    See also
    ========
    function used for the numerical integration : numpy.trapz
    """
    invres_array = 4. / nep_array

    if flim is None:
        inf_index = None
        sup_index = None
    else:
        finf, fsup = flim

        inf_index = max(np.where(freq_array<=finf)[0])
        sup_index = min(np.where(freq_array>=fsup)[0])

    invres_trapz = invres_array[inf_index:sup_index]
    freq_trapz = freq_array[inf_index:sup_index]

    invres_int = np.trapz(invres_trapz, freq_trapz)
    res = (invres_int)**-0.5

    return res


def res_ref(system, eval_dict, fs, L, ref_bath, flim):
    """ Return the resolution value. No need for byproduct (nep_array, etc)
    for this function.

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
    L : float
        Time length of the window in second.
    ref_bath : ethem.RealBath
        Reference bath where the measure takes place.
    flim : tuple of float
        The tuple flim = (finf, fsup) contains the lower and upper
        limit of integration.

    Return
    ======
    res : float
        Resolution value. Be careful to the unit ! Should be the same as
        the energy referenced in the "perturbation object"
        affecting the system.

    See also
    ========
    ethem.nep_ref, ethem.nep_to_res
    """
    freq_array, nep_array = nep_ref(system, eval_dict, fs, L, ref_bath)
    res = nep_to_res(freq_array, nep_array, flim)

    return res


def res_ref_param(system, param, eval_dict, ref_bath, fs, L, flim=None):

    freq_array = np.flip(np.arange(fs/2., 0, -L**-1), axis=0)

    nep_array_fun = nep_ref_param(system, param, eval_dict, ref_bath)

    def res_ref_fun(p):
        nep_array = nep_array_fun(p)(freq_array)
        res = nep_to_res(freq_array, nep_array, flim=flim)
        return res

    return res_ref_fun
