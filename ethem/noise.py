#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: misiak

Contains the functions which extracts the equations and matrices from the
electro-thermal system.
"""

import sympy as sy
from .core_classes import RealBath, System
from .evaluation import lambda_fun, lambdify_fun
from .steady_state import solve_sse_param

def noise_flux_vects():
    """ Returns a dictionnary of the different noise power affecting
    the system. The keys indicates the source of the noise. The values are
    noise perturbation vectors (LPSD) .

    Return LPSD/perturbation, so that the sign holds a physical meaning.

    Returns
    =======
    vects : dict
        Dictionnary with keys labelling the noise source, and the values
        being the symbolic noise matrix associated.
    """
    bath_list = System.bath_list
    num = len(bath_list)
    vects = dict()
    # internal noise of the thermal/electric bath
    for b in bath_list:
        noi_dict = b.noise_sys

        for key, noi in noi_dict.iteritems():
            if noi != 0:
                vec = sy.zeros(num, 1)

                ind = bath_list.index(b)
                vec[ind] = noi

                vects[key] = vec

    # listing the links between the baths, with no duplicates
    source = list()
    for b in bath_list:
        source += b.link_in + b.link_out
    source = list(set(source))

    # noise from the links
    for s in source:
        noi_dict = s.noise_flux

        for key, noi in noi_dict.iteritems():
            if noi != 0:
                vec = sy.zeros(num, 1)

                if isinstance(s.from_bath, RealBath):
                    ind1 = bath_list.index(s.from_bath)
                    vec[ind1] = noi

                if isinstance(s.to_bath, RealBath):
                    ind2 = bath_list.index(s.to_bath)
                    vec[ind2] = -noi

                vects[key] = vec

    return vects


def noise_flux_fun(eval_dict):
    """ Returns a dictionnary of the different noise power function affecting
    the system. The keys indicates the source of the noise. The values are
    noise perturbation vectors (LPSD) .

    Return LPSD/perturbation, so that the sign holds a physical meaning.

    Parameters
    ==========
    eval_dict : dict
        Evaluation dictionnary.

    Returns
    =======
    fun_dict : dict
        Dictionnary with keys labelling the noise source, and the values
        being the noise matrix function of the frequencies.
    """
    noise_dict = noise_flux_vects()

    fun_dict = dict()

    def fun_maker(noi):

        # FIXING SYMPY LAMBDIFY BROADCASTING
        noi[0] += 1e-40 * System.freq

        noise_num = noi.subs(eval_dict)

        noise_fun_simple = sy.lambdify(System.freq, noise_num, modules="numpy")

        noise_fun_array = lambda frange: lambda_fun(noise_fun_simple, frange)

        return noise_fun_array

    for key, noise in noise_dict.iteritems():

        fun_dict[key] = fun_maker(noise)

    return fun_dict


def noise_flux_fun_param(param, eval_dict, auto_ss=True):

    npar = len(param)

    char_dict = eval_dict.copy()

    for p in param:
        try:
            char_dict.pop(p)
        except:
            pass

    phi = tuple(System.phi_vect)

    noise_dict = noise_flux_vects()

    args_lambda = (System.freq,) + phi + param

    if auto_ss:
        ss_fun = solve_sse_param(param, eval_dict)

    noise_dict_simple = dict()
    for key, noi in noise_dict.iteritems():

        # FIXING SYMPY LAMBDIFY BROADCASTING
        noi[0] += 1e-40 * System.freq

        noise_num = noi.subs(char_dict)
        noise_fun_simple = sy.lambdify(args_lambda, noise_num, modules="numpy")
        noise_dict_simple[key] = noise_fun_simple

    def noise_flux_fun_aux(p, sol_ss=[]):

        assert len(p) == npar

        if auto_ss:
            sol_ss = ss_fun(p).x
        else:
            assert len(sol_ss) == len(phi)

        args = tuple(sol_ss) + tuple(p)

        def array_maker(nfun):
            nfun_complex = lambda f: nfun(f, *args)
            nfun_array = lambda frange: lambda_fun(nfun_complex, frange)
            return nfun_array

        noise_flux_dict = dict()
        for key, nfun in noise_dict_simple.iteritems():

            noise_flux_dict[key] = array_maker(nfun)

        return noise_flux_dict

    return noise_flux_fun_aux


def noise_obs_fun(ref_bath, eval_dict):
    """ Returns a dictionnary of the different observationnal noise function
    affecting the system. The keys indicates the source of the noise.
    The values are noise perturbation vectors (LPSD) .

    Return LPSD/perturbation, so that the sign holds a physical meaning.

    Parameters
    ==========
    eval_dict : dict
        Evaluation dictionnary.

    Returns
    =======
    fun_dict : dict
        Dictionnary with keys labelling the noise source, and the values
        being the noise matrix function of the frequencies.
    """
    noise_dict = ref_bath.noise_obs

    fun_dict = dict()

    def fun_maker(noi):

        noise_num = noi.subs(eval_dict)

        noise_fun_simple = sy.lambdify(System.freq, noise_num, modules="numpy")

        noise_fun_array = lambda frange: lambdify_fun(noise_fun_simple, frange)

        return noise_fun_array

    for key, noise in noise_dict.iteritems():

        fun_dict[key] = fun_maker(noise)

    return fun_dict


def noise_obs_param(param, eval_dict, ref_bath, auto_ss=True):

    npar = len(param)

    char_dict = eval_dict.copy()

    for p in param:
        try:
            char_dict.pop(p)
        except:
            pass

    phi = tuple(System.phi_vect)

    noise_dict = ref_bath.noise_obs

    args_lambda = (System.freq,) + phi + param

    if auto_ss:
        ss_fun = solve_sse_param(param, eval_dict)

    noise_dict_simple = dict()
    for key, noi in noise_dict.iteritems():

        noise_num = noi.subs(char_dict)
        noise_fun_simple = sy.lambdify(args_lambda, noise_num, modules="numpy")
        noise_dict_simple[key] = noise_fun_simple

    def noise_obs_fun(p, sol_ss=[]):

        assert len(p) == npar

        if auto_ss:
            sol_ss = ss_fun(p).x
        else:
            assert len(sol_ss) == len(phi)

        args = tuple(sol_ss) + tuple(p)

        def array_maker(nfun):
            nfun_complex = lambda f: nfun(f, *args)
            nfun_array = lambda frange: lambdify_fun(nfun_complex, frange)
            return nfun_array

        noise_obs_dict = dict()
        for key, nfun in noise_dict_simple.iteritems():

            noise_obs_dict[key] = array_maker(nfun)

        return noise_obs_dict

    return noise_obs_fun
