#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: misiak

Contains the functions which extracts the equations and matrices from the
electro-thermal system.
"""

import sympy as sy
from .et_scheme import RealBath
from .evaluation import lambda_fun, lambdify_fun
from .et_scheme import f

def noise_flux_vects(bath_list):
    """ Returns a dictionnary of the different noise power affecting
    the system. The keys indicates the source of the noise. The values are
    noise perturbation vectors (LPSD) .

    Return LPSD/perturbation, so that the sign holds a physical meaning.
    """
    vects = dict()
    # internal noise of the thermal/electric bath
    for b in bath_list:
        noi_dict = b.noise_sys

        for key, noi in noi_dict.iteritems():
            if noi != 0:
                vec = sy.zeros(4, 1)

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
                vec = sy.zeros(4, 1)

                if isinstance(s.from_bath, RealBath):
                    ind1 = bath_list.index(s.from_bath)
                    vec[ind1] = noi

                if isinstance(s.to_bath, RealBath):
                    ind2 = bath_list.index(s.to_bath)
                    vec[ind2] = -noi

                vects[key] = vec

    return vects


def noise_flux_fun(bath_list, eval_dict):
    """ pass
    """
    noise_dict = noise_flux_vects(bath_list)

    fun_dict = dict()

    def fun_maker(noi):

        # FIXING SYMPY LAMBDIFY BROADCASTING
        noi[0] += 1e-40 * f

        noise_num = noi.subs(eval_dict)

        noise_fun_simple = sy.lambdify(f, noise_num, modules="numpy")

        noise_fun_array = lambda frange: lambda_fun(noise_fun_simple, frange)

        return noise_fun_array

    for key, noise in noise_dict.iteritems():

        fun_dict[key] = fun_maker(noise)

    return fun_dict


def noise_obs_fun(ref_bath, eval_dict):
    """ pass
    """
    noise_dict = ref_bath.noise_obs

    fun_dict = dict()

    def fun_maker(noi):

        noise_num = noi.subs(eval_dict)

        noise_fun_simple = sy.lambdify(f, noise_num, modules="numpy")

        noise_fun_array = lambda frange: lambdify_fun(noise_fun_simple, frange)

        return noise_fun_array

    for key, noise in noise_dict.iteritems():

        fun_dict[key] = fun_maker(noise)

    return fun_dict



