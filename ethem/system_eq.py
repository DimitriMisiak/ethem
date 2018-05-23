#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: misiak

Contains the functions which extracts the equations and matrices from the
electro-thermal system.
"""

import sympy as sy
import sys

def phi_vect(bath_list):
    """ Returns the temperature/voltage vectors in the order of bath_list.
    """
    main_quant_list =  [b.main_quant for b in bath_list]
    return sy.Matrix(len(bath_list), 1, main_quant_list)


def capacity_matrix(bath_list):
    """ Returns the diagonal matrix containing the thermal/electric capacity
    in the same order as tehe bath_list.
    """
    capa_list = [b.capacity for b in bath_list]
    return sy.diag(*capa_list)


def sse(bath_list):
    """ Returns the system of steady state equations.
    Quite the same as the ete function, only without the thermal capacity.

    Examples:
    =========
    The equation C*dT/dt = a*f(T) is describes by this function with the
    return :
    a * f(T)
    """
    power_list = [bath.eq().args[1] for bath in bath_list]
    return sy.Matrix(power_list)


def ete(bath_list):
    """ Returns the expression of the temperature derivative from the
    Electro-Thermal Equations.

    Examples:
    =========
    The equation dT/dt = a*f(T) is describes by this function with the
    return :
    a * f(T)
    """
    power_list = [bath.eq().args[1]/bath.capacity for bath in bath_list]
    return sy.Matrix(power_list)


def sym_check(bath_list):
    """ Saving the main symbolic result of the system in a txt file.
    """
    phi = phi_vect(bath_list)

    eteq = ete(bath_list)

#    cmeq = cm(bath_list)
#
    capa_matrix = capacity_matrix(bath_list)
#
#    icimeq = capa_matrix * cim(bath_list)
#
#    zeq = icimeq[-1, -1]**-1

    original_stdout = sys.stdout

    def pprint(x):
        sy.pprint(x, wrap_line=False, use_unicode=False)

    try:
        sys.stdout = open('check_output.txt', 'w')

        print "\nphi_vect :", phi

        print "\ncapacity_matrix :", capa_matrix

        print "\nete :\n"
        pprint(eteq)

        print "\nsse :\n"
        pprint(sse(bath_list))

#        print "\ncm :\n"
#        pprint(cmeq)
#
#        print "\nicimeq :\n"
#        pprint(icimeq)
#
#        print "\nZeq :\n"
#        pprint(zeq)

        print '\n END OF PPRINT.'
    finally:
        sys.stdout = original_stdout

