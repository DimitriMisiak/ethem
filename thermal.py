#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
List of convenience physical sympy function usually used in
electro-thermal modelling.

@author: misiak
"""

import sympy as sy

# physical constant
kB = sy.symbols('kB')


def tfn_noise(G, T1, T2):
    """ Power Spectral Density level of the Thermal Fluctuation Noise in
    a thermal link (assumed to be white and gaussian).

    Parameters
    ==========
    G : conductance

    T1, T2  : temperatures of the 2 linked baths

    Return
    ======
    The PSD level in [Watt^2/Hz].

    .. math::
        PSD_{TFN} = 2 k_B G (T_1^2 + T_2^2)
    """
    return 2 * kB * G * (T1**2 + T2**2)


def johnson_noise(R, T):
    """ Power Spectral Density level of the Johnson Noise in a resistor
    (assumed to be white and gaussian).

    Parameters
    ==========
    R : resistivity

    T : temperature

    Return
    ======
    The PSD level in [V^2/Hz].

    .. math::
        PSD_{Johnson} = 4 k_B R T
    """
    return 4 * kB * R * T


def ntd_char(R0, T0, T):
    """ NTD resistor characteristic.

    Parameters
    ==========
    R0 : resistance constant

    T0 : temperature constant

    T : NTD temperature

    Return
    ======
    The NTD resistance in [Ohms].

    .. math::
        R_{NTD} = R_0 \exp(\sqrt{T_0 \over T})
    """
    return R0 * sy.exp((T0/T)**0.5)


def nbsi_char(Rn, Tc, sig, T):
    """ Early approximation of the NbSi supraconductor characteristic.

    Parameters
    ==========
    Rn : normal resistance at high temperature

    Tc : supraconductor transistion temperature

    sig : width of the transition

    T : nbsi temperature

    """
    return Rn / ( 1 + sy.exp( -(T-Tc)/sig ) )


def joule_power(V, R):
    """ Joule power in resistance calculated from its voltage and resistance.

    Parameters
    ==========
    V : voltage

    R : resistance

    Return
    ======
    The Joule Power in [Watt].

    .. math::
        P_J = {V^2 \over R_{NTD}}
    """
    return V**2/R


def kapitsa_power(g, n, from_temp, to_temp):
    """ Kapitsa power through a thermal link.

    Parameters
    ==========
    g : kapitsa conductivity coefficient

    n : kapitsa exponant

    from_temp : temperature of from_bath

    to_temp : temperature of to_bath

    Return
    ======
    The kapitsa power in [Watt].

    .. math::
        P_{kapitsa} = g(T_{from}^n - T_{to}^n)
    """
    return g*(from_temp**n - to_temp**n)


def event_power(E, s, t):
    """ Power deposited by an event.

    Parameters
    ==========
    E : Energy

    s : Time constant

    t : Time

    Return
    ======
    The power injected by an event in [Watt] if E in [Joule].

    .. math::
        P_{event} = \Theta(t) {E \over s} \exp({-t \over s})

    """
    return sy.Heaviside(t) * E/s * sy.exp(-t/s)
