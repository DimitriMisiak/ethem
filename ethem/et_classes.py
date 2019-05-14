#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""
@author: misiak

Establishing the classes used to simulate a electro-thermal system.
Should be completd by a GUI in order to easily implete the system
electro-thermal scheme.
"""

import sympy as sy
import abc
import sys
import os

from .core_classes import System, Element, Bath, RealBath, Link
from .thermal import event_power


class Thermostat(Bath):
    """ Bath subclass defining a object characterized by
    its temperature.
    This thermal object has its temperature as 'main_quant' and its power as
    'main_flux'.
    """
    def __init__(self, label):
        super(Thermostat, self).__init__(label)
        self.temperature = sy.symbols('T_'+ self.label)
        self.power = 0

    @property
    def main_quant(self):
        return self.temperature

    @property
    def main_flux(self):
        return self.power


class Voltstat(Bath):
    """ Bath subclass defining a object characterized by
    its voltage.
    This electric object has its voltage as 'main_quant' and its current as
    'main_flux'.
    """
    def __init__(self, label):
        super(Voltstat, self).__init__(label)
        self.voltage = sy.symbols('V_'+ self.label)
        self.current = 0

    @property
    def main_quant(self):
        return self.voltage

    @property
    def main_flux(self):
        return self.current


class ThermalBath(Thermostat, RealBath):
    """ Subclass of Thermostat and RealBath, which is defined
    by its thermal capacity.
    """
    def __init__(self, label):
        super(ThermalBath, self).__init__(label)
        self.th_capacity = sy.symbols('C_'+self.label)

    @property
    def capacity(self):
        return self.th_capacity


class Capacitor(Voltstat, RealBath):
    """ Subclass of Voltstat and RealBath, which is defined
    by its electric capacity.
    """
    def __init__(self, label):
        super(Capacitor, self).__init__(label)
        self.el_capacity = sy.symbols('C_'+label)

    @property
    def capacity(self):
        return self.el_capacity


class ThermalLink(Link):
    """ Link subclass defining an thermal link characterized by
    its conductivity and the power passing through.
    The expression of the power is passed by default. Feel free to change it
    once the ThermalLink object is created.
    """
    def __init__(self, from_bath, to_bath, label):

        # can only attach ThermalLink to Thermostat
        assert isinstance(from_bath, Thermostat)
        assert isinstance(to_bath, Thermostat)

        super(ThermalLink, self).__init__(from_bath, to_bath, label)

        # default power expression
        self.power = sy.Function('P_' + self.label)(
                self.from_bath.temperature, self.to_bath.temperature
        )

    @property
    def main_flux(self):
        return self.power

    @property
    def conductance(self):
        # default choice for the conductance
        return (self.power).diff(self.from_bath.temperature)

    @property
    def temperature_diff(self):
#        return self.from_bath.temperature - self.to_bath.temperature
        return self.main_quant_diff


class Resistor(Link):
    """ Link subclass defining an electric link characterized by
    its resistivity and the current passing through.
    The Resistor instances also possesses an attribute temperature in
    prevision of the evaluation of the Johnson Noise.
    """
    def __init__(self, from_bath, to_bath, label):

        # can only attach Resistor to Voltstat
        assert isinstance(from_bath, Voltstat)
        assert isinstance(to_bath, Voltstat)

        super(Resistor, self).__init__(from_bath, to_bath, label)

        self.resistivity = sy.symbols( 'R_' + self.label)

        self.temperature = sy.symbols('T_R_' + self.label)

    @property
    def voltage(self):
        return self.main_quant_diff

    @property
    def current(self):
        return self.resistivity**-1 * self.voltage

    @property
    def main_flux(self):
        return self.current


class Perturbation(object):
    """ Perturbation class.

    Parameters
    ----------
    energy : sympy.symbols
        Energy symbol.
    Fraction : list
        List of the fraction of the energy going in each bath. Its length
        must match the number of bath in the system.
    tau_therm : list
        List of the thermalization time in each bath. Its length must match
        the number of bath in the system.
    """
    def __init__(self, energy, fraction, tau_therm):

        bath_list = System.bath_list
        num = len(bath_list)

        assert len(fraction) == num
        assert len(tau_therm) == num

        per = sy.zeros(len(bath_list), 1)

        for i in range(num):
            per[i] = fraction[i] * event_power(energy, tau_therm[i], System.time)

        self.matrix = per
        self.energy = energy
        self. fraction = fraction
        self.tau_therm = tau_therm

        System.perturbation = self
