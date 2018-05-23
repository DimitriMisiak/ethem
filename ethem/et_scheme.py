#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""
@author: misiak

Establishing the classes used to simulate a electro-thermal system.
Shoul be completd by a GUI in order to easily implete the system
electro-thermal scheme.
"""

import sympy as sy
import abc

### Defining time and frequency variables
t, f = sy.symbols('t, f')


class Bath(object):
    """ Abstract Base Class which is the parent class for the RealBath,
    Thermostat and Voltstat classes. It introduces the properties 'main_quant'
    (representing the temperature or voltage), 'main_flux' (representing
    the power or current). It features the attributes 'link_in'
    and 'link_out', which are list referencing the link attached to the bath.
    It also features the attribute 'label' to have a specific symbol notation
    for each bath.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, label):
        """ Required for co-operative subclassing
        """
        super(Bath, self).__init__()

        self.label = label

        # listing the attached links, automatically set by the
        # link __init__ function.
        self.link_in = []
        self.link_out = []

    @abc.abstractproperty
    def main_quant(self):
        """ Main physical quantity of the bath.
        Should be :
        - temperature for ThermalBath classes
        - voltage for Capacitor classes
        - current for Coil classes (work with TES ?)
        """
        return

    @abc.abstractproperty
    def main_flux(self):
        """ Main physical flux of the bath. It is the temporal derivative of
        the main quantity of the bath multiplied by the capacity.
        Should be :
        - power for ThermalBath classes
        - current for Capacitor classes
        - voltage for Coil classes (work with TES ?)
        """
        return


class RealBath(Bath):
    """ Bath subclass which is an Abstract Base Class which is one of the
    parent class for ThermalBath and Capacitor classes.
    It features the attributes 'noise_sys' and 'noise_obs', which are
    dict containing the independant noise sources, respectively intrinsic to
    the element (will see the response of the system, same units as main_flux)
    and intrinsic the the measure of the main_quant (independant from the
    system, relative to the observer-mean to measure the quantity like an
    amplifier, same units as main_quant).
    It defines the property 'capacity' of the bath, linking the 'maint_quant'
    temporal variation with the 'main_flux'.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, label):
        # required for co-operative subclassing
        super(RealBath, self).__init__(label)

        # empty dictionnary of the intrinsic noise power
        self.noise_sys = dict()
        # empty dictionnary of the measure noise of main_quant
        self.noise_obs = dict()

    @abc.abstractproperty
    def capacity(self):
        """ Capacity of the bath.
        Should be :
        - thermal capacity for ThermalBath classes
        - electric capacity for Capacitor classes
        - electric inductance for Coil classes (work with TES ?)
        """
        return

    def eq(self):
        """ Return the equilibrium equation in the bath.

        Return:
        =======
        eq : sympy.core.relational.Equality
            Equality describing the equilibrium equation. Access the right
            term of this equation with eq.args[1] (or eq.args[-1]).

        """
        # intrinsic power
        power = self.main_flux
        # power from link_in
        power += sum([lin.main_flux for lin in self.link_in])
        # power from link_out
        power -= sum([lin.main_flux for lin in self.link_out])

        # time derivative
        var = self.capacity * sy.Derivative(self.main_quant, t)

        # equation in bath
        eq = sy.Eq(var, power, evaluate=False)
        return eq


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


class Link(object):
    """ Abstract Base Class which is the parent class for ThermalLink.
    It introduces the property 'main_flux' (representing
    the power or current flowing through the link).
    It features the attributes 'from_bath' and 'to_bath', which are the baths
    the link is attached to.
    It also features the attributes 'noise_flux', a dictionnary which lists
    all the independant noise power source affecting the attached bath in
    correlation.
    A Link object is also defined by a label for specific symbol notations.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, from_bath, to_bath, label):
        """ Required for co-operative subclassing.
        Automatically update the link_in and link_out attributes of the
        from_bath and the to_bath.
        """
        super(Link, self).__init__()

        # can only attach Link to Bath
        assert isinstance(from_bath, Bath)
        assert isinstance(to_bath, Bath)

        self.from_bath = from_bath
        self.to_bath = to_bath
        self.label = label

        # updating the from_bath and to_bath attributes
        self.from_bath.link_out.append(self)
        self.to_bath.link_in.append(self)

        # empty noise dictionnary
        self.noise_flux = dict()

    @abc.abstractproperty
    def main_flux(self):
        """ Main physical flux flowing through the link.
        Should be :
        - power for ThermalBath classes
        - current for Capacitor classes
        - voltage for Coil classes (work with TES ?)
        """
        return


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
        self.conductivity = sy.symbols('G_'+ self.label)
        self.power = self.conductivity * (self.from_bath.temperature
                                          - self.to_bath.temperature)

    @property
    def main_flux(self):
        return self.power

    @property
    def conductance(self):
        # default choice for the conductance
        return (self.power).diff(self.from_bath.temperature)


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
    def current(self):
        return self.resistivity**-1 * (self.from_bath.voltage
                                       - self.to_bath.voltage)

    @property
    def main_flux(self):
        return self.current
