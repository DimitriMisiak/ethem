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

class System(object):
    """ Un-instanced class representing the system environment in which the
    different Element instances interact with each others.
    """

    # List to keep track of the different elements of the system
    elements_list = []

    # time and frequency symbols
    time, freq = sy.symbols('t, f')

    @classmethod
    def checkpoint(cls):
        """ Fast checking the elements of the system by printing theier label.
        """
        for e in cls.elements_list:
            print('The label of this bath is ', e.label)

    @classmethod
    def subclass_list(cls, subcls):
        """ Define or update the attribute 'bath_list', a list tracking all
        the RealBath in the Elements instances.
        """
        return [b for b in cls.elements_list if isinstance(b, subcls)]

    @classmethod
    def build_phi_vect(cls):
        """ Returns the main_quant vectors in the order of bath_list.
        Also defines the attribute 'phi_vect'.
        """
        main_quant_list =  [b.main_quant for b in System.bath_list]
        main_quant_mat = sy.Matrix(len(System.bath_list), 1, main_quant_list)

        System.phi_vect = main_quant_mat
        return System.phi_vect

    @classmethod
    def build_capacity_matrix(cls):
        """ Returns the diagonal matrix containing the thermal/electric capacity
        in the same order as the bath_list.
        Also defines the attribute 'capacity_matrix'.
        """
        capa_list = [b.capacity for b in System.bath_list]
        capa_mat = sy.diag(*capa_list)

        System.capacity_matrix = capa_mat
        return System.capacity_matrix

    @classmethod
    def build_steady_state_eq(cls):
        """ Returns the system of steady state equations.
        Quite the same as the ete function, only without the thermal capacity.
        Also defines the attribute 'sseq'.
        Examples:
        =========
        The equation C*dT/dt = a*f(T) is describes by this function with the
        return :
        a * f(T)
        """
        power_list = [bath.eq().args[1] for bath in System.bath_list]
        power_mat = sy.Matrix(power_list)

        System.sseq = power_mat
        return System.sseq

    @classmethod
    def build_eletro_thermal_eq(cls):
        """ Returns the expression of the temperature derivative from the
        Electro-Thermal Equations.
        Also defines the attributes 'eteq'.
        Examples:
        =========
        The equation dT/dt = a*f(T) is describes by this function with the
        return :
        a * f(T)
        """
        power_list = [b.eq().args[1]/b.capacity for b in System.bath_list]
        power_mat = sy.Matrix(power_list)

        System.eteq = power_mat
        return System.eteq

    @classmethod
    def build_coupling_matrix(cls):
        """ Returns the coupling matrix. Such that:
        dPhi/dt = -CM*Phi + F(t)
        Also defines the attribute 'coupling_matrix'.
        """
        bath_num = len(System.bath_list)

        coup_list = list()
        for bath in System.bath_list:
            flux = bath.eq().args[1]

            coup = sy.zeros(1, bath_num)

            for j, quant in enumerate(System.phi_vect):
                coup[j] = flux.diff(quant) / bath.capacity

            coup_list.append(coup)

        # M = reduce(lambda x,y: x.col_join(y), coup_list)
        M = coup_list[0]
        for vect in coup_list[1:]:
            M = M.col_join(vect)

        ### minus sign to obtain the coupling matrix as :
        ### dPhi/dt = - M * Phi
        coup_mat = -M

        System.coupling_matrix = coup_mat
        return System.coupling_matrix

    @classmethod
    def build_admittance_mat(cls):
        """ Returns the complex admittance matrix. It si the inverse of the
        complex impedance matrix.
        dPhi/dt = -CM*Phi + F(t) <=> A*Phi = tf[F](w)
        with A = CM + 1j*w*Id
        Also define the attribute 'admittance_matrix'.
        """
        cm_mat = System.coupling_matrix

        deri = sy.eye(cm_mat.shape[0]) * sy.I * 2 * sy.pi * System.freq

        capa_matrix = System.capacity_matrix

        admit = capa_matrix*(cm_mat + deri)

        System.admittance_matrix = admit
        return System.admittance_matrix

    @classmethod
    def build_sym(cls, savepath='ethem_system_output.txt'):
        """ Call other classmethod to defines or update all the symbolic
        attributes of the System. Also, pretty_print thiese symbolic attributes
        in a txt file.
        """

        # printing the name of the method and pprint the symbolic expression.
        def pprint(method):
            print('\n{} :\n'.format(method.__name__))
            sy.pprint(method(), wrap_line=False, use_unicode=False)


        # saving the original printing backend.
        original_stdout = sys.stdout

        # printing into a txt file.
        try:

#            # creating save directory of the save file.
#            build_path(savepath)

            # redirecting the printing to the txt file.
            sys.stdout = open(savepath, 'w')

            # defining the attribute 'bath_list'
            System.bath_list = System.subclass_list(RealBath)

            # printing and defining all the symbolic attributes
            pprint(System.build_phi_vect)
            pprint(System.build_capacity_matrix)
            pprint(System.build_steady_state_eq)
            pprint(System.build_eletro_thermal_eq)
            pprint(System.build_coupling_matrix)
            pprint(System.build_admittance_mat)

            # String to conclude the txt file, and assure a good pprint.
            print('\n END OF PPRINT.')

            # reverting to the original printing backend
            sys.stdout = original_stdout
            # for the user
            print('Building System Done (written in {})'.format(savepath))

        finally:
            # even if an error occur, the priting backend is reverted to
            # the original one.
            sys.stdout = original_stdout


class Element(object):
    """ Parent class with the __init__ function registering the instance in
    the attribute 'elements_list' of the class 'System'. Also, an Element
    instance is described by an attribute 'label'.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, label):
        self.label = label
        self.name =  '{}_{}'.format(self.__class__.__name__, self.label)

        # listing all the elements
        System.elements_list.append(self)

        # creating an attribute to the system for each element
        setattr(System, self.name, self)


class Bath(Element):
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
        super(Bath, self).__init__(label)

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

        # perturbation of the bath is to zero (no perturation)
        self.perturbation = 0

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
        var = self.capacity * sy.Derivative(self.main_quant, System.time)

        # equation in bath
        eq = sy.Eq(var, power, evaluate=False)
        return eq

class Link(Element):
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
        super(Link, self).__init__(label)

        # can only attach Link to Bath
        assert isinstance(from_bath, Bath)
        assert isinstance(to_bath, Bath)

        self.from_bath = from_bath
        self.to_bath = to_bath

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

    @property
    def main_quant_diff(self):
        """ Difference of main_quant between the from_bath and the to_bath
        """
        return self.from_bath.main_quant - self.to_bath.main_quant
