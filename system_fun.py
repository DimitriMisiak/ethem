#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""
@author: misiak

Establishing the classes used to simulate a electro-thermal system.
Should be completd by a GUI in order to easily implete the system
electro-thermal scheme.
"""

from .et_classes import ThermalBath, ThermalLink

def dynamic_check(system, eval_dict):

    for e in system.elements_list:
        if isinstance(e, ThermalBath):
            print('In the bath {}:\n{}={}\n'.format(
                    e.name,
                    str(e.capacity), e.capacity.subs(eval_dict)
            ))

        if isinstance(e, ThermalLink):
            print('In the bath {}:\n{}={}\n'.format(
                    e.name,
                    str(e.conductance), e.conductance.subs(eval_dict)
            ))

