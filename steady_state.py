#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: misiak

Functions to access the steady state.
"""

import warnings
import numpy as np
import sympy as sy
from scipy.optimize import root
from scipy.integrate import odeint

from .et_classes import ThermalBath, Thermostat, Capacitor


def phi_init(system, eval_dict):
    """ Return the intial vector used for the search of the steady-state
    solution. For the capacitor, the initial voltage is set to 0. For the
    thermal bath, the initial temperature is set to the lowest thermostat
    temperature of the system. If none exists, the intial temperature is
    set to 0.

    Parameters
    ==========
    eval_dict : dict
        Evaluation dictionnary.

    Return
    ======
    init : list of float
        Initial vector.
    """
    bath_list = system.bath_list

    v0 = 0.0 #V
    t0 = 0.0 #K

    # retreive all the Thermostat in the System
    thermo_list = system.subclass_list(Thermostat)
    # keep only the pure Thermostas class, not the subclass
    pure_list = list(set(thermo_list) - set(bath_list))
    if len(pure_list):
        temp_list = [(b.temperature).subs(eval_dict) for b in pure_list]
        t0 = float(min(temp_list))

    init = []
    for b in bath_list:
        if isinstance(b, Capacitor):
            init.append(v0)
        elif isinstance(b, ThermalBath):
            init.append(t0)
        else:
            raise Exception('Invalid class in System.bath_list'
                            '(only valid: Capacitor and ThermalBath).'
                            'Class is '.format(type(b)))

    return init


def solve_sse_perf(fun, x0, safe_odeint=True, **kwargs):
    """ Solve the stationnary state of the given system of equations, using
    a numerical integration and then a root finder from a given starting
    vector.

    Parameters
    ----------
    fun : callable
        A vector function to find a stationnary state of.
    x0 : array_like
        Initial condition.
    safe_odeint : bool, optional
        Switch on and off the use of the numerical integration with odeint.
        By default, set to True, using odeint.

    **kwargs parameters are passed to the root finder scipy.optimize.root.

    Return
    ------
    sol :scipy.optimize.optimize.OptimizeResult
        The solution represented as a OptimizeResult object.
        Important attributes are 'x' the solution array, 'success' a boolean
        flag indicating if the algorithm exited successfully and 'message'
        which describes the cause of the termination.

    See also
    --------
    scipy.integrate.odeint, scipy.optimize.root
    """
    if not safe_odeint:

        x0_root = x0

    if safe_odeint:

        system_eq_odeint = lambda x,t: fun(x)

        time_odeint = np.linspace(0., 10., 10)

        inte = odeint(system_eq_odeint, x0, time_odeint)

        x0_root = inte[-1]

    # Resolution with scipy.optimize.root
    sol = root(fun, x0_root, **kwargs)

    return sol


def solve_sse(system, eval_dict, x0=None, safe_odeint=True, **kwargs):
    """ Solve the steady-state for the eth.System, automatically fetching
    the initial vector and the system of equations.

    Parameters
    ----------
    eval_dict :dict
        Evaluation dictionnary. Contains the evaluation values
        for the system characteristics symbols.
    x0 : array_like, optional
        Initial vector for the resolution. By default, set to None. For a x0
        set to None, the initial vector is returned by the function
        eth.phi_init.
        Fixing x0 requires some comprehension from the user, but is faster.
    safe_odeint : bool, optional
        Switch on and off the use of the numerical integration with odeint.
        By default, set to True, using odeint.

    **kwargs parameters are passed to the steady-state solver solve_see_perf.

    Return
    ------
    sol :scipy.optimize.optimize.OptimizeResult
        The solution represented as a OptimizeResult object.
        Important attributes are 'x' the solution array, 'success' a boolean
        flag indicating if the algorithm exited successfully and 'message'
        which describes the cause of the termination.

    See also
    --------
    eth.phi_init, eth.solve_sse_perf
    """
    # Quantities to be evaluated by the resolution
    phi = system.phi_vect

    if x0 is None:
        x0 = phi_init(system, eval_dict)
    else:
        # checking that the initial vector is adapted in length
        assert len(phi) == len(x0)

    # Steady state equations
    eteq = system.eteq.subs(eval_dict)

    # checking that all symbols the desired symbols are evaluated
    # if an error is raised, a term is missing from the given dictionnary
    assert set(phi) == set(eteq.free_symbols)

    eteq_list = list(eteq)

    # process the sympy equations into a function adaptated to scipy root
    funk = sy.lambdify(phi, eteq_list, 'math')
    system_eq = lambda x: funk(*x)

    sol = solve_sse_perf(system_eq, x0, safe_odeint, **kwargs)

    return sol


def solve_sse_param(system, param, eval_dict):
    """ Return an auxiliary function solving the steady-state of eth.System
    for a configuration of the given parameters. This is efficient to
    compute current-voltage curve or do multiple resolution of the
    steady-state.

    Parameters
    ----------
    param : tuple of sympy.symbols
        Tuple of symbols associated to the parameters of the returned function.
        In orde to quickly compute a IV curve, this param should be:
        (cryo_temp, bias_voltage).
    eval_dict : dict
        Evaluation dictionnary. Contains the evaluation values
        for the system characteristics symbols.

    Others settings of the steady-state resolution are passed as parameters
    for the returned function. See solve_sse_fun doc.

    Return
    ------
    sol_see_fun : function
        Actual function solving the steady state with the given parameters
        values. See its doc for more info.

    See also
    --------
    Doc of the auxiliary returned function solve_sse_fun
    eth.phi_init, eth.solve_sse_perf
    """
    npar = len(param)

    char_dict = eval_dict.copy()

    for p in param:
        try:
            char_dict.pop(p)
        except:
            pass

    phi = tuple(system.phi_vect)
    nphi = len(phi)

    eteq_list = list((system.eteq).subs(char_dict))
    eteq_lambda = sy.lambdify(phi+param, eteq_list, 'math')

    def solve_sse_fun(x, phi0=None, safe_odeint=True, **kwargs):
        """ Solve the steady-state for the eth.System according to the given
        configuration x.

        Parameters
        ----------
        x : tuple of floats
            Values for the parameters param.
        phi0 : array_like, optional
            Initial vector for the resolution. By default, set to None.
            For a phi0 set to None, the initial vector is returned by a
            custom routine using eth.phi_init, checking the parameters of
            the original function to build a proper phi0.
        safe_odeint : bool, optional
            Switch on and off the use of the numerical integration with odeint.
            By default, set to True, using odeint.

        **kwargs parameters are passed to the steady-state
        solver solve_see_perf.

        Return
        ------
        sol :scipy.optimize.optimize.OptimizeResult
            The solution represented as a OptimizeResult object.
            Important attributes are 'x' the solution array, 'success' a
            boolean flag indicating if the algorithm exited successfully
            and 'message' which describes the cause of the termination.

        See also
        --------
        eth.phi_init, eth.solve_sse_perf, eth.solve_sse_param
        """
        assert len(x) == npar

        if phi0 is None:
            param_dict = {s:v for s,v in zip(param, x)}
            char_dict.update(param_dict)
            phi0 = phi_init(system, char_dict)
        else:
            assert len(phi0) == nphi, "Invalid initial vector size."

        def aux(phi):
            args = tuple(phi) + tuple(x)
            return eteq_lambda(*args)

        sol = solve_sse_perf(aux, phi0)

        return sol

    return solve_sse_fun


def dict_sse(system, eval_dict, **kwargs):
    """ Return an evaluation dictionnary with the steady-state solution.

    Parameters
    ----------
    eval_dict : dict
        Evaluation dictionnary.

    **kwargs parameters are passed to eth.solve_sse function.

    Return
    ------
    eval_dict_sse : dict
        Updated copy of the evaluation dictionnary.

    See also
    --------
    eth.solve_sse
    """
    sol_ss = solve_sse(system, eval_dict, **kwargs)

    if not sol_ss.success:
        warnings.warn("""ethem.solve_see was not successful.\n
                      The scipy.root message is: \n
                      {}
                      """.format(sol_ss.message))

    sse_dict = {bath.main_quant:sol for bath,sol in zip(system.bath_list, sol_ss.x)}

    eval_dict_sse = eval_dict.copy()
    eval_dict_sse.update(sse_dict)

    return eval_dict_sse
