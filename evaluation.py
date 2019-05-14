#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Group of function manipulating the sympy lambdified expression in order
to properly work with numpy braodcasting ability.

@author: misiak
"""

import sympy as sy
import numpy as np

def lambda_fun_mat(icimeq_funk, x):
    """ Return the Inverse Complex Impedance Matrix for each frequency.

    Parameters
    ==========
    x : numpy.ndarray
        Frequency array.

    Returns
    =======
    abo_fix : numpy.ndarray
        3d numpy.ndarray of shape (x.size, len(bath_list), len(bath_list)).
    """
    # input must be 1d numpy.ndarray
    assert isinstance(x, np.ndarray) and x.ndim == 1

    # result from the lambdified function with array input
    abo = icimeq_funk(x)

    # for each row in the array
    for i,a in enumerate(abo):
        # fix the term not properly broadcast in the lambdified funtion
        a_fix = np.broadcast_arrays(*a)
        abo[i] = a_fix

    # cleaning to work with a proper array (no array of arrays of arrays !)
    abo_fix = np.array(abo.tolist())

    # roll the axis to obtain an array of matrix, and not a matrix of arrays
    abo_fix = np.moveaxis(abo_fix, -1, 0)

    return abo_fix


def lambda_fun(funk, x):
    """ Return the sympy lambdified function for each frequency using numpy
    efficient broadcast system.

    Parameters
    ==========
    x : numpy.ndarray
        Frequency array.

    Returns
    =======
    abo_fix : numpy.ndarray
        2D numpy.ndarray of shape (x.size, len(bath_list)).
    """
    # input must be 1d numpy.ndarray
    assert isinstance(x, np.ndarray) and x.ndim == 1

    # result from the lambdified function with array input
    abo = funk(x)

    if isinstance(abo, float):
        abo = np.array([[abo]])

    if abo.ndim < 3:
        # reshaping from (n,1) to (n,)
        abo = abo.reshape(abo.shape[0])

        # fix the term not properly broadcasted in the lambdified funtion
        abo_fix = np.array(np.broadcast_arrays(x, *abo)[1:])

    elif abo.ndim == 3:
        abo_fix = abo.reshape(abo.shape[0], -1)

    # roll the axis to obtain an array of matrix, and not a matrix of arrays
    abo_fix = np.moveaxis(abo_fix, -1, 0)

    return abo_fix


def lambdify_fun(funk, x):
    """ Broadcasting system for the lambdify function
    New version for test white noise.
    WARNING ! Meant to be merged with the previous function 'lambda_fun'.
    """

    abo = funk(x)

    if isinstance(abo, float):
        abo *= np.ones(x.shape)

    return abo

