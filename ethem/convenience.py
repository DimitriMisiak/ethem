#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: misiak

This script defines small basic functions needed for convenience purpose.
Could serve as a buffer script for functions in very early development
(try to avoid that thx).
"""

import os


def build_path(p):
    """ Create the path if not existing.

    Parameters
    ==========
    p : str
        Filepath.

    See also
    ========
    os.path.dirname, os.makedirs, os.path.isdir
    """
    folder_path = os.path.dirname(p)
    try:
        os.makedirs(folder_path)
    except OSError:
        if not os.path.isdir(folder_path):
            raise


