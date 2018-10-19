#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 09:57:01 2018

@author: ilikecarbs

%%%%%%%%%%%%%%%%%%%%%%%%%%%
        ARPES_utils
%%%%%%%%%%%%%%%%%%%%%%%%%%%

**Useful helper functions**

.. note::
        To-Do:
            -
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as la
from scipy import special
from scipy.ndimage.filters import gaussian_filter
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.cm as cm
import time
from joblib import Parallel, delayed
import multiprocessing


def find(array, val):
    """returns array[_val], _val

    **Searches entry in array closest to val.**

    Args
    ----
    :array:     entry value in array
    :_val:      index of entry

    Return
    ------
    :array[_val]:   entry value in array
    :_val:          index of entry
    """

    array = np.asarray(array)
    _val = (np.abs(array - val)).argmin()

    return array[_val], _val


def area(x, y):
    """returns a

    **Calculates enclosed area with Green's theorem**

    Args
    ----
    :x:     x-data
    :y:     y-data

    Return
    ------
    :a:     area
    """

    a = 0
    x0, y0 = x[0], y[0]
    for i in range(len(x)-1):
        x1 = x[i+1]
        y1 = y[i+1]
        dx = x1 - x0
        dy = y1 - y0
        a += 0.5 * (y0 * dx - x0 * dy)
        x0 = x1
        y0 = y1
    return a


"""
%%%%%%%%%%%%%%%%%%%%%
  Useful functions
%%%%%%%%%%%%%%%%%%%%%
"""


def FDsl(x, *p):
    """returns FDsl

    **Fermi Dirac function on a sloped**

    Args
    ----
    :x:     energy axis
    :p0:    kB * T
    :p1:    EF
    :p2:    Amplitude
    :p3:    Constant background
    :p4:    Slope

    Return
    ------
    :FDsl:  Fermi Dirac function on a sloped background
    """

    FDsl = p[3] + (p[2] + p[4] * x) * (np.exp((x - p[1]) / p[0]) + 1) ** -1

    return FDsl


def poly_n(x, n, *p):
    """returns poly_n

    **Polynomial n-th order**

    Args
    ----
    :x:       x
    :n:       order
    :p[n]:    coefficients

    Return
    ------
    :poly_n:  polynomial n-th order
    """

    poly_n = 0

    # Loop over orders
    for i in range(n+1):
        poly_n += p[i] * x ** i

    return poly_n


def lor_n(x, n, *p):
    """returns lor_n

    **n Lorentzians on a quadratic background**

    Args
    ----
    :x:          momentum
    :n:          number of Lorentzians
    :p[0:n-1]:   center
    :p[n:2*n-1]: HWHM
    :p[2*n:-4]:  amplitudes

    :p[-3]:      constant
    :p[-2]:      slope
    :p[-1]:      quadratic

    Return
    ------
    :lor_n:        n Lorentzians
    """

    lor_n = 0

    # Loop over Lorentzians
    for i in range(n):
        lor_n += (p[i+2*n] / (np.pi * p[i+n] *
                  (1 + ((x - p[i]) / p[i+n]) ** 2)))
    lor_n += p[-3] + p[-2] * x + p[-1] * x ** 2

    return lor_n


def gauss_n(x, n, *p):
    """returns gauss_n

    **n Gaussians on a quadratic background**

    Args
    ----
    :x:          momentum axis
    :n:          number of Gaussians
    :p[0:n-1]:   center
    :p[n:2*n-1]: width
    :p[2*n:-4]:  amplitudes

    :p[-3]:      constant
    :p[-2]:      slope
    :p[-1]:      quadratic

    Return
    ------
    :gauss_n:      n Gaussians
    """

    gauss_n = 0

    # Loop over Gaussians
    for i in range(n):
        gauss_n += p[i+2*n] * np.exp(-(x - p[i]) ** 2 / (2 * p[i+n] ** 2))

    gauss_n += p[-3] + p[-2] * x + p[-1] * x ** 2

    return gauss_n


"""
%%%%%%%%%%%%%%%%%%%%%
  Wrapper functions
%%%%%%%%%%%%%%%%%%%%%
"""


def poly_1(x, *p):
    """returns poly_1

    **wrapper function of poly_n with n=1**

    Args
    ----
    :x:          momentum axis
    :p[0]:       constant
    :p[1]:       slope

    Return
    ------
    poly_1       polynomial first order
    """

    poly_1 = poly_n(x, 1, *p)

    return poly_1


def poly_2(x, *p):
    """returns poly_2

    **wrapper function of poly_n with n=2**

    Args
    ----
    :x:          momentum axis
    :p[0]:       constant
    :p[1]:       slope
    :p[2]:       quadratic part

    Return
    ------
    :poly_2:       polynomial second order
    """

    poly_2 = poly_n(x, 2, *p)

    return poly_2


def lor(x, *p):
    """returns lor

    **wrapper function of lor_n with n=1**

    Args
    ----
    :x:      momentum
    :p[0]:   center
    :p[1]:   HWHM
    :p[2]:   amplitudes

    :p[3]:   constant
    :p[4]:   slope
    :p[5]:   quadratic

    Return
    ------
    :lor:      single Lorentzian
    """

    lor = lor_n(x, 1, *p)

    return lor


def lor_2(x, *p):
    """returns lor_2

    **wrapper function of lor_n with n=2**

    Args
    ----
    :x:          momentum
    :p[0:1]:     center
    :p[2:3]:     HWHM
    :p[4:5]:     amplitudes

    :p[-3]:      constant
    :p[-2]:      slope
    :p[-1]:      quadratic

    Return
    ------
    :lor_2:      2 Lorentzians
    """

    lor_2 = lor_n(x, 2, *p)

    return lor_2


def lor_4(x, *p):
    """returns lor_4

    **wrapper function of lor_n with n=4**

    Args
    ----
    :x:          momentum
    :p[0:3]:     center
    :p[4:7]:     HWHM
    :p[8:-4]:    amplitudes

    :p[-3]:      constant
    :p[-2]:      slope
    :p[-1]:      quadratic

    Return
    ------
    :lor_4:      4 Lorentzians
    """

    lor_4 = lor_n(x, 4, *p)

    return lor_4


def lor_6(x, *p):
    """returns lor_6

    **wrapper function of lor_n with n=6**

    Args
    ----
    :x:          momentum
    :p[0:5]:     center
    :p[6:11]:    HWHM
    :p[12:-4]:   amplitudes

    :p[-3]:      constant
    :p[-2]:      slope
    :p[-1]:      quadratic

    Return
    ------
    :lor_6:      6 Lorentzians
    """

    lor_6 = lor_n(x, 6, *p)

    return lor_6


def lor_7(x, *p):
    """returns lor_7

    **wrapper function of lor_n with n=7**

    Args
    ----
    :x:          momentum
    :n:          number of Lorentzians
    :p[0:6]:     center
    :p[7:12]:    HWHM
    :p[13:-4]:   amplitudes

    :p[-3]:      constant
    :p[-2]:      slope
    :p[-1]:      quadratic

    Return
    ------
    :lor_7:      7 Lorentzians
    """

    lor_7 = lor_n(x, 7, *p)

    return lor_7


def lor_8(x, *p):
    """returns lor_8

    **wrapper function of lor_n with n=8**

    Args
    ----
    :x:          momentum
    :p[0:7]:     center
    :p[8:15]:    HWHM
    :p[16:-4]:   amplitudes

    :p[-3]:      constant
    :p[-2]:      slope
    :p[-1]:      quadratic

    Return
    ------
    :lor_8:      8 Lorentzians
    """

    lor_8 = lor_n(x, 8, *p)

    return lor_8


def gauss(x, *p):
    """returns gauss

    **wrapper function of gauss_n with n=1**

    Args
    ----
    :x:        momentum axis
    :p[0]:     center
    :p[1]:     width
    :p[2]:     amplitudes

    :p[-3]:    constant
    :p[-2]:    slope
    :p[-1]:    quadratic

    Return
    ------
    :gauss:    Gaussian
    """

    gauss = gauss_n(x, 1, *p)

    return gauss


def gauss_2(x, *p):
    """returns gauss_2

    **wrapper function of gauss_n with n=2**

    Args
    ----
    :x:          momentum axis
    :p[0:2]:     center
    :p[3:5]:     width
    :p[6:8]:     amplitudes

    :p[-3]:      constant
    :p[-2]:      slope
    :p[-1]:      quadratic

    Return
    ------
    :gauss_2:    2 Gaussians
    """

    gauss_2 = gauss_n(x, 2, *p)

    return gauss_2


"""
%%%%%%%%%%%%%%%%%%%%%%%
  External functions
%%%%%%%%%%%%%%%%%%%%%%%
"""
