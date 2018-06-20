#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 19 12:21:42 2018

@author: denyssutter
"""

import numpy as np

def FDsl(x, p0, p1, p2, p3, p4):
    """
    Fermi Dirac Function sloped
    p3 + (p2 + p4 * x) * (np.exp((x - p1) / p0) + 1) ** -1
    """
    return p3 + (p2 + p4 * x) * (np.exp((x - p1) / p0) + 1) ** -1

def poly2(x, p0, p1, p2, p3):
    """
    Polynomial second order
    p1 + p2 * (x - p0) + p3 * (x - p0)**2 
    """
    return p1 + p2 * (x - p0) + p3 * (x - p0)**2 
