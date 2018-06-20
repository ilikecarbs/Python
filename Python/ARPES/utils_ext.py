#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 21:58:21 2018

@author: denyssutter
"""

import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# +----------+ #
# | Colormap | # ===============================================================
# +----------+ #
def rainbow_light():
    # Rainbox ligth colormap from ALS
    # ------------------------------------------------------------------------------
    
    # Load the colormap data from file
    filepath = '/Users/denyssutter/Documents/Python/ARPES/cmap/rainbow_light.dat'
    data = np.loadtxt(filepath)
    colors = np.array([(i[0], i[1], i[2]) for i in data])
    
    # Normalize the colors
    colors /= colors.max()
    
    # Build the colormap
    rainbow_light = LinearSegmentedColormap.from_list('rainbow_light', colors, 
                                                      N=len(colors))
    return rainbow_light

