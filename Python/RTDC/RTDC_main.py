#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 15:37:10 2018

@author: denyssutter

%%%%%%%%%%%%%%%
   RTDC_main
%%%%%%%%%%%%%%%

**Script for calling subscripts and create figures**

.. note::
        To-Do:
            - In the fit procedure, the initial radius for the undeformed cell
              is given as follows: Measure area A and extract radius r_0 from
              a circle with area A. Perhaps finding a way to fit r_0 would
              be an improvement.
"""

import os
import numpy as np
import RTDC


# %%

"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Generate Coefficients
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""

RTDC.Coefficients(res=100, N=40)

# %%

"""
%%%%%%%%%%%%%%%%
    Figures
%%%%%%%%%%%%%%%%
"""

RTDC.Cell_Def(Q=1.2e-11,
              r_0=7.658e-6,
              save_data=True)

# %%

RTDC.Stream_Func(Q=1.2e-11,
                 r_0=8e-6)

# %%

"""
%%%%%%%%%%%%%%%%%%%%%%%
    Fit Data (shell)
%%%%%%%%%%%%%%%%%%%%%%%

**Fit procedure for model of an elastic shell with surface tension.**

Demonstration with coordinates coord_sh (extracted from RTDC.Cell_Def).
When using real data, feed data as (x_0, z_0).
z_0: Flow axis
x_0: Perpendicular to flow axis.

**Note:**
-   Rotate input data such that flow axis is vertical. Use for example:

    import RTDC_utils as utils
    x_0, z_0 = utils.R2(x_old, z_old, th=np.pi/2)

"""

# Directory path
data_dir = '/Users/denyssutter/Documents/Denys/Zivi/data/'

os.chdir(data_dir)
coord_sh = np.loadtxt('coord_sh.dat')

x_0 = coord_sh[0, :] + 2e-6  # x-data in meters with some distortions in x-pos.
z_0 = coord_sh[1, :] + 2e-6  # x-data in meters with some distortions in z-pos.

RTDC.Fit_Shell(x_0, z_0,
               Eh_ini=1,
               Q=1.2e-11,
               gamma_pre=.1,
               it_max=500,
               alpha=2e-2)

# %%

"""
%%%%%%%%%%%%%%%%%%%%%%%%
    Fit Data (sphere)
%%%%%%%%%%%%%%%%%%%%%%%%

**Fit procedure for model of a sphere.**

Demonstration with coordinates coord_sh (extracted from RTDC.Cell_Def).
When using real data, feed data as (x_0, z_0).
z_0: Flow axis
x_0: Perpendicular to flow axis.

**Note:**
-   Rotate input data such that flow axis is vertical. Use for example:

    import RTDC_utils as utils
    x_0, z_0 = utils.R2(x_old, z_old, th=np.pi/2)

"""

os.chdir(data_dir)
coord_sp = np.loadtxt('coord_sp.dat')

x_0 = coord_sp[0, :] + 2e-6  # x-data in meters with some distortions in x-pos.
z_0 = coord_sp[1, :] + 2e-6  # x-data in meters with some distortions in z-pos.

RTDC.Fit_Sphere(x_0, z_0,
                E_0_ini=1e3,
                Q=1.2e-11,
                it_max=500,
                alpha=3e1)

# %%

"""
%%%%%%%%%%%%%%%%%%%%%%%%%%
    Area-vs-Deformation
%%%%%%%%%%%%%%%%%%%%%%%%%%
"""
RTDC.Area_vs_Def(Q=1.2e-11,
                 eh=3.4e-3,
                 e_0=270)
