#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 15:37:10 2018

@author: denyssutter
"""

import RTDC

#RTDC.Cell_Def(Q=1.2e-11, r_0=7.658e-6, save_data=False)
#RTDC.Stream_Func(Q=1.2e-11, r_0=9e-6)
#RTDC.Coefficients(res=100, N=40)
#RTDC.Fit_Shell(Eh_ini=1, Q=1.2e-11, gamma_pre=.1, it_max=500, alpha=1e-2)
RTDC.Fit_Sphere(E_0_ini=1e3, Q=1.2e-11, it_max=500, alpha=3e1)