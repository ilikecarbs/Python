#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 15:37:10 2018

@author: denyssutter
"""

import RT_DC

#RT_DC.Cell_Def(Q=1.2e-11, r_0=7.658e-6)
#RT_DC.Stream_Func(Q=1.2e-11, r_0=9e-6)
#RT_DC.Coefficients(res=100, N=40)
RT_DC.Fit_Shell(E_ini=1, Q=1.2e-11, gamma_pre=.1, it_max=1000, alpha=5e-3)
