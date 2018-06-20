#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 10 13:20:31 2018

@author: denyssutter
"""

import os
os.chdir('/Users/denyssutter/Documents/Python/ARPES')

import numpy as np
from ARPES import DLS


file = 62087
gold = 62081
#file = 62090
#gold = 62091
mat = 'CSRO20'
year = 2017
sample = 'S6'

D = DLS(file, mat, year, sample)

#%%
#u.gold(gold, mat, year, sample, Ef_ini=17.63, BL='DLS')
D.norm(gold)

#%%
D.FS(e = -0.01, ew = .02, norm = True)

D.ang2kFS(D.ang, Ekin=22-4.5, a=5.33, b=5.33, c=11, V0=0, thdg=0, tidg=0, phidg=0)

#%%
D.plt_FS(coord = True)

#%%
D.plt_spec(norm = True)

#%%
import os
os.chdir('/Users/denyssutter/Documents/Python/ARPES')

import models as mdl

tb = mdl.TB(a = 1*np.pi, kpoints = 200)

#param = mdl.paramSRO()
param = mdl.paramCSRO20()

#tb.simple(param)
tb.CSRO(param)
#tb.SRO(param)

tb.plt_cont_TB_CSRO20()


#%%
import utils as ut

ut.FS_GUI(D = D.int_norm)
#%%

    
    
    






















