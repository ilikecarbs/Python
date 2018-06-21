#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 15:14:29 2018

@author: denyssutter
"""
import os
os.chdir('/Users/denyssutter/Documents/library/Python/ARPES')
import utils_plt as uplt
import matplotlib.pyplot as plt


"""
fig1: DFT plot Ca2RuO4: figure 3 of Nature Comm.
fig2: DMFT pot Ca2RuO4: figure 3 of Nature Comm.
fig3: DFT plot orbitally selective Mott scenario
fig4: DFT plot uniform gap scnenario
"""
#[7974,8048,7993,8028]

uplt.fig3()



#%%
plt.savefig(
'/Users/denyssutter/Documents/PhD/PhD_Denys/Chapter_Ca214/Figs/Raster/fig3.png', 
dpi = 300,bbox_inches="tight")

#%%
import os
os.chdir('/Users/denyssutter/Documents/library/Python/ARPES')

from ARPES import DLS
import utils as ut
file = 47974
gold = 48000
mat = 'Ca2RuO4'
year = 2016
sample = 'T10'

D = DLS(file, mat, year, sample)
D.shift(gold)
D.norm(gold)
D.ang2k(D.ang, Ekin=65-4.5, a=3.89, b=3.89, c=11, V0=0, thdg=0, tidg=0, phidg=0)

#%%
ut.gold(gold, mat, year, sample, Ef_ini=60.4, BL='DLS')


#%%
D.plt_spec(norm = 'shift')
import os
os.chdir('/Users/denyssutter/Documents/library/Python/ARPES')

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
os.chdir('/Users/denyssutter/Documents/library/Python/ARPES')

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

    
    
    






















