#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Created on Tue Jun 19 15:14:29 2018

@author: denyssutter
"""

import os
os.chdir('/Users/denyssutter/Documents/library/Python/ARPES')
import utils_plt as uplt
import utils_math as umath
import utils as u
import matplotlib.pyplot as plt
from ARPES import DLS
import numpy as np
import time
import matplotlib.cm as cm


rainbow_light = uplt.rainbow_light
cm.register_cmap(name='rainbow_light', cmap=rainbow_light)
 
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.serif']=['Computer Modern Roman']
plt.rc('font', **{'family': 'serif', 'serif': ['STIXGeneral']})

font = {'family': 'serif',
        'style': 'normal',
        'color':  [0,0,0],
        'weight': 'ultralight',
        'size': 12,
        }

#%%
"""
fig1: DFT plot Ca2RuO4: figure 3 of Nature Comm.
fig2: DMFT pot Ca2RuO4: figure 3 of Nature Comm.
fig3: DFT plot orbitally selective Mott scenario
fig4: DFT plot uniform gap scnenario
"""


uplt.fig1(
        colmap = cm.bone_r, print_fig = False
        )


#%%
    
#    plt.figure(100, figsize = (5,5))
#    plt.plot([-1, -1], [-1, 1], 'k--')
#    plt.plot([1, 1], [-1, 1], 'k--')
#    plt.plot([-1, 1], [1, 1], 'k--')
#    plt.plot([-1, 1], [-1, -1], 'k--')
#    plt.plot(D.k[0], D.k[1])
#    plt.show()

#u.gold(gold, mat, year, sample, Ef_ini=60.4, BL='DLS')


#%%

os.chdir('/Users/denyssutter/Documents/library/Python/ARPES')

file = 62087
gold = 62081
#file = 62090
#gold = 62091
mat = 'CSRO20'
year = 2017
sample = 'S6'

D = DLS(file, mat, year, sample)
#u.gold(gold, mat, year, sample, Ef_ini=17.63, BL='DLS')
D.norm(gold)

#%%
D.FS(e = -0.0, ew = .02, norm = True)
D.ang2kFS(D.ang, Ekin=22-4.5, a=5.33, b=5.33, c=11, V0=0, thdg=0, tidg=0, phidg=0)
D.plt_FS(coord = True)

#%%
os.chdir('/Users/denyssutter/Documents/library/Python/ARPES')

start = time.time()
tb = umath.TB(a = np.pi, kpoints = 200)

#param = mdl.paramSRO()
param = umath.paramCSRO20()

#tb.simple(param)
tb.CSRO(param)
#tb.SRO(param)

tb.plt_cont_TB_CSRO20()

print(time.time()-start)
#%%

    
    
    






















