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
import ARPES
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
fig5: Experimental Data of Nature Comm.
fig6: Constant energy map CaRuO4 of alpha branch
"""


uplt.fig5(
        colmap=cm.ocean_r, print_fig = True
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
import os
os.chdir('/Users/denyssutter/Documents/library/Python/ARPES')
import matplotlib.cm as cm
import h5py
import numpy as np

#7991 7992

file = 'CRO_SIS_0048'
mat = 'Ca2RuO4'
year = 2015
sample = 'data'
folder = ''.join(['/Users/denyssutter/Documents/Denys/',str(mat),
                  '/SIS',str(year),'/',str(sample),'/'])
filename = ''.join([str(file),'.h5'])
path = folder + filename
f = h5py.File(path,'r')
data  = (f['Electron Analyzer/Image Data'])
intensity = np.array(data)
hv  = np.array(f['Other Instruments/hv'])
pol  = np.array(f['Other Instruments/Tilt'])
d1, d2, d3 = data.shape
e_i, de = data.attrs['Axis0.Scale']
a_i, da = data.attrs['Axis1.Scale']
en = np.arange(e_i, e_i + d1 * de, de)
ang = np.arange(a_i, a_i + d2 * da, da)

#%%

import os
os.chdir('/Users/denyssutter/Documents/library/Python/ARPES')
import matplotlib.cm as cm
import ARPES

#7991 7992

file = 'CRO_SIS_0048'
mat = 'Ca2RuO4'
year = 2015
sample = 'data'

D = ARPES.SIS(file, mat, year, sample)
#%%

import os
os.chdir('/Users/denyssutter/Documents/library/Python/ARPES')
import matplotlib.cm as cm
import ARPES

#7991 7992

file = 'CSRO_P1_0032'
mat = 'CSRO20'
year = 2017
sample = 'data'

D = ARPES.SIS(file, mat, year, sample)
#%%

os.chdir('/Users/denyssutter/Documents/library/Python/ARPES')

file = 62087
gold = 62081
#file = 62090
#gold = 62091
mat = 'CSRO20'
year = 2017
sample = 'S6'

D = ARPES.DLS(file, mat, year, sample)
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

    
    
    






















