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
fig7: Photon energy dependence Ca2RuO4: figure 2 of Nature Comm.
"""


uplt.fig3(
        colmap=cm.ocean_r, print_fig = True
        )

#%%

import os
os.chdir('/Users/denyssutter/Documents/library/Python/ARPES')
import matplotlib.cm as cm
os.chdir('/Users/denyssutter/Documents/PhD/data')
xz_data = np.loadtxt('DMFT_CRO_xz.dat')
yz_data = np.loadtxt('DMFT_CRO_yz.dat')
xy_data = np.loadtxt('DMFT_CRO_xy.dat')
os.chdir('/Users/denyssutter/Documents/library/Python/ARPES')

#%%
m, n = 8000, 351 #dimensions energy, full k-path
bot, top = 2500, 5000 #restrict energy window
DMFT_data = np.array([xz_data, yz_data, xy_data]) #combine data
DMFT_spec = np.reshape(DMFT_data[:, :, 2], (3, n, m)) #reshape into n,m
DMFT_spec = DMFT_spec[:, :, bot:top] #restrict data to bot, top
DMFT_en   = np.linspace(-8, 8, m) #define energy data
DMFT_en   = DMFT_en[bot:top] #restrict energy data
#[0, 56, 110, 187, 241, 266, 325, 350]  = [G,X,S,G,Y,T,G,Z]
DMFT_spec = np.transpose(DMFT_spec, (0,2,1)) #transpose
DMFT_k = np.arange(0, 351, 1)
plt.figure(1009, figsize=(8, 6), clear=True)
plt.tick_params(direction='in', length=1.5, width=.5, colors='k')  
plt.pcolormesh(DMFT_k, DMFT_en, DMFT_spec[0, :, :], cmap=cm.bone_r,
             vmin = 0, vmax = .3)
plt.plot([0, 350], [0, 0], 'k:')
plt.xlim(xmax=350, xmin=0)
plt.xticks([0, 56, 110, 187, 241, 266, 325, 350], 
           ('$\Gamma$', 'X', 'S', '$\Gamma$', 'Y', 'T', '$\Gamma$', 'Z'))
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
D.restrict(bot=0, top=1, left=.1, right=.9)

D.FS(e = -0.0, ew = .02, norm = True)
D.ang2kFS(D.ang, Ekin=22-4.5, lat_unit=True, a=5.33, b=5.33, c=11, 
          V0=0, thdg=8.7, tidg=-4, phidg=0)
D.plt_FS(coord = True)

#%%

"""
Test Script for Tight binding models
"""

os.chdir('/Users/denyssutter/Documents/library/Python/ARPES')

start = time.time()
tb = umath.TB(a = np.pi, kpoints = 200)  #Initialize tight binding model

####SRO TB hopping parameters###
#param = umath.paramSRO()  
param = umath.paramCSRO20()  

###Calculate and Plot FS###
#tb.simple(param) 
#tb.SRO(param) 
tb.CSRO(param)


#tb.plt_cont_TB_SRO()
tb.plt_cont_TB_CSRO20()

print(time.time()-start)
#%%

    
    
    






















