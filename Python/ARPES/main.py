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
plt.rcParams['xtick.top'] = plt.rcParams['xtick.bottom'] = True
plt.rcParams['ytick.right'] = plt.rcParams['ytick.left'] = True
    
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
fig8: Polarization dependence Ca2RuO4: figure 2 of Nature Comm.
fig9: DMFT plot Ca2RuO4 dxy/dxz,yz: figure 4 of Nature Comm.
"""


uplt.fig5(
        colmap=cm.bone_r, print_fig = True
        )

#%%
os.chdir('/Users/denyssutter/Documents/library/Python/ARPES')
mat = 'Ca2RuO4'
year = 2016
sample = 'T10'
plt.figure(1005, figsize = (10, 10), clear = True)
files = np.array([47974, 48048, 47993, 48028])
gold = 48000

###Plotting###
#Setting which axes should be ticked and labelled
plt.rcParams['xtick.labelbottom'] = True
plt.rcParams['xtick.labeltop'] = False
scale = .02
v_scale = 1.3
k_seg_1 = np.array([0, 4.442882938158366, 8.885765876316732])
k_seg_2 = np.array([0, 3.141592653589793, 6.283185307179586])
k_seg_3 = np.array([0, 4.442882938158366])
k_seg_4 = np.array([0, 3.141592653589793, 6.283185307179586, 9.42477796076938])

n = 0
for file in files:
    n += 1
    D = ARPES.DLS(file, mat, year, sample)
    D.shift(gold)
    D.norm(gold)
    D.restrict(bot=.6, top=1, left=0, right=1)
    D.flatten(norm=True)
    if n == 1:
        plt.rcParams['ytick.labelright'] = False
        plt.rcParams['ytick.labelleft'] = True
        ax = plt.subplot(1, 4, n) 
        ax.set_position([.1, .3, k_seg_1[-1] * scale, .3])
        pos = ax.get_position()
        D.ang2k(D.ang, Ekin=65-4.5, lat_unit=True, a=3.89, b=3.89, c=11, 
                V0=0, thdg=-4, tidg=0, phidg=0)
        plt.tick_params(direction='in', length=1.5, width=.5, colors='k')  
        plt.contourf(D.ks, D.en_norm+.1, D.int_norm, 300,
                   cmap=cm.bone_r, 
                   vmin=v_scale * 0.01 * np.max(D.int_norm), 
                   vmax=v_scale * 0.5 * np.max(D.int_norm))
        plt.xlim(xmax = 1, xmin = -1)
        plt.ylabel('$\omega$ (meV)', fontdict = font)
        plt.xticks([-1, 0, 1], ('S', '$\Gamma$', 'S'))
    elif n == 2:
        plt.rcParams['ytick.labelright'] = False
        plt.rcParams['ytick.labelleft'] = False
        ax = plt.subplot(1, 4, n)
        ax.set_position([pos.x0 + k_seg_1[-1] * scale, pos.y0, 
                         k_seg_2[-1] * scale, pos.height])
        D.ang2k(D.ang, Ekin=65-4.5, lat_unit=True, a=3.89, b=3.89, c=11, 
                V0=0, thdg=-7.5, tidg=8.5, phidg=45)
        plt.tick_params(direction='in', length=1.5, width=.5, colors='k')  
        plt.contourf(D.ks, D.en_norm+.1, D.int_norm, 300,
                   cmap=cm.bone_r,
                   vmin=v_scale * 0.0 * np.max(D.int_norm), 
                   vmax=v_scale * 0.54 * np.max(D.int_norm))
        plt.xlim(xmax = 0, xmin = -1)
        plt.xticks([-1, -.5, 0], ('', 'X', 'S'))
    elif n == 3:
        plt.rcParams['ytick.labelright'] = False
        plt.rcParams['ytick.labelleft'] = False
        ax = plt.subplot(1, 4, n)
        ax.set_position([pos.x0 + k_seg_2[-1] * scale, pos.y0, 
                         k_seg_3[-1] * scale, pos.height])
        D.ang2k(D.ang, Ekin=65-4.5, lat_unit=True, a=3.89, b=3.89, c=11, 
                V0=0, thdg=5, tidg=12.5, phidg=0)
        plt.tick_params(direction='in', length=1.5, width=.5, colors='k')  
        plt.contourf(D.ks, D.en_norm+.1, np.flipud(D.int_norm), 300,
                   cmap=cm.bone_r, 
                   vmin=v_scale * 0.01 * np.max(D.int_norm), 
                   vmax=v_scale * 0.7 * np.max(D.int_norm))
        plt.xlim(xmax = 1, xmin = 0)
        plt.xticks([0, 1], ('', '$\Gamma$'))
    elif n == 4:
        plt.rcParams['ytick.labelright'] = False
        plt.rcParams['ytick.labelleft'] = False
        ax = plt.subplot(1, 4, n)
        ax.set_position([pos.x0 + k_seg_3[-1] * scale, pos.y0, 
                         k_seg_4[-1] * scale, pos.height])
        D.ang2k(D.ang, Ekin=65-4.5, lat_unit=True, a=3.89, b=3.89, c=11, 
                V0=0, thdg=-9.5, tidg=0, phidg=45)
        plt.tick_params(direction='in', length=1.5, width=.5, colors='k')  
        plt.contourf(D.ks, D.en_norm+.1, np.flipud(D.int_norm), 300,
                   cmap=cm.bone_r, 
                   vmin=v_scale * 0.01 * np.max(D.int_norm), 
                   vmax=v_scale * 0.53 * np.max(D.int_norm))
        plt.xlim(xmax = 1.5, xmin = 0)
        plt.xticks([0, 0.5, 1, 1.5], ('', 'X', '$\Gamma$', 'X'))
    
    pos = ax.get_position()
    plt.ylim(ymax = 0, ymin = -2.5)
    plt.show()
cax = plt.axes([pos.x0 + k_seg_4[-1] * scale + 0.01,
                pos.y0, 0.01, pos.height])
cbar = plt.colorbar(cax = cax, ticks = None)
cbar.set_ticks([])


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

    
    
    






















