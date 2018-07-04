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
(L):   long loading time!!!
fig1:  DFT plot Ca2RuO4: figure 3 of Nature Comm.
fig2:  (L): DMFT pot Ca2RuO4: figure 3 of Nature Comm.
fig3:  DFT plot orbitally selective Mott scenario
fig4:  DFT plot uniform gap scnenario
fig5:  Experimental Data of Nature Comm.
fig6:  Constant energy map CaRuO4 of alpha branch
fig7:  Photon energy dependence Ca2RuO4: figure 2 of Nature Comm.
fig8:  Polarization dependence Ca2RuO4: figure 2 of Nature Comm.
fig9:  (L): DMFT plot Ca2RuO4 dxy/dxz,yz: figure 4 of Nature Comm.
fig10: (L): DFT plot Ca2RuO4: spaghetti and spectral representation
"""


uplt.fig10(
        print_fig = True
        )

#%%
import pandas as pd
from numpy import linalg as la

###Load DFT spaghetti Plot###
os.chdir('/Users/denyssutter/Documents/PhD/data')
DFT_data = pd.read_table('DFT_CRO.dat', sep='\t')
DFT_data = DFT_data.replace({'{': '', '}': ''}, regex=True)
DFT_data = DFT_data.values
os.chdir('/Users/denyssutter/Documents/library/Python/ARPES')
###Build k-axis segments###
G = (0, 0, 0); X = (np.pi, 0, 0); Y = (0, np.pi, 0)
Z = (0, 0, np.pi); T = (0, np.pi, np.pi); S = (np.pi, np.pi, 0)    
###Data along path in k-space###
k_pts = np.array([G, X, S, G, Y, T, G, Z])
k_seg = [0]
for k in range(len(k_pts)-1):
    diff = abs(np.subtract(k_pts[k], k_pts[k + 1]))
    k_seg.append(k_seg[k] + la.norm(diff)) #extending list cummulative
###Spaceholders DFT spaghetti plot###
(M, N) = DFT_data.shape
data = np.zeros((M, N, 3))
en = np.zeros((M, N)) 
xz = np.zeros((M, N))
k = np.linspace(0, 350, M)
###Load Data spectral representation###
os.chdir('/Users/denyssutter/Documents/PhD/data')
DFT_spec = pd.read_csv('DFT_CRO_all.dat').values
os.chdir('/Users/denyssutter/Documents/library/Python/ARPES')
(m, n) = DFT_spec.shape
DFT_en = np.linspace(-3, 1.5, m)
DFT_k = np.linspace(0, 350, n)

def fig10a():
    ax = plt.subplot(121)
    ax.set_position([.1, .3, .35 , .35])
    plt.tick_params(direction='in', length=1.5, width=.5, colors='k') 
    plt.plot(0, 3, 'bo')
    plt.plot(50, 3, 'ro')
    for m in range(M):
        for n in range(N):
            data[m][n][:] = np.asfarray(DFT_data[m][n].split(','))
            en[m][n] = data[m][n][1]
            xz[m][n] = data[m][n][2]
            plt.plot(k[m], en[m, n], 'o', markersize=3, 
                     color=(xz[m, n], 0, (1-xz[m, n])))
    plt.plot([0, 350], [0, 0], 'k:')
    plt.text(10, 1.15, r'(a)', fontsize=12)
    plt.xlim(xmax=350, xmin=0)
    plt.ylim(ymax=1.5, ymin=-3)
    plt.xticks(k_seg / k_seg[-1] * 350, 
               ('$\Gamma$', 'X', 'S', '$\Gamma$', 'Y', 'T', '$\Gamma$', 'Z'));
    plt.yticks(np.arange(-3, 2, 1.))
    plt.ylabel('$\omega$ (eV)', fontdict = font)
    plt.legend(('$d_{xy}$', '$d_{xz/yz}$'), frameon=False)

def fig10b():
    ax = plt.subplot(122)
    ax.set_position([.1 + .38, .3, .35 , .35])
    plt.tick_params(direction='in', length=1.5, width=.5, colors='k') 
    plt.contourf(DFT_k, DFT_en, DFT_spec, 300, cmap=cm.bone_r,
                 vmin = 0, vmax = 25)
    plt.plot([0, 350], [0, 0], 'k:')
    plt.text(10, 1.15, r'(b)', fontsize=12)
    plt.xlim(xmax=350, xmin=0)
    plt.ylim(ymax=1.5, ymin=-3)
    plt.xticks(k_seg / k_seg[-1] * 350, 
               ('$\Gamma$', 'X', 'S', '$\Gamma$', 'Y', 'T', '$\Gamma$', 'Z'));
    plt.yticks(np.arange(-3, 2, 1.), [])
    pos = ax.get_position()
    cax = plt.axes([pos.x0+pos.width+0.01 ,
                    pos.y0, 0.01, pos.height])
    cbar = plt.colorbar(cax = cax, ticks = None)
    cbar.set_ticks([])

plt.figure(1010, figsize=(8,8), clear=True)
fig10a()
fig10b()
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

    
    
    






















