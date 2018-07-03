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


uplt.fig6(
        colmap=cm.ocean_r, print_fig = False
        )

#%%

import os
os.chdir('/Users/denyssutter/Documents/library/Python/ARPES')
import matplotlib.cm as cm
import ARPES
from scipy.stats import exponnorm

#7991 7992

file = 'CRO_SIS_0048'
mat = 'Ca2RuO4'
year = 2015
sample = 'data'

D = ARPES.SIS(file, mat, year, sample)
D.ang2k(D.ang, Ekin=65-4.5, lat_unit=True, a=3.89, b=3.89, c=11, 
        V0=0, thdg=-4, tidg=0, phidg=0)

#D.plt_hv()
int1 = D.int[11, :, :]
int2 = D.int[16, :, :] * 3.9
val, _edc = u.find(D.k[0], 1)
val, _mdc = u.find(D.en, -2.2)
val, _mdcw = u.find(D.en, -2.3)
edc1 = int1[_edc, :]
edc2 = int2[_edc, :]
mdc = np.sum(int1[:, _mdcw:_mdc], axis=1)
mdc = mdc / np.max(mdc)
plt.tick_params(direction='in', length=1.5, width=.5, colors='k')  
#plt.rcParams['ytick.labelleft'] = True
#plt.rcParams['xtick.labelbottom'] = True
plt.figure(2007, figsize=(8, 6), clear=True)
#lor2(x, p0, p1, p2, p3, p4, p5, p6, p7, p8)
plt.plot(D.k[0], mdc, 'bo')

plt.figure(1007, figsize=(8, 6), clear=True)


def fig7a():
    ax = plt.subplot(1, 3, 1) 
    ax.set_position([.1, .3, .2 , .6])
    plt.tick_params(direction='in', length=1.5, width=.5, colors='k')  
    plt.contourf(D.k[0], D.en, np.transpose(int1), 100, cmap=cm.ocean_r,
                 vmin = 0, vmax = 1.4e4)
    plt.plot([-1, 1.66], [0, 0], 'k:')
    plt.plot([1, 1], [-2.5, .5], 'g--', linewidth=1)
    plt.xlim(xmax = 1.66, xmin = -1)
    plt.ylim(ymax = 0.5, ymin = -2.5)
    plt.ylabel('$\omega$ (meV)', fontdict = font)
    plt.xticks([-1, 0, 1], ('S', '$\Gamma$', 'S'))
    plt.yticks(np.arange(-2.5, .5, .5))
    plt.text(-.9, 0.3, r'(a)', fontsize=15)
#    plt.plot(D.k[0], mdc)

def fig7b():
    ax = plt.subplot(1, 3, 2) 
    ax.set_position([.32, .3, .2 , .6])
    plt.tick_params(direction='in', length=1.5, width=.5, colors='k')  
    plt.contourf(D.k[0], D.en+.07, np.transpose(int2), 100, cmap=cm.ocean_r,
                 vmin = 0, vmax = 1.4e4)
    plt.plot([-1, 1.66], [0, 0], 'k:')
    plt.plot([1, 1], [-2.5, .5], 'g--', linewidth=1)
    plt.xlim(xmax = 1.66, xmin = -1)
    plt.ylim(ymax = 0.5, ymin = -2.5)
    plt.xticks([-1, 0, 1], ('S', '$\Gamma$', 'S'))
    plt.yticks(np.arange(-2.5, .5, .5), ())
    plt.text(-.9, 0.3, r'(b)', fontsize=15)
    
    pos = ax.get_position()
    cax = plt.axes([pos.x0+pos.width+0.01 ,
                        pos.y0, 0.01, pos.height])
    cbar = plt.colorbar(cax = cax, ticks = None)
    cbar.set_ticks([])
    cbar.set_clim(np.min(D.int), np.max(D.int))
    
def fig7c():
    xx = np.linspace(1, -5, 200)
    ax = plt.subplot(1, 3, 3) 
    ax.set_position([.57, .3, .2 , .6])
    plt.tick_params(direction='in', length=1.5, width=.5, colors='k')  
    plt.plot(edc1, D.en, 'o', markersize=3, color=(0, 0, .8))
    plt.plot(edc2, D.en, 'd', markersize=3, color='C0')
    plt.fill_between([0, 1.5e4], 0, -.2, color='C3', alpha=0.2)
    plt.fill(7.4e3 * exponnorm.pdf(-xx, K=2, loc=.63, scale = .2), xx, 
             alpha = .2, fc=(0, 0, .8))
    plt.fill(1.3e4 * exponnorm.pdf(-xx, K=2, loc=1.34, scale = .28), xx, 
             alpha = .2, fc='C0')
    plt.plot([0, 1.5e4], [0, 0], 'k:')
    plt.plot([0, 1.5e4], [-.2, -.2], 'k:', linewidth=.2)
    plt.text(1e3, -0.15, r'$\Delta$', fontsize=12)
    plt.text(7e2, 0.3, r'(c)', fontsize=15)
    plt.text(6e3, -.9, r'$\mathcal{A}$', fontsize=15)
    plt.text(6e3, -1.75, r'$\mathcal{B}$', fontsize=15)
    plt.xlim(xmax = 1.2e4, xmin = 0)
    plt.ylim(ymax = 0.5, ymin = -2.5)
    plt.xticks([])
    plt.yticks(np.arange(-2.5, .5, .5), ())
    plt.xlabel('Intensity (a.u)', fontdict = font)
    
fig7a()
fig7b()
fig7c()




#%%

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import exponnorm

fig, ax = plt.subplots(1, 1)


K = 1.5
mean, var, skew, kurt = exponnorm.stats(K, moments='mvsk')
x = np.linspace(exponnorm.ppf(0.01, K), exponnorm.ppf(0.99, K), 100)
ax.plot(x, exponnorm.pdf(x, K, loc=-1, scale = 1),
        'r-', lw=5, alpha=0.6, label='exponnorm pdf')
plt.xticks()           # Get locations and labels
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

    
    
    






















