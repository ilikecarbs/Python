#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Created on Tue Jun 19 15:14:29 2018

@author: denyssutter
"""

import os
os.chdir('/Users/denyssutter/Documents/library/Python/ARPES')
import utils_plt
import utils_math 
import utils
import matplotlib.pyplot as plt
import ARPES
import numpy as np
import time
import matplotlib.cm as cm


rainbow_light = utils_plt.rainbow_light
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
--------   Ca2RuO4 Figures   --------
CROfig1:   DFT plot Ca2RuO4: figure 3 of Nature Comm.
CROfig2:   (L): DMFT pot Ca2RuO4: figure 3 of Nature Comm.
CROfig3:   DFT plot orbitally selective Mott scenario
CROfig4:   DFT plot uniform gap scnenario
CROfig5:   Experimental Data of Nature Comm.
CROfig6:   Constant energy map CaRuO4 of alpha branch
CROfig7:   Photon energy dependence Ca2RuO4: figure 2 of Nature Comm.
CROfig8:   Polarization dependence Ca2RuO4: figure 2 of Nature Comm.
CROfig9:   (L): DMFT plot Ca2RuO4 dxy/dxz,yz: figure 4 of Nature Comm.
CROfig10:  (L): DFT plot Ca2RuO4: spaghetti and spectral representation
CROfig11:  Multiplet analysis Ca2RuO4
CROfig12:  Constant energy maps oxygen band -5.2eV
CROfig13:  Constant energy maps alpha band -0.5eV
CROfig14:  Constant energy maps gamma band -2.4eV

--------   Ca1.8Sr0.2RuO4 Figures --------
CSROfig1:  Experimental data: Figure 1 CSRO20 paper
CSROfig2:  Experimental PSI data: Figure 2 CSCRO20 paper
"""

utils_plt.CSROfig2(print_fig=True)

#%%
from scipy.optimize import curve_fit


os.chdir('/Users/denyssutter/Documents/library/Python/ARPES')

file = 25
file_LH = 19
file_LV = 20
gold = 14
mat = 'CSRO20'
year = 2017
sample = 'S1'

D = ARPES.Bessy(file, mat, year, sample)
LH = ARPES.Bessy(file_LH, mat, year, sample)
LV = ARPES.Bessy(file_LV, mat, year, sample)

#utils.gold(gold, mat, year, sample, Ef_ini=35.72, BL='Bessy')

colmap = cm.ocean_r
D.norm(gold)
LH.norm(gold)
LV.norm(gold)

#D.shift(gold)
#LH.shift(gold)
#LV.shift(gold)
D.restrict(bot=.7, top=.9, left=0, right=1)
LH.restrict(bot=.55, top=.85, left=0, right=1)
LV.restrict(bot=.55, top=.85, left=0, right=1)

D.ang2k(D.ang, Ekin=35-4.5, lat_unit=True, a=5.5, b=5.5, c=11, 
          V0=0, thdg=2.4, tidg=0, phidg=-45)
LH.ang2k(LH.ang, Ekin=35-4.5, lat_unit=True, a=5.5, b=5.5, c=11, 
          V0=0, thdg=2.4, tidg=0, phidg=-45)
LV.ang2k(LV.ang, Ekin=35-4.5, lat_unit=True, a=5.5, b=5.5, c=11, 
          V0=0, thdg=2.4, tidg=0, phidg=-45)
c = (0, 238 / 256, 118 / 256)

data = (D.int_norm, LH.int_norm, LV.int_norm)
en = (D.en_norm - .008, LH.en_norm, LV.en_norm)
ks = (D.ks, LH.ks, LV.ks)
k = (D.k[0], LH.k[0], LV.k[0])
#%%
###MDC###


plt.figure(2003, figsize=(8, 8), clear=True)
for j in [1, 2]:
    mdc_val = .003
    mdcw_val = .006
    mdc = np.zeros(k[j].shape)
    for i in range(len(k[j])):
        val, _mdc = utils.find(en[j][i, :], mdc_val)
        val, _mdcw = utils.find(en[j][i, :], mdc_val - mdcw_val)
        mdc[i] = np.sum(data[j][i, _mdcw:_mdc])
    mdc = mdc / np.max(mdc)
    mdc[0] = 0
    mdc[-1] = 0
    
    ax = plt.subplot(2, 3, j + 1) 
    ax.set_position([.08 + j * .26, .5, .25, .25])
    plt.tick_params(direction='in', length=1.5, width=.5, colors='k')
    plt.contourf(ks[j], en[j], data[j], 200, cmap=colmap,
                 vmin=.05 * np.max(data[j]), vmax=.35 * np.max(data[j]))
    plt.plot([np.min(ks[j]), np.max(ks[j])], [0, 0], 'k:')
    plt.yticks(np.arange(-.1, .05, .02), ('-100', '-80', '-60', '-40', '-20', '0',
               '20', '40'))
    plt.xticks([-1.5, -1, -.5, 0, .5, 1.0])
    plt.xlim(xmax=np.max(ks[j]), xmin=np.min(ks[j]))   
    plt.ylim(ymax=.05, ymin=-.1)
    plt.ylabel('$\omega\,(\mathrm{meV})$', fontdict = font)
    plt.xlabel('$k_x \,(\pi/a)$', fontdict = font)
    plt.plot(k[j], mdc / 30 + .001, 'o', markersize=1.5, color='C9')
    plt.fill(k[j], mdc / 30 + .001, alpha=.2, color='C9')
    plt.text(-1, .038, r'(a)', fontsize=12)

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

#%%
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
tb = utils_math.TB(a = np.pi, kbnd = 2, kpoints = 200)  #Initialize tight binding model

####SRO TB hopping parameters###
#param = umath.paramSRO()  
param = utils_math.paramCSRO20()  

###Calculate and Plot FS###
#tb.simple(param) 
#tb.SRO(param) 
tb.CSRO(param)


#tb.plt_cont_TB_SRO()
tb.plt_cont_TB_CSRO20()

print(time.time()-start)
#%%










