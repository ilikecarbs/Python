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
"""

utils_plt.CROfig8(print_fig=True)

#%%
colmap = cm.ocean_r
p65 = '0618_00113'
s65 = '0618_00114'
p120 = '0618_00115'
s120 = '0618_00116'
mat = 'Ca2RuO4'
year = 2016
sample = 'data'
files = [p120, s120, p65, s65]
lbls1 = ['(a)', '(b)', '(c)', '(d)']
lbls2 = [r'$120\,\mathrm{eV}$', r'$120\,\mathrm{eV}$', r'$65\,\mathrm{eV}$', r'$65\,\mathrm{eV}$']
lbls3 = [r'$\bar{\pi}$-pol.', r'$\bar{\sigma}$-pol.', r'$\bar{\pi}$-pol.', r'$\bar{\sigma}$-pol.']

th = 25
ti = -.5
phi = -25.
c = (0, 238 / 256, 118 / 256)

###Plotting###
plt.figure(1012, figsize=(10, 10), clear=True)

for i in range(4):
    D = ARPES.ALS(files[i], mat, year, sample) #frist scan
    D.ang2kFS(D.ang, Ekin=D.hv - 4.5 - 5.2, lat_unit=True, a=4.8, b=5.7, c=11, 
                    V0=0, thdg=th, tidg=ti, phidg=phi)
    en = D.en - 2.1 #energy off set (Fermi level not specified)
    e = -5.2; ew = 0.1
    e_val, e_ind = utils.find(en, e)
    ew_val, ew_ind = utils.find(en, e-ew)
    FSmap = np.sum(D.int[:, :, ew_ind:e_ind], axis=2) #creating FS map
    ax = plt.subplot(1, 5, i + 2) 
    ax.set_position([.06 + (i * .23), .3, .22, .3])
    if i == 2:
        ax.set_position([.06 + (i * .23), .3, .16, .3])
    elif i == 3:
        ax.set_position([.06 + (2 * .23) + .17, .3, .16, .3])
    plt.tick_params(direction='in', length=1.5, width=.5, colors='k')
    plt.contourf(D.kx, D.ky, FSmap, 300, cmap = colmap,
                   vmin = .25 * np.max(FSmap), vmax = .95 * np.max(FSmap))
    plt.grid(alpha=0.5)
    plt.xticks(np.arange(-10, 10, 2))
    plt.xlabel('$k_x$ ($\pi/a$)', fontdict = font)
    plt.plot([-1, -1], [-1, 1], 'k-')
    plt.plot([1, 1], [-1, 1], 'k-')
    plt.plot([-1, 1], [1, 1], 'k-')
    plt.plot([-1, 1], [-1, -1], 'k-')
    plt.plot([-2, 0], [0, 2], 'k--', linewidth=.5)
    plt.plot([-2, 0], [0, -2], 'k--', linewidth=.5)
    plt.plot([2, 0], [0, 2], 'k--', linewidth=.5)
    plt.plot([2, 0], [0, -2], 'k--', linewidth=.5)
    if i == 0:
        plt.ylabel('$k_y$ ($\pi/a$)', fontdict = font)
        plt.yticks(np.arange(-10, 10, 2))
        plt.plot([-1, 1], [-1, 1], linestyle=':', color=c, linewidth=1)
        plt.plot([-1, 1], [1, 1], linestyle=':', color=c, linewidth=1)
        plt.plot([-1, 0], [1, 2], linestyle=':', color=c, linewidth=1)
        plt.plot([0, 0], [2, -1], linestyle=':', color=c, linewidth=1)
        ax.arrow(-1, -1, .3, .3, head_width=0.3, head_length=0.3, fc=c, ec='k')
        ax.arrow(0, -.4, 0, -.3, head_width=0.3, head_length=0.3, fc=c, ec='k')
    else:
        plt.yticks(np.arange(-10, 10, 2), [])
    if any(x==i for x in [0, 1]):
        x_pos = -2.7
    else:
        x_pos = -1.9
    plt.text(x_pos, 5.6, lbls1[i], fontsize=12)
    plt.text(x_pos, 5.0, lbls2[i], fontsize=10)
    plt.text(x_pos, 4.4, lbls3[i], fontsize=10) 
    plt.text(-0.15, -0.1, r'$\Gamma$',
             fontsize=12, color='r')
    plt.text(-0.15, 1.9, r'$\Gamma$',
             fontsize=12, color='r')
    plt.text(.9, .9, r'S',
             fontsize=12, color='r')
    plt.text(-0.15, .9, r'X',
             fontsize=12, color='r')
    plt.xlim(xmin=-3, xmax=4)
    if any(x==i for x in [2, 3]):
        plt.xlim(xmin=-2.2, xmax=2.9)
    plt.ylim(ymin=-3.3, ymax=6.2)

#%%

#pos = ax.get_position()
#cax = plt.axes([pos.x0+pos.width+0.03 ,
#                    pos.y0, 0.03, pos.height])
#cbar = plt.colorbar(cax = cax, ticks = None)
#cbar.set_ticks([])
#plt.show()
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










