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

os.chdir('/Users/denyssutter/Documents/library/Python/ARPES')
file = 8
gold = 14
mat = 'CSRO20'
year = 2017
sample = 'S1'
D = ARPES.Bessy(file, mat, year, sample)
D.norm(gold)
D.FS(e = 0.07, ew = .02, norm = True)
D.ang2kFS(D.ang, Ekin=36, lat_unit=True, a=5.5, b=5.5, c=11, 
          V0=0, thdg=2.7, tidg=-1.5, phidg=42)
FS = D.map
for i in range(FS.shape[1]):
    FS[:, i] = np.divide(FS[:, i], np.sum(FS[:, i]))  #Flatten
plt.figure(20004, figsize=(8, 8), clear=True)
ax = plt.subplot(1, 3, 2) 
ax.set_position([.3, .3, .4, .4])
plt.tick_params(direction='in', length=1.5, width=.5, colors='k')  
plt.contourf(D.kx, D.ky, FS, 300,
           cmap=cm.ocean_r)
plt.grid(alpha=.5)
#%%

#%%
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

D.ang2k(D.ang, Ekin=40, lat_unit=True, a=5.5, b=5.5, c=11, 
          V0=0, thdg=2.7, tidg=0, phidg=42)
LH.ang2k(LH.ang, Ekin=40, lat_unit=True, a=5.5, b=5.5, c=11, 
          V0=0, thdg=2.7, tidg=0, phidg=42)
LV.ang2k(LV.ang, Ekin=40, lat_unit=True, a=5.5, b=5.5, c=11, 
          V0=0, thdg=2.7, tidg=0, phidg=42)
c = (0, 238 / 256, 118 / 256)

data = (D.int_norm, LH.int_norm, LV.int_norm)
en = (D.en_norm - .008, LH.en_norm, LV.en_norm)
ks = (D.ks, LH.ks, LV.ks)
k = (D.k[0], LH.k[0], LV.k[0])
b_par = (np.array([0, .0037, .0002, .002]),
         np.array([0, .0037, .0002, .002]),
         np.array([0, .0037+.0005, .0002, .002]))

def figCSROfig3abc():
    lbls = [r'(a) C$^+$-pol.', r'(b) $\bar{\pi}$-pol.', r'(c) $\bar{\sigma}$-pol.']
    for j in range(3): 
        plt.figure(20003)
        ax = plt.subplot(2, 3, j + 1) 
        ax.set_position([.08 + j * .26, .5, .25, .25])
        mdc_val = -.005
        mdcw_val = .015
        mdc = np.zeros(k[j].shape)
        for i in range(len(k[j])):
            val, _mdc = utils.find(en[j][i, :], mdc_val)
            val, _mdcw = utils.find(en[j][i, :], mdc_val - mdcw_val)
            mdc[i] = np.sum(data[j][i, _mdcw:_mdc])
        
        b_mdc = utils_math.poly2(k[j], b_par[j][0], b_par[j][1], b_par[j][2], b_par[j][3])
    #    B_mdc = np.transpose(
    #            np.broadcast_to(b_mdc, (data[j].shape[1], data[j].shape[0])))
        plt.plot(k[j], mdc, 'bo')
        plt.plot(k[j], b_mdc, 'k--')
        plt.figure(2003)
        ax = plt.subplot(2, 3, j + 1) 
        ax.set_position([.08 + j * .26, .5, .25, .25])
        plt.tick_params(direction='in', length=1.5, width=.5, colors='k')
        if j == 0:
            plt.contourf(ks[j], en[j], data[j], 100, cmap=colmap,
                         vmin=.05 * np.max(data[j]), vmax=.35 * np.max(data[j]))
            mdc = mdc / np.max(mdc)
            plt.yticks(np.arange(-.1, .05, .02), ('-100', '-80', '-60', '-40', '-20',
                   '0', '20', '40'))
            plt.ylabel('$\omega\,(\mathrm{meV})$', fontdict = font)
        else:
            plt.contourf(ks[j], en[j], data[j], 100, cmap=colmap,
                         vmin=.3 * np.max(data[1]), vmax=.6 * np.max(data[1]))
            mdc = (mdc - b_mdc) / .005
            plt.yticks(np.arange(-.1, .05, .02), [])
        mdc[0] = 0
        mdc[-1] = 0
        plt.plot([np.min(ks[j]), np.max(ks[j])], [0, 0], 'k:')
        plt.plot([np.min(ks[j]), np.max(ks[j])], [mdc_val, mdc_val], 
                  linestyle='-.', color=c, linewidth=.5)
        plt.xticks(np.arange(-1, .5, .5), [])
        plt.xlim(xmax=np.max(ks[j]), xmin=np.min(ks[j]))   
        plt.ylim(ymax=.05, ymin=-.1)
        plt.plot(k[j], mdc / 30 + .001, 'o', markersize=1.5, color='C9')
        plt.fill(k[j], mdc / 30 + .001, alpha=.2, color='C9')
        plt.text(-1.2, .038, lbls[j], fontsize=12)
    pos = ax.get_position()
    cax = plt.axes([pos.x0+pos.width+0.01 ,
                        pos.y0, 0.01, pos.height])
    cbar = plt.colorbar(cax = cax, ticks = None)
    cbar.set_ticks([])
    cbar.set_clim(np.min(LV.int_norm), np.max(LV.int_norm))
plt.figure(2003, figsize=(8, 8), clear=True)
plt.figure(20003, figsize=(8, 8), clear=True)
figCSROfig3abc()
#%%
os.chdir('/Users/denyssutter/Documents/PhD/data')
xz_data = np.loadtxt('DMFT_CSRO_xz.dat')
yz_data = np.loadtxt('DMFT_CSRO_yz.dat')
xy_data = np.loadtxt('DMFT_CSRO_xy.dat')
os.chdir('/Users/denyssutter/Documents/library/Python/ARPES')
#%%
m, n = 8000, 351 #dimensions energy, full k-path
bot, top = 2000, 5000 #restrict energy window
DMFT_data = np.array([xz_data, yz_data, xy_data]) #combine data
DMFT_spec = np.reshape(DMFT_data[:, :, 2], (3, n, m)) #reshape into n,m
#DMFT_spec = DMFT_spec[:, :, bot:top] #restrict data to bot, top
DMFT_en   = np.linspace(-8, 8, m) #define energy data
#DMFT_en   = DMFT_en[bot:top] #restrict energy data
#[0, 56, 110, 187, 241, 266, 325, 350]  = [G,X,S,G,Y,T,G,Z]
DMFT_spec = np.transpose(DMFT_spec, (0,2,1)) #transpose
#DMFT_spec = np.sum(DMFT_spec, axis=0) #sum up over orbitals

###Data used:
SG = DMFT_spec[0, :, 110:187]
GS = np.fliplr(SG)

DMFT = np.concatenate((GS, SG, GS), axis=1)
#DMFT = GS
DMFT_k = np.linspace(-2, 1, DMFT.shape[1])

scale = .02
plt.figure(20003, figsize=(10, 10), clear = True)
###Plotting###

plt.tick_params(direction='in', length=1.5, width=.5, colors='k')    
plt.contourf(DMFT_k, DMFT_en, DMFT, 300, cmap = colmap,
               vmin=0, vmax=.5*np.max(DMFT))
plt.xlim(xmax=np.max(ks[0]), xmin=np.min(ks[0]))   
plt.ylim(ymax=.05, ymin=-.1)
#%%
os.chdir('/Users/denyssutter/Documents/library/Python/ARPES')
file = 62151
gold = 62081
mat = 'CSRO20'
year = 2017
sample = 'S6'
D = ARPES.DLS(file, mat, year, sample)
D.norm(gold)
#D.restrict(bot=0, top=1, left=.12, right=.9)
D.FS(e = 0.0, ew = .02, norm = True)
D.ang2kFS(D.ang, Ekin=22-4.5, lat_unit=True, a=5.5, b=5.5, c=11, 
          V0=0, thdg=12, tidg=-2.5, phidg=45)
FS = D.map
for i in range(FS.shape[1]):
    FS[:, i] = np.divide(FS[:, i], np.sum(FS[:, i]))  #Flatten
plt.figure(20004, figsize=(8, 8), clear=True)
ax = plt.subplot(1, 3, 2) 
ax.set_position([.3, .3, .4, .4])
plt.tick_params(direction='in', length=1.5, width=.5, colors='k')  
plt.contourf(D.kx, D.ky, FS, 300,
           cmap=cm.ocean_r)
plt.grid(alpha=.5)

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










