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
CSROfig3:  (L): Polarization and orbital characters. Figure 3 in paper
"""

utils_plt.CSROfig3(print_fig=False)

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
colmap = cm.ocean_r
os.chdir('/Users/denyssutter/Documents/library/Python/ARPES')
files = [25, 26, 27, 28]
gold = 14
mat = 'CSRO20'
year = 2017
sample = 'S1'
edc_e_val = -.9  #EDC espilon band
edcw_e_val = .01
edc_a_val = -.34  #EDC alpha band
edcw_a_val = .01
c = (0, 238 / 256, 118 / 256)
top_e = .005; top_a = .005
bot_e = -.015; bot_a = -.015
left_e = -1.1; left_a = -.5
right_e = -.7; right_a = -.2

spec = ()
en = ()
k = ()
int_e = np.zeros((4)) 
int_a = np.zeros((4))
T = np.array([1.3, 10., 20., 30.])
EDC_e = () #EDC alpha band
EDC_a = () #EDC alpha band
_EDC_e = () #Index EDC epsilon band
_EDC_a = () #Index EDC alpha band
_Top_e = (); _Top_a = ()
_Bot_e = (); _Bot_a = ()
_Left_e = (); _Left_a = ()
_Right_e = (); _Right_a = ()


for j in range(4): 
    D = ARPES.Bessy(files[j], mat, year, sample)
    D.norm(gold)
#    D.restrict(bot=.7, top=.9, left=.33, right=.5)
#    D.restrict(bot=.7, top=.9, left=.0, right=1)
#    D.bkg(norm=True)
    if j == 0:
        D.ang2k(D.ang, Ekin=40, lat_unit=True, a=5.5, b=5.5, c=11, 
                  V0=0, thdg=2.5, tidg=0, phidg=42)
        int_norm = D.int_norm * 1.5
    else: 
        D.ang2k(D.ang, Ekin=40, lat_unit=True, a=5.5, b=5.5, c=11, 
                  V0=0, thdg=2.9, tidg=0, phidg=42)
        int_norm = D.int_norm
        
    en_norm = D.en_norm - .008
    val, _edc_e = utils.find(D.ks[:, 0], edc_e_val)
    val, _edcw_e = utils.find(D.ks[:, 0], edc_e_val - edcw_e_val)
    val, _edc_a = utils.find(D.ks[:, 0], edc_a_val)
    val, _edcw_a = utils.find(D.ks[:, 0], edc_a_val - edcw_a_val)
    val, _top_e = utils.find(en_norm[0, :], top_e)
    val, _top_a = utils.find(en_norm[0, :], top_a)
    val, _bot_e = utils.find(en_norm[0, :], bot_e)
    val, _bot_a = utils.find(en_norm[0, :], bot_a)
    val, _left_e = utils.find(D.ks[:, 0], left_e)
    val, _left_a = utils.find(D.ks[:, 0], left_a)
    val, _right_e = utils.find(D.ks[:, 0], right_e)
    val, _right_a = utils.find(D.ks[:, 0], right_a)
    
    edc_e = np.sum(int_norm[_edcw_e:_edc_e, :], axis=0)
    edc_a = np.sum(int_norm[_edcw_a:_edc_a, :], axis=0)
    int_e[j] = np.sum(int_norm[_left_e:_right_e, _bot_e:_top_e])
    int_a[j] = np.sum(int_norm[_left_a:_right_a, _bot_a:_top_a])
    spec = spec + (int_norm,)
    en = en + (en_norm,)
    k = k + (D.ks,)
    EDC_e = EDC_e + (edc_e,)
    EDC_a = EDC_a + (edc_a,)
    _EDC_e = _EDC_e + (_edc_e,)
    _EDC_a = _EDC_a + (_edc_a,)
    _Top_e = _Top_e + (_top_e,)
    _Top_a = _Top_a + (_top_a,)
    _Bot_e = _Bot_e + (_bot_e,)
    _Bot_a = _Bot_a + (_bot_a,)
    _Left_e = _Left_e + (_left_e,)
    _Left_a = _Left_a + (_left_a,)
    _Right_e = _Right_e + (_right_e,)
    _Right_a = _Right_a + (_right_a,)

int_e = int_e / int_e[0]
int_a = int_a / int_a[0]
lbls = [r'(a) $T=1.3\,$K', r'(b) $T=10\,$K', r'(c) $T=20\,$K', r'(d) $T=30\,$K']
#%%
def CSROfig4abcd():
    for j in range(4): 
        ax = plt.subplot(2, 4, j + 1) 
        ax.set_position([.08 + j * .21, .5, .2, .2])
        plt.tick_params(direction='in', length=1.5, width=.5, colors='k')
        plt.contourf(k[j], en[j], spec[j], 100, cmap=colmap,
                         vmin=.05 * np.max(spec[j]), vmax=.35 * np.max(spec[j]))
        if j == 0:
            plt.yticks(np.arange(-.1, .03, .02), ('-100', '-80', '-60', '-40', '-20',
                   '0', '20'))
            plt.ylabel('$\omega\,(\mathrm{meV})$', fontdict = font)
        else:
            plt.yticks(np.arange(-.1, .05, .02), [])
        plt.plot([np.min(k[j]), np.max(k[j])], [0, 0], 'k:')
        plt.plot([k[j][_EDC_e[j], 0], k[j][_EDC_e[j], 0]], [en[j][0, 0], en[j][0, -1]],
                 linestyle='-.', color=c, linewidth=.5)
        plt.plot([k[j][_EDC_a[j], 0], k[j][_EDC_a[j], 0]], [en[j][0, 0], en[j][0, -1]],
                 linestyle='-.', color=c, linewidth=.5)
        
        plt.plot([k[j][_Left_e[j], 0], k[j][_Left_e[j], 0]], 
                 [en[j][0, _Top_e[j]], en[j][0, _Bot_e[j]]],
                 linestyle='--', color='r', linewidth=.5)
        plt.plot([k[j][_Right_e[j], 0], k[j][_Right_e[j], 0]], 
                 [en[j][0, _Top_e[j]], en[j][0, _Bot_e[j]]],
                 linestyle='--', color='r', linewidth=.5)
        plt.plot([k[j][_Left_e[j], 0], k[j][_Right_e[j], 0]], 
                 [en[j][0, _Top_e[j]], en[j][0, _Top_e[j]]],
                 linestyle='--', color='r', linewidth=.5)
        plt.plot([k[j][_Left_e[j], 0], k[j][_Right_e[j], 0]], 
                 [en[j][0, _Bot_e[j]], en[j][0, _Bot_e[j]]],
                 linestyle='--', color='r', linewidth=.5)
        
        plt.plot([k[j][_Left_a[j], 0], k[j][_Left_a[j], 0]], 
                 [en[j][0, _Top_a[j]], en[j][0, _Bot_a[j]]],
                 linestyle='--', color='C1', linewidth=.5)
        plt.plot([k[j][_Right_a[j], 0], k[j][_Right_a[j], 0]], 
                 [en[j][0, _Top_a[j]], en[j][0, _Bot_a[j]]],
                 linestyle='--', color='C1', linewidth=.5)
        plt.plot([k[j][_Left_a[j], 0], k[j][_Right_a[j], 0]], 
                 [en[j][0, _Top_a[j]], en[j][0, _Top_a[j]]],
                 linestyle='--', color='r', linewidth=.5)
        plt.plot([k[j][_Left_a[j], 0], k[j][_Right_a[j], 0]], 
                 [en[j][0, _Bot_a[j]], en[j][0, _Bot_a[j]]],
                 linestyle='--', color='r', linewidth=.5)
        
        ax.xaxis.tick_top()
        plt.xticks(np.arange(-1, .5, 1.), [r'S', r'$\Gamma$'])
        plt.xlim(xmax=0.05, xmin=np.min(k[j]))   
        plt.ylim(ymax=.03, ymin=-.1)
        plt.text(-1.25, .018, lbls[j], fontsize=10)
        
    pos = ax.get_position()
    cax = plt.axes([pos.x0+pos.width + 0.01 ,
                        pos.y0, 0.01, pos.height])
    cbar = plt.colorbar(cax = cax, ticks = None)
    cbar.set_ticks([])
    cbar.set_clim(np.min(D.int_norm), np.max(D.int_norm))

def CSROfig4e():
    plt.figure(20004, figsize=(8, 8), clear=True)
    ax = plt.subplot(2, 4, 5) 
    ax.set_position([.08 + 0 * .21, .29, .2, .2])
    for j in range(4):
        plt.plot(en[j][_EDC_e[j]], EDC_e[j])
def CSROfig4h():
    ax = plt.subplot(2, 4, 8) 
    ax.set_position([.08 + 3 * .21, .29, .2, .2])
    plt.tick_params(direction='in', length=1.5, width=.5, colors='k')
    plt.plot(T, int_e, 'ro')
    plt.plot(T, int_a, 'C1o')
    plt.plot([1.3, 32], [1, .75], 'r--', linewidth=.5)
    plt.plot([1.3, 32], [1, 1], 'C1--', linewidth=.5)
    plt.xticks(T)
    plt.xlim(xmax=32, xmin=0)
    plt.grid(True, alpha=.2)
    plt.yticks([.8, .9, 1.])
    plt.xlabel(r'$T$ (K)', fontdict = font)
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position('right')
    plt.ylabel(r'$\int_\boxdot \mathcal{A}(k, \omega, T) \, \slash \quad \int_\boxdot \mathcal{A}(k, \omega, 1.3\,\mathrm{K})$', 
               fontdict = font, fontsize=8)
    
#plt.figure(2004, figsize=(8, 8), clear=True)
#CSROfig4abcd()
CSROfig4e()
#CSROfig4h()
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










