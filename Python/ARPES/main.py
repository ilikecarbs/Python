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

CSROfig1:  Experimental data: Figure 1 CSCRO20 paper
"""

uplt.CSROfig1(print_fig=True)


#%%
###Load Data###
os.chdir('/Users/denyssutter/Documents/library/Python/ARPES')
file = 62087
gold = 62081
mat = 'CSRO20'
year = 2017
sample = 'S6'
D = ARPES.DLS(file, mat, year, sample)
D.norm(gold)
D.restrict(bot=0, top=1, left=.12, right=.9)
D.FS(e = 0.02, ew = .03, norm = True)
D.ang2kFS(D.ang, Ekin=22-4.5, lat_unit=True, a=5.5, b=5.5, c=11, 
          V0=0, thdg=8.7, tidg=4, phidg=88)
FS = np.flipud(D.map)
file = 62090
gold = 62091
A1 = ARPES.DLS(file, mat, year, sample)
A1.norm(gold)
A1.ang2k(A1.ang, Ekin=22-4.5, lat_unit=True, a=5.5, b=5.5, c=11, 
          V0=0, thdg=9.3, tidg=0, phidg=90)
file = 62097
gold = 62091
A2 = ARPES.DLS(file, mat, year, sample)
A2.norm(gold)
A2.ang2k(A1.ang, Ekin=22-4.5, lat_unit=True, a=5.5, b=5.5, c=11, 
          V0=0, thdg=6.3, tidg=-16, phidg=90)
c = (0, 238 / 256, 118 / 256)
###MDC###
mdc_val = -.004
mdcw_val = .002
mdc = np.zeros(A1.ang.shape)
for i in range(len(A1.ang)):
    val, _mdc = u.find(A1.en_norm[i, :], mdc_val)
    val, _mdcw = u.find(A1.en_norm[i, :], mdc_val - mdcw_val)
    mdc[i] = np.sum(A1.int_norm[i, _mdcw:_mdc])
mdc = mdc / np.max(mdc)
plt.figure(20001, figsize=(4, 4), clear=True)
###Fit MDC###
delta = 1e-5
p_mdc_i = np.array(
            [-1.4, -1.3, -1.1, -.9, -.7, -.6, -.3, .3,
             .05, .05, .05, .05, .05, .05, .1, .1, 
             .3, .3, .4, .4, .5, .5, .1, .1,
             .33, 0.02, .02])
from scipy.optimize import curve_fit
bounds_bot = np.concatenate((p_mdc_i[0:-3] - np.inf, p_mdc_i[-3:27] - delta))
bounds_top = np.concatenate((p_mdc_i[0:-3] + np.inf, p_mdc_i[-3:27] + delta))
p_mdc_bounds = (bounds_bot, bounds_top)
p_mdc, cov_mdc = curve_fit(
        umath.lor8, A1.k[1], mdc, p_mdc_i, bounds=p_mdc_bounds)
b_mdc = umath.poly2(A1.k[1], 0, p_mdc[-3], p_mdc[-2], p_mdc[-1])
f_mdc = umath.lor8(A1.k[1], *p_mdc)
plt.plot(A1.k[1], mdc, 'bo')
plt.plot(A1.k[1], f_mdc)
plt.plot(A1.k[1], b_mdc, 'k--')
    
def CSROfig1a():
    ax = plt.subplot(1, 3, 1) 
    ax.set_position([.08, .3, .28, .35])
    plt.tick_params(direction='in', length=1.5, width=.5, colors='k')
    plt.contourf(A1.en_norm, A1.kys, A1.int_norm, 100, cmap=cm.ocean_r,
                 vmin=.1 * np.max(A1.int_norm), vmax=.8 * np.max(A1.int_norm))
    plt.plot([0, 0], [np.min(A1.kys), np.max(A1.kys)], 'k:')
    plt.plot([-.005, -.005], [np.min(A1.kys), np.max(A1.kys)], linestyle='--',
             color=c, linewidth=.5)
    plt.xlim(xmax=.03, xmin=-.06)
    plt.ylim(ymax=np.max(D.ky), ymin=np.min(D.ky))   
    plt.xticks(np.arange(-.06, .03, .02), ('-60', '-40', '-20', '0', '20'))
    plt.yticks([-1.5, -1, -.5, 0, .5])
    plt.xlabel('$\omega\,(\mathrm{meV})$', fontdict = font)
    plt.ylabel('$k_x \,(\pi/a)$', fontdict = font)
    plt.plot((mdc - b_mdc) / 30 + .001, A1.k[1], 'o', markersize=1.5, color='C9')
    plt.fill((f_mdc - b_mdc) / 30 + .001, A1.k[1], alpha=.2, color='C9')
    plt.text(-.058, .56, r'(a)', fontsize=12)
    plt.text(.024, -.03, r'$\Gamma$', fontsize=12, color='r')
    plt.text(.024, -1.03, r'Y', fontsize=12, color='r')
    cols = ['k', 'b', 'b', 'b', 'b', 'm', 'C1', 'C1']
    lbls = [r'$\beta$', r'$\gamma$', r'$\gamma$', r'$\gamma$', r'$\gamma$',
            r'$\beta$', r'$\alpha$', r'$\alpha$']
    corr = np.array([.003, .002, .002, 0, -.001, 0, .004, .003])
    for i in range(8):
        plt.plot((umath.lor(A1.k[1], p_mdc[i], p_mdc[i + 8], p_mdc[i + 16], 
                 p_mdc[-3], p_mdc[-2], p_mdc[-1]) - b_mdc) / 30 + .001, 
                 A1.k[1], linewidth=.5, color=cols[i])
        plt.text(p_mdc[i + 16] / 20 + corr[i], p_mdc[i]-.03, lbls[i], 
                 fontdict=font, fontsize=10, color=cols[i])
    plt.plot((f_mdc - b_mdc) / 30 + .001, A1.k[1], color=c, linewidth=.5)

def CSROfig1c():
    ax = plt.subplot(1, 3, 3) 
    ax.set_position([.66, .3, .217, .35])
    plt.tick_params(direction='in', length=1.5, width=.5, colors='k')
    plt.contourf(-np.transpose(np.fliplr(A2.en_norm)), np.transpose(A2.kys), 
                 np.transpose(np.fliplr(A2.int_norm)), 100, cmap=cm.ocean_r,
                 vmin=.1 * np.max(A2.int_norm), vmax=.8 * np.max(A2.int_norm))
    plt.plot([0, 0], [np.min(A2.kys), np.max(A2.kys)], 'k:')
    plt.xlim(xmin=-.01, xmax=.06)
    plt.ylim(ymax=np.max(D.ky), ymin=np.min(D.ky))  
    plt.xticks(np.arange(0, .06, .02), ('0', '-20', '-40', '-60'))
    plt.yticks([-1.5, -1, -.5, 0, .5], [])
    plt.xlabel('$\omega\,(\mathrm{meV})$', fontdict = font)
    plt.text(-.0085, .56, r'(c)', fontsize=12)
    plt.text(-.008, -.03, r'X', fontsize=12, color='r')
    plt.text(-.008, -1.03, r'S', fontsize=12, color='r')
    pos = ax.get_position()
    cax = plt.axes([pos.x0+pos.width+0.01 ,
                        pos.y0, 0.01, pos.height])
    cbar = plt.colorbar(cax = cax, ticks = None)
    cbar.set_ticks([])
    cbar.set_clim(np.min(A2.int_norm), np.max(A2.int_norm))

def CSROfig1b():
    for i in range(FS.shape[1]):
        FS[:, i] = np.divide(FS[:, i], np.sum(FS[:, i]))  #Flatten
    ax = plt.subplot(1, 3, 2) 
    ax.set_position([.37, .3, .28, .35])
    plt.tick_params(direction='in', length=1.5, width=.5, colors='k')  
    plt.contourf(D.kx, D.ky, FS, 300, vmax=.9 * np.max(FS), vmin=.3 * np.max(FS),
               cmap=cm.ocean_r)
    plt.xlabel('$k_y \,(\pi/a)$', fontdict = font)
    #plt.axis('equal')
    plt.text(-.65, .56, r'(b)', fontsize=12, color='w')
    plt.text(-.05, -.03, r'$\Gamma$', fontsize=12, color='r')
    plt.text(-.05, -1.03, r'Y', fontsize=12, color='r')
    plt.text(.95, -.03, r'X', fontsize=12, color='r')
    plt.text(.95, -1.03, r'S', fontsize=12, color='r')
    plt.plot(A1.k[0], A1.k[1], linestyle='--', color=c)
    plt.plot(A2.k[0], A2.k[1], linestyle='--', color=c)
    ###Tight Binding Model###
    tb = umath.TB(a = np.pi, kbnd = 2, kpoints = 200)#Initialize 
    param = umath.paramCSRO20()  #Load parameters
    tb.CSRO(param)  #Calculate bandstructure
    bndstr = tb.bndstr  #Load bandstructure
    coord = tb.coord  #Load coordinates
    X = coord['X']; Y = coord['Y']   
    Axy = bndstr['Axy']; Bxz = bndstr['Bxz']; Byz = bndstr['Byz']
    en = (Axy, Bxz, Byz)  #Loop over sheets
    n = 0
    for i in en:
        n += 1
        C = plt.contour(X, Y, i, colors = 'black', linestyles = ':', levels = 0)
        p = C.collections[0].get_paths()
        p = np.asarray(p)
        axy = np.arange(0, 4, 1) #indices of same paths
        bxz = np.arange(16, 24, 1)
        byz = np.array([16, 17, 20, 21])
        if n == 1:
            ind = axy; col = 'r'
        elif n == 2:
            ind = bxz; col = 'b'
        elif n == 3:
            ind = byz; col = 'k'
            v = p[18].vertices
            plt.plot(v[:, 0], v[:, 1], linestyle = ':', color = 'm', markersize=1)
            v = p[19].vertices
            plt.plot(v[:, 0], v[:, 1], linestyle = ':', color = 'C1')
        for j in ind:
            v = p[j].vertices
            plt.plot(v[:, 0], v[:, 1], linestyle = ':', color = col)
    plt.xticks([-.5, 0, .5, 1])
    plt.yticks([-1.5, -1, -.5, 0, .5], [])
    plt.xlim(xmax=np.max(D.kx), xmin=np.min(D.kx))
    plt.ylim(ymax=np.max(D.ky), ymin=np.min(D.ky))     
    
plt.figure(2001, figsize=(8, 8), clear=True)
CSROfig1a()
CSROfig1b()
CSROfig1c()
  
#%%
tb = umath.TB(a = np.pi, kbnd = 2, kpoints = 200)#Initialize 
param = umath.paramCSRO20()  #Load parameters
tb.CSRO(param)  #Calculate bandstructure
bndstr = tb.bndstr  #Load bandstructure
coord = tb.coord  #Load coordinates
X = coord['X']; Y = coord['Y']   
Axy = bndstr['Axy']; Bxz = bndstr['Bxz']; Byz = bndstr['Byz']
en = (Axy, Bxz, Byz)  #Loop over sheets
n = 0
for i in en:
    n += 1
    C = plt.contour(X, Y, i, colors = 'black', linestyles = ':', levels = 0)
    p = C.collections[0].get_paths()
    p = np.asarray(p)
    axy = np.arange(0, 4, 1) #indices of same paths
    bxz = np.arange(16, 24, 1)
    byz = np.array([16, 17, 20, 21])
    if n == 1:
        ind = axy; col = 'r'
    elif n == 2:
        ind = bxz; col = 'b'
    elif n == 3:
        ind = byz; col = 'k'
        v = p[18].vertices
        plt.plot(v[:, 0], v[:, 1], linestyle = ':', color = 'm', markersize=1)
        v = p[19].vertices
        plt.plot(v[:, 0], v[:, 1], linestyle = ':', color = 'C1')
    for j in ind:
        v = p[j].vertices
        plt.plot(v[:, 0], v[:, 1], linestyle = ':', color = col)
plt.xticks([-.5, 0, .5, 1])
plt.yticks([-1.5, -1, -.5, 0, .5], [])
plt.xlim(xmax=np.max(D.kx), xmin=np.min(D.kx))
plt.ylim(ymax=np.max(D.ky), ymin=np.min(D.ky))   
p = C.collections[0].get_paths()
p = np.asarray(p)
plt.figure()
for j in range(p.size):
    v = p[j].vertices
    plt.plot(v[:, 0], v[:, 1], linestyle = ':', color = col)
    plt.text(v[0, 0], v[0, 1], str(j))
    
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
tb = umath.TB(a = np.pi, kbnd = 2, kpoints = 200)  #Initialize tight binding model

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

    
    
    






















