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
from scipy.stats import exponnorm
from scipy.optimize import curve_fit

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
CSROfig4:  (L): Temperature dependence. Figure 4 in paper
CSROfig5:  (L): Analysis Z epsilon band
"""

utils_plt.CSROfig5(print_fig=True)

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
en, EDCn_e, EDCn_b, EDC_e, EDC_b, Bkg_e, Bkg_b, _EDC_e, _EDC_b = utils_plt.CSROfig4()
#%%
from scipy import integrate
d = 1e-6
D = 1e6
p_edc_i = np.array([6.9e-1, 7.3e-3, 4.6, 4.7e-3, 4.1e-2, 2.6e-3,
                    1e0, -.2, .3, 1, -.1, 1e-1])
bounds_fl = ([p_edc_i[0] - D, p_edc_i[1] - d, p_edc_i[2] - d,
              p_edc_i[3] - D, p_edc_i[4] - D, p_edc_i[5] - D],
             [p_edc_i[0] + D, p_edc_i[1] + d, p_edc_i[2] + d, 
              p_edc_i[3] + D, p_edc_i[4] + D, p_edc_i[5] + D])

plt.figure('20005a', figsize=(10, 10), clear=True)
titles = [r'$T=1.3\,$K', r'$T=10\,$K', r'$T=20\,$K', r'$T=30\,$K']
lbls = [r'(a)', r'(b)', r'(c)', r'(d)',
        r'(e)', r'(f)', r'(g)', r'(h)',
        r'(i)', r'(j)', r'(k)', r'(l)']
cols = ([0, 1, 1], [0, .7, .7], [0, .4, .4], [0, 0, 0])
cols_r = ([0, 0, 0], [0, .4, .4], [0, .7, .7], [0, 1, 1])
xx = np.arange(-2, .5, .001)
Z = np.ones((4))
for j in range(4):
    ###First row###
    Bkg = Bkg_e[j]
    Bkg[0] = 0
    Bkg[-1] = 0
    ax = plt.subplot(5, 4, j + 1) 
    ax.set_position([.08 + j * .21, .61, .2, .2])
    plt.tick_params(direction='in', length=1.5, width=.5, colors='k')
    plt.plot(en[j][_EDC_e[j]], EDC_e[j], 'o', markersize=1, color=cols[j])
    plt.fill(en[j][_EDC_e[j]], Bkg, '--', linewidth=1, color='C8', alpha=.3)
    plt.yticks([])
    plt.xticks(np.arange(-.8, .2, .2), [])
    plt.xlim(xmin=-.8, xmax=.1)
    plt.ylim(ymin=0, ymax=.02)
    plt.text(-.77, .001, r'Background')
    plt.text(-.77, .0183, lbls[j])
    plt.title(titles[j], fontsize=15)
    if j == 0:
        plt.ylabel(r'Intensity (a.u.)')
    ###Third row#
    ax = plt.subplot(5, 4, j + 13) 
    ax.set_position([.08 + j * .21, .18, .2, .2])
    plt.tick_params(direction='in', length=1.5, width=.5, colors='k')
    plt.plot(en[j][_EDC_e[j]], EDCn_e[j], 'o', markersize=1, color=cols[j])
    
    p_fl, cov_edc = curve_fit(
            utils_math.FL_simple, en[j][_EDC_e[j]][900:-1], 
            EDCn_e[j][900:-1], 
            p_edc_i[0: -6], bounds=bounds_fl)
    f_fl = utils_math.FL_simple(xx, *p_fl)
        
    plt.yticks([])
    plt.xticks(np.arange(-.8, .2, .1))
    plt.xlim(xmin=-.1, xmax=.05)
    plt.ylim(ymin=0, ymax=1.1)
    plt.xlabel(r'$\omega$ (eV)')
    if j == 0:
        plt.ylabel(r'Intensity (a.u.)')
        plt.text(-.095, .2, 
                 r'$\int \, \, \mathcal{A}_\mathrm{coh.}(k\approx k_\mathrm{F}^{\bar\epsilon}, \omega) \, \mathrm{d}\omega$')
    plt.text(-.095, 1.01, lbls[j + 8])
    ###Second row###
    ax = plt.subplot(5, 4, j + 9) 
    ax.set_position([.08 + j * .21, .4, .2, .2])
    plt.tick_params(direction='in', length=1.5, width=.5, colors='k')
    plt.plot(en[j][_EDC_e[j]], EDCn_e[j], 'o', markersize=1, color=cols[j])
    
    bounds = (np.concatenate((p_fl - D, p_edc_i[6:] - D), axis=0),
              np.concatenate((p_fl + D, p_edc_i[6:] + D), axis=0))
    bnd = 300
    p_edc, cov_edc = curve_fit(
            utils_math.Full_mod, en[j][_EDC_e[j]][bnd:-1], EDCn_e[j][bnd:-1], 
            np.concatenate((p_fl, p_edc_i[-6:]), axis=0), bounds=bounds)
    f_edc = utils_math.Full_mod(xx, *p_edc)
    plt.plot(xx, f_edc,'--', color=cols_r[j], linewidth=1.5)
    f_mod = utils_math.gauss_mod(xx, *p_edc[-6:])
    f_fl = utils_math.FL_simple(xx, *p_edc[0:6]) 
    plt.fill(xx, f_mod, alpha=.3, color=cols[j])
    plt.yticks([])
    plt.xticks(np.arange(-.8, .2, .2))
    plt.xlim(xmin=-.8, xmax=.1)
    plt.ylim(ymin=0, ymax=2.2)
    if j == 0:
        plt.ylabel(r'Intensity (a.u.)')
        plt.text(-.68, .3, 
                 r'$\int \, \, \mathcal{A}_\mathrm{inc.}(k\approx k_\mathrm{F}^{\bar\epsilon}, \omega) \, \mathrm{d}\omega$')
    plt.text(-.77, 2.03, lbls[j + 4])  
    ###Third row###
    ax = plt.subplot(5, 4, j + 13) 
    plt.fill(xx, f_fl, alpha=.3, color=cols[j])
    p=plt.plot(xx, f_edc,'--', color=cols_r[j],  linewidth=2)
    plt.legend(p, [r'$\mathcal{A}_\mathrm{coh.} + \mathcal{A}_\mathrm{inc.}$'], frameon=False)
    ###Calculate Z###
    A_edc = integrate.trapz(f_edc, xx)
    A_mod = integrate.trapz(f_mod, xx)
    A_fl = integrate.trapz(f_fl, xx)
    Z[j] = A_fl / A_mod
print(('\n Z = ' + str(np.round(Z, 3))))
#%%
os.chdir('/Users/denyssutter/Documents/library/Python/ARPES')
#file = 62151
file = 62087
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










