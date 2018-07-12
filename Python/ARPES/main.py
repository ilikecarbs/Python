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
CSROfig6:  Analysis MDC's beta band
CSROfig7:  Background subtraction
CSROfig8:  Extraction LDA Fermi velocity
"""
#--------
#utils_plt.CROfig1(print_fig=True)
#utils_plt.CROfig2(print_fig=True)
#utils_plt.CROfig3(print_fig=True)
#utils_plt.CROfig4(print_fig=True)
#utils_plt.CROfig5(print_fig=True)
#utils_plt.CROfig6(print_fig=True)
#utils_plt.CROfig7(print_fig=True)
#utils_plt.CROfig8(print_fig=True)
#utils_plt.CROfig9(print_fig=True)
#utils_plt.CROfig10(print_fig=True)
#utils_plt.CROfig11(print_fig=True)
#utils_plt.CROfig12(print_fig=True)
#utils_plt.CROfig13(print_fig=True)
#utils_plt.CROfig14(print_fig=True)
#--------
#utils_plt.CSROfig1(print_fig=True)
#utils_plt.CSROfig2(print_fig=True)
#utils_plt.CSROfig3(print_fig=True)
#utils_plt.CSROfig4(print_fig=True)
#Z = utils_plt.CSROfig5(print_fig=True)
#(Pos, ePos, Width, eWidth) = utils_plt.CSROfig6(print_fig=True)
#utils_plt.CSROfig7(print_fig=True)
v_LDA = utils_plt.CSROfig8(print_fig=True)

#%%
v_LDA = 2.3411686586990417 
colmap = cm.ocean_r
os.chdir('/Users/denyssutter/Documents/library/Python/ARPES')
files = [25, 26, 27, 28]
gold = 14
mat = 'CSRO20'
year = 2017
sample = 'S1'

spec = ()
espec = ()
en = ()
k = ()
Width = ()
eWidth = ()
Pos = ()
ePos = ()
T = np.array([1.3, 10., 20., 30.])
mdc_t_val = .001
mdc_b_val = -.1
n_spec = 4
n_show = 30
scale = 5e-5
vLDA   = 2.1909
c = (0, 238 / 256, 118 / 256)
cols = ([0, 1, 1], [0, .7, .7], [0, .4, .4], [0, 0, 0])
cols_r = ([0, 0, 0], [0, .4, .4], [0, .7, .7], [0, 1, 1])
for j in range(n_spec): 
    D = ARPES.Bessy(files[j], mat, year, sample)
    D.norm(gold)
    D.restrict(bot=.7, top=.9, left=.31, right=.6)
    D.bkg(norm=True)
    if j == 0:
        D.ang2k(D.ang, Ekin=40, lat_unit=True, a=5.5, b=5.5, c=11, 
                  V0=0, thdg=2.5, tidg=0, phidg=42)
        int_norm = D.int_norm * 1.5
        eint_norm = D.eint_norm * 1.5
    else: 
        D.ang2k(D.ang, Ekin=40, lat_unit=True, a=5.5, b=5.5, c=11, 
                  V0=0, thdg=2.9, tidg=0, phidg=42)
        int_norm = D.int_norm
        eint_norm = D.eint_norm        
    en_norm = D.en_norm - .008
    spec = spec + (int_norm,)
    espec = espec + (eint_norm,)
    en = en + (en_norm,)
    k = k + (D.ks,)
    
plt.figure('2007', figsize=(10, 10), clear=True)
titles = [r'$T=1.3\,$K', r'$T=10\,$K', r'$T=20\,$K', r'$T=30\,$K']
lbls = [r'(a)', r'(b)', r'(c)', r'(d)',
            r'(e)', r'(f)', r'(g)', r'(h)',
            r'(i)', r'(j)', r'(k)', r'(l)']
for j in range(n_spec):
    ###First row###
    ax = plt.subplot(3, 4, j + 1) 
    ax.set_position([.08 + j * .21, .66, .2, .2])
    plt.tick_params(direction='in', length=1.5, width=.5, colors='k')
    plt.contourf(k[j], en[j], spec[j], 200, cmap=colmap,
                     vmin=.0 * np.max(spec[0]), vmax=1 * np.max(spec[0]))
    if j == 0:
        plt.ylabel('$\omega\,(\mathrm{meV})$', fontdict = font)
        plt.yticks(np.arange(-.2, .1, .05), ('-200', '-150', '-100', '-50', '0', '50'))
    else:
        plt.yticks(np.arange(-.2, .1, .05), [])
    plt.xticks(np.arange(-.8, -.2, .2), [])
    plt.xlim(xmax=-.1, xmin=-.6)
    plt.ylim(ymax=.05, ymin=-.15)
    plt.text(-.585, .035, lbls[j])
    plt.title(titles[j], fontsize=15)
    


#%%
from scipy import integrate
d = 1e-6
plt.figure(2005, figsize=(10, 10), clear=True)
D = 1e6
p_edc_i = np.array([6.9e-1, 7.3e-3, 4.6, 4.7e-3, 4.1e-2, 2.6e-3,
                    1e0, -.2, .3, 1, -.1, 1e-1])
bounds_fl = ([p_edc_i[0] - D, p_edc_i[1] - d, p_edc_i[2] - d,
              p_edc_i[3] - D, p_edc_i[4] - D, p_edc_i[5] - D],
             [p_edc_i[0] + D, p_edc_i[1] + d, p_edc_i[2] + d, 
              p_edc_i[3] + D, p_edc_i[4] + D, p_edc_i[5] + D])

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
    
    p_fl, cov_fl = curve_fit(
            utils_math.FL_simple, en[j][_EDC_e[j]][900:-1], 
            EDCn_e[j][900:-1], 
            p_edc_i[0: -6], 
            bounds=bounds_fl, sigma=eEDCn_e[j][900:-1])
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
    p = plt.plot(xx, f_edc,'--', color=cols_r[j],  linewidth=2)
    plt.legend(p, [r'$\mathcal{A}_\mathrm{coh.} + \mathcal{A}_\mathrm{inc.}$'], frameon=False)
    ###Calculate Z###
    A_mod = integrate.trapz(f_mod, xx)
    A_fl = integrate.trapz(f_fl, xx)
    Z[j] = A_fl / A_mod

#%%
from uncertainties import ufloat
import uncertainties.unumpy as unumpy

xx = np.arange(-2, .5, .01)
def uFL_simple(x, p0, p1, p2, p3, p4, p5,
           ep0, ep1, ep2, ep3, ep4, ep5):
    
    ReS = ufloat(p0, ep0) * x
    ImS = ufloat(p1, ep1) + ufloat(p2, ep2) * x ** 2;
    
    return (ufloat(p4, ep4) * 1 / np.pi * 
            ImS / ((x - ReS - ufloat(p3, ep3)) ** 2 + ImS ** 2) * 
            (unumpy.exp((x - 0) / ufloat(p5, ep5)) + 1) ** -1)
        
perr_fl = np.sqrt(np.diag(cov_fl))
pfull_fl = np.concatenate((p_fl, perr_fl), axis=0)

uf_fl = uFL_simple(xx, *pfull_fl)

plt.figure(20005, figsize=(8, 8), clear=True)
plt.subplot(121)
plt.errorbar(xx, unumpy.nominal_values(uf_fl), yerr=unumpy.std_devs(uf_fl))
plt.subplot(122)
plt.plot(xx, unumpy.nominal_values(uf_fl))


    
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
os.chdir('/Users/denyssutter/Documents/library/Python/ARPES')
file = 62151
#file = 62087
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










