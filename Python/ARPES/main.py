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
(Loc_en, Loc, eLoc, Width, eWidth) = utils_plt.CSROfig6(print_fig=True)
#utils_plt.CSROfig7(print_fig=True)
#k_F, v_LDA = utils_plt.CSROfig8(print_fig=True)

#%%
colmap=cm.ocean_r

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
Z = ()
Re = ()
Width = ()
eWidth = ()
Loc_en = ()
Loc = ()
eLoc = ()
mdc_t_val = .001
mdc_b_val = -.1
n_spec = 1
scale = 5e-5
c = (0, 238 / 256, 118 / 256)
cols = ([0, 1, 1], [0, .7, .7], [0, .4, .4], [0, 0, 0])
cols_r = ([0, 0, 0], [0, .4, .4], [0, .7, .7], [0, 1, 1])
Re_cols = ['khaki', 'darkkhaki', 'goldenrod', 'darkgoldenrod']
Re_cols_r = ['darkgoldenrod', 'goldenrod', 'darkkhaki', 'khaki']
v_LDA = 2.3411686586990417 
xx = np.arange(-.4, .25, .01)
for j in range(n_spec): 
    D = ARPES.Bessy(files[j], mat, year, sample)
    D.norm(gold)
    D.restrict(bot=.7, top=.9, left=.31, right=.6)
    D.bkg(norm=True)
    if j == 0:
        D.ang2k(D.ang, Ekin=40 - 4.5, lat_unit=False, a=5.5, b=5.5, c=11, 
                  V0=0, thdg=2.5, tidg=0, phidg=42)
        int_norm = D.int_norm * 1.5
        eint_norm = D.eint_norm * 1.5
    else: 
        D.ang2k(D.ang, Ekin=40 - 4.5, lat_unit=False, a=5.5, b=5.5, c=11, 
                  V0=0, thdg=2.9, tidg=0, phidg=42)
        int_norm = D.int_norm
        eint_norm = D.eint_norm        
    en_norm = D.en_norm - .008
    spec = spec + (int_norm,)
    espec = espec + (eint_norm,)
    en = en + (en_norm,)
    k = k + (D.ks * np.sqrt(2),)
    
plt.figure('2006', figsize=(10, 10), clear=True)
titles = [r'$T=1.3\,$K', r'$T=10\,$K', r'$T=20\,$K', r'$T=30\,$K']
lbls = [r'(a)', r'(b)', r'(c)', r'(d)',
            r'(e)', r'(f)', r'(g)', r'(h)',
            r'(i)', r'(j)', r'(k)', r'(l)',
            r'(k)', r'(l)', r'(m)', r'(n)']
for j in range(n_spec):
    val, _mdc_t = utils.find(en[j][0, :], mdc_t_val)
    val, _mdc_b = utils.find(en[j][0, :], mdc_b_val)
    mdc_seq = np.arange(_mdc_t,_mdc_b, -1)
    loc = np.zeros((_mdc_t - _mdc_b))
    eloc = np.zeros((_mdc_t - _mdc_b))
    width = np.zeros((_mdc_t - _mdc_b))
    ewidth = np.zeros((_mdc_t - _mdc_b))
    ###First row###
    ax = plt.subplot(4, 4, j + 1) 
    ax.set_position([.08 + j * .21, .76, .2, .2])
    plt.tick_params(direction='in', length=1.5, width=.5, colors='k')
    plt.contourf(k[j], en[j], spec[j], 200, cmap=colmap,
                     vmin=.0 * np.max(spec[0]), vmax=1 * np.max(spec[0]))
    plt.plot([-1, 0], [en[j][0, mdc_seq[2]], en[j][0, mdc_seq[2]]], '-.',
             color=c, linewidth=.5)
    plt.plot([-1, 0], [en[j][0, mdc_seq[50]], en[j][0, mdc_seq[50]]], '-.',
             color=c, linewidth=.5)
    plt.plot([-1, 0], [en[j][0, mdc_seq[100]], en[j][0, mdc_seq[100]]], '-.',
             color=c, linewidth=.5)
    plt.plot([-1, 0], [0, 0], 'k:')
    if j == 0:
        plt.ylabel('$\omega\,(\mathrm{meV})$', fontdict = font)
        plt.yticks(np.arange(-.2, .1, .05), 
                   ('-200', '-150', '-100', '-50', '0', '50'))
        plt.text(-.43, .009, r'MDC maxima (Lorentzian fit)', color='C8')
    else:
        plt.yticks(np.arange(-.2, .1, .05), [])
    plt.xticks(np.arange(-1, 0, .1), [])
    plt.xlim(xmax=-.05, xmin=-.45)
    plt.ylim(ymax=.05, ymin=-.15)
    plt.text(-.44, .035, lbls[j])
    plt.title(titles[j], fontsize=15)
    ###Second row###
    ax = plt.subplot(4, 4, j + 5) 
    ax.set_position([.08 + j * .21, .55, .2, .2])
    plt.tick_params(direction='in', length=1.5, width=.5, colors='k')
    n = 0
    p_mdc = []
    for i in mdc_seq:
        _sl1 = 10
        _sl2 = 155
        n += 1
        mdc_k = k[j][:, i]
        mdc_int = spec[j][:, i]
        mdc_eint = espec[j][:, i]
        if any(x==n for x in [1, 50, 100]):
            plt.errorbar(mdc_k, mdc_int - scale * n**1.15, mdc_eint, 
                         linewidth=.5, capsize=.1, color=cols[j], fmt='o', ms=.5)
#            plt.errorbar([mdc_k[_sl1], mdc_k[_sl2]], 
#                         [mdc_int[_sl1] - scale * n**1.15, mdc_int[_sl2] - scale * n**1.15], 
#                         [mdc_eint[_sl1], mdc_eint[_sl2]], 
#                         linewidth=.5, capsize=.1, color='r', fmt='o', ms=2)
        ###Fit MDC###
        d = 1e-2
        eps = 1e-8
        D = 1e5
        const_i = mdc_int[_sl2] 
        slope_i = (mdc_int[_sl1] - mdc_int[_sl2])/(mdc_k[_sl1] - mdc_k[_sl2])
        
        p_mdc_i = np.array(
                    [-.27, 5e-2, 1e-3,
                     const_i, slope_i, .0])
        if n > 70:
            p_mdc_i = p_mdc
            bounds_bot = np.array([
                            p_mdc_i[0] - d, p_mdc_i[1] - d, p_mdc_i[2] - D, 
                            p_mdc_i[3] - D, p_mdc_i[4] - eps, p_mdc_i[5] - eps])
            bounds_top = np.array([
                            p_mdc_i[0] + d, p_mdc_i[1] + d, p_mdc_i[2] + D, 
                            p_mdc_i[3] + D, p_mdc_i[4] + eps, p_mdc_i[5] + eps])
        else:
            bounds_bot = np.array([
                            p_mdc_i[0] - D, p_mdc_i[1] - D, p_mdc_i[2] - D, 
                            p_mdc_i[3] - D, p_mdc_i[4] - D, p_mdc_i[5] - eps])
            bounds_top = np.array([
                            p_mdc_i[0] + D, p_mdc_i[1] + D, p_mdc_i[2] + D, 
                            p_mdc_i[3] + D, p_mdc_i[4] + D, p_mdc_i[5] + eps])
        bounds = (bounds_bot, bounds_top)
        
        p_mdc, c_mdc = curve_fit(
            utils_math.lorHWHM, mdc_k, mdc_int, p0=p_mdc_i, bounds=bounds)
        err_mdc = np.sqrt(np.diag(c_mdc))
        loc[n - 1] = p_mdc[0]
        eloc[n - 1] = err_mdc[0]
        width[n - 1] = p_mdc[1]
        ewidth[n - 1] = err_mdc[1]
        b_mdc = utils_math.poly2(mdc_k, 0, *p_mdc[-3:])
        f_mdc = utils_math.lorHWHM(mdc_k, *p_mdc)
        if any(x==n for x in [1, 50, 100]):
            plt.plot(mdc_k, f_mdc - scale * n**1.15, '--', color=cols_r[j])
            plt.plot(mdc_k, b_mdc - scale * n**1.15, 'C8-', linewidth=2, alpha=.3)
    if j == 0:
        plt.ylabel('Intensity (a.u.)', fontdict = font)
        plt.text(-.43, -.0092, r'Background', color='C8')
    plt.yticks([])
    plt.xticks(np.arange(-1, 0, .1))
    plt.xlim(xmax=-.05, xmin=-.45)
    plt.ylim(ymin=-.01, ymax = .003)
    plt.text(-.44, .0021, lbls[j + 4])
    plt.xlabel(r'$k_{\Gamma - \mathrm{S}}\,(\mathrm{\AA}^{-1})$', fontdict=font)
    ###First row again###
    ax = plt.subplot(4, 4, j + 1) 
    loc_en = en[j][0, mdc_seq]
    plt.plot(loc, loc_en, 'C8o', ms=.5)
    if j == 3:
        pos = ax.get_position()
        cax = plt.axes([pos.x0+pos.width+0.01 ,
                            pos.y0, 0.01, pos.height])
        cbar = plt.colorbar(cax = cax, ticks = None)
        cbar.set_ticks([])
        cbar.set_clim(np.min(int_norm), np.max(int_norm))
    ###Third row###
    ax = plt.subplot(4, 4, j + 9) 
    ax.set_position([.08 + j * .21, .29, .2, .2])
    plt.tick_params(direction='in', length=1.5, width=.5, colors='k')
    plt.errorbar(-en[j][0][mdc_seq], width, ewidth,
                 color=cols[j], linewidth=.5, capsize=2, fmt='o', ms=2)
    
    im_bot = np.array([0 - eps, 1 - D, -.1 - d, 1 - D])
    im_top = np.array([0 + eps, 1 + D, -.1 + d, 1 + D])
    im_bounds = (im_bot, im_top)
    p_im, c_im = curve_fit(
            utils_math.poly2, -en[j][0][mdc_seq], width, bounds=im_bounds)
    plt.plot(-en[j][0][mdc_seq], utils_math.poly2(-en[j][0][mdc_seq], *p_im),
             '--', color=cols_r[j])
    if j == 0:
        plt.ylabel('HWHM $(\mathrm{\AA}^{-1})$', fontdict = font)
        plt.yticks(np.arange(0, 1, .05))
        plt.text(.005, .05, r'Quadratic fit', fontdict = font)
    else:
        plt.yticks(np.arange(0, 1, .05), [])
    plt.xticks(np.arange(0, .1, .02), [])
    plt.xlim(xmax=-en[j][0][mdc_seq[-1]], xmin=-en[j][0][mdc_seq[0]])
    plt.ylim(ymax=.13, ymin=0)
    plt.text(.0025, .12, lbls[j + 8])
    ###Fourth row###
    k_F = loc[0] 
    p0 = -k_F * v_LDA
    yy = p0 + xx * v_LDA
    en_LDA = p0 + loc * v_LDA
    re = loc_en - en_LDA
    
    ax = plt.subplot(4, 4, j + 13) 
    ax.set_position([.08 + j * .21, .08, .2, .2])
    plt.tick_params(direction='in', length=1.5, width=.5, colors='k')
    plt.errorbar(-loc_en, re, ewidth * v_LDA,
                 color=Re_cols[j], linewidth=.5, capsize=2, fmt='o', ms=2)
    _bot = 0
    _top = 15
    re_bot = np.array([0 - eps, 1 - D])
    re_top = np.array([0 + eps, 1 + D])
    re_bounds = (re_bot, re_top)
    p_re, c_re = curve_fit(
            utils_math.poly1, -loc_en[_bot:_top], re[_bot:_top], 
            bounds=re_bounds)
    dre = -p_re[1]
    plt.plot(-loc_en, utils_math.poly1(-loc_en, *p_re),
             '--', color=Re_cols_r[j])
    z = 1 / (1 - 1 / dre)
    
    if j == 0:
        plt.ylabel(r'Re$\Sigma$ (meV)', 
                   fontdict = font)
        plt.yticks(np.arange(0, .15, .05), ('0', '50', '100'))
#        plt.text(.005, .05, r'Quadratic fit', fontdict = font)
    else:
        plt.yticks(np.arange(0, .15, .05), [])
    plt.xticks(np.arange(0, .1, .02), ('0', '-20', '-40', '-60', '-80', '-100'))
    plt.xlabel(r'$\omega$ (meV)', fontdict = font)
    plt.xlim(xmax=-en[j][0][mdc_seq[-1]], xmin=-en[j][0][mdc_seq[0]])
    plt.ylim(ymax=.15, ymin=0)
    plt.text(.0025, .14, lbls[j + 12])
    
    Z = Z + (z,)
    Re = Re + (re,)
    Loc_en = Loc_en + (loc_en,)
    Loc = Loc + (loc,)
    eLoc = eLoc + (eloc,)
    Width = Width + (width,)
    eWidth = eWidth + (ewidth,)
print(Z)
#%%
#(Loc_en, Loc, eLoc, Width, eWidth) = utils_plt.CSROfig6()
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
Re = ()
n_spec = 1

c = (0, 238 / 256, 118 / 256)
cols = ([0, 1, 1], [0, .7, .7], [0, .4, .4], [0, 0, 0])
cols_r = ([0, 0, 0], [0, .4, .4], [0, .7, .7], [0, 1, 1])
for j in range(n_spec): 
    D = ARPES.Bessy(files[j], mat, year, sample)
    D.norm(gold)
    D.restrict(bot=.7, top=.9, left=.31, right=.6)
    D.bkg(norm=True)
    if j == 0:
        D.ang2k(D.ang, Ekin=40-4.5, lat_unit=False, a=5.5, b=5.5, c=11, 
                  V0=0, thdg=2.5, tidg=0, phidg=42)
        int_norm = D.int_norm * 1.5
        eint_norm = D.eint_norm * 1.5
    else: 
        D.ang2k(D.ang, Ekin=40-4.5, lat_unit=False, a=5.5, b=5.5, c=11, 
                  V0=0, thdg=2.9, tidg=0, phidg=42)
        int_norm = D.int_norm
        eint_norm = D.eint_norm        
    en_norm = D.en_norm - .008
    k_norm = D.ks * np.sqrt(2)
    spec = spec + (int_norm,)
    espec = espec + (eint_norm,)
    en = en + (en_norm,)
    k = k + (k_norm,)
    
plt.figure('2007', figsize=(10, 10), clear=True)
titles = [r'$T=1.3\,$K', r'$T=10\,$K', r'$T=20\,$K', r'$T=30\,$K']
lbls = [r'(a)', r'(b)', r'(c)', r'(d)',
            r'(e)', r'(f)', r'(g)', r'(h)',
            r'(i)', r'(j)', r'(k)', r'(l)']
xx = np.arange(-.4, .25, .01)
Im_cols = np.array([[0, 1, 1], [0, .7, .7], [0, .4, .4], [0, 0, 0]])
Im_cols_r = np.flipud(Re_cols)
Re_cols = ['khaki', 'darkkhaki', 'goldenrod', 'darkgoldenrod']
Re_cols_r = ['darkgoldenrod', 'goldenrod', 'darkkhaki', 'khaki']
for j in range(n_spec):
    loc_j = Loc[j] 
    k_F = loc_j[0] 
    p0 = -k_F * v_LDA
    yy = p0 + xx * v_LDA
    en_LDA = p0 + loc_j * v_LDA
    re = Loc_en[j] - en_LDA
    ###First row###
    ax = plt.subplot(3, 4, j + 1) 
    ax.set_position([.08 + j * .21, .66, .2, .2])
    plt.tick_params(direction='in', length=1.5, width=.5, colors='k')
    plt.contourf(k[j], en[j], spec[j], 200, cmap=colmap,
                     vmin=.0 * np.max(spec[0]), vmax=1 * np.max(spec[0]))
    plt.plot(xx, yy, 'C9-', lw=.5)
    plt.plot(loc_j, Loc_en[j], 'C8o', ms=1)
    if j == 0:
        ax.arrow(-1, -1, .3, .3, head_width=0.3, head_length=0.3, fc=c, ec='k')
        plt.ylabel('$\omega\,(\mathrm{meV})$', fontdict = font)
        plt.yticks(np.arange(-.2, .1, .05), ('-200', '-150', '-100', '-50', '0', '50'))
    else:
        plt.yticks(np.arange(-.2, .1, .05), [])
    plt.xticks(np.arange(-1, 0, .1))
    plt.xlim(xmax=-.05, xmin=-.45)
    plt.ylim(ymax=.05, ymin=-.15)
    plt.xlabel(r'$k_{\Gamma - \mathrm{S}}\,(\mathrm{\AA}^{-1})$', fontdict=font)
    plt.text(-.44, .035, lbls[j])
    plt.title(titles[j], fontsize=15)
    
    Re = Re + (re,)
    Z = .55
    ax = plt.subplot(3, 4, j + 5) 
    ax.set_position([.08 + j * .21, .4, .2, .2])
    plt.tick_params(direction='in', length=1.5, width=.5, colors='k')
    plt.errorbar(-Loc_en[j], re / Z, eWidth[j] * v_LDA / Z,
                 color=Re_cols[j], linewidth=.5, capsize=2, fmt='o', ms=2)
    
    if j == 0:
        plt.errorbar(-Loc_en[j], Width[j] * v_LDA  - 0.048, eWidth[j] * v_LDA,
                     color=Im_cols[j], linewidth=.5, capsize=2, fmt='d', ms=2)
        plt.ylabel(r'Self energy $\Sigma$ (eV)', 
                   fontdict = font)
        plt.yticks(np.arange(0, .3, .05))
        plt.text(.005, .2, r'Re$\Sigma / (1-Z)$', color=Re_cols[1], fontsize=12)
        plt.text(.07, .019, r'Im$\Sigma$', color=Im_cols[1], fontsize=12)
        
    else:
        plt.errorbar(-Loc_en[j], Width[j] * v_LDA  - 0.042, eWidth[j] * v_LDA, 
                 color=Im_cols[j], linewidth=.5, capsize=2, fmt='d', ms=2)
        
        plt.yticks(np.arange(0, .1, .05), [])
    plt.xlabel('$\omega\,(\mathrm{meV})$', fontdict = font)
    plt.xticks(np.arange(0, .12, .02), ('0', '20', '40', '60', '80'))
    plt.xlim(xmin=0, xmax=.1)
    plt.ylim(ymax=.25, ymin=0)
    plt.text(.003, .23, lbls[j + 4])
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










