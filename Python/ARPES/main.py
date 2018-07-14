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
CSROfig5:  (L): Analysis Z epsilon band (load=True)
CSROfig6:  Analysis MDC's beta band (load=True)
CSROfig7:  Background subtraction
CSROfig8:  Extraction LDA Fermi velocity
CSROfig9:  ReSigma vs ImSigma (load=True)
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
#utils_plt.CSROfig5(print_fig=True)
utils_plt.CSROfig6(print_fig=True)
#utils_plt.CSROfig7(print_fig=True)
#utils_plt.CSROfig8(print_fig=True)
#utils_plt.CSROfig9(print_fig=True)
#%%
os.chdir('/Users/denyssutter/Documents/PhD/data')
en = np.loadtxt('Data_CSROfig4_en.dat');
EDCn_e = np.loadtxt('Data_CSROfig4_EDCn_e.dat');
EDCn_b = np.loadtxt('Data_CSROfig4_EDCn_b.dat');
EDC_e = np.loadtxt('Data_CSROfig4_EDC_e.dat');
EDC_b = np.loadtxt('Data_CSROfig4_EDC_b.dat');
Bkg_e = np.loadtxt('Data_CSROfig4_Bkg_e.dat');
Bkg_b = np.loadtxt('Data_CSROfig4_Bkg_b.dat');
_EDC_e = np.loadtxt('Data_CSROfig4__EDC_e.dat', dtype=np.int32);
_EDC_b = np.loadtxt('Data_CSROfig4__EDC_b.dat', dtype=np.int32);
eEDCn_e = np.loadtxt('Data_CSROfig4_eEDCn_e.dat');
eEDCn_b = np.loadtxt('Data_CSROfig4_eEDCn_b.dat');
eEDC_e = np.loadtxt('Data_CSROfig4_eEDC_e.dat');
eEDC_b = np.loadtxt('Data_CSROfig4_eEDC_b.dat');
dims = np.loadtxt('Data_CSROfig4_dims.dat',dtype=np.int32);
print('\n ~ Data loaded (en, EDCs + normalized + indices + Bkgs)',
      '\n', '==========================================')  
os.chdir('/Users/denyssutter/Documents/library/Python/ARPES')
#%%
en = np.reshape(np.ravel(en), (dims[0], dims[1], dims[2]))
EDCn_e = np.reshape(np.ravel(EDCn_e), (dims[0], dims[2]))
EDCn_b = np.reshape(np.ravel(EDCn_b), (dims[0], dims[2]))
EDC_e = np.reshape(np.ravel(EDC_e), (dims[0], dims[2]))
EDC_b = np.reshape(np.ravel(EDC_b), (dims[0], dims[2]))
Bkg_e = np.reshape(np.ravel(Bkg_e), (dims[0], dims[2]))
Bkg_b = np.reshape(np.ravel(Bkg_b), (dims[0], dims[2]))
_EDC_e = np.reshape(np.ravel(_EDC_e), (dims[0]))
_EDC_b = np.reshape(np.ravel(_EDC_b), (dims[0]))
eEDCn_e = np.reshape(np.ravel(eEDCn_e), (dims[0], dims[2]))
eEDCn_b = np.reshape(np.ravel(eEDCn_b), (dims[0], dims[2]))
eEDC_e = np.reshape(np.ravel(eEDC_e), (dims[0], dims[2]))
eEDC_b = np.reshape(np.ravel(eEDC_b), (dims[0], dims[2]))


j = 0
en[j][_EDC_e[j]]
#%%
os.chdir('/Users/denyssutter/Documents/PhD/data')
xz_lda = np.loadtxt('LDA_CSRO_xz.dat')
os.chdir('/Users/denyssutter/Documents/library/Python/ARPES')
#%%
#[0, 56, 110, 187, 241, 266, 325, 350]  = [G,X,S,G,Y,T,G,Z]
m, n = 8000, 351 #dimensions energy, full k-path
size = 187 - 110
bot, top = 3840, 4055 #restrict energy window
data = np.array([xz_lda]) #combine data
spec = np.reshape(data[0, :, 2], (n, m)) #reshape into n,m
spec = spec[110:187, bot:top] #restrict data to bot, top
#spec = np.flipud(spec)
spec_en = np.linspace(-8, 8, m) #define energy data
spec_en = spec_en[bot:top] #restrict energy data
spec_en = np.broadcast_to(spec_en, (spec.shape))
spec_k = np.linspace(-np.sqrt(2) * np.pi / 5.5, 0, size)
spec_k = np.transpose(
            np.broadcast_to(spec_k, (spec.shape[1], spec.shape[0])))
max_pts = np.ones((size))
for i in range(size):
    max_pts[i] = spec[i, :].argmax()
ini = 40
fin = 48
max_pts = max_pts[ini:fin]

max_k = spec_k[ini:fin, 0]
max_en = spec_en[0, max_pts.astype(int)]
p_max, c_max = curve_fit(
                utils_math.poly1, max_k, max_en)
v_LDA = p_max[1]
ev_LDA = np.sqrt(np.diag(c_max))[1]
k_F = -p_max[0] / p_max[1]
xx = np.arange(-.43, -.27, .01)
plt.figure(2008, figsize=(8, 8), clear=True)
ax = plt.subplot(1, 3, 1) 
ax.set_position([.2, .24, .5, .3])
plt.tick_params(direction='in', length=1.5, width=.5, colors='k') 
plt.contourf(spec_k, spec_en, spec, 200, cmap=cm.ocean_r) 
plt.plot(xx, utils_math.poly1(xx, *p_max), 'C9--', linewidth=1)
plt.plot(max_k, max_en, 'ro', ms=2)
plt.plot([np.min(spec_k), 0], [0, 0], 'k:')
plt.yticks(np.arange(-1, 1, .1))
plt.ylim(ymax=.1, ymin=-.3)
plt.ylabel(r'$\omega$ (eV)', fontdict=font)
plt.xlabel(r'$k_{\Gamma - \mathrm{S}}\, (\mathrm{\AA}^{-1})$', fontdict=font)
plt.text(-.3, -.15, '$v_\mathrm{LDA}=$' + str(np.round(v_LDA,2)) + ' eV$\,\mathrm{\AA}$')
pos = ax.get_position()
cax = plt.axes([pos.x0+pos.width + 0.01 ,
                    pos.y0, 0.01, pos.height])
cbar = plt.colorbar(cax = cax, ticks = None)
cbar.set_ticks([])
cbar.set_clim(np.min(spec), np.max(spec))
#%%
os.chdir('/Users/denyssutter/Documents/PhD/data')
np.savetxt('Data_CSROfig8_v_LDA.dat', [v_LDA, ev_LDA])
os.chdir('/Users/denyssutter/Documents/library/Python/ARPES')
#%%
v_LDA = 2.3411686586990417
plt.figure('2010', figsize=(8, 8), clear=True)
e_cols = np.array([[0, 1, 1], [0, .7, .7], [0, .4, .4], [0, 0, 0]])
b_cols = ['khaki', 'darkkhaki', 'goldenrod', 'darkgoldenrod']
T = np.array([1.3, 10, 20, 30])
eZ_b = np.asarray(eZ_b)

os.chdir('/Users/denyssutter/Documents/PhD/data')
C_B = np.genfromtxt('Data_C_Braden.csv', delimiter=',')
C_M = np.genfromtxt('Data_C_Maeno.csv', delimiter=',')
R_1 = np.genfromtxt('Data_R_1.csv', delimiter=',')
R_2 = np.genfromtxt('Data_R_2.csv', delimiter=',')
os.chdir('/Users/denyssutter/Documents/library/Python/ARPES')

hbar = 1.0545717e-34
NA = 6.022141e23
kB = 1.38065e-23
a = 5.33e-10
m_e = 9.109383e-31
m_LDA = 1.6032
gamma = (np.pi * NA * kB ** 2 * a ** 2 / (3 * hbar ** 2)) * m_LDA * m_e
Z_B = gamma / C_B[:, 1] 
Z_M = gamma / C_M[:, 1] * 1e3



xx = np.array([1e-3, 1e4])
yy = 2.3 * xx ** 2
ax = plt.subplot(1, 2, 1) 
ax.set_position([.2, .3, .3, .3])
plt.tick_params(direction='in', length=1.5, width=.5, colors='k')
plt.errorbar(T, Z_b, eZ_b * v_LDA,
             color='C1', linewidth=.5, capsize=2, fmt='o', ms=2)
plt.errorbar(T, Z_e, Z_e / v_LDA,
             color='r', linewidth=.5, capsize=2, fmt='d', ms=2)
plt.plot(C_B[:, 0], Z_B, 'o', ms=1, color='slateblue')
plt.plot(C_M[:, 0], Z_M, 'o', ms=1, color='cadetblue')
ax.set_xscale("log", nonposx='clip')
plt.yticks(np.arange(0, .5, .1))
plt.xlim(xmax=40, xmin=1)
plt.ylim(ymax=.35, ymin=0)
plt.xlabel(r'$T$ (K)')
plt.ylabel(r'$Z$')
###Inset###
ax = plt.subplot(1, 2, 2) 
ax.set_position([.28, .38, .15, .1])
plt.tick_params(direction='in', length=1.5, width=.5, colors='k')
plt.loglog(np.sqrt(R_1[:, 0]), R_1[:, 1], 'C8o', ms=3)
plt.loglog(np.sqrt(R_2[:, 0]), R_2[:, 1], 'C8o', ms=3)
plt.loglog(xx, yy, 'k--', lw=1)
plt.ylabel(r'$\rho\,(\mu \Omega \mathrm{cm})$')
plt.xlim(xmax=1e1, xmin=1e-1)
plt.ylim(ymax=1e4, ymin=1e-2)
plt.show()
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










