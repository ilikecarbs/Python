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
from scipy.optimize import curve_fit
colmap = cm.ocean_r
os.chdir('/Users/denyssutter/Documents/library/Python/ARPES')
file = 'CSRO_P1_0032'
mat = 'CSRO20'
year = 2017
sample = 'data'
D = ARPES.SIS(file, mat, year, sample) 
D.FS(e = 19.3, ew = .01, norm = False)
D.FS_flatten(ang=False)
"""
D.FS_restrict(bot=0, top=1, left=0, right=1)
"""
D.ang2kFS(D.ang, Ekin=D.hv-4.5, lat_unit=True, a=5.5, b=5.5, c=11, 
          V0=0, thdg=-1.2, tidg=1, phidg=0)
D.plt_FS(coord=True)

###Diagonal MDC###
bnd = .75
c = (0, 238 / 256, 118 / 256)
mdc_d = np.zeros(D.ang.size)
for i in range(D.ang.size):
    val, _mdc_d = utils.find(D.ky[:, i], D.kx[0, i])
    mdc_d[i] = D.map[_mdc_d, i]
mdc_d = mdc_d / np.max(mdc_d)
###Fit MDC###
plt.figure(20002, figsize=(4, 4), clear=True)
delta = 1e-5
p_mdc_i = np.array(
            [-.6, -.4, -.2, .2, .4, .6,
             .05, .05, .05, .05, .05, .05,
             .3, .3, .4, .4, .5, .5, 
             .33, -0.2, .02])
bounds_bot = np.concatenate((p_mdc_i[0:-3] - np.inf, p_mdc_i[-3:27] - delta))
bounds_top = np.concatenate((p_mdc_i[0:-3] + np.inf, p_mdc_i[-3:27] + delta))
p_mdc_bounds = (bounds_bot, bounds_top)
p_mdc, cov_mdc = curve_fit(
        utils_math.lor6, D.kx[0, :], mdc_d, p_mdc_i, bounds=p_mdc_bounds)
b_mdc = utils_math.poly2(D.kx[0, :], 0, p_mdc[-3], p_mdc[-2], p_mdc[-1])
f_mdc = utils_math.lor6(D.kx[0, :], *p_mdc) - b_mdc
f_mdc[0] = 0
f_mdc[-1] = 0
plt.plot(D.kx[0, :], mdc_d, 'bo')
plt.plot(D.kx[0, :], f_mdc + b_mdc)
plt.plot(D.kx[0, :], b_mdc, 'k--')
###Plotting
plt.figure(2002, figsize=(8, 8), clear=True)
ax = plt.subplot(1, 3, 1) 
ax.set_position([.08, .2, .4, .4])
plt.tick_params(direction='in', length=1.5, width=.5, colors='k')
plt.contourf(D.kx, D.ky, D.map_flat, 100, cmap=colmap,
             vmin=.6 * np.max(D.map_flat), vmax=1.0 * np.max(D.map_flat))
plt.plot([-bnd, bnd], [-bnd, bnd], linestyle='--', color=c, linewidth=.5)
###Tight Binding Model###
tb = utils_math.TB(a = np.pi, kbnd = 2, kpoints = 200)#Initialize 
param = utils_math.paramCSRO20()  #Load parameters
tb.CSRO(param)  #Calculate bandstructure
bndstr = tb.bndstr  #Load bandstructure
coord = tb.coord  #Load coordinates
X = coord['X']; Y = coord['Y']   
Axy = bndstr['Axy']; Bxz = bndstr['Bxz']; Byz = bndstr['Byz']
en = (Axy, Byz)  #Loop over sheets

C = plt.contour(X, Y, Byz, colors = 'black', linestyles = ':', 
                alpha=0, levels = -0.00)
p = C.collections[0].get_paths()
p = np.asarray(p)
axy = np.arange(0, 4, 1) #indices of same paths
bxz = np.arange(16, 24, 1)
byz = np.array([16, 17, 20, 21])

ind = byz; col = 'k'
v = p[18].vertices
plt.plot(v[:, 0], v[:, 1], linestyle = ':', color = 'm', 
         linewidth=1)
v = p[19].vertices
plt.plot(v[:, 0], v[:, 1], linestyle = ':', color = 'C1', 
         linewidth=1)
for j in ind:
    v = p[j].vertices
    plt.plot(v[:, 0], v[:, 1], linestyle = ':', color = col, 
             linewidth=1)
plt.xticks(np.arange(-10, 10, .5))
plt.yticks(np.arange(-10, 10, .5))
plt.axis('equal')

plt.xlim(xmax=bnd, xmin=-bnd)
plt.ylim(ymax=bnd, ymin=-bnd)   
ax = plt.subplot(1, 4, 3) 
ax.set_position([.08, .65, .4, .2])
plt.plot(D.kx[0, :], mdc_d, 'o', markersize=1.5, color='C9')
plt.tick_params(direction='in', length=1.5, width=.5, colors='k')
plt.xlim(xmax=bnd, xmin=-bnd)
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










