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
CROfig1:  DFT plot Ca2RuO4: figure 3 of Nature Comm.
CROfig2:  (L): DMFT pot Ca2RuO4: figure 3 of Nature Comm.
CROfig3:  DFT plot orbitally selective Mott scenario
CROfig4:  DFT plot uniform gap scnenario
CROfig5:  Experimental Data of Nature Comm.
CROfig6:  Constant energy map CaRuO4 of alpha branch
CROfig7:  Photon energy dependence Ca2RuO4: figure 2 of Nature Comm.
CROfig8:  Polarization dependence Ca2RuO4: figure 2 of Nature Comm.
CROfig9:  (L): DMFT plot Ca2RuO4 dxy/dxz,yz: figure 4 of Nature Comm.
CROfig10: (L): DFT plot Ca2RuO4: spaghetti and spectral representation
CROfig11:  Multiplet analysis Ca2RuO4
"""

uplt.fig5(print_fig=True)

#%%
tb = umath.TB(a = np.pi, kbnd = 2, kpoints = 200)  #Initialize tight binding model
param = umath.paramCSRO20()  
tb.CSRO(param)
bndstr = tb.bndstr
coord = tb.coord   
X = coord['X']; Y = coord['Y']   
Axz = bndstr['Axz']; Ayz = bndstr['Ayz']; Axy = bndstr['Axy']
Bxz = bndstr['Bxz']; Byz = bndstr['Byz']; Bxy = bndstr['Bxy']
en = (Axy, Bxz, Byz)
plt.figure(10005, figsize=(5, 5), clear=True)
n = 0
for i in en:
    n += 1
    C = plt.contour(X, Y, i, colors = 'black', linestyles = ':', levels = 0)
    p = C.collections[0].get_paths()
    p = np.asarray(p)
    axy = np.arange(0, 4, 1)
    bxz = np.arange(16, 24, 1)
    byz1 = np.array([16, 17, 20, 21])
    byz2 = np.array([18])
    byz3 = np.array([19])
    if n == 1:
        ind = axy; col = 'r'
    elif n == 2:
        ind = bxz; col = 'b'
    elif n == 3:
        ind = byz1; col = 'k'
    for j in ind:
        v = p[j].vertices
        plt.plot(v[:, 0], v[:, 1], marker = '.', color = col)
        
#%%
os.chdir('/Users/denyssutter/Documents/library/Python/ARPES')

file = 62087
gold = 62081
mat = 'CSRO20'
year = 2017
sample = 'S6'

D = ARPES.DLS(file, mat, year, sample)
D.norm(gold)
D.restrict(bot=0, top=1, left=.12, right=.9)

#%%
D.FS(e = 0.02, ew = .03, norm = True)

D.ang2kFS(D.ang, Ekin=22-4.5, lat_unit=True, a=5.5, b=5.5, c=11, 
          V0=0, thdg=8.7, tidg=-4, phidg=92)

plt.figure(1005, figsize = (8, 8), clear = True)
plt.tick_params(direction='in', length=1.5, width=.5, colors='k')

FS = D.map
for i in range(FS.shape[1]):
    FS[:, i] = np.divide(FS[:, i], np.sum(FS[:, i]))  
    
ax = plt.subplot(1, 3, 2) 
ax.set_position([.35, .3, .3, .37])
pos = ax.get_position()
plt.tick_params(direction='in', length=1.5, width=.5, colors='k')  
plt.contourf(D.kx, D.ky, FS, 300,
           cmap=cm.ocean_r)
plt.xlim(xmax=np.max(D.kx), xmin=np.min(D.kx))
plt.ylim(ymax=np.max(D.ky), ymin=np.min(D.ky))
plt.ylabel('$k_x$', fontdict = font)
plt.axis('equal')
#%%

    
#%%
p = C.collections[0].get_paths()
p = np.asarray(p)
#v = p.vertices
#x = v[:, 0]
#y = v[:, 1]

plt.figure(20000, clear=True)
for i in ind:
    v = p[i].vertices
    plt.plot(v[:, 0], v[:, 1], color = col)
#    if i == 19:
#        plt.plot(v[:, 0], v[:, 1], 'ko')

    plt.text(v[-10, 0], v[-10, 1], str(i))

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

    
    
    






















