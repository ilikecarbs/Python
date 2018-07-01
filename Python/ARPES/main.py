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

font = {'family': 'serif',
        'style': 'normal',
        'color':  [0,0,0],
        'weight': 'ultralight',
        'size': 12,
        }

#%%
"""
fig1: DFT plot Ca2RuO4: figure 3 of Nature Comm.
fig2: DMFT pot Ca2RuO4: figure 3 of Nature Comm.
fig3: DFT plot orbitally selective Mott scenario
fig4: DFT plot uniform gap scnenario
fig5: Experimental Data of Nature Comm.
"""


uplt.fig1(
        colmap = cm.bone_r, print_fig = False
        )


#%%
    
#    plt.figure(100, figsize = (5,5))
#    plt.plot([-1, -1], [-1, 1], 'k--')
#    plt.plot([1, 1], [-1, 1], 'k--')
#    plt.plot([-1, 1], [1, 1], 'k--')
#    plt.plot([-1, 1], [-1, -1], 'k--')
#    plt.plot(D.k[0], D.k[1])
#    plt.show()

#u.gold(gold, mat, year, sample, Ef_ini=60.4, BL='DLS')

#%%
from astropy.io import fits
import os
os.chdir('/Users/denyssutter/Documents/library/Python/ARPES')
import numpy as np
import matplotlib.cm as cm

rainbow_light = uplt.rainbow_light
cm.register_cmap(name='rainbow_light', cmap=rainbow_light)
rainbow_light_2 = uplt.rainbow_light_2
cm.register_cmap(name='rainbow_light_2', cmap=rainbow_light_2)

file1 = '0619_00161'
file2 = '0619_00162'
mat = 'Ca2RuO4'
year = 2016
sample = 'data'

th = 20
ti = -2
phi = 21
a = 5.5
D1 = ARPES.ALS(file1, mat, year, sample)
D2 = ARPES.ALS(file2, mat, year, sample)
D1.ang2kFS(D1.ang, Ekin=D1.hv-4.5-4.7, lat_unit=True, a=a, b=a, c=11, 
                V0=0, thdg=th, tidg=ti, phidg=phi)
D2.ang2kFS(D2.ang, Ekin=D2.hv-4.5-4.7, lat_unit=True, a=a, b=a, c=11, 
                V0=0, thdg=th, tidg=ti, phidg=phi)

data = np.concatenate((D1.int, D2.int), axis=0)
pol = np.concatenate((D1.pol, D2.pol), axis=0)
kx = np.concatenate((D1.kx, D2.kx), axis=0)
ky = np.concatenate((D1.ky, D2.ky), axis=0)
en = D1.en-2.3

e = -2.1; ew = 0.4
e_val, e_ind = u.find(en, e)
ew_val, ew_ind = u.find(en, e-ew)
FSmap = np.sum(data[:, :, ew_ind:e_ind], axis=2)
        
plt.figure(1006, figsize=(4, 6), clear=True)
plt.tick_params(direction='in', length=1.5, width=.5, colors='k')
plt.contourf(kx, ky, FSmap, 100, cmap = cm.ocean_r,
               vmin = .5 * np.max(FSmap), vmax = .93 * np.max(FSmap))
plt.xlabel('$k_x$ ($\pi/a$)', fontdict = font)
plt.ylabel('$k_y$ ($\pi/b$)', fontdict = font)
#plt.xlim((-1.1, 1.1))
#plt.ylim((-1.1, 3.1))
plt.axis('equal')
plt.grid(alpha=0.2)
plt.xticks(np.arange(-10,10,1))
plt.yticks(np.arange(-10,10,1))
plt.plot([-1, -1], [-1, 1], 'k-')
plt.plot([1, 1], [-1, 1], 'k-')
plt.plot([-1, 1], [1, 1], 'k-')
plt.plot([-1, 1], [-1, -1], 'k-')
plt.plot([-1, 1], [-1, 1], 'g:')
plt.plot([-1, 1], [1, 1], 'g:')
plt.plot([-1, 0], [1, 2], 'g:')
plt.plot([0, 0], [2, -1], 'g:')
ax = plt.axes()
ax.arrow(-1, -1, .3, .3, head_width=0.2, head_length=0.2, fc='g', ec='k')
ax.arrow(0, -.4, 0, -.3, head_width=0.2, head_length=0.2, fc='g', ec='k')

#plt.plot(0, 0, 'ko', markersize=3)
plt.text(-0.15, -0.15, r'$\Gamma$',
         fontsize=12, color='r')
plt.text(.85, .85, r'S',
         fontsize=12, color='r')
plt.text(-0.15, .85, r'X',
         fontsize=12, color='r')
pos = ax.get_position()
cax = plt.axes([pos.x0+pos.width+0.03 ,
                    pos.y0, 0.03, pos.height])
cbar = plt.colorbar(cax = cax, ticks = None)
cbar.set_ticks([])
plt.show()


#%%
D1.FS(e = 2.3-4.7, ew = .4, norm = False)
D1.ang2kFS(D.ang, Ekin=D.hv-4.5, lat_unit=False, a=5.33, b=5.33, c=11, 
                V0=0, thdg=20.5, tidg=0, phidg=0)
D1.plt_FS(coord = False)
#%%



folder = ''.join(['/Users/denyssutter/Documents/Denys/',str(mat),
                  '/ALS',str(year),'/',str(sample),'/'])
filename = ''.join([str(year),file,'.fits'])
path = folder + filename

f = fits.open(path)
hdr = f[0].header
mode = hdr['NM_0_0']
data = f[1].data

px_per_en = hdr['SSPEV_0']
e_i = hdr['SSX0_0']
e_f = hdr['SSX1_0']
a_i = hdr['SSY0_0']
a_f = hdr['SSY1_0']
Ef = hdr['SSKE0_0']
ang_per_px = 0.193
binning = 2

npol = data.size
(nen, nang) = data[0][-1].shape

intensity = np.zeros((npol, nang, nen))
ens = np.zeros((npol, nang, nen))
angs = np.zeros((npol, nang, nen))
pols = np.zeros((npol, nang, nen))

en = (np.arange(e_i, e_f, 1) - Ef) / px_per_en
ang = np.arange(a_i, a_f, 1) * ang_per_px / binning
ang = np.arange(0, nang, 1)
pol = np.arange(0, npol, 1)

for i in range(npol):
    pol[i] = data[i][1]
    intensity[i, :, :] = np.transpose(data[i][-1])
    
pols  = np.transpose(np.broadcast_to(pol, (ang.size, en.size, pol.size)),
                                (2, 0, 1))

angs  = np.transpose(np.broadcast_to(
                            ang, (pol.size, en.size, ang.size)), (0, 2, 1))


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

#%%
D.FS(e = -0.0, ew = .02, norm = True)
D.ang2kFS(D.ang, Ekin=22-4.5, a=5.33, b=5.33, c=11, V0=0, thdg=0, tidg=0, phidg=0)
D.plt_FS(coord = True)

#%%
os.chdir('/Users/denyssutter/Documents/library/Python/ARPES')

start = time.time()
tb = umath.TB(a = np.pi, kpoints = 200)

#param = mdl.paramSRO()
param = umath.paramCSRO20()

#tb.simple(param)
tb.CSRO(param)
#tb.SRO(param)

tb.plt_cont_TB_CSRO20()

print(time.time()-start)
#%%

    
    
    






















