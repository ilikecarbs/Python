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
---------  Ca2RuO4 Figures   ---------
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

---------  Ca1.8Sr0.2RuO4 Figures ---------
CSROfig1:  Experimental data: Figure 1 CSRO20 paper
CSROfig2:  Experimental PSI data: Figure 2 CSCRO20 paper
CSROfig3:  (L): Polarization and orbital characters. Figure 3 in paper
CSROfig4:  (L): Temperature dependence. Figure 4 in paper
CSROfig5:  (L): Analysis Z epsilon band (load=True)
CSROfig6:  Analysis MDC's beta band (load=True)
CSROfig7:  Background subtraction
CSROfig8:  Extraction LDA Fermi velocity
CSROfig9:  ReSigma vs ImSigma (load=True)
CSROfig10: Quasiparticle Z

---------  To-Do ---------
CSRO: kz dependence
CSRO: CSRO30 vs CSRO20 (FS and cuts)
CSRO: TB FSmaps
CSRO: TB DOS
CSRO: TB specific heat
CSRO: DOS calculations
CSRO: FS maps DMFT
CSRO: Symmetrization
CSRO: FS area counting 
"""
#--------
#utils_plt.CROfig1()
#utils_plt.CROfig2()
#utils_plt.CROfig3()
#utils_plt.CROfig4()
#utils_plt.CROfig5()
#utils_plt.CROfig6()
#utils_plt.CROfig7()
#utils_plt.CROfig8()
#utils_plt.CROfig9()
#utils_plt.CROfig10()
#utils_plt.CROfig11()
#utils_plt.CROfig12()
#utils_plt.CROfig13()
#utils_plt.CROfig14()
#--------
#utils_plt.CSROfig1()
#utils_plt.CSROfig2()
#utils_plt.CSROfig3()
#utils_plt.CSROfig4()
#utils_plt.CSROfig5()
#utils_plt.CSROfig6()
#utils_plt.CSROfig7()
#utils_plt.CSROfig8()
#utils_plt.CSROfig9()
#utils_plt.CSROfig10()

#%%



#%%
# Use Green's theorem to compute the area
# enclosed by the given contour.
def area(vs):
    a = 0
    x0,y0 = vs[0]
    for [x1,y1] in vs[1:]:
        dx = x1-x0
        dy = y1-y0
        a += 0.5*(y0*dx - x0*dy)
        x0 = x1
        y0 = y1
    print(dx, dy)
    return a


# Generate some test data.
delta = 0.01
x = np.arange(-3.1, 3.1, delta)
y = np.arange(-3.1, 3.1, delta)
X, Y = np.meshgrid(x, y)
r = np.sqrt(X**2 + Y**2)

# Plot the data
levels = [1.0,2.0,3.0]
cs = plt.contour(X,Y,r,levels=levels)
plt.clabel(cs, inline=1, fontsize=10)

# Get one of the contours from the plot.
for i in range(len(levels)):
    contour = cs.collections[i]
    vs = contour.get_paths()[0].vertices
    # Compute area enclosed by vertices.
    a = area(vs)
    plt.axis('equal')
    print("r = " + str(levels[i]) + ": a =" + str(a))
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
D.FS_flatten(ang=True)
D.plt_FS(coord=True)


#%%
os.chdir('/Users/denyssutter/Documents/library/Python/ARPES')
file = 62151
gold = 62081
mat = 'CSRO20'
year = 2017
sample = 'S6'

D = ARPES.DLS(file, mat, year, sample)
D.norm(gold)
D.restrict(bot=0, top=1, left=.12, right=.9)
D.FS(e = 0.0, ew = .02, norm = True)
D.ang2kFS(D.ang, Ekin=22-4.5, lat_unit=True, a=5.5, b=5.5, c=11, 
          V0=0, thdg=12, tidg=-2.5, phidg=45)
D.FS_flatten(ang=True)
D.plt_FS(coord=True)

#%%

"""
Test Script for Tight binding models
"""

os.chdir('/Users/denyssutter/Documents/library/Python/ARPES')

start = time.time()
tb = utils_math.TB(a = np.pi, kbnd = 2, kpoints = 200)  #Initialize tight binding model

####SRO TB hopping parameters###
param = utils_math.paramSRO()  
#param = utils_math.paramCSRO20()  

###Calculate and Plot FS###
#tb.simple(param) 
tb.SRO(param) 
#tb.CSRO(param)


#tb.plt_cont_TB_SRO()
#tb.plt_cont_TB_CSRO20()

#print(time.time()-start)
#%%
from numpy import linalg as la
kpoints = 50
e0 = 0
a = np.pi
kbnd = 2

x = np.linspace(-kbnd, kbnd, kpoints)
y = np.linspace(-kbnd, kbnd, kpoints)
[X, Y] = np.meshgrid(x,y)
coord = dict([('x', x), ('y', y), ('X', X), ('Y', Y)])

Pyz = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]])
Pxz = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
Pxy = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 1]])

t1 = param['t1']; t2 = param['t2']; t3 = param['t3']
t4 = param['t4']; t5 = param['t5']; t6 = param['t6']
mu = param['mu']; l = param['l']
x = coord['x']; y = coord['y']; X = coord['X']; Y = coord['Y']
#Hopping terms
fyz = - 2 * t2 * np.cos(X * a) - 2 * t1 * np.cos(Y * a)
fxz = - 2 * t1 * np.cos(X * a) - 2 * t2 * np.cos(Y * a)
fxy = - 2 * t3 * (np.cos(X * a) + np.cos(Y * a)) - \
        4 * t4 * (np.cos(X * a) * np.cos(Y * a)) - \
        2 * t5 * (np.cos(2 * X * a) + np.cos(2 * Y * a))
off = - 4 * t6 * (np.sin(X * a) * np.sin(Y * a))
#Placeholders energy eigenvalues
yz = np.ones((len(x), len(y))); xz = np.ones((len(x), len(y)))
xy = np.ones((len(x), len(y)))
#Tight binding Hamiltonian
def H(i,j):
    H = np.array([[fyz[i,j] - mu, off[i,j] + complex(0,l), -l],
                  [off[i,j] - complex(0,l), fxz[i,j] - mu, complex(0,l)],
                  [-l, -complex(0,l), fxy[i,j] - mu]])
    return H
#Diagonalization of symmetric Hermitian matrix on k-mesh
for i in range(len(x)):
    for j in range(len(y)):
        eval, evec = la.eigh(H(i,j))
        eval = np.real(eval)
        yz[i,j] = eval[0]; xz[i,j] = eval[1]; xy[i,j] = eval[2]
        
        
#wyz(i) = sum(conj(evec(:,n)).*(Pyz*evec(:,n))); 
#wxz(i) = sum(conj(evec(:,n)).*(Pxz*evec(:,n))); 
#wxy(i) = sum(conj(evec(:,n)).*(Pxy*evec(:,n))); 
#X = coord['X']; Y = coord['Y']   
#xz = bndstr['xz']; yz = bndstr['yz']; xy = bndstr['xy']
en = (xz, yz, xy)
plt.figure(20011, figsize=(5, 5), clear=True)
plt.tick_params(direction='in', length=1.5, width=.5, colors='k')  
n = 0
for i in en:
    n = n + 1
    plt.contour(X, Y, i, colors = 'black', linestyles = ':', levels = e0)
    plt.axis('equal')
plt.xticks(np.arange(-3, 3, 1))
plt.yticks(np.arange(-3, 3, 1))
plt.xlim(xmax=2, xmin=-2)
plt.ylim(ymax=2, ymin=-2)
plt.xlabel(r'$k_x (\pi/a)$', fontdict=font)
plt.ylabel(r'$k_y (\pi/b)$', fontdict=font)
plt.grid(True, alpha=.2)
plt.show()  
#%%










