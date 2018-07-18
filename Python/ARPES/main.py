#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Created on Tue Jun 19 15:14:29 2018

@author: denyssutter

%%%%%%%%%%%%%%%%%%%%
        main
%%%%%%%%%%%%%%%%%%%%

Content:
1. Load all relevant modules
2. Plot figures for thesis (uncomment relevant figure)
3. Load current file in a running experiment
4. Current projects

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
#from scipy.stats import exponnorm
#from scipy.optimize import curve_fit

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
CSROfig11: Tight binding model CSRO
CSROfig12: Tight binding model SRO
CSROfig13: TB along high symmetry directions, orbitally resolved

---------  To-Do ---------
CSRO: kz dependence
CSRO: CSRO30 vs CSRO20 (FS and cuts)
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
#utils_plt.CSROfig11()
#utils_plt.CSROfig12()
#utils_plt.CSROfig13()

#%%
"""
Loading Current Data:
"""

os.chdir('/Users/denyssutter/Documents/library/Python/ARPES')
mat = 'CaMn2Sb2'
year = 2018
file = 'S3_FSM_fine_hv90_T230'
mode = 'FSM'
D = ARPES.CASS(file, mat, year, mode)
utils.gold(file='S3_5', mat='CaMn2Sb2', year=2018, sample=1, Ef_ini=86.4, BL='CASS')
D.norm(gold='S3_5')
#%%
#D.plt_hv()
D.FS(e = -.2, ew = .1, norm = True)
D.ang2kFS(D.ang, Ekin=90-4.5, lat_unit=False, a=1, b=1, c=1, 
          V0=0, thdg=-6, tidg=24.5, phidg=-0)
#D.FS_flatten(ang=True)
D.plt_FS(coord=True)
D.plt_FS_polcut(norm=True, p=24.6, pw=.5)

#%%

"""
Test Script for Tight binding models
"""

os.chdir('/Users/denyssutter/Documents/library/Python/ARPES')

start = time.time()
kbnd = 1
tb = utils_math.TB(a = np.pi, kbnd = kbnd, kpoints = 50)  #Initialize tight binding model

####SRO TB hopping parameters###
#param = utils_math.paramSRO()  
param = utils_math.paramCSRO20()  

###Calculate and Plot FS###
#tb.simple(param) 
#tb.SRO(param, e0=0, vertices=False, proj=False) 
tb.CSRO(param, e0=0, vertices=False, proj=False) 

#tb.plt_cont_TB_SRO()
#tb.plt_cont_TB_CSRO20()

print(time.time()-start)
#%%
"""
Project: Density of states
"""

os.chdir('/Users/denyssutter/Documents/PhD/data')
Axz_dos = np.loadtxt('Data_Axz_kpts_5000.dat')
Ayz_dos = np.loadtxt('Data_Ayz_kpts_5000.dat')
Axy_dos = np.loadtxt('Data_Axy_kpts_5000.dat')
Bxz_dos = np.loadtxt('Data_Bxz_kpts_5000.dat')
Byz_dos = np.loadtxt('Data_Byz_kpts_5000.dat')
Bxy_dos = np.loadtxt('Data_Bxy_kpts_5000.dat')
os.chdir('/Users/denyssutter/Documents/library/Python/ARPES')
bands = (Axz_dos, Ayz_dos, Axy_dos, Bxz_dos, Byz_dos, Bxy_dos)
#%%

En = ()
DOS = ()
N_bands = ()
N_full= ()
plt.figure('DOS', figsize=(8, 8), clear=True)
n = 0
for band in bands:
    n += 1
    ax = plt.subplot(2, 3, n) 
    plt.tick_params(direction='in', length=1.5, width=.5, colors='k') 
    dos, bins, patches = plt.hist(np.ravel(band), bins=150, density=True,
                                alpha=.2, color='C8')
    en = np.zeros((len(dos)))
    for i in range(len(dos)):
        en[i] = (bins[i] + bins[i + 1]) / 2
    ef, _ef = utils.find(en, 0.00)
    n_full = np.trapz(dos, x=en)
    n_band = np.trapz(dos[:_ef], x=en[:_ef])
    plt.plot(en, dos, color='k', lw=.5)
    plt.fill_between(en[:_ef], dos[:_ef], 0, color='C1', alpha=.5)
    if n < 4:
        ax.set_position([.1 + (n - 1) * .29, .5, .28 , .23])
        plt.xticks(np.arange(-.6, .3, .1), [])
    else:
        ax.set_position([.1 + (n - 4) * .29, .26, .28 , .23])
        plt.xticks(np.arange(-.6, .3, .1))
    if n == 5:
        plt.xlabel(r'$\omega$ (eV)', fontdict=font)
    if any(x==n for x in [1, 4]):
        plt.ylabel('Intensity (a.u)', fontdict=font)
    plt.yticks(np.arange(0, 40, 10), [])
    plt.xlim(xmin=-.37, xmax=.21)
    N_full = N_full + (n_full,)
    N_bands = N_bands + (n_band,)
    En = En + (en,)
    DOS = DOS + (dos,)
N = np.sum(N_bands)
print(N)
plt.show()


#%%
"""
Project: Advanced TB plot along direction
"""
k_pts = 200
x_GS = np.linspace(0, 1, k_pts)
y_GS = np.linspace(0, 1, k_pts)

x_SX = np.linspace(1, 0, k_pts)
y_SX = np.ones(k_pts)

x_XG = np.zeros(k_pts)
y_XG = np.linspace(1, 0, k_pts)

x = (x_GS, x_SX, x_XG)
y = (y_GS, y_SX, y_XG)
plt.figure('TB_eval', figsize=(6, 6), clear=True)
for i in range(len(x)):
    en, spec, bndstr = utils_math.CSRO_eval(x[i], y[i])
    k = np.sqrt(x[i] ** 2 + y[i] ** 2)
    v_bnd = .1
    if i != 0:
        ax = plt.subplot(2, 3, i + 1)
        ax.set_position([.1 + i * .2 + .2 * (np.sqrt(2)-1), .2, .2 , .4])
        plt.tick_params(direction='in', length=1.5, width=.5, colors='k') 
        k = -k
    else:
        ax = plt.subplot(2, 3, i + 1)
        ax.set_position([.1 + i * .2 , .2, .2 * np.sqrt(2) , .4])
        plt.tick_params(direction='in', length=1.5, width=.5, colors='k') 
    for j in range(len(bndstr)):
        plt.plot(k, bndstr[j])
    if i == 0:
        plt.xticks([0, np.sqrt(2)], ('$\Gamma$', 'S'))
        plt.yticks(np.arange(-1, 1, .2))
    elif i==1:
        plt.xticks([-np.sqrt(2), -1], ('', 'X'))
        plt.yticks(np.arange(-1, 1, .2), [])
    elif i==2:
        plt.xticks([-1, 0], ('', '$\Gamma$'))
        plt.yticks(np.arange(-1, 1, .2), [])
    plt.plot([k[0], k[-1]], [0, 0], 'k:')    
    plt.xlim(xmin=k[0], xmax=k[-1])
    plt.ylim(ymax=np.max(en), ymin=np.min(en))

plt.show()
#%%
"""
Project: Heat capacity
"""
DOS = dos
En = en
EF = 0

nbins = len(En);
T = np.arange(.01, 12, .005)
kB = 8.6173303e-5
C = np.ones(len(T)); Cp = np.ones(len(T))
U = np.ones(len(T))
stp = 10
expnt = np.ones((nbins,len(T)))
FD = np.ones((nbins, len(T)))
dFD = np.ones((nbins, len(T)))
expntext = np.ones((stp * nbins - (stp - 1), len(T)))
FDext = np.ones((stp * nbins - (stp - 1), len(T)))
dFDext = np.ones((stp * nbins - (stp - 1),len(T)))
J       = 1.60218e-19
mols    = 6.022140857e23

dE      = En[1] - En[0]
Enext   = np.arange(En[1], En[-1], dE / stp)
#DOSext  = interp1(En,DOS,Enext,'spline');

plt.plot(En, DOS)
#plot([EF+2*kB*T(end) EF+2*kB*T(end)],[0 max(DOS)],'r--');
#plot([EF-2*kB*T(end) EF-2*kB*T(end)],[0 max(DOS)],'r--');

for t in range(len(T)):
   expnt[:, t] = (En - EF) / (kB * T[t])
#   expntext[:, t] = (Enext - EF) / (kB * T[t])
   FD[:, t] = 1 / (np.exp(expnt[:,t]) + 1) 
#   FDext[:, t] = 1 / (np.exp(expntext[:,t]) + 1) 

for e in range(nbins):
   dFD[e, 0:-1] = np.diff(FD[e, :]) / (T[2] - T[1]) 
   dFD[e, -1] = dFD[e, -2]

#for e in np.arange(0, nbins - (stp - 1), stp):
#   dFDext[e, :-1] = np.diff(FDext[e, :]) / (T[2] - T[1])
#   dFDext[e, -1] = dFDext[e, -2]


Cpext = Cp;Uext = U;Cext = C;
for t in range(len(T)):
   Cp[t] = np.trapz((En - EF) * DOS * dFD[:,t], x=En)
   U[t] = np.trapz((En - EF) * DOS * FD[:,t], x=En)  
#   Cpext[t] = trapz(Enext,(Enext-EF).*DOSext.*dFDext(:,t)'); %Integrate + Derivative inside integral (dFD)
#   Uext[t] = trapz(Enext,(Enext-EF).*DOSext.*FDext(:,t)');  %Integrate

C[:-1] = np.diff(U[:]) / (T[2] - T[1]) 
C[-1] = C[-2] 
#Cext(1:end-1) = diff(Uext(:))./(T(2)-T(1)); Cext(end) = Cext(end-1); %heat cap. derivative after integration
pre = J*mols*1000; 

#gammap = pre * Cpext / T
gamma  = pre * C / T
plt.figure(r'$C_el$')
plt.plot(T, gamma)
#%%
"""
Project: Luttinger' theorem (FS counting)
"""
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
"""
Project: Experimental setup with orbitals
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import sph_harm 

# nur fuer den Seiteneffekt: plt.gca(projection = '3d') funktioniert sonst nicht
from mpl_toolkits.mplot3d import Axes3D 
from matplotlib import cm

theta_1d = np.linspace(0,   np.pi,  91) # 2 GRAD Schritte
phi_1d   = np.linspace(0, 2*np.pi, 181) # 2 GRAD Schritte

theta_2d, phi_2d = np.meshgrid(theta_1d, phi_1d)
xyz_2d = np.array([np.sin(theta_2d) * np.sin(phi_2d),
                  np.sin(theta_2d) * np.cos(phi_2d),
                  np.cos(theta_2d)]) 

colormap = cm.ScalarMappable( cmap=plt.get_cmap("cool"))
colormap.set_clim(-.45, .45)
limit = .5

def show_Y_lm(l, m):
    print("Y_%i_%i" % (l,m)) # zeigen, dass was passiert
    plt.figure()
    ax = plt.gca(projection = "3d")
    
    plt.title("$Y^{%i}_{%i}$" % (m,l))
    Y_lm = sph_harm(m,l, phi_2d, theta_2d)
    r = np.abs(Y_lm.real)*xyz_2d
    ax.plot_surface(r[0], r[1], r[2], 
                    facecolors=colormap.to_rgba(Y_lm.real), 
                    rstride=2, cstride=2)
    ax.set_xlim(-limit,limit)
    ax.set_ylim(-limit,limit)
    ax.set_zlim(-limit,limit)
    ax.set_aspect("equal")
    #ax.set_axis_off()
    

# Vorsicht: diese Schleifen erzeugen 16 plots (in 16 Fenstern)!
for l in range(0,4):
    for m in range(-l,l+1):
        show_Y_lm(l,m)

show_Y_lm(l=5,m=0)
show_Y_lm(l=5,m=4)        
show_Y_lm(l=6,m=6)
        
plt.show()








