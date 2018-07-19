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
CSROfig14: (L): TB and density of states
CSROfig15: DMFT FS
CSROfig16: (L): DMFT bandstructure calculation
CSROfig17: (L): LDA bandstructure calculation

---------  To-Do ---------

CSRO: TB with cuts
CSRO: Symmetrization
CSRO: FS area counting 
CSRO: kz dependence
CSRO: CSRO30 vs CSRO20 (FS and cuts)
CSRO: TB specific heat
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
#utils_plt.CSROfig14()
#utils_plt.CSROfig15()
#utils_plt.CSROfig16()
#utils_plt.CSROfig17()
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

