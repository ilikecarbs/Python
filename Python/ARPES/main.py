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
CSROfig11: Tight binding model CSRO
CSROfig12: Tight binding model SRO

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

#%%

os.chdir('/Users/denyssutter/Documents/library/Python/ARPES')
mat = 'CaMn2Sb2'
year = 2018
sample = 'S2_hv100_hv130_T230'
mode = 'hv'

file = 1
D = ARPES.CASS(file, mat, year, sample, mode)
#%%
D.plt_hv()
#D.FS(e = 86.2, ew = .02, norm = False)
#D.ang2kFS(D.ang, Ekin=82, lat_unit=False, a=1, b=1, c=1, 
#          V0=0, thdg=0, tidg=15, phidg=-7)
#D.FS_flatten(ang=False)
#D.plt_FS(coord=True)

#%%
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
#%%
with open(path_info) as f:
    for line in f.readlines():
        if 'hv' in line:
            hv_raw = line.strip(('hv (eV):'))
            try:
                hv = np.float32(hv_raw.split())
                print(hv)
            except ValueError:   
                 print('')
            
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
#bndstr = tb.bndstr
#coord = tb.coord   
#X = coord['X']; Y = coord['Y']   
#Axz = bndstr['Axz']; Ayz = bndstr['Ayz']; Axy = bndstr['Axy']
#Bxz = bndstr['Bxz']; Byz = bndstr['Byz']; Bxy = bndstr['Bxy']
#bands = (Axz, Ayz, Axy, Bxz, Byz, Bxy)
#os.chdir('/Users/denyssutter/Documents/PhD/data')
#np.savetxt('Data_Axz_kpts_5000.dat', Axz)
#np.savetxt('Data_Ayz_kpts_5000.dat', Ayz)
#np.savetxt('Data_Axy_kpts_5000.dat', Axy)
#np.savetxt('Data_Bxz_kpts_5000.dat', Bxz)
#np.savetxt('Data_Byz_kpts_5000.dat', Byz)
#np.savetxt('Data_Bxy_kpts_5000.dat', Bxy)
#os.chdir('/Users/denyssutter/Documents/library/Python/ARPES')
#%%
os.chdir('/Users/denyssutter/Documents/PhD/data')
Axz = np.loadtxt('Data_Axz_kpts_5000.dat')
Ayz = np.loadtxt('Data_Ayz_kpts_5000.dat')
Axy = np.loadtxt('Data_Axy_kpts_5000.dat')
Bxz = np.loadtxt('Data_Bxz_kpts_5000.dat')
Byz = np.loadtxt('Data_Byz_kpts_5000.dat')
Bxy = np.loadtxt('Data_Bxy_kpts_5000.dat')
os.chdir('/Users/denyssutter/Documents/library/Python/ARPES')
bands = (Axz, Ayz, Axy, Bxz, Byz, Bxy)
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
    dos, bins, patches = plt.hist(np.ravel(band), bins=150, density=True,
                                alpha=.2, color='C8')
    en = np.zeros((len(dos)))
    for i in range(len(dos)):
        en[i] = (bins[i] + bins[i + 1]) / 2
    ef, _ef = utils.find(en, 0.00)
    n_full = np.trapz(dos, x=en)
    n_band = np.trapz(dos[:_ef], x=en[:_ef])
    plt.tick_params(direction='in', length=1.5, width=.5, colors='k')
    plt.plot(en, dos, color='k', lw=.5)
    plt.fill_between(en[:_ef], dos[:_ef], 0, color='C1', alpha=.5)
    plt.xlim(xmin=np.min(en), xmax=np.max(en))
    N_full = N_full + (n_full,)
    N_bands = N_bands + (n_band,)
    En = En + (en,)
    DOS = DOS + (dos,)
N = np.sum(N_bands)
print(N)
plt.show()

#%%
from numpy import linalg as la
a = np.pi

#x = np.linspace(-1, 1, 200)
#y = np.linspace(-1, 1, 200)
x = np.linspace(0, 2, 200)
y = np.zeros(len(x))

#Load TB parameters
t1 = param['t1']; t2 = param['t2']; t3 = param['t3']
t4 = param['t4']; t5 = param['t5']; t6 = param['t6']
mu = param['mu']; l = param['l']
#Hopping terms
fx = -2 * np.cos((x + y) / 2 * a)
fy = -2 * np.cos((x - y) / 2 * a)
f4 = -2 * t4 * (np.cos(x * a) + np.cos(y * a))
f5 = -2 * t5 * (np.cos((x + y) * a) + np.cos((x - y) * a))
f6 = -2 * t6 * (np.cos(x * a) - np.cos(y * a))
#Placeholders energy eigenvalues
Ayz = np.ones(len(x)); Axz = np.ones(len(x))
Axy = np.ones(len(x)); Byz = np.ones(len(x)) 
Bxz = np.ones(len(x)); Bxy = np.ones(len(x))
wAyz = np.ones((len(x), 6)); wAxz = np.ones((len(x), 6))
wAxy = np.ones((len(x), 6)); wByz = np.ones((len(x), 6))
wBxz = np.ones((len(x), 6)); wBxy = np.ones((len(x), 6))

###Projectors###
PAyz = np.array([[1,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],
                 [0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0]])
PAxz = np.array([[0,0,0,0,0,0],[0,1,0,0,0,0],[0,0,0,0,0,0],
                 [0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0]])
PAxy = np.array([[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,1,0,0,0],
                 [0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0]])
PByz = np.array([[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],
                 [0,0,0,1,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0]])
PBxz = np.array([[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],
                 [0,0,0,0,0,0],[0,0,0,0,1,0],[0,0,0,0,0,0]])
PBxy = np.array([[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],
                 [0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,1]])
    
#TB submatrix
def A(i):
    A = np.array([[-mu, complex(0,l) + f6[i], -l],
                  [-complex(0,l) + f6[i], -mu, complex(0,l)],
                  [-l, -complex(0,l), -mu + f4[i] + f5[i]]])
    return A
#TB submatrix
def B(i): 
    B = np.array([[t2 * fx[i] + t1 * fy[i], 0, 0],
                  [0, t1 * fx[i] + t2 * fy[i], 0],
                  [0, 0, t3 * (fx[i] + fy[i])]])
    return B
#Tight binding Hamiltonian
def H(i):
    C1 = np.concatenate((A(i), B(i)), 1)
    C2 = np.concatenate((B(i), A(i)), 1)
    H  = np.concatenate((C1, C2), 0)
    return H
#Diagonalization of symmetric Hermitian matrix on k-mesh
plt.figure('TB_eval', figsize=(6, 6), clear=True)
for i in range(len(x)):
    eval, evec = la.eigh(H(i))
    eval = np.real(eval)
    Ayz[i] = eval[0]; Axz[i] = eval[1]; Axy[i] = eval[2]
    Byz[i] = eval[3]; Bxz[i] = eval[4]; Bxy[i] = eval[5]
    en = (Ayz[i], Axz[i], Axy[i], Byz[i], Bxz[i], Bxy[i])
    n = 0
    for en_value in en:
        wAyz[i, n] = np.real(np.sum(np.conj(evec[:, n]) * (PAyz * evec[:, n]))) 
        wAxz[i, n] = np.real(np.sum(np.conj(evec[:, n]) * (PAxz * evec[:, n]))) 
        wAxy[i, n] = np.real(np.sum(np.conj(evec[:, n]) * (PAxy * evec[:, n]))) 
        wByz[i, n] = np.real(np.sum(np.conj(evec[:, n]) * (PByz * evec[:, n]))) 
        wBxz[i, n] = np.real(np.sum(np.conj(evec[:, n]) * (PBxz * evec[:, n]))) 
        wBxy[i, n] = np.real(np.sum(np.conj(evec[:, n]) * (PBxy * evec[:, n]))) 
        plt.plot(x[i], en_value, 'o', ms=3,
                 color=[wAxz[i, n] + wBxz[i, n], wAyz[i, n] + wByz[i, n], wAxy[i, n] + wBxy[i, n]])
        n += 1
        
plt.plot([x[0], x[-1]], [0, 0], 'k:')
plt.show()

#%%
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
xy = np.ones((len(x), len(y))); wyz = np.ones((len(x)))
wxz = np.ones((len(x))); wxy = np.ones((len(x)))

#Tight binding Hamiltonian
def H(i,j):
    H = np.array([[fyz[i, j] - mu, off[i, j] + complex(0, l), -l],
                  [off[i, j] - complex(0, l), fxz[i, j] - mu, complex(0, l)],
                  [-l, -complex(0, l), fxy[i, j] - mu]])
    return H
plt.figure(20011, figsize=(5, 5), clear=True)
plt.tick_params(direction='in', length=1.5, width=.5, colors='k')  
#Diagonalization of symmetric Hermitian matrix on k-mesh
#for n in range(1):
for i in range(len(x)):
    for j in range(len(y)):
        eval, evec = la.eigh(H(i, j))
        eval = np.real(eval)
        yz[i, j] = eval[0]; xz[i, j] = eval[1]; xy[i, j] = eval[2]
#            wyz[i] = np.sum(np.conj(evec[:, n]) * (Pyz * evec[:, n])) 
#            wxz[i] = np.sum(np.conj(evec[:, n]) * (Pxz * evec[:, n])) 
#            wxy[i] = np.sum(np.conj(evec[:, n]) * (Pxy * evec[:, n])) 
#            plt.contour(X[i, j], Y[i, j], yz[i, j], levels=e0)
            
c = plt.contour(X, Y, xz, colors = 'black', linestyles = ':', 
                            alpha=1, levels = 0)
p = c.collections[0].get_paths()
p = np.asarray(p)

plt.figure('20011a', figsize=(5, 5), clear=True)
for i in range(2):
    v = p[i].vertices
    plt.plot(v[:, 0], v[:, 1])
    plt.axis('equal')
    plt.show()
#%%
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










