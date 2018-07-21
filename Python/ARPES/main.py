#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Created on Tue Jun 19 15:14:29 2018

@author: ilikecarbs

%%%%%%%%%%%%%%%%%%%%%
        main
%%%%%%%%%%%%%%%%%%%%%

**Development environment and plotting figures for dissertation**

.. note::
        To-Do:
            -
"""

import os
import utils_plt
import utils
import matplotlib.pyplot as plt
import ARPES
import numpy as np
import time
import matplotlib.cm as cm
from scipy.stats import exponnorm
from scipy.optimize import curve_fit

os.chdir('/Users/denyssutter/Documents/library/Python/ARPES')

rainbow_light = utils.rainbow_light
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

# %%
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
CSROfig18: CSRO30 Experimental band structure
CSROfig19: CSRO30 Gamma - S cut epsilon pocket

---------  To-Do ---------

CSRO: TB with cuts
CSRO: Symmetrization
CSRO: FS area counting
CSRO: kz dependence
CSRO: TB specific heat
"""
# --------
# utils_plt.CROfig1()
# utils_plt.CROfig2()
# utils_plt.CROfig3()
# utils_plt.CROfig4()
# utils_plt.CROfig5()
# utils_plt.CROfig6()
# utils_plt.CROfig7()
# utils_plt.CROfig8()
# utils_plt.CROfig9()
# utils_plt.CROfig10()
# utils_plt.CROfig11()
# utils_plt.CROfig12()
# utils_plt.CROfig13()
# utils_plt.CROfig14()
# --------
utils_plt.CSROfig1()
# utils_plt.CSROfig2()
# utils_plt.CSROfig3()
# utils_plt.CSROfig4()
# utils_plt.CSROfig5()
# utils_plt.CSROfig6()
# utils_plt.CSROfig7()
# utils_plt.CSROfig8()
# utils_plt.CSROfig9()
# utils_plt.CSROfig10()
# utils_plt.CSROfig11()
# utils_plt.CSROfig12()
# utils_plt.CSROfig13()
# utils_plt.CSROfig14()
# utils_plt.CSROfig15()
# utils_plt.CSROfig16()
# utils_plt.CSROfig17()
# utils_plt.CSROfig18()
# utils_plt.CSROfig19()
# %%


def func(x, n, *p):
    y = n + p[0] * x + p[1]
    return y


def wrapper(x, *p):
    n = 3
    return func(x, n, *p)

n=1
p = np.array([1, -1])
x = np.linspace(0, 1, 100)
#y = func(x, n, *p)
y = wrapper(x, *p)


plt.plot(x, y)

# %%


def get_text(name):
    return "lorem ipsum, {0} dolor sit amet".format(name)


def p_decorate(func):
    def func_wrapper(name):
        return "<p>{0}</p>".format(func(name))
    return func_wrapper


my_get_text = p_decorate(get_text)

print(my_get_text("John"))

# %%
"""
Project: Heat capacity
"""
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

#%%
"""
Project: GUI
"""
from pyqtgraph.Qt import QtGui, QtCore, QtWidgets
import pyqtgraph as pg 

#def FS_GUI(D): 
    # Interpret image data as row-major instead of col-major
pg.setConfigOptions(imageAxisOrder='row-major')

app = QtCore.QCoreApplication.instance()
if app is None:
    app = QtWidgets.QApplication(sys.argv)
    
## Create window with two ImageView widgets
mw = QtGui.QMainWindow()
mw.resize(1500,800)
mw.setWindowTitle('pyqtgraph example: DataSlicing')
cw = QtGui.QWidget()
mw.setCentralWidget(cw)
l = QtGui.QGridLayout()
cw.setLayout(l)

imv1 = pg.ImageView()
imv2 = pg.ImageView()
l.addWidget(imv1, 0, 0, 0, 1)
l.addWidget(imv2, 0, 1, 0, 1)
mw.show()
data = np.transpose(D.int_norm,(2,0,1))
roi = pg.LineSegmentROI([[10, 64], [120,64]], pen='r')
imv1.addItem(roi)

def update():
    global data, imv1, imv2
    d = roi.getArrayRegion(data, imv1.imageItem, axes=(0,1))
    imv2.setImage(d)
    
roi.sigRegionChanged.connect(update)

## Display the data
imv1.setImage(data, scale = (1, 6))
imv1.setHistogramRange(-0.01, 0.01)
imv1.setLevels(-0.003, 0.003)

#imv1.scale(0.2, 0.2)

update()

## Start Qt event loop unless running in interactive mode.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
    
#%%
        #def lor2(x, p0, p1, 
#         p2, p3, 
#         p4, p5, 
#         p6, p7, p8):
#    """
#    Two lorentzians on a quadratic background
#    """
#    return (p4 / (1 + ((x - p0) / p2) ** 2) + 
#            p5 / (1 + ((x - p1) / p3) ** 2) +
#            p6 + p7 * x + p8 * x ** 2)
#
#def lor4(x, p0, p1, p2, p3, 
#         p4, p5, p6, p7, 
#         p8, p9, p10, p11, 
#         p12, p13, p14):
#    """
#    Four lorentzians on a quadratic background
#    """
#    return (p8 / (1 + ((x - p0) / p4)  ** 2) + 
#            p9 / (1 + ((x - p1) / p5)  ** 2) +
#            p10 / (1 + ((x - p2) / p6)  ** 2) +
#            p11 / (1 + ((x - p3) / p7)  ** 2) +
#            p12 + p13 * x + p14 * x ** 2)
#    
#def lor6(x, p0, p1, p2, p3, p4, p5, 
#         p6, p7, p8, p9, p10, p11, 
#         p12, p13, p14, p15, p16, p17, 
#         p18, p19, p20):
#    """
#    Six lorentzians on a quadratic background
#    """
#    return (p12 / (1 + ((x - p0) / p6)  ** 2) + 
#            p13 / (1 + ((x - p1) / p7)  ** 2) +
#            p14 / (1 + ((x - p2) / p8)  ** 2) +
#            p15 / (1 + ((x - p3) / p9)  ** 2) +
#            p16 / (1 + ((x - p4) / p10) ** 2) +
#            p17 / (1 + ((x - p5) / p11) ** 2) +
#            p18 + p19 * x + p20 * x ** 2)
#
#def lor7(x, p0, p1, p2, p3, p4, p5, p6,
#         p7, p8, p9, p10, p11, p12, p13, 
#         p14, p15, p16, p17, p18, p19, p20,
#         p21, p22, p23):
#    """
#    Seven lorentzians on a quadratic background
#    """
#    return (p14 / (1 + ((x - p0) / p7)  ** 2) + 
#            p15 / (1 + ((x - p1) / p8)  ** 2) +
#            p16 / (1 + ((x - p2) / p9)  ** 2) +
#            p17 / (1 + ((x - p3) / p10) ** 2) +
#            p18 / (1 + ((x - p4) / p11) ** 2) +
#            p19 / (1 + ((x - p5) / p12) ** 2) +
#            p20 / (1 + ((x - p6) / p13) ** 2) +
#            p21 + p22 * x + p23 * x ** 2)
#    
#def lor8(x, p0, p1, p2, p3, p4, p5, p6, p7, 
#         p8, p9, p10, p11, p12, p13, p14, p15, 
#         p16, p17, p18, p19, p20, p21, p22, p23, 
#         p24, p25, p26):
#    """
#    Eight lorentzians on a quadratic background
#    """
#    return (p16 / (1 + ((x - p0) / p8)  ** 2) + 
#            p17 / (1 + ((x - p1) / p9)  ** 2) +
#            p18 / (1 + ((x - p2) / p10) ** 2) +
#            p19 / (1 + ((x - p3) / p11) ** 2) +
#            p20 / (1 + ((x - p4) / p12) ** 2) +
#            p21 / (1 + ((x - p5) / p13) ** 2) +
#            p22 / (1 + ((x - p6) / p14) ** 2) +
#            p23 / (1 + ((x - p7) / p15) ** 2) +
#            p24 + p25 * x + p26 * x ** 2)
#        
#def gauss2(x, p0, p1, p2, p3, p4, p5, p6, p7, p8):
#    """
#    Two gaussians on a quadratic background
#    """
#    return (p4 * np.exp(-(x - p0) ** 2 / (2 * p2 ** 2)) + 
#            p5 * np.exp(-(x - p1) ** 2 / (2 * p3 ** 2)) +
#            p6 + p7 * x + p8 * x ** 2)    