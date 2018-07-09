#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 11:30:51 2018

@author: denyssutter
"""    
    
import os
os.chdir('/Users/denyssutter/Documents/library/Python/ARPES')
import ARPES
import utils
import utils_math
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
import matplotlib.pyplot as plt
from numpy import linalg as la
import matplotlib.cm as cm
from scipy.stats import exponnorm
from scipy.optimize import curve_fit

plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.serif']=['Computer Modern Roman']
plt.rc('font', **{'family': 'serif', 'serif': ['STIXGeneral']})
font = {'family': 'serif',
        'style': 'normal',
        'color':  [0,0,0],
        'weight': 'ultralight',
        'size': 12,
        }

# +----------+ #
# | Colormap | # ===============================================================
# +----------+ #

def rainbow_light():
    filepath = '/Users/denyssutter/Documents/library/Python/ARPES/cmap/rainbow_light.dat'
    data = np.loadtxt(filepath)
    colors = np.array([(i[0], i[1], i[2]) for i in data])
    
    # Normalize the colors
    colors /= colors.max()
    
    # Build the colormap
    rainbow_light = LinearSegmentedColormap.from_list('rainbow_light', colors, 
                                                      N=len(colors))
    return rainbow_light

def rainbow_light_2():
    filepath = '/Users/denyssutter/Documents/library/Python/ARPES/cmap/rainbow_light_2.dat'
    data = np.loadtxt(filepath)
    colors = np.array([(i[0], i[1], i[2]) for i in data])
    
    # Normalize the colors
    colors /= colors.max()
    
    # Build the colormap
    rainbow_light_2 = LinearSegmentedColormap.from_list('rainbow_light', colors, 
                                                      N=len(colors))
    return rainbow_light_2

def orbitals():
    colors = np.zeros((100,3))
    for i in range(100):
        colors[i,:] = [i/100, 0, 1 - i/100]
        
    # Normalize the colors
    colors /= colors.max()
    
    # Build the colormap
    orbitals = LinearSegmentedColormap.from_list('orbitals', colors, 
                                                      N=len(colors))
    return orbitals
    
rainbow_light = rainbow_light()
cm.register_cmap(name='rainbow_light', cmap=rainbow_light)
 
rainbow_light_2 = rainbow_light_2()
cm.register_cmap(name='rainbow_light_2', cmap=rainbow_light_2)

orbitals = orbitals()
cm.register_cmap(name='orbitals', cmap=orbitals)

def plt_spec(self, norm):
    if norm == True:
        k = self.angs
        en = self.en_norm
        dat = self.int_norm
    elif norm == 'shift':
        k = self.angs
        en = self.en_shift
        dat = self.int_shift
    elif norm == False:
        k = self.ang
        en = self.en
        dat = np.transpose(self.int)
    plt.figure(20006, figsize=(10, 10), clear=True)
    plt.tick_params(direction='in', length=1.5, width=.5, colors='k')
    plt.contourf(k, en, dat, 100, cmap = cm.ocean_r)
    if norm == True:
        plt.plot([np.min(k), np.max(k)], [0, 0], 'k:')
    plt.xlabel('$k_x$')   
    plt.ylabel('\omega')
    plt.show()

def plt_FS_polcut(self, norm, p, pw):
    if norm == True:
        k = self.angs
        en = self.en_norm
        dat = self.int_norm
    elif norm == 'shift':
        k = self.angs
        en = self.en_shift
        dat = self.int_shift
    elif norm == False:
        k = self.ang
        en = self.en
        dat = np.transpose(self.int)
    p_val, p_ind = utils.find(self.pol, p)
    pw_val, pw_ind = utils.find(self.pol, p - pw)
    spec = np.sum(dat[:, : , pw_ind:p_ind], axis=2)
    plt.figure(20005, figsize=(10, 10), clear=True)
    plt.tick_params(direction='in', length=1.5, width=.5, colors='k')
    plt.contourf(k, en, spec, 100, cmap = cm.ocean_r)
    plt.plot([np.min(k), np.max(k)], [-2.8, -2.8], 'k:')
    if norm == True:
        plt.plot([np.min(k), np.max(k)], [0, 0], 'k:')
    plt.xlabel('$k$')   
    plt.ylabel('\omega')
    plt.show()

def plt_hv(self, a, aw):
    k = self.ang
    hv = self.hv
    en = self.en
    dat = self.int
    a_val, a_ind = utils.find(self.ang, a)
    aw_val, aw_ind = utils.find(self.ang, a - aw)
    spec = np.sum(dat[:, aw_ind:a_ind+1, :], axis=1)
    ###Plotting###
    n = np.ceil(np.sqrt(self.hv.size))
    plt.figure(2004, figsize=(10, 10), clear=True)
    plt.tick_params(direction='in', length=1.5, width=.5, colors='k')
    for i in range(self.hv.size):
        plt.subplot(n, n, i+1)
        plt.contourf(k, en, np.transpose(dat[i, :, :]), 100, cmap = cm.ocean_r)
        plt.xticks((0, 0), ('', ''))
        plt.title(str(np.round(hv[i], 0))+" eV")
    plt.figure(20007, figsize=(10, 10), clear=True)
    plt.contourf(hv, en, np.transpose(spec), 100, cmap = cm.ocean_r)
    plt.show()
        
def plt_FS(self, coord):
    if coord == True:
        kx = self.kx
        ky = self.ky
        dat = self.map
    elif coord == False:
        kx = self.ang
        ky = self.pol
        dat = self.map
    plt.figure(20000, figsize=(10,10), clear=True)
    plt.tick_params(direction='in', length=1.5, width=.5, colors='k')
    plt.contourf(kx, ky, dat, 100, cmap = cm.ocean_r)
    plt.axis('equal')
    plt.colorbar()
    plt.show()

def plt_cont_TB_simple(self, e0):
    bndstr = self.bndstr
    coord = self.coord   
    X = coord['X']; Y = coord['Y']   
    en = bndstr['en']
    plt.figure(20003, figsize=(10, 10), clear=True)
    plt.contour(X, Y, en, levels = e0)
  
def plt_cont_TB_SRO(self, e0):
    bndstr = self.bndstr
    coord = self.coord   
    X = coord['X']; Y = coord['Y']   
    xz = bndstr['xz']; yz = bndstr['yz']; xy = bndstr['xy']
    en = (xz, yz, xy)
    plt.figure(20002, figsize=(10, 3), clear=True)
    n = 0
    for i in en:
        n = n + 1
        plt.subplot(1, 3, n)
        plt.contour(X, Y, i, colors = 'black', linestyles = ':', levels = e0)
        plt.axis('equal')
  
def plt_cont_TB_CSRO20(self, e0):   
    bndstr = self.bndstr
    coord = self.coord   
    X = coord['X']; Y = coord['Y']   
    Axz = bndstr['Axz']; Ayz = bndstr['Ayz']; Axy = bndstr['Axy']
    Bxz = bndstr['Bxz']; Byz = bndstr['Byz']; Bxy = bndstr['Bxy']
    en = (Axz, Ayz, Axy, Bxz, Byz, Bxy)
    plt.figure(20001, figsize=(6, 4), clear=True)
    n = 0
    for i in en:
        n = n + 1
        plt.subplot(2, 3, n)
        plt.contour(X, Y, i, colors = 'black', linestyles = ':', levels = e0)
        plt.axis('equal')
        
def CRO_theory_plot(k_pts, data_en, data, colmap, v_max, fignr):
    c = len(data)
    scale = .02
    plt.figure(fignr, figsize=(10, 10), clear = True)
    for k in range(len(data)): #looping over segments of k-path
        c = len(data[k])
        m, n = data[k][0].shape
        data_kpath = np.zeros((1, c * n)) #Placeholders
        data_spec  = np.zeros((m, c * n)) #Placeholders
        k_seg = [0] #Used to mark k-points along in k-path 
        for i in range(c):
            diff = abs(np.subtract(k_pts[k][i], k_pts[k][i+1])) #distances in k-space
            k_seg.append(k_seg[i] + la.norm(diff)) #extending list cummulative
            data_spec[:, n * i : n * (i+1)] = data[k][i] #Feeding in data
        data_kpath = np.linspace(0, k_seg[-1], c * n) #Full k-path for plotting
        
        ###Plotting###
        #Setting which axes should be ticked and labelled
        plt.rcParams['xtick.labelbottom'] = True
        plt.rcParams['xtick.labeltop'] = False
        
        #Subplot sensitive formatting
        if k==0:
            plt.rcParams['ytick.labelright'] = False
            plt.rcParams['ytick.labelleft'] = True
            ax = plt.subplot(1, len(data), k+1) 
            ax.set_position([.1, .3, k_seg[-1] * scale, .3])
            pos = ax.get_position()
            k_prev = k_seg[-1] 
            
        else:
            plt.rcParams['ytick.labelright'] = False
            plt.rcParams['ytick.labelleft'] = False
            ax = plt.subplot(1, len(data), k+1)
            ax.set_position([pos.x0 + k_prev * scale, pos.y0, 
                             k_seg[-1] * scale, pos.height])
        k_prev = k_seg[-1] #For formatting subplot axis
        pos = ax.get_position()
        
        #Labels
        if k == 0:
            plt.ylabel('$\omega$ (eV)', fontdict = font)
            plt.xticks(k_seg, ('S', '$\Gamma$', 'S'))
        elif k == 1:
            plt.xticks(k_seg, ('', 'X', 'S'))
        elif k == 2:
            plt.xticks(k_seg, ('', '$\Gamma$'))
        elif k == 3:
            plt.xticks(k_seg, ('', 'X', '$\Gamma$', 'X'))
        plt.tick_params(direction='in', length=1.5, width=.5, colors='k')    
        plt.contourf(data_kpath, data_en, data_spec, 300, cmap = colmap,
                       vmin=0, vmax=v_max*np.max(data_spec))
        plt.ylim(ymax = 0, ymin = -2.5)
    cax = plt.axes([pos.x0 + k_prev * scale + 0.01,
                    pos.y0, 0.01, pos.height])
    cbar = plt.colorbar(cax = cax, ticks = None)
    cbar.set_ticks([])
    cbar.set_clim(np.min(data_spec), np.max(data_spec))
    ax.set_position([pos.x0, pos.y0, k_prev * scale, pos.height])
    
def CRO_FS_plot(colmap, e, v_min, fignr):
    """
    Constant energy maps Oxygen bands
    """
    p65 = '0618_00113'
    s65 = '0618_00114'
    p120 = '0618_00115'
    s120 = '0618_00116'
    mat = 'Ca2RuO4'
    year = 2016
    sample = 'data'
    files = [p120, s120, p65, s65]
    lbls1 = ['(a)', '(b)', '(c)', '(d)']
    lbls2 = [r'$120\,\mathrm{eV}$', r'$120\,\mathrm{eV}$', r'$65\,\mathrm{eV}$', r'$65\,\mathrm{eV}$']
    lbls3 = [r'$\bar{\pi}$-pol.', r'$\bar{\sigma}$-pol.', r'$\bar{\pi}$-pol.', r'$\bar{\sigma}$-pol.']
    th = 25
    ti = -.5
    phi = -25.
    c = (0, 238 / 256, 118 / 256)
    ###Plotting###
    plt.figure(fignr, figsize=(10, 10), clear=True)
    for i in range(4):
        D = ARPES.ALS(files[i], mat, year, sample) #frist scan
        D.ang2kFS(D.ang, Ekin=D.hv - 4.5 + e, lat_unit=True, a=4.8, b=5.7, c=11, 
                        V0=0, thdg=th, tidg=ti, phidg=phi)
        en = D.en - 2.1 #energy off set (Fermi level not specified)
        ew = 0.1
        e_val, e_ind = utils.find(en, e)
        ew_val, ew_ind = utils.find(en, e-ew)
        FSmap = np.sum(D.int[:, :, ew_ind:e_ind], axis=2) #creating FS map
        ax = plt.subplot(1, 5, i + 2) 
        ax.set_position([.06 + (i * .23), .3, .22, .3])
        if i == 2:
            ax.set_position([.06 + (i * .23), .3, .16, .3])
        elif i == 3:
            ax.set_position([.06 + (2 * .23) + .17, .3, .16, .3])
        plt.tick_params(direction='in', length=1.5, width=.5, colors='k')
        plt.contourf(D.kx, D.ky, FSmap, 300, cmap = colmap,
                       vmin = v_min * np.max(FSmap), vmax = .95 * np.max(FSmap))
        plt.grid(alpha=0.5)
        plt.xticks(np.arange(-10, 10, 2))
        plt.xlabel('$k_x$ ($\pi/a$)', fontdict = font)
        plt.plot([-1, -1], [-1, 1], 'k-')
        plt.plot([1, 1], [-1, 1], 'k-')
        plt.plot([-1, 1], [1, 1], 'k-')
        plt.plot([-1, 1], [-1, -1], 'k-')
        plt.plot([-2, 0], [0, 2], 'k--', linewidth=.5)
        plt.plot([-2, 0], [0, -2], 'k--', linewidth=.5)
        plt.plot([2, 0], [0, 2], 'k--', linewidth=.5)
        plt.plot([2, 0], [0, -2], 'k--', linewidth=.5)
        if i == 0:
            plt.ylabel('$k_y$ ($\pi/a$)', fontdict = font)
            plt.yticks(np.arange(-10, 10, 2))
            plt.plot([-1, 1], [-1, 1], linestyle=':', color=c, linewidth=1)
            plt.plot([-1, 1], [1, 1], linestyle=':', color=c, linewidth=1)
            plt.plot([-1, 0], [1, 2], linestyle=':', color=c, linewidth=1)
            plt.plot([0, 0], [2, -1], linestyle=':', color=c, linewidth=1)
            ax.arrow(-1, -1, .3, .3, head_width=0.3, head_length=0.3, fc=c, ec='k')
            ax.arrow(0, -.4, 0, -.3, head_width=0.3, head_length=0.3, fc=c, ec='k')
        else:
            plt.yticks(np.arange(-10, 10, 2), [])
        if any(x==i for x in [0, 1]):
            x_pos = -2.7
        else:
            x_pos = -1.9
        plt.text(x_pos, 5.6, lbls1[i], fontsize=12)
        plt.text(x_pos, 5.0, lbls2[i], fontsize=10)
        plt.text(x_pos, 4.4, lbls3[i], fontsize=10) 
        plt.text(-0.2, -0.15, r'$\Gamma$',
                 fontsize=12, color='r')
        plt.text(-0.2, 1.85, r'$\Gamma$',
                 fontsize=12, color='r')
        plt.text(.85, .85, r'S',
                 fontsize=12, color='r')
        plt.text(-0.2, .9, r'X',
                 fontsize=12, color='r')
        plt.xlim(xmin=-3, xmax=4)
        if any(x==i for x in [2, 3]):
            plt.xlim(xmin=-2.2, xmax=2.9)
        plt.ylim(ymin=-3.3, ymax=6.2)
        
    pos = ax.get_position()
    cax = plt.axes([pos.x0+pos.width+0.01 ,
                        pos.y0, 0.01, pos.height])
    cbar = plt.colorbar(cax = cax, ticks = None)
    cbar.set_ticks([])
    cbar.set_clim(np.min(FSmap), np.max(FSmap))
"""
Figures Dissertation Ca2RuO4 (CRO)
"""

def CROfig1(colmap = cm.bone_r, print_fig = False):
    """
    Prepare and plot DFT data of Ca2RuO4 (final)
    """
    os.chdir('/Users/denyssutter/Documents/PhD/data')
    GS = pd.read_csv('DFT_CRO_GS_final.dat').values
    SG = np.fliplr(GS)
    GX = pd.read_csv('DFT_CRO_GX_final.dat').values
    XG = np.fliplr(GX)
    XS = pd.read_csv('DFT_CRO_YS_final.dat').values
    SX = np.fliplr(XS)
    os.chdir('/Users/denyssutter/Documents/library/Python/ARPES')
    
    ###k-points
    G = (0, 0)
    X = (np.pi, 0)
    S = (np.pi, np.pi)
    
    ###Data along path in k-space
    k_pts = np.array([[S, G, S], [S, X, S], [S, G], [G, X, G, X]])
    DFT = np.array([[SG, GS], [SX, XS], [SG], [GX, XG, GX]])
    DFT_en = np.linspace(-2.5,0,500)
    CRO_theory_plot(k_pts, DFT_en, DFT, colmap, v_max = 1, fignr=1001) #Plot data
    if print_fig == True:
        plt.savefig(
                '/Users/denyssutter/Documents/PhD/PhD_Denys/Figs/CROfig1.png', 
                dpi = 300,bbox_inches="tight")
    
def CROfig2(colmap = cm.bone_r, print_fig = False):
    """
    Prepare and plot DMFT data of Ca2RuO4 
    """
    os.chdir('/Users/denyssutter/Documents/PhD/data')
    xz_data = np.loadtxt('DMFT_CRO_xz.dat')
    yz_data = np.loadtxt('DMFT_CRO_yz.dat')
    xy_data = np.loadtxt('DMFT_CRO_xy.dat')
    os.chdir('/Users/denyssutter/Documents/library/Python/ARPES')
    m, n = 8000, 351 #dimensions energy, full k-path
    bot, top = 2500, 5000 #restrict energy window
    DMFT_data = np.array([xz_data, yz_data, xy_data]) #combine data
    DMFT_spec = np.reshape(DMFT_data[:, :, 2], (3, n, m)) #reshape into n,m
    DMFT_spec = DMFT_spec[:, :, bot:top] #restrict data to bot, top
    DMFT_en   = np.linspace(-8, 8, m) #define energy data
    DMFT_en   = DMFT_en[bot:top] #restrict energy data
    #[0, 56, 110, 187, 241, 266, 325, 350]  = [G,X,S,G,Y,T,G,Z]
    DMFT_spec = np.transpose(DMFT_spec, (0,2,1)) #transpose
    DMFT_spec = np.sum(DMFT_spec, axis=0) #sum up over orbitals
    
    ###Data used:
    GX = DMFT_spec[:, 0:56] 
    XG = np.fliplr(GX)
    XS = DMFT_spec[:, 56:110]
    SX = np.fliplr(XS)
    SG = DMFT_spec[:, 110:187]
    GS = np.fliplr(SG)
    
    ###k-points
    G = (0, 0)
    X = (np.pi, 0)
    S = (np.pi, np.pi)
    
    ###Data along path in k-space
    k_pts = np.array([[S, G, S], [S, X, S], [S, G], [G, X, G, X]])
    DMFT = np.array([[SG, GS], [SX, XS], [SG], [GX, XG, GX]])
    
    CRO_theory_plot(k_pts, DMFT_en, DMFT, colmap, v_max = .5, fignr=1002) #Plot data
    if print_fig == True:
        plt.savefig(
                '/Users/denyssutter/Documents/PhD/PhD_Denys/Figs/CROfig2.png', 
                dpi = 300,bbox_inches="tight")

def CROfig3(colmap = cm.bone_r, print_fig = False):
    """
    Prepare and plot DFT data of Ca2RuO4 (OSMT)
    """
    
    os.chdir('/Users/denyssutter/Documents/PhD/data')
    GS = pd.read_csv('DFT_CRO_GS_OSMT.dat').values
    SG = np.fliplr(GS)
    GX = pd.read_csv('DFT_CRO_GX_OSMT.dat').values
    XG = np.fliplr(GX)
    XS = pd.read_csv('DFT_CRO_YS_OSMT.dat').values
    SX = np.fliplr(XS)
    os.chdir('/Users/denyssutter/Documents/library/Python/ARPES')
    
    ###k-points
    G = (0, 0)
    X = (np.pi, 0)
    S = (np.pi, np.pi)
    
    ###Data along path in k-space
    k_pts = np.array([[S, G, S], [S, X, S], [S, G], [G, X, G, X]])
    DFT = np.array([[SG, GS], [SX, XS], [SG], [GX, XG, GX]])
    DFT_en = np.linspace(-2.5,0,500)
    
    CRO_theory_plot(k_pts, DFT_en, DFT, colmap, v_max = 1, fignr=1003) #Plot data
    if print_fig == True:
        plt.savefig(
                '/Users/denyssutter/Documents/PhD/PhD_Denys/Figs/CROfig3.png', 
                dpi = 300,bbox_inches="tight")

def CROfig4(colmap = cm.bone_r, print_fig = False):
    """
    Prepare and plot DFT data of Ca2RuO4 (OSMT)
    """
    
    os.chdir('/Users/denyssutter/Documents/PhD/data')
    GS = pd.read_csv('DFT_CRO_GS_uni.dat').values
    SG = np.fliplr(GS)
    GX = pd.read_csv('DFT_CRO_GX_uni.dat').values
    XG = np.fliplr(GX)
    XS = pd.read_csv('DFT_CRO_YS_uni.dat').values
    SX = np.fliplr(XS)
    os.chdir('/Users/denyssutter/Documents/library/Python/ARPES')
    
    ###k-points
    G = (0, 0)
    X = (np.pi, 0)
    S = (np.pi, np.pi)
    
    ###Data along path in k-space
    k_pts = np.array([[S, G, S], [S, X, S], [S, G], [G, X, G, X]])
    DFT = np.array([[SG, GS], [SX, XS], [SG], [GX, XG, GX]])
    DFT_en = np.linspace(-2.5,0,500)
    
    CRO_theory_plot(k_pts, DFT_en, DFT, colmap, v_max = 1, fignr=1004) #Plot data
    if print_fig == True:
        plt.savefig(
                '/Users/denyssutter/Documents/PhD/PhD_Denys/Figs/CROfig4.png', 
                dpi = 300,bbox_inches="tight")
    
def CROfig5(colmap = cm.ocean_r, print_fig = False):
    """
    Plot experimental Data Ca2RuO4
    """
    os.chdir('/Users/denyssutter/Documents/library/Python/ARPES')
    mat = 'Ca2RuO4'
    year = 2016
    sample = 'T10'
    plt.figure(1005, figsize = (10, 10), clear = True)
    files = np.array([47974, 48048, 47993, 48028])
    gold = 48000
    
    ###Plotting###
    #Setting which axes should be ticked and labelled
    scale = .02
    v_scale = 1.3
    k_seg_1 = np.array([0, 4.442882938158366, 8.885765876316732])
    k_seg_2 = np.array([0, 3.141592653589793, 6.283185307179586])
    k_seg_3 = np.array([0, 4.442882938158366])
    k_seg_4 = np.array([0, 3.141592653589793, 6.283185307179586, 9.42477796076938])
    
    n = 0
    for file in files:
        n += 1
        D = ARPES.DLS(file, mat, year, sample)
        D.shift(gold)
        D.norm(gold)
        D.restrict(bot=.6, top=1, left=0, right=1)
        D.flatten(norm=True)
        if n == 1:
            ax = plt.subplot(1, 4, n) 
            ax.set_position([.1, .3, k_seg_1[-1] * scale, .3])
            pos = ax.get_position()
            D.ang2k(D.ang, Ekin=65-4.5, lat_unit=True, a=3.89, b=3.89, c=11, 
                    V0=0, thdg=-4, tidg=0, phidg=0)
            plt.tick_params(direction='in', length=1.5, width=.5, colors='k')  
            plt.contourf(D.kxs, D.en_norm+.1, D.int_norm, 300,
                       cmap=colmap, 
                       vmin=v_scale * 0.01 * np.max(D.int_norm), 
                       vmax=v_scale * 0.5 * np.max(D.int_norm))
            plt.xlim(xmax = 1, xmin = -1)
            plt.ylabel('$\omega$ (eV)', fontdict = font)
            plt.xticks([-1, 0, 1], ('S', '$\Gamma$', 'S'))
            plt.yticks(np.arange(-2.5, 0.001, .5))
        elif n == 2:
            ax = plt.subplot(1, 4, n)
            ax.set_position([pos.x0 + k_seg_1[-1] * scale, pos.y0, 
                             k_seg_2[-1] * scale, pos.height])
            D.ang2k(D.ang, Ekin=65-4.5, lat_unit=True, a=3.89, b=3.89, c=11, 
                    V0=0, thdg=-7.5, tidg=8.5, phidg=45)
            plt.tick_params(direction='in', length=1.5, width=.5, colors='k')  
            plt.contourf(D.kxs, D.en_norm+.1, D.int_norm, 300,
                       cmap=colmap,
                       vmin=v_scale * 0.0 * np.max(D.int_norm), 
                       vmax=v_scale * 0.54 * np.max(D.int_norm))
            plt.xlim(xmax = 0, xmin = -1)
            plt.xticks([-1, -.5, 0], ('', 'X', 'S'))
            plt.yticks(np.arange(-2.5, 0, .5), [])
        elif n == 3:
            ax = plt.subplot(1, 4, n)
            ax.set_position([pos.x0 + k_seg_2[-1] * scale, pos.y0, 
                             k_seg_3[-1] * scale, pos.height])
            D.ang2k(D.ang, Ekin=65-4.5, lat_unit=True, a=3.89, b=3.89, c=11, 
                    V0=0, thdg=5, tidg=12.5, phidg=0)
            plt.tick_params(direction='in', length=1.5, width=.5, colors='k')  
            plt.contourf(D.kxs, D.en_norm+.1, np.flipud(D.int_norm), 300,
                       cmap=colmap, 
                       vmin=v_scale * 0.01 * np.max(D.int_norm), 
                       vmax=v_scale * 0.7 * np.max(D.int_norm))
            plt.xlim(xmax = 1, xmin = 0)
            plt.xticks([0, 1], ('', '$\Gamma$'))
            plt.yticks(np.arange(-2.5, 0, .5), [])
        elif n == 4:
            ax = plt.subplot(1, 4, n)
            ax.set_position([pos.x0 + k_seg_3[-1] * scale, pos.y0, 
                             k_seg_4[-1] * scale, pos.height])
            D.ang2k(D.ang, Ekin=65-4.5, lat_unit=True, a=3.89, b=3.89, c=11, 
                    V0=0, thdg=-9.5, tidg=0, phidg=45)
            plt.tick_params(direction='in', length=1.5, width=.5, colors='k')  
            plt.contourf(D.kxs, D.en_norm+.1, np.flipud(D.int_norm), 300,
                       cmap=colmap, 
                       vmin=v_scale * 0.01 * np.max(D.int_norm), 
                       vmax=v_scale * 0.53 * np.max(D.int_norm))
            plt.xlim(xmax = 1.5, xmin = 0)
            plt.xticks([0, 0.5, 1, 1.5], ('', 'X', '$\Gamma$', 'X'))
            plt.yticks(np.arange(-2.5, 0, .5), [])
        
        pos = ax.get_position()
        plt.ylim(ymax = .001, ymin = -2.5)
        plt.show()
    cax = plt.axes([pos.x0 + k_seg_4[-1] * scale + 0.01,
                    pos.y0, 0.01, pos.height])
    cbar = plt.colorbar(cax = cax, ticks = None)
    cbar.set_ticks([])
    cbar.set_clim(np.min(D.int_norm), np.max(D.int_norm))
    if print_fig == True:
        plt.savefig(
                '/Users/denyssutter/Documents/PhD/PhD_Denys/Figs/CROfig5.png', 
                dpi = 300,bbox_inches="tight")
    
def CROfig6(colmap = cm.ocean_r, print_fig = False):
    """
    Constant energy map of Ca2RuO4 of alpha branch over two BZ's
    """
    file1 = '0619_00161'
    file2 = '0619_00162'
    mat = 'Ca2RuO4'
    year = 2016
    sample = 'data'
    th = 20
    ti = -2
    phi = 21
    a = 5.5
    D1 = ARPES.ALS(file1, mat, year, sample) #frist scan
    D2 = ARPES.ALS(file2, mat, year, sample) #second scan
    D1.ang2kFS(D1.ang, Ekin=D1.hv-4.5-4.7, lat_unit=True, a=a, b=a, c=11, 
                    V0=0, thdg=th, tidg=ti, phidg=phi)
    D2.ang2kFS(D2.ang, Ekin=D2.hv-4.5-4.7, lat_unit=True, a=a, b=a, c=11, 
                    V0=0, thdg=th, tidg=ti, phidg=phi)    
    data = np.concatenate((D1.int, D2.int), axis=0) #combining two scans
    kx = np.concatenate((D1.kx, D2.kx), axis=0)
    ky = np.concatenate((D1.ky, D2.ky), axis=0)
    en = D1.en-2.3 #energy off set (Fermi level not specified)
    e = -2.2; ew = 0.2
    e_val, e_ind = utils.find(en, e)
    ew_val, ew_ind = utils.find(en, e-ew)
    FSmap = np.sum(data[:, :, ew_ind:e_ind], axis=2) #creating FS map
    
    ###Plotting###
    plt.figure(1006, figsize=(3.5, 5), clear=True)
    plt.tick_params(direction='in', length=1.5, width=.5, colors='k')
    plt.contourf(kx, ky, FSmap, 300, cmap = colmap,
                   vmin = .5 * np.max(FSmap), vmax = .95 * np.max(FSmap))
    plt.xlabel('$k_x$ ($\pi/a$)', fontdict = font)
    plt.ylabel('$k_y$ ($\pi/b$)', fontdict = font)
    plt.axis('equal')
    plt.grid(alpha=0.3)
    plt.xticks(np.arange(-10,10,1))
    plt.yticks(np.arange(-10,10,1))
    plt.plot([-1, -1], [-1, 1], 'k-')
    plt.plot([1, 1], [-1, 1], 'k-')
    plt.plot([-1, 1], [1, 1], 'k-')
    plt.plot([-1, 1], [-1, -1], 'k-')
    c = (0, 238 / 256, 118 / 256)
    plt.plot([-1, 1], [-1, 1], linestyle=':', color=c, linewidth=3)
    plt.plot([-1, 1], [1, 1], linestyle=':', color=c, linewidth=3)
    plt.plot([-1, 0], [1, 2], linestyle=':', color=c, linewidth=3)
    plt.plot([0, 0], [2, -1], linestyle=':', color=c, linewidth=3)
    ax = plt.axes()
    ax.arrow(-1, -1, .3, .3, head_width=0.2, head_length=0.2, fc=c, ec='k')
    ax.arrow(0, -.5, 0, -.3, head_width=0.2, head_length=0.2, fc=c, ec='k')
    plt.text(-0.1, -0.1, r'$\Gamma$',
             fontsize=20, color='r')
    plt.text(-0.1, 1.9, r'$\Gamma$',
             fontsize=20, color='r')
    plt.text(.9, .9, r'S',
             fontsize=20, color='r')
    plt.text(-0.1, .9, r'X',
             fontsize=20, color='r')
    plt.xlim(xmin=-1.1, xmax=1.1)
    plt.ylim(ymin=-1.1, ymax=3.1)
    pos = ax.get_position()
    cax = plt.axes([pos.x0+pos.width+0.03 ,
                        pos.y0, 0.03, pos.height])
    cbar = plt.colorbar(cax = cax, ticks = None)
    cbar.set_ticks([])
    plt.show()
    if print_fig == True:
        plt.savefig(
                    '/Users/denyssutter/Documents/PhD/PhD_Denys/Figs/CROfig6.png', 
                    dpi = 300,bbox_inches="tight")

def CROfig7(colmap = cm.ocean_r, print_fig = False):
    """
    Photon energy dependence Ca2RuO4 
    """
    file = 'CRO_SIS_0048'
    mat = 'Ca2RuO4'
    year = 2015
    sample = 'data'
    D = ARPES.SIS(file, mat, year, sample)
    D.ang2k(D.ang, Ekin=65-4.5, lat_unit=True, a=3.89, b=3.89, c=11, 
            V0=0, thdg=-4, tidg=0, phidg=0)
    int1 = D.int[11, :, :]
    int2 = D.int[16, :, :] * 3.9
    edc_val = 1
    mdc_val = -2.2
    mdcw_val = .1
    val, _edc = utils.find(D.k[0], edc_val)
    val, _mdc = utils.find(D.en, mdc_val)
    val, _mdcw = utils.find(D.en, mdc_val - mdcw_val)
    edc1 = int1[_edc, :]
    edc2 = int2[_edc, :]
    mdc = np.sum(int1[:, _mdcw:_mdc], axis=1)
    mdc = mdc / np.max(mdc)
    
    plt.figure(10007, figsize=(4, 4), clear=True)
    ###Fit MDC###
    delta = 1e-5
    p_mdc_i = [-.3, .35, 
               .1, .1, 
               1, 1, 
               .695, 0.02, -.02]
    p_mdc_bounds = ([-.3, .2,
                     0, 0,
                     0, 0,
                     p_mdc_i[-3]-delta, p_mdc_i[-2]-delta, p_mdc_i[-1]-delta],
                    [-.2, .5,
                     .12, .12,
                     np.inf, np.inf,
                     p_mdc_i[-3]+delta, p_mdc_i[-2]+delta, p_mdc_i[-1]+delta])
    p_mdc, cov_mdc = curve_fit(
            utils_math.gauss2, D.k[0], mdc, p_mdc_i, bounds=p_mdc_bounds)
    b_mdc = utils_math.poly2(D.k[0], 0, p_mdc[-3], p_mdc[-2], p_mdc[-1])
    f_mdc = utils_math.gauss2(D.k[0], *p_mdc)
    plt.plot(D.k[0], mdc, 'bo')
    plt.plot(D.k[0], f_mdc)
    plt.plot(D.k[0], b_mdc, 'k--')
    ###Plot Panels###
    def CROfig7a():
        ax = plt.subplot(1, 3, 1) 
        ax.set_position([.1, .3, .2 , .6])
        plt.tick_params(direction='in', length=1.5, width=.5, colors='k')  
        plt.contourf(D.k[0], D.en, np.transpose(int1), 300, cmap=colmap,
                     vmin = 0, vmax = 1.4e4)
        plt.plot([-1, 1.66], [0, 0], 'k:')
        plt.plot([-1, 1.66], [mdc_val - mdcw_val / 2, mdc_val - mdcw_val / 2],
                 linestyle='-.', color=(0, 238 / 256, 118 / 256), linewidth=.5)
        plt.plot([edc_val, edc_val], [-2.5, .5], linestyle='-.', 
                 color=(0, 238 / 256, 118 / 256), linewidth=.5)
        plt.xlim(xmax = 1.66, xmin = -1)
        plt.ylim(ymax = 0.5, ymin = -2.5)
        plt.ylabel('$\omega$ (eV)', fontdict = font)
        plt.xticks([-1, 0, 1], ('S', '$\Gamma$', 'S'))
        plt.yticks(np.arange(-2.5, .5, .5))
        plt.text(-.9, 0.3, r'(a)', fontsize=15)
        plt.text(.22, .1, r'$\mathcal{C}$', fontsize=15)
        plt.plot(D.k[0], (mdc - b_mdc) * 1.5, 'o', markersize=1, color='C9')
        plt.fill(D.k[0], (f_mdc - b_mdc) * 1.5, alpha=.2, color='C9')
    
    def CROfig7b():
        ax = plt.subplot(1, 3, 2) 
        ax.set_position([.32, .3, .2 , .6])
        plt.tick_params(direction='in', length=1.5, width=.5, colors='k')  
        plt.contourf(D.k[0], D.en+.07, np.transpose(int2), 300, cmap=colmap,
                     vmin = 0, vmax = 1.4e4)
        plt.plot([-1, 1.66], [0, 0], 'k:')
        plt.plot([edc_val, edc_val], [-2.5, .5], linestyle='-.', 
                 color=(0, 238 / 256, 118 / 256), linewidth=.5)
        plt.xlim(xmax = 1.66, xmin = -1)
        plt.ylim(ymax = 0.5, ymin = -2.5)
        plt.xticks([-1, 0, 1], ('S', '$\Gamma$', 'S'))
        plt.yticks(np.arange(-2.5, .5, .5), ())
        plt.text(-.9, 0.3, r'(b)', fontsize=15)
        
        pos = ax.get_position()
        cax = plt.axes([pos.x0+pos.width+0.01 ,
                            pos.y0, 0.01, pos.height])
        cbar = plt.colorbar(cax = cax, ticks = None)
        cbar.set_ticks([])
        cbar.set_clim(np.min(D.int), np.max(D.int))
        
    def CROfig7c():
        xx = np.linspace(1, -5, 200)
        ax = plt.subplot(1, 3, 3) 
        ax.set_position([.57, .3, .2 , .6])
        plt.tick_params(direction='in', length=1.5, width=.5, colors='k')  
        plt.plot(edc1, D.en, 'o', markersize=3, color=(0, 0, .8))
        plt.plot(edc2, D.en, 'd', markersize=3, color='C0')
        plt.fill_between([0, 1.5e4], 0, -.2, color='C3', alpha=0.2)
        plt.fill(7.4e3 * exponnorm.pdf(-xx, K=2, loc=.63, scale = .2), xx, 
                 alpha = .2, fc=(0, 0, .8))
        plt.fill(1.3e4 * exponnorm.pdf(-xx, K=2, loc=1.34, scale = .28), xx, 
                 alpha = .2, fc='C0')
        plt.plot([0, 1.5e4], [0, 0], 'k:')
        plt.plot([0, 1.5e4], [-.2, -.2], 'k:', linewidth=.2)
        plt.text(1e3, -0.15, r'$\Delta$', fontsize=12)
        plt.text(7e2, 0.3, r'(c)', fontsize=15)
        plt.text(6e3, -.9, r'$\mathcal{A}$', fontsize=15)
        plt.text(6e3, -1.75, r'$\mathcal{B}$', fontsize=15)
        plt.xlim(xmax = 1.2e4, xmin = 0)
        plt.ylim(ymax = 0.5, ymin = -2.5)
        plt.xticks([])
        plt.yticks(np.arange(-2.5, .5, .5), ())
        plt.legend(('63$\,$eV', '78$\,$eV'), frameon=False)
        plt.xlabel('Intensity (a.u)', fontdict = font)
        
    plt.figure(1007, figsize=(8, 6), clear=True)
    CROfig7a()
    CROfig7b()
    CROfig7c()
    if print_fig == True:
        plt.savefig(
                    '/Users/denyssutter/Documents/PhD/PhD_Denys/Figs/CROfig7.png', 
                    dpi = 300,bbox_inches="tight")

def CROfig8(colmap = cm.ocean_r, print_fig = False):
    """
    Polarization dependence Ca2RuO4
    """
    file1 = 47991
    file2 = 47992
    mat = 'Ca2RuO4'
    year = 2016
    sample = 'T10'
    D1 = ARPES.DLS(file1, mat, year, sample)
    D2 = ARPES.DLS(file2, mat, year, sample)
    D1.norm(48000)
    D2.norm(48000)
    D1.restrict(bot=.6, top=1, left=0, right=1)
    D2.restrict(bot=.6, top=1, left=0, right=1)
    D1.flatten(norm=True)
    D2.flatten(norm=True)
    D1.ang2k(D1.ang, Ekin=65-4.5, lat_unit=True, a=3.89, b=3.89, c=11, 
                        V0=0, thdg=5, tidg=12.5, phidg=0)
    D2.ang2k(D2.ang, Ekin=65-4.5, lat_unit=True, a=3.89, b=3.89, c=11, 
                        V0=0, thdg=5, tidg=12.5, phidg=0)
    edc_val = .35
    val, _edc = utils.find(np.flipud(D1.k[0]), edc_val)
    edc1 = D1.int_norm[_edc, :]
    edc2 = D2.int_norm[_edc, :]
    ###Plot Panels###
    def CROfig8a():
        ax = plt.subplot(1, 3, 1) 
        ax.set_position([.1, .3, .2 , .6])
        plt.tick_params(direction='in', length=1.5, width=.5, colors='k')  
        plt.contourf(D1.kxs, D1.en_norm+.1, np.flipud(D1.int_norm), 300, 
                     cmap=colmap, vmin = 0, vmax = .007)
        plt.plot([-1, 1.66], [0, 0], 'k:')
        plt.plot([edc_val, edc_val], [-2.5, .5], linestyle='-.', 
                 color=(0, 238 / 256, 118 / 256), linewidth=.5)
        plt.xlim(xmax = 1, xmin = 0)
        plt.ylim(ymax = 0.5, ymin = -2.5)
        plt.ylabel('$\omega$ (eV)', fontdict = font)
        plt.xticks([0, 1], ('S', '$\Gamma$'))
        plt.yticks(np.arange(-2.5, .5, .5))
        plt.text(.05, 0.3, r'(a)', fontsize=15)
        plt.arrow(-1, -1, 0, -.3, head_width=0.2, head_length=0.2, fc='g', ec='k')
    
    def CROfig8b():
        ax = plt.subplot(1, 3, 2) 
        ax.set_position([.32, .3, .2 , .6])
        plt.tick_params(direction='in', length=1.5, width=.5, colors='k')  
        plt.contourf(D2.kxs, D2.en_norm+.1, np.flipud(D2.int_norm), 300, 
                     cmap=colmap, vmin = 0, vmax = .007)
        plt.plot([-1, 1.66], [0, 0], 'k:')
        plt.plot([edc_val, edc_val], [-2.5, .5], linestyle='-.', 
                 color=(0, 238 / 256, 118 / 256), linewidth=.5)
        plt.xlim(xmax = 1, xmin = 0)
        plt.ylim(ymax = 0.5, ymin = -2.5)
        plt.xticks([0, 1], ('S', '$\Gamma$'))
        plt.yticks(np.arange(-2.5, .5, .5), ())
        plt.text(.05, 0.3, r'(b)', fontsize=15)
        
        pos = ax.get_position()
        cax = plt.axes([pos.x0+pos.width+0.01 ,
                            pos.y0, 0.01, pos.height])
        cbar = plt.colorbar(cax = cax, ticks = None)
        cbar.set_ticks([])
        cbar.set_clim(np.min(D2.int_norm), np.max(D2.int_norm))
        
    def CROfig8c():
        xx = np.linspace(1, -5, 200)
        ax = plt.subplot(1, 3, 3) 
        ax.set_position([.57, .3, .2 , .6])
        plt.tick_params(direction='in', length=1.5, width=.5, colors='k')  
        plt.plot(edc1, D1.en_norm[_edc, :]+.1, 'o', markersize=3, color=(0, 0, .8))
        plt.plot(edc2 * .8, D2.en_norm[_edc, :]+.1, 'd', markersize=3, color='C0')
        plt.fill_between([0, 1e-2], 0, -.2, color='C3', alpha=0.2)
        plt.fill(5.5e-3 * exponnorm.pdf(-xx, K=2, loc=.6, scale = .2), xx, 
                 alpha = .2, fc=(0, 0, .8))
        plt.fill(5.5e-3 * exponnorm.pdf(-xx, K=2, loc=1.45, scale = .25), xx, 
                 alpha = .2, fc='C0')
        plt.plot([0, 1e-2], [0, 0], 'k:')
        plt.plot([0, 1e-2], [-.2, -.2], 'k:', linewidth=.2)
        plt.text(7e-4, -0.15, r'$\Delta$', fontsize=12)
        plt.text(5e-4, 0.3, r'(c)', fontsize=15)
        plt.text(3.3e-3, -.9, r'$\mathcal{A}$', fontsize=15)
        plt.text(3.3e-3, -1.75, r'$\mathcal{B}$', fontsize=15)
        plt.xlim(xmax = .007, xmin = 0)
        plt.ylim(ymax = 0.5, ymin = -2.5)
        plt.xticks([])
        plt.yticks(np.arange(-2.5, .5, .5), ())
        plt.legend(('$\sigma$-pol.', '$\pi$-pol.'), frameon=False)
        plt.xlabel('Intensity (a.u)', fontdict = font)
        
    plt.figure(1008, figsize=(8, 6), clear=True)
    CROfig8a()
    CROfig8b()
    CROfig8c()
    if print_fig == True:
        plt.savefig(
                    '/Users/denyssutter/Documents/PhD/PhD_Denys/Figs/CROfig8.png', 
                    dpi = 300,bbox_inches="tight")
        
def CROfig9(colmap = cm.bone_r, print_fig = False):
    """
    DMFT plot dxz/yz, dxy Ca2RuO4
    """
    os.chdir('/Users/denyssutter/Documents/PhD/data')
    xz_data = np.loadtxt('DMFT_CRO_xz.dat')
    yz_data = np.loadtxt('DMFT_CRO_yz.dat')
    xy_data = np.loadtxt('DMFT_CRO_xy.dat')
    os.chdir('/Users/denyssutter/Documents/library/Python/ARPES')
    m, n = 8000, 351 #dimensions energy, full k-path
    bot, top = 2500, 5000 #restrict energy window
    DMFT_data = np.array([xz_data, yz_data, xy_data]) #combine data
    DMFT_spec = np.reshape(DMFT_data[:, :, 2], (3, n, m)) #reshape into n,m
    DMFT_spec = DMFT_spec[:, :, bot:top] #restrict data to bot, top
    DMFT_en   = np.linspace(-8, 8, m) #define energy data
    DMFT_en   = DMFT_en[bot:top] #restrict energy data
    #[0, 56, 110, 187, 241, 266, 325, 350]  = [G,X,S,G,Y,T,G,Z]
    DMFT_spec = np.transpose(DMFT_spec, (0,2,1)) #transpose
    DMFT_k = np.arange(0, 351, 1)
    plt.figure(1009, figsize=(8, 8), clear=True)
    for i in range(2):
        ax = plt.subplot(1, 2, i + 1)
        ax.set_position([.1 + (i * .38), .3, .35 , .35])
        plt.tick_params(direction='in', length=1.5, width=.5, colors='k')  
        plt.contourf(DMFT_k, DMFT_en, DMFT_spec[i + 1, :, :], 300, cmap=colmap,
                     vmin = 0, vmax = .3)
        plt.plot([0, 350], [0, 0], 'k:')
        plt.xlim(xmax=350, xmin=0)
        plt.ylim(ymax=1.5, ymin=-3)
        plt.xticks([0, 56, 110, 187, 241, 266, 325, 350], 
                   ('$\Gamma$', 'X', 'S', '$\Gamma$', 'Y', 'T', '$\Gamma$', 'Z'));
        if i == 0:
            plt.text(10, -2.8, r'(a) $d_{\gamma z}$', fontsize=12)
            plt.text(198, -.65, r'$U+J_\mathrm{H}$', fontsize=12)
            plt.arrow(188, 0, 0, .7, head_width=8, head_length=0.2, fc='g', ec='g')
            plt.arrow(188, 0, 0, -1.7, head_width=8, head_length=0.2, fc='g', ec='g')
            plt.yticks(np.arange(-3, 2, 1.))
            plt.ylabel('$\omega$ (eV)', fontdict = font)
        elif i == 1:
            plt.text(10, -2.8, r'(b) $d_{xy}$', fontsize=12)
            plt.text(263, -1, r'$3J_\mathrm{H}$', fontsize=12)
            plt.arrow(253, -.8, 0, .22, head_width=8, head_length=0.2, fc='g', ec='g')
            plt.arrow(253, -.8, 0, -.5, head_width=8, head_length=0.2, fc='g', ec='g')
            plt.yticks(np.arange(-3, 2, 1.), [])
            pos = ax.get_position()
            cax = plt.axes([pos.x0+pos.width+0.01 ,
                            pos.y0, 0.01, pos.height])
            cbar = plt.colorbar(cax = cax, ticks = None)
            cbar.set_ticks([])
            cbar.set_clim(np.min(DMFT_spec), 0.4 * np.max(DMFT_spec))
    if print_fig == True:
        plt.savefig(
                    '/Users/denyssutter/Documents/PhD/PhD_Denys/Figs/CROfig9.png', 
                    dpi = 300,bbox_inches="tight")
        
def CROfig10(colmap = cm.bone_r, print_fig = False):
    """
    DFT plot of Ca2RuO4: spaghetti and spectral representation plot
    """
    ###Load DFT spaghetti Plot###
    os.chdir('/Users/denyssutter/Documents/PhD/data')
    DFT_data = pd.read_table('DFT_CRO.dat', sep='\t')
    DFT_data = DFT_data.replace({'{': '', '}': ''}, regex=True)
    DFT_data = DFT_data.values
    os.chdir('/Users/denyssutter/Documents/library/Python/ARPES')
    ###Build k-axis segments###
    G = (0, 0, 0); X = (np.pi, 0, 0); Y = (0, np.pi, 0)
    Z = (0, 0, np.pi); T = (0, np.pi, np.pi); S = (np.pi, np.pi, 0)    
    ###Data along path in k-space###
    k_pts = np.array([G, X, S, G, Y, T, G, Z])
    k_seg = [0]
    for k in range(len(k_pts)-1):
        diff = abs(np.subtract(k_pts[k], k_pts[k + 1]))
        k_seg.append(k_seg[k] + la.norm(diff)) #extending list cummulative
    ###Spaceholders DFT spaghetti plot###
    (M, N) = DFT_data.shape
    data = np.zeros((M, N, 3))
    en = np.zeros((M, N)) 
    xz = np.zeros((M, N))
    k = np.linspace(0, 350, M)
    ###Load Data spectral representation###
    os.chdir('/Users/denyssutter/Documents/PhD/data')
    DFT_spec = pd.read_csv('DFT_CRO_all.dat').values
    os.chdir('/Users/denyssutter/Documents/library/Python/ARPES')
    (m, n) = DFT_spec.shape
    DFT_en = np.linspace(-3, 1.5, m)
    DFT_k = np.linspace(0, 350, n)
    
    def CROfig10a():
        ax = plt.subplot(121)
        ax.set_position([.1, .3, .35 , .35])
        plt.tick_params(direction='in', length=1.5, width=.5, colors='k') 
        plt.plot(0, 3, 'bo')
        plt.plot(50, 3, 'ro')
        for m in range(M):
            for n in range(N):
                data[m][n][:] = np.asfarray(DFT_data[m][n].split(','))
                en[m][n] = data[m][n][1]
                xz[m][n] = data[m][n][2]
                plt.plot(k[m], en[m, n], 'o', markersize=3, 
                         color=(xz[m, n], 0, (1-xz[m, n])))
        plt.plot([0, 350], [0, 0], 'k:')
        plt.text(10, 1.15, r'(a)', fontsize=12)
        plt.xlim(xmax=350, xmin=0)
        plt.ylim(ymax=1.5, ymin=-3)
        plt.xticks(k_seg / k_seg[-1] * 350, 
                   ('$\Gamma$', 'X', 'S', '$\Gamma$', 'Y', 'T', '$\Gamma$', 'Z'));
        plt.yticks(np.arange(-3, 2, 1.))
        plt.ylabel('$\omega$ (eV)', fontdict = font)
        plt.legend(('$d_{xy}$', '$d_{\gamma z}$'), frameon=False)
    
    def CROfig10b():
        ax = plt.subplot(122)
        ax.set_position([.1 + .38, .3, .35 , .35])
        plt.tick_params(direction='in', length=1.5, width=.5, colors='k') 
        plt.contourf(DFT_k, DFT_en, DFT_spec, 300, cmap=colmap,
                     vmin = 0, vmax = 25)
        plt.plot([0, 350], [0, 0], 'k:')
        plt.text(10, 1.15, r'(b)', fontsize=12)
        plt.xlim(xmax=350, xmin=0)
        plt.ylim(ymax=1.5, ymin=-3)
        plt.xticks(k_seg / k_seg[-1] * 350, 
                   ('$\Gamma$', 'X', 'S', '$\Gamma$', 'Y', 'T', '$\Gamma$', 'Z'));
        plt.yticks(np.arange(-3, 2, 1.), [])
        pos = ax.get_position()
        cax = plt.axes([pos.x0+pos.width+0.01 ,
                        pos.y0, 0.01, pos.height])
        cbar = plt.colorbar(cax = cax, ticks = None)
        cbar.set_ticks([])
        cbar.set_clim(np.min(DFT_spec), np.max(DFT_spec))

    plt.figure(1010, figsize=(8,8), clear=True)
    CROfig10a()
    CROfig10b()
    if print_fig == True:
        plt.savefig(
                    '/Users/denyssutter/Documents/PhD/PhD_Denys/Figs/CROfig10.png', 
                    dpi = 300,bbox_inches="tight")
        
def CROfig11(print_fig = False):
    """
    Multiplet analysis Ca2RuO4
    """
    plt.figure(1011, figsize=(8, 8), clear=True)
    ax = plt.subplot(111)
    ax.set_position([.1, .3, .52 , .15])
    plt.tick_params(direction='in', length=0, width=.5, colors='k') 
    off = [0, 3, 5, 8, 9.25, 11.25, 12.5]
    n = 0
    for i in off:
        n += 1
        plt.plot([0 + i, 1 + i], [0, 0], 'k-')
        plt.plot([0 + i, 1 + i], [-.5, -.5], 'k-')
        plt.plot([0 + i, 1 + i], [-2.5, -2.5], 'k-')
        
        if any(x==n for x in [1, 2, 3, 4, 6, 7]):
            ax.arrow(.33 + i, -.5, 0, 1, head_width=0.2, head_length=0.4,
                     linewidth=1.5, fc='r', ec='r')
        if any(x==n for x in [6]):
            ax.arrow(.1 + i, .8, 0, -1, head_width=0.2, head_length=0.4,
                     linewidth=1.5, fc='r', ec='r')
        if any(x==n for x in [1, 2, 3, 5, 6, 7]):
            ax.arrow(.66 + i, -1, 0, 1, head_width=0.2, head_length=0.4,
                     linewidth=1.5, fc='r', ec='r')
        if any(x==n for x in [7]):
            ax.arrow(.9 + i, .3, 0, -1, head_width=0.2, head_length=0.4,
                     linewidth=1.5, fc='r', ec='r')
        if any(x==n for x in [1, 2, 4, 5, 6, 7]):
            ax.arrow(.33 + i, -3, 0, 1, head_width=0.2, head_length=0.4,
                     linewidth=1.5, fc='r', ec='r')
        if any(x==n for x in [1, 3, 4, 5, 6, 7]):  
            ax.arrow(.66 + i, -1.7, 0, -1, head_width=0.2, head_length=0.4,
                     linewidth=1.5, fc='r', ec='r')
            
    plt.fill_between([2, 7], 4, -4, color='C0', alpha=0.2)
    plt.fill_between([7, 14.3], 4, -4, color=(0, 0, .8), alpha=0.2)
    plt.text(-1.7, -2.7, r'$d_{xy}$', fontsize=12)
    plt.text(-1.7, -.3, r'$d_{\gamma z}$', fontsize=12)
    plt.text(4., 1.5, r'$3J_\mathrm{H}$', fontsize=12)
    plt.text(9.9, 1.5, r'$U+J_\mathrm{H}$', fontsize=12)
    plt.text(-1.7, 3, r'$| d_4; S=1,\alpha = xy\rangle$', fontsize=8)
    plt.text(2.6, 3, r'$| d_3; \frac{3}{2},\gamma z\rangle$', fontsize=8)
    plt.text(4.6, 3, r'$| d_3; \frac{1}{2},\gamma z\rangle$', fontsize=8)
    plt.text(8.4, 3, r'$| d_3; \frac{1}{2}, xy\rangle$', fontsize=8)
    plt.text(11.5, 3, r'$| d_5; \frac{1}{2}, xy\rangle$', fontsize=8)
    plt.xticks([])
    plt.yticks([])
    plt.xlim(xmax=14.3, xmin=-2)
    plt.ylim(ymax=4, ymin=-4)
    if print_fig == True:
        plt.savefig(
                    '/Users/denyssutter/Documents/PhD/PhD_Denys/Figs/CROfig11.png', 
                    dpi = 600,bbox_inches="tight")
        
def CROfig12(colmap = cm.ocean_r, print_fig = False):
    """
    Constant energy maps oxygen band 
    """
    CRO_FS_plot(colmap, e=-5.2, v_min=.25, fignr=1012)
    if print_fig == True:
        plt.savefig(
                    '/Users/denyssutter/Documents/PhD/PhD_Denys/Figs/CROfig12.png', 
                    dpi = 300,bbox_inches="tight")

def CROfig13(colmap = cm.ocean_r, print_fig = False):
    """
    Constant energy maps alpha band
    """
    CRO_FS_plot(colmap, e=-.5, v_min=.05, fignr=1013)
    if print_fig == True:
        plt.savefig(
                    '/Users/denyssutter/Documents/PhD/PhD_Denys/Figs/CROfig13.png', 
                    dpi = 300,bbox_inches="tight")
        
def CROfig14(colmap = cm.ocean_r, print_fig = False):
    """
    Constant energy maps gamma band 
    """
    CRO_FS_plot(colmap, e=-2.4, v_min=.4, fignr=1014)
    if print_fig == True:
        plt.savefig(
                    '/Users/denyssutter/Documents/PhD/PhD_Denys/Figs/CROfig14.png', 
                    dpi = 300,bbox_inches="tight")
"""
Figures Dissertation Ca1.8Sr0.2RuO4 (CSRO)
"""     

def CSROfig1(colmap = cm.ocean_r, print_fig = False):
    """
    Experimental data: Figure 1 CSCRO20 paper
    """
    ###Load Data###
    os.chdir('/Users/denyssutter/Documents/library/Python/ARPES')
    file = 62087
    gold = 62081
    mat = 'CSRO20'
    year = 2017
    sample = 'S6'
    D = ARPES.DLS(file, mat, year, sample)
    D.norm(gold)
    D.restrict(bot=0, top=1, left=.12, right=.9)
    D.FS(e = 0.02, ew = .03, norm = True)
    D.ang2kFS(D.ang, Ekin=22-4.5, lat_unit=True, a=5.33, b=5.55, c=11, 
              V0=0, thdg=8.7, tidg=4, phidg=88)
    FS = np.flipud(D.map)
    file = 62090
    gold = 62091
    A1 = ARPES.DLS(file, mat, year, sample)
    A1.norm(gold)
    A1.ang2k(A1.ang, Ekin=22-4.5, lat_unit=True, a=5.33, b=5.55, c=11, 
              V0=0, thdg=9.3, tidg=0, phidg=90)
    file = 62097
    gold = 62091
    A2 = ARPES.DLS(file, mat, year, sample)
    A2.norm(gold)
    A2.ang2k(A1.ang, Ekin=22-4.5, lat_unit=True, a=5.33, b=5.55, c=11, 
              V0=0, thdg=6.3, tidg=-16, phidg=90)
    c = (0, 238 / 256, 118 / 256)
    ###MDC###
    mdc_val = -.004
    mdcw_val = .002
    mdc = np.zeros(A1.ang.shape)
    for i in range(len(A1.ang)):
        val, _mdc = utils.find(A1.en_norm[i, :], mdc_val)
        val, _mdcw = utils.find(A1.en_norm[i, :], mdc_val - mdcw_val)
        mdc[i] = np.sum(A1.int_norm[i, _mdcw:_mdc])
    mdc = mdc / np.max(mdc)
    plt.figure(20001, figsize=(4, 4), clear=True)
    ###Fit MDC###
    delta = 1e-5
    p_mdc_i = np.array(
                [-1.4, -1.3, -1.1, -.9, -.7, -.6, -.3, .3,
                 .05, .05, .05, .05, .05, .05, .1, .1, 
                 .3, .3, .4, .4, .5, .5, .1, .1,
                 .33, 0.02, .02])
    bounds_bot = np.concatenate((p_mdc_i[0:-3] - np.inf, p_mdc_i[-3:27] - delta))
    bounds_top = np.concatenate((p_mdc_i[0:-3] + np.inf, p_mdc_i[-3:27] + delta))
    p_mdc_bounds = (bounds_bot, bounds_top)
    p_mdc, cov_mdc = curve_fit(
            utils_math.lor8, A1.k[1], mdc, p_mdc_i, bounds=p_mdc_bounds)
    b_mdc = utils_math.poly2(A1.k[1], 0, p_mdc[-3], p_mdc[-2], p_mdc[-1])
    f_mdc = utils_math.lor8(A1.k[1], *p_mdc) - b_mdc
    f_mdc[0] = 0
    f_mdc[-1] = 0
    plt.plot(A1.k[1], mdc, 'bo')
    plt.plot(A1.k[1], f_mdc)
    plt.plot(A1.k[1], b_mdc, 'k--')
        
    def CSROfig1a():
        ax = plt.subplot(1, 3, 1) 
        ax.set_position([.08, .3, .28, .35])
        plt.tick_params(direction='in', length=1.5, width=.5, colors='k')
        plt.contourf(A1.en_norm, A1.kys, A1.int_norm, 100, cmap=colmap,
                     vmin=.1 * np.max(A1.int_norm), vmax=.8 * np.max(A1.int_norm))
        plt.plot([0, 0], [np.min(A1.kys), np.max(A1.kys)], 'k:')
        plt.plot([-.005, -.005], [np.min(A1.kys), np.max(A1.kys)], linestyle='-.',
                 color=c, linewidth=.5)
        plt.xlim(xmax=.03, xmin=-.06)
        plt.ylim(ymax=np.max(D.ky), ymin=np.min(D.ky))   
        plt.xticks(np.arange(-.06, .03, .02), ('-60', '-40', '-20', '0', '20'))
        plt.yticks([-1.5, -1, -.5, 0, .5])
        plt.xlabel('$\omega\,(\mathrm{meV})$', fontdict = font)
        plt.ylabel('$k_x \,(\pi/a)$', fontdict = font)
        plt.plot((mdc - b_mdc) / 30 + .001, A1.k[1], 'o', markersize=1.5, color='C9')
        plt.fill(f_mdc / 30 + .001, A1.k[1], alpha=.2, color='C9')
        plt.text(-.058, .56, r'(a)', fontsize=12)
        plt.text(.024, -.03, r'$\Gamma$', fontsize=12, color='r')
        plt.text(.024, -1.03, r'Y', fontsize=12, color='r')
        cols = ['k', 'b', 'b', 'b', 'b', 'm', 'C1', 'C1']
        lbls = [r'$\bar{\beta}$', r'$\bar{\gamma}$', r'$\bar{\gamma}$', 
                r'$\bar{\gamma}$', r'$\bar{\gamma}$',
                r'$\bar{\beta}$', r'$\bar{\alpha}$', r'$\bar{\alpha}$']
        corr = np.array([.004, .002, .002, 0, -.001, 0, .003, .003])
        p_mdc[6 + 16] *= 1.5
        for i in range(8):
            plt.plot((utils_math.lor(A1.k[1], p_mdc[i], p_mdc[i + 8], p_mdc[i + 16], 
                     p_mdc[-3], p_mdc[-2], p_mdc[-1]) - b_mdc) / 30 + .001, 
                     A1.k[1], linewidth=.5, color=cols[i])
#            if i == 6:
#                p_mdc[7 + 16] /= 2
            plt.text(p_mdc[i + 16] / 20 + corr[i], p_mdc[i]-.03, lbls[i], 
                     fontdict=font, fontsize=10, color=cols[i])
        plt.plot(f_mdc / 30 + .001, A1.k[1], color=c, linewidth=.5)
    
    def CSROfig1c():
        ax = plt.subplot(1, 3, 3) 
        ax.set_position([.66, .3, .217, .35])
        plt.tick_params(direction='in', length=1.5, width=.5, colors='k')
        plt.contourf(-np.transpose(np.fliplr(A2.en_norm)), np.transpose(A2.kys), 
                     np.transpose(np.fliplr(A2.int_norm)), 100, cmap=colmap,
                     vmin=.1 * np.max(A2.int_norm), vmax=.8 * np.max(A2.int_norm))
        plt.plot([0, 0], [np.min(A2.kys), np.max(A2.kys)], 'k:')
        plt.xlim(xmin=-.01, xmax=.06)
        plt.ylim(ymax=np.max(D.ky), ymin=np.min(D.ky))  
        plt.xticks(np.arange(0, .06, .02), ('0', '-20', '-40', '-60'))
        plt.yticks([-1.5, -1, -.5, 0, .5], [])
        plt.xlabel('$\omega\,(\mathrm{meV})$', fontdict = font)
        plt.text(-.0085, .56, r'(c)', fontsize=12)
        plt.text(-.008, -.03, r'X', fontsize=12, color='r')
        plt.text(-.008, -1.03, r'S', fontsize=12, color='r')
        pos = ax.get_position()
        cax = plt.axes([pos.x0+pos.width+0.01 ,
                            pos.y0, 0.01, pos.height])
        cbar = plt.colorbar(cax = cax, ticks = None)
        cbar.set_ticks([])
        cbar.set_clim(np.min(A2.int_norm), np.max(A2.int_norm))
    
    def CSROfig1b():
        for i in range(FS.shape[1]):
            FS[:, i] = np.divide(FS[:, i], np.sum(FS[:, i]))  #Flatten
        ax = plt.subplot(1, 3, 2) 
        ax.set_position([.37, .3, .28, .35])
        plt.tick_params(direction='in', length=1.5, width=.5, colors='k')  
        plt.contourf(D.kx, D.ky, FS, 300, vmax=.9 * np.max(FS), vmin=.3 * np.max(FS),
                   cmap=colmap)
        plt.xlabel('$k_y \,(\pi/a)$', fontdict = font)
        plt.text(-.65, .56, r'(b)', fontsize=12, color='w')
        plt.text(-.05, -.03, r'$\Gamma$', fontsize=12, color='r')
        plt.text(-.05, -1.03, r'Y', fontsize=12, color='r')
        plt.text(.95, -.03, r'X', fontsize=12, color='r')
        plt.text(.95, -1.03, r'S', fontsize=12, color='r')
        lblmap = [r'$\bar{\alpha}$', r'$\bar{\beta}$', r'$\bar{\gamma}$', 
                  r'$\bar{\delta}$', r'$\bar{\epsilon}$']
        lblx = np.array([.25, .43, .66, .68, .8])
        lbly = np.array([-.25, -.43, -.23, -.68, -.8])
        lblc = ['C1', 'm', 'b', 'k', 'r']
        for k in range(5):
            plt.text(lblx[k], lbly[k], lblmap[k], fontsize=12, color=lblc[k])
        plt.plot(A1.k[0], A1.k[1], linestyle='-.', color=c, linewidth=.5)
        plt.plot(A2.k[0], A2.k[1], linestyle='-.', color=c, linewidth=.5)
        ###Tight Binding Model###
        tb = utils_math.TB(a = np.pi, kbnd = 2, kpoints = 200)#Initialize 
        param = utils_math.paramCSRO20()  #Load parameters
        tb.CSRO(param)  #Calculate bandstructure
        bndstr = tb.bndstr  #Load bandstructure
        coord = tb.coord  #Load coordinates
        X = coord['X']; Y = coord['Y']   
        Axy = bndstr['Axy']; Bxz = bndstr['Bxz']; Byz = bndstr['Byz']
        en = (Axy, Bxz, Byz)  #Loop over sheets
        n = 0
        for i in en:
            n += 1
            C = plt.contour(X, Y, i, colors = 'black', linestyles = ':', 
                            alpha=0, levels = 0)
            p = C.collections[0].get_paths()
            p = np.asarray(p)
            axy = np.arange(0, 4, 1) #indices of same paths
            bxz = np.arange(16, 24, 1)
            byz = np.array([16, 17, 20, 21])
            if n == 1:
                ind = axy; col = 'r'
            elif n == 2:
                ind = bxz; col = 'b'
            elif n == 3:
                ind = byz; col = 'k'
                v = p[18].vertices
                plt.plot(v[:, 0], v[:, 1], linestyle = ':', color = 'm', 
                         linewidth=1)
                v = p[2].vertices
                plt.plot(v[:, 0], v[:, 1], linestyle = ':', color = 'm', 
                         linewidth=1)
                v = p[19].vertices
                plt.plot(v[:, 0], v[:, 1], linestyle = ':', color = 'C1', 
                         linewidth=1)
            for j in ind:
                v = p[j].vertices
                plt.plot(v[:, 0], v[:, 1], linestyle = ':', color = col, 
                         linewidth=1)
        plt.xticks([-.5, 0, .5, 1])
        plt.yticks([-1.5, -1, -.5, 0, .5], [])
        plt.xlim(xmax=np.max(D.kx), xmin=np.min(D.kx))
        plt.ylim(ymax=np.max(D.ky), ymin=np.min(D.ky))     
        
    plt.figure(2001, figsize=(8, 8), clear=True)
    CSROfig1a()
    CSROfig1b()
    CSROfig1c()
    if print_fig == True:
            plt.savefig(
                        '/Users/denyssutter/Documents/PhD/PhD_Denys/Figs/CSROfig1.png', 
                        dpi = 600,bbox_inches="tight")

def CSROfig2(colmap = cm.ocean_r, print_fig = False):
    """
    Experimental PSI data: Figure 2 CSCRO20 paper
    """
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
    ###Diagonal MDC###
    bnd = .72
    c = (0, 238 / 256, 118 / 256)
    mdc_d = np.zeros(D.ang.size)
    mdc = np.zeros(D.pol.size)
    for i in range(D.ang.size):
        val, _mdc_d = utils.find(D.ky[:, i], D.kx[0, i])
        mdc_d[i] = D.map[_mdc_d, i]
    val, _mdc = utils.find(D.kx[0, :], .02)
    val, _mdcw = utils.find(D.kx[0, :], -.02)
    mdc = np.sum(D.map[:, _mdcw:_mdc], axis=1)
    mdc_d = mdc_d / np.max(mdc_d)
    mdc = mdc / np.max(mdc)
    ###Fit MDC's###
    plt.figure(20002, figsize=(4, 4), clear=True)
    delta = 1e-5
    p_mdc_d_i = np.array(
                [-.6, -.4, -.2, .2, .4, .6,
                 .05, .05, .05, .05, .05, .05,
                 .3, .3, .4, .4, .5, .5, 
                 .59, -0.2, .04])
    bounds_bot = np.concatenate(
                        (p_mdc_d_i[0:-3] - np.inf, p_mdc_d_i[-3:21] - delta))
    bounds_top = np.concatenate(
                        (p_mdc_d_i[0:-3] + np.inf, p_mdc_d_i[-3:21] + delta))
    p_mdc_d_bounds = (bounds_bot, bounds_top)
    p_mdc_d, cov_mdc = curve_fit(
            utils_math.lor6, D.kx[0, :], mdc_d, p_mdc_d_i, bounds=p_mdc_d_bounds)
    b_mdc_d = utils_math.poly2(D.kx[0, :], 0, p_mdc_d[-3], p_mdc_d[-2], p_mdc_d[-1])
    f_mdc_d = utils_math.lor6(D.kx[0, :], *p_mdc_d) - b_mdc_d
    f_mdc_d[0] = 0
    f_mdc_d[-1] = 0
    plt.subplot(211)
    plt.plot(D.kx[0, :], mdc_d, 'bo')
    plt.plot(D.kx[0, :], f_mdc_d + b_mdc_d)
    plt.plot(D.kx[0, :], b_mdc_d, 'k--')
    delta = 5e-2
    p_mdc_i = np.array(
                [-.6,  -.2, .2, .6,
                 .05, .05, .05, .05,
                 .3, .3, .4, .4,
                 .6, -0.15, .1])
    bounds_bot = np.concatenate((p_mdc_i[0:-3] - np.inf, p_mdc_i[-3:15] - delta))
    bounds_top = np.concatenate((p_mdc_i[0:-3] + np.inf, p_mdc_i[-3:15] + delta))
    p_mdc_bounds = (bounds_bot, bounds_top)
    p_mdc, cov_mdc = curve_fit(
            utils_math.lor4, D.ky[:, 0], mdc, p_mdc_i, bounds=p_mdc_bounds)
    b_mdc = utils_math.poly2(D.ky[:, 0], 0, p_mdc[-3], p_mdc[-2], p_mdc[-1])
    f_mdc = utils_math.lor4(D.ky[:, 0], *p_mdc) - b_mdc
    f_mdc[0] = 0
    f_mdc[-1] = 0
    plt.subplot(212)
    plt.plot(D.ky[:, 0], mdc, 'bo')
    plt.plot(D.ky[:, 0], f_mdc + b_mdc)
    plt.plot(D.ky[:, 0], b_mdc, 'k--')
    def CSROfig2a():
        ax = plt.subplot(1, 4, 1) 
        ax.set_position([.08, .605, .4, .15])
        plt.plot(D.kx[0, :], mdc_d - b_mdc_d + .01, 'o', markersize=1.5, color='C9')
        plt.fill(D.kx[0, :], f_mdc_d + .01, alpha=.2, color='C9')
        corr = np.array([.04, .03, .07, .08, .07, .05])
        cols = ['k', 'm', 'C1', 'C1', 'm', 'k']
        lbls = [r'$\bar{\delta}$', r'$\bar{\beta}$', r'$\bar{\alpha}$', 
                r'$\bar{\alpha}$', r'$\bar{\beta}$', r'$\bar{\delta}$']
        for i in range(6):
            plt.plot(D.kx[0, :], (utils_math.lor(D.kx[0, :], p_mdc_d[i], 
                     p_mdc_d[i + 6], p_mdc_d[i + 12], p_mdc_d[-3], p_mdc_d[-2], 
                     p_mdc_d[-1]) - b_mdc_d) + .01, linewidth=.5, color=cols[i])
            plt.text(p_mdc_d[i] - .02, p_mdc_d[i + 12] + corr[i], lbls[i], 
                         fontdict=font, fontsize=10, color=cols[i])
        plt.tick_params(direction='in', length=1.5, width=.5, colors='k')
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position('top')
        plt.xticks(np.arange(-10, 10, .5))
        plt.yticks([])
        plt.xlim(xmax=bnd, xmin=-bnd)
        plt.ylim(ymax=.42, ymin=0)
        plt.xlabel(r'$k_x = k_y \, (\pi/a)$')
        plt.ylabel(r'Intensity (a.u.)')
        plt.text(-.7, .36, r'(a)', fontsize=12)
    
    def CSROfig2b():
        ax = plt.subplot(1, 4, 3) 
        ax.set_position([.08, .2, .4, .4])
        plt.tick_params(direction='in', length=1.5, width=.5, colors='k')
        plt.contourf(D.kx, D.ky, D.map_flat, 100, cmap=colmap,
                     vmin=.6 * np.max(D.map_flat), vmax=1.0 * np.max(D.map_flat))
        plt.plot([-bnd, bnd], [-bnd, bnd], linestyle='-.', color=c, linewidth=.5)
        plt.plot([0, 0], [-bnd, bnd], linestyle='-.', color=c, linewidth=.5)
        ax.arrow(.55, .55, 0, .13, head_width=0.03, head_length=0.03, fc=c, ec=c)
        plt.xticks(np.arange(-10, 10, .5))
        plt.yticks(np.arange(-10, 10, .5),[])
        
        
        plt.axis('equal')
        plt.xlabel(r'$k_x \, (\pi/a)$')
        plt.text(-.7, .63, r'(b)', fontsize=12, color='w')
        plt.xlim(xmax=bnd, xmin=-bnd)
        plt.ylim(ymax=bnd, ymin=-bnd)  
        pos = ax.get_position()
        cax = plt.axes([pos.x0 - .02 ,
                        pos.y0, 0.01, pos.height])
        cbar = plt.colorbar(cax = cax, ticks = None)
        cbar.set_ticks([])
        cbar.set_clim(np.min(D.map_flat), np.max(D.map_flat))
        ###Tight Binding Model###
        ax = plt.subplot(1, 4, 3) 
        ax.set_position([.08, .2, .4, .4])
        tb = utils_math.TB(a = np.pi, kbnd = 2, kpoints = 200)#Initialize 
        param = utils_math.paramCSRO20()  #Load parameters
        tb.CSRO(param)  #Calculate bandstructure
        bndstr = tb.bndstr  #Load bandstructure
        coord = tb.coord  #Load coordinates
        X = coord['X']; Y = coord['Y']   
        Byz = bndstr['Byz']
        C = plt.contour(X, Y, Byz, colors = 'black', linestyles = ':', 
                        alpha=0, levels = -0.00)
        p = C.collections[0].get_paths()
        p = np.asarray(p)
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
        
    def CSROfig2c():
        ax = plt.subplot(1, 4, 4) 
        ax.set_position([.485, .2, .15, .4])
        plt.plot(mdc - b_mdc, D.ky[:, 0], 'o', markersize=1.5, color='C9')
        plt.fill(f_mdc + .01, D.ky[:, 0], alpha=.2, color='C9')
        corr = np.array([.03, .06, .04, .04])
        cols = ['m', 'C1', 'C1', 'm']
        lbls = [r'$\bar{\beta}$', r'$\bar{\alpha}$', r'$\bar{\alpha}$', 
                r'$\bar{\beta}$']
        for i in range(4):
            plt.plot((utils_math.lor(D.ky[:, 0], p_mdc[i], p_mdc[i + 4], 
                                     p_mdc[i + 8], p_mdc[-3], p_mdc[-2],
                                     p_mdc[-1]) - b_mdc) + .01, 
                     D.ky[:, 0], linewidth=.5, color=cols[i])
            plt.text(p_mdc[i + 8] + corr[i], p_mdc[i] - .02, lbls[i], 
                         fontdict=font, fontsize=10, color=cols[i])
        plt.tick_params(direction='in', length=1.5, width=.5, colors='k')
        plt.yticks(np.arange(-10, 10, .5))
        ax.yaxis.tick_right()
        plt.xticks([])
        plt.ylim(ymax=bnd, ymin=-bnd)
        plt.xlim(xmax=.42, xmin=0)
        ax.yaxis.set_label_position('right')
        plt.ylabel(r'$k_y \, (\pi/b)$')
        plt.xlabel(r'Intensity (a.u.)')
        plt.text(.33, .63, r'(c)', fontsize=12)
        
    ###Plotting
    plt.figure(2002, figsize=(8, 8), clear=True)
    CSROfig2a()
    CSROfig2b()
    CSROfig2c()
    if print_fig == True:
            plt.savefig(
                        '/Users/denyssutter/Documents/PhD/PhD_Denys/Figs/CSROfig2.png', 
                        dpi = 600,bbox_inches="tight")
        