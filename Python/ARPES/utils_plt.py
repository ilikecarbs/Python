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
    plt.figure(2006, figsize=(10, 10), clear=True)
    plt.tick_params(direction='in', length=1.5, width=.5, colors='k')
    plt.contourf(k, en, dat, 100, cmap = cm.ocean_r)
    if norm == True:
        plt.plot([np.min(k), np.max(k)], [0, 0], 'k:')
    plt.xlabel('$k_x$')   
    plt.ylabel('\omega')
    plt.show()

def plt_FS_poliut(self, norm, p, pw):
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
    plt.figure(2005, figsize=(10, 10), clear=True)
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
    plt.figure(2007, figsize=(10, 10), clear=True)
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
    plt.figure(2000, figsize=(10,10), clear=True)
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
    plt.figure(2003, figsize=(10, 10), clear=True)
    plt.contour(X, Y, en, levels = e0)
  
def plt_cont_TB_SRO(self, e0):
    bndstr = self.bndstr
    coord = self.coord   
    X = coord['X']; Y = coord['Y']   
    xz = bndstr['xz']; yz = bndstr['yz']; xy = bndstr['xy']
    en = (xz, yz, xy)
    plt.figure(2002, figsize=(10, 3), clear=True)
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
    plt.figure(2001, figsize=(6, 4), clear=True)
    n = 0
    for i in en:
        n = n + 1
        plt.subplot(2, 3, n)
        plt.contour(X, Y, i, levels = e0)
        plt.axis('equal')
        
def CRO_theory_plot(k_pts, data_en, data, colmap, v_max):
    c = len(data)
    scale = .02
    plt.figure(1001, figsize=(10, 10), clear = True)
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
            plt.ylabel('$\omega$ (meV)', fontdict = font)
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
    ax.set_position([pos.x0, pos.y0, k_prev * scale, pos.height])

def fig1(colmap = cm.bone_r, print_fig = False):
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
    CRO_theory_plot(k_pts, DFT_en, DFT, colmap, v_max = 1) #Plot data
    if print_fig == True:
        plt.savefig(
                '/Users/denyssutter/Documents/PhD/PhD_Denys/Figs/fig1.png', 
                dpi = 300,bbox_inches="tight")
    
def fig2(colmap = cm.bone_r, print_fig = False):
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
    
    CRO_theory_plot(k_pts, DMFT_en, DMFT, colmap, v_max = .5) #Plot data
    if print_fig == True:
        plt.savefig(
                '/Users/denyssutter/Documents/PhD/PhD_Denys/Figs/fig2.png', 
                dpi = 300,bbox_inches="tight")

def fig3(colmap = cm.bone_r, print_fig = False):
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
    
    CRO_theory_plot(k_pts, DFT_en, DFT, colmap, v_max = 1) #Plot data
    if print_fig == True:
        plt.savefig(
                '/Users/denyssutter/Documents/PhD/PhD_Denys/Figs/fig3.png', 
                dpi = 300,bbox_inches="tight")

def fig4(colmap = cm.bone_r, print_fig = False):
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
    
    CRO_theory_plot(k_pts, DFT_en, DFT, colmap, v_max = 1) #Plot data
    if print_fig == True:
        plt.savefig(
                '/Users/denyssutter/Documents/PhD/PhD_Denys/Figs/fig4.png', 
                dpi = 300,bbox_inches="tight")
    
def fig5(colmap = cm.ocean_r, print_fig = False):
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
    plt.rcParams['xtick.labelbottom'] = True
    plt.rcParams['xtick.labeltop'] = False
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
            plt.rcParams['ytick.labelright'] = False
            plt.rcParams['ytick.labelleft'] = True
            ax = plt.subplot(1, 4, n) 
            ax.set_position([.1, .3, k_seg_1[-1] * scale, .3])
            pos = ax.get_position()
            D.ang2k(D.ang, Ekin=65-4.5, lat_unit=True, a=3.89, b=3.89, c=11, 
                    V0=0, thdg=-4, tidg=0, phidg=0)
            plt.tick_params(direction='in', length=1.5, width=.5, colors='k')  
            plt.contourf(D.ks, D.en_norm+.1, D.int_norm, 300,
                       cmap=colmap, 
                       vmin=v_scale * 0.01 * np.max(D.int_norm), 
                       vmax=v_scale * 0.5 * np.max(D.int_norm))
            plt.xlim(xmax = 1, xmin = -1)
            plt.ylabel('$\omega$ (meV)', fontdict = font)
            plt.xticks([-1, 0, 1], ('S', '$\Gamma$', 'S'))
        elif n == 2:
            plt.rcParams['ytick.labelright'] = False
            plt.rcParams['ytick.labelleft'] = False
            ax = plt.subplot(1, 4, n)
            ax.set_position([pos.x0 + k_seg_1[-1] * scale, pos.y0, 
                             k_seg_2[-1] * scale, pos.height])
            D.ang2k(D.ang, Ekin=65-4.5, lat_unit=True, a=3.89, b=3.89, c=11, 
                    V0=0, thdg=-7.5, tidg=8.5, phidg=45)
            plt.tick_params(direction='in', length=1.5, width=.5, colors='k')  
            plt.contourf(D.ks, D.en_norm+.1, D.int_norm, 300,
                       cmap=colmap,
                       vmin=v_scale * 0.0 * np.max(D.int_norm), 
                       vmax=v_scale * 0.54 * np.max(D.int_norm))
            plt.xlim(xmax = 0, xmin = -1)
            plt.xticks([-1, -.5, 0], ('', 'X', 'S'))
        elif n == 3:
            plt.rcParams['ytick.labelright'] = False
            plt.rcParams['ytick.labelleft'] = False
            ax = plt.subplot(1, 4, n)
            ax.set_position([pos.x0 + k_seg_2[-1] * scale, pos.y0, 
                             k_seg_3[-1] * scale, pos.height])
            D.ang2k(D.ang, Ekin=65-4.5, lat_unit=True, a=3.89, b=3.89, c=11, 
                    V0=0, thdg=5, tidg=12.5, phidg=0)
            plt.tick_params(direction='in', length=1.5, width=.5, colors='k')  
            plt.contourf(D.ks, D.en_norm+.1, np.flipud(D.int_norm), 300,
                       cmap=colmap, 
                       vmin=v_scale * 0.01 * np.max(D.int_norm), 
                       vmax=v_scale * 0.7 * np.max(D.int_norm))
            plt.xlim(xmax = 1, xmin = 0)
            plt.xticks([0, 1], ('', '$\Gamma$'))
        elif n == 4:
            plt.rcParams['ytick.labelright'] = False
            plt.rcParams['ytick.labelleft'] = False
            ax = plt.subplot(1, 4, n)
            ax.set_position([pos.x0 + k_seg_3[-1] * scale, pos.y0, 
                             k_seg_4[-1] * scale, pos.height])
            D.ang2k(D.ang, Ekin=65-4.5, lat_unit=True, a=3.89, b=3.89, c=11, 
                    V0=0, thdg=-9.5, tidg=0, phidg=45)
            plt.tick_params(direction='in', length=1.5, width=.5, colors='k')  
            plt.contourf(D.ks, D.en_norm+.1, np.flipud(D.int_norm), 300,
                       cmap=colmap, 
                       vmin=v_scale * 0.01 * np.max(D.int_norm), 
                       vmax=v_scale * 0.53 * np.max(D.int_norm))
            plt.xlim(xmax = 1.5, xmin = 0)
            plt.xticks([0, 0.5, 1, 1.5], ('', 'X', '$\Gamma$', 'X'))
        
        pos = ax.get_position()
        plt.ylim(ymax = 0, ymin = -2.5)
        plt.show()
    cax = plt.axes([pos.x0 + k_seg_4[-1] * scale + 0.01,
                    pos.y0, 0.01, pos.height])
    cbar = plt.colorbar(cax = cax, ticks = None)
    cbar.set_ticks([])
    if print_fig == True:
        plt.savefig(
                '/Users/denyssutter/Documents/PhD/PhD_Denys/Figs/fig5.png', 
                dpi = 300,bbox_inches="tight")
    
def fig6(colmap = cm.ocean_r, print_fig = False):
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
    plt.plot([-1, 1], [-1, 1], 'g:', linewidth=3)
    plt.plot([-1, 1], [1, 1], 'g:', linewidth=3)
    plt.plot([-1, 0], [1, 2], 'g:', linewidth=3)
    plt.plot([0, 0], [2, -1], 'g:', linewidth=3)
    ax = plt.axes()
    ax.arrow(-1, -1, .3, .3, head_width=0.2, head_length=0.2, fc='g', ec='k')
    ax.arrow(0, -.5, 0, -.3, head_width=0.2, head_length=0.2, fc='g', ec='k')
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
                    '/Users/denyssutter/Documents/PhD/PhD_Denys/Figs/fig6.png', 
                    dpi = 300,bbox_inches="tight")

def fig7(colmap = cm.ocean_r, print_fig = False):
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
    
    plt.figure(2007, figsize=(4, 4), clear=True)
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
    def fig7a():
        ax = plt.subplot(1, 3, 1) 
        ax.set_position([.1, .3, .2 , .6])
        plt.tick_params(direction='in', length=1.5, width=.5, colors='k')  
        plt.contourf(D.k[0], D.en, np.transpose(int1), 300, cmap=colmap,
                     vmin = 0, vmax = 1.4e4)
        plt.plot([-1, 1.66], [0, 0], 'k:')
        plt.plot([-1, 1.66], [mdc_val - mdcw_val / 2, mdc_val - mdcw_val / 2],
                 'g--', linewidth=1)
        plt.plot([edc_val, edc_val], [-2.5, .5], 'g--', linewidth=1)
        plt.xlim(xmax = 1.66, xmin = -1)
        plt.ylim(ymax = 0.5, ymin = -2.5)
        plt.ylabel('$\omega$ (meV)', fontdict = font)
        plt.xticks([-1, 0, 1], ('S', '$\Gamma$', 'S'))
        plt.yticks(np.arange(-2.5, .5, .5))
        plt.text(-.9, 0.3, r'(a)', fontsize=15)
        plt.text(.22, .1, r'$\mathcal{C}$', fontsize=15)
        plt.plot(D.k[0], (mdc - b_mdc) * 1.5, 'o', markersize=1, color='C9')
        plt.fill(D.k[0], (f_mdc - b_mdc) * 1.5, alpha=.2, color='C9')
    
    def fig7b():
        ax = plt.subplot(1, 3, 2) 
        ax.set_position([.32, .3, .2 , .6])
        plt.tick_params(direction='in', length=1.5, width=.5, colors='k')  
        plt.contourf(D.k[0], D.en+.07, np.transpose(int2), 300, cmap=colmap,
                     vmin = 0, vmax = 1.4e4)
        plt.plot([-1, 1.66], [0, 0], 'k:')
        plt.plot([edc_val, edc_val], [-2.5, .5], 'g--', linewidth=1)
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
        
    def fig7c():
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
    fig7a()
    fig7b()
    fig7c()
    if print_fig == True:
        plt.savefig(
                    '/Users/denyssutter/Documents/PhD/PhD_Denys/Figs/fig7.png', 
                    dpi = 300,bbox_inches="tight")

def fig8(colmap = cm.ocean_r, print_fig = False):
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
    def fig8a():
        ax = plt.subplot(1, 3, 1) 
        ax.set_position([.1, .3, .2 , .6])
        plt.tick_params(direction='in', length=1.5, width=.5, colors='k')  
        plt.contourf(D1.ks, D1.en_norm+.1, np.flipud(D1.int_norm), 300, 
                     cmap=colmap, vmin = 0, vmax = .007)
        plt.plot([-1, 1.66], [0, 0], 'k:')
        plt.plot([edc_val, edc_val], [-2.5, .5], 'g--', linewidth=1)
        plt.xlim(xmax = 1, xmin = 0)
        plt.ylim(ymax = 0.5, ymin = -2.5)
        plt.ylabel('$\omega$ (meV)', fontdict = font)
        plt.xticks([0, 1], ('S', '$\Gamma$'))
        plt.yticks(np.arange(-2.5, .5, .5))
        plt.text(.05, 0.3, r'(a)', fontsize=15)
        plt.arrow(-1, -1, 0, -.3, head_width=0.2, head_length=0.2, fc='g', ec='k')
    
    def fig8b():
        ax = plt.subplot(1, 3, 2) 
        ax.set_position([.32, .3, .2 , .6])
        plt.tick_params(direction='in', length=1.5, width=.5, colors='k')  
        plt.contourf(D2.ks, D2.en_norm+.1, np.flipud(D2.int_norm), 300, 
                     cmap=colmap, vmin = 0, vmax = .007)
        plt.plot([-1, 1.66], [0, 0], 'k:')
        plt.plot([edc_val, edc_val], [-2.5, .5], 'g--', linewidth=1)
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
        
    def fig8c():
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
    fig8a()
    fig8b()
    fig8c()
    if print_fig == True:
        plt.savefig(
                    '/Users/denyssutter/Documents/PhD/PhD_Denys/Figs/fig8.png', 
                    dpi = 300,bbox_inches="tight")
        
def fig9(colmap = cm.bone_r, print_fig = False):
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
            plt.text(10, -2.8, r'(a) $d_{xz/yz}$', fontsize=12)
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
                    '/Users/denyssutter/Documents/PhD/PhD_Denys/Figs/fig9.png', 
                    dpi = 300,bbox_inches="tight")
        
def fig10(colmap = cm.bone_r, print_fig = False):
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
    
    def fig10a():
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
        plt.legend(('$d_{xy}$', '$d_{xz/yz}$'), frameon=False)
    
    def fig10b():
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
    fig10a()
    fig10b()
    if print_fig == True:
        plt.savefig(
                    '/Users/denyssutter/Documents/PhD/PhD_Denys/Figs/fig10.png', 
                    dpi = 300,bbox_inches="tight")
        
if __name__ == "__main__":
    fig3() 
        
        
        