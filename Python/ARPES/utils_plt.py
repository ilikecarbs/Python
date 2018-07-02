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
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
import matplotlib.pyplot as plt
from numpy import linalg as la
import matplotlib.cm as cm


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

rainbow_light = rainbow_light()
cm.register_cmap(name='rainbow_light', cmap=rainbow_light)
 
rainbow_light_2 = rainbow_light_2()
cm.register_cmap(name='rainbow_light_2', cmap=rainbow_light_2)

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
        plt.rcParams['xtick.top'] = plt.rcParams['xtick.bottom'] = True
        plt.rcParams['ytick.right'] = plt.rcParams['ytick.left'] = True
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
        plt.pcolormesh(data_kpath, data_en, data_spec, cmap = colmap,
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
    plt.rcParams['xtick.top'] = plt.rcParams['xtick.bottom'] = True
    plt.rcParams['ytick.right'] = plt.rcParams['ytick.left'] = True
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
        D.flatten(norm='spec')
        if n == 1:
            plt.rcParams['ytick.labelright'] = False
            plt.rcParams['ytick.labelleft'] = True
            ax = plt.subplot(1, 4, n) 
            ax.set_position([.1, .3, k_seg_1[-1] * scale, .3])
            pos = ax.get_position()
            
            D.ang2k(D.ang, Ekin=65-4.5, lat_unit=True, a=3.89, b=3.89, c=11, 
                    V0=0, thdg=-4, tidg=0, phidg=0)
            plt.tick_params(direction='in', length=1.5, width=.5, colors='k')  
            plt.pcolormesh(D.ks, D.en_norm+.1, D.int_flat, 
                       cmap=colmap, 
                       vmin=v_scale * 0.01 * np.max(D.int_flat), 
                       vmax=v_scale * 0.5 * np.max(D.int_flat))
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
            plt.pcolormesh(D.ks, D.en_norm+.1, D.int_flat, 
                       cmap=colmap,
                       vmin=v_scale * 0.0 * np.max(D.int_flat), 
                       vmax=v_scale * 0.54 * np.max(D.int_flat))
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
            plt.pcolormesh(D.ks, D.en_norm+.1, np.flipud(D.int_flat), 
                       cmap=colmap, 
                       vmin=v_scale * 0.01 * np.max(D.int_flat), 
                       vmax=v_scale * 0.7 * np.max(D.int_flat))
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
            plt.pcolormesh(D.ks, D.en_norm+.1, np.flipud(D.int_flat), 
                       cmap=colmap, 
                       vmin=v_scale * 0.01 * np.max(D.int_flat), 
                       vmax=v_scale * 0.53 * np.max(D.int_flat))
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
    plt.contourf(kx, ky, FSmap, 100, cmap = cm.ocean_r,
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
        
if __name__ == "__main__":
    fig3() 
        
        
        