#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 11:30:51 2018

@author: ilikecarbs

%%%%%%%%%%%%%%%%%%%%%
   PhD_chapter_CRO
%%%%%%%%%%%%%%%%%%%%%

**Plotting helper functions**

.. note::
        To-Do:
            -
"""

import os
import ARPES
import utils
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import linalg as la
import matplotlib.cm as cm
from scipy.stats import exponnorm
from scipy.optimize import curve_fit


# Set standard fonts
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.serif'] = ['Computer Modern Roman']
plt.rc('font', **{'family': 'serif', 'serif': ['STIXGeneral']})
font = {'family': 'serif',
        'style': 'normal',
        'color':  [0, 0, 0],
        'weight': 'ultralight',
        'size': 12,
        }
kwargs_ex = {'cmap': cm.ocean_r}
kwargs_th = {'cmap': cm.bone_r}


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

def fig1(colmap=cm.bone_r, print_fig=True):
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
    plt.show()
    
def CROfig2(colmap=cm.bone_r, print_fig=True):
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
    plt.show()
    
def CROfig3(colmap=cm.bone_r, print_fig=True):
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
    plt.show()
    
def CROfig4(colmap=cm.bone_r, print_fig=True):
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
    plt.show()
    
def CROfig5(colmap=cm.ocean_r, print_fig=True):
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
    plt.show()
    
def CROfig6(colmap=cm.ocean_r, print_fig=True):
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
    plt.show()
        
def CROfig7(colmap=cm.ocean_r, print_fig=True):
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
            utils.gauss_2, D.k[0], mdc, p_mdc_i, bounds=p_mdc_bounds)
    b_mdc = utils.poly_2(D.k[0], p_mdc[-3], p_mdc[-2], p_mdc[-1])
    f_mdc = utils.gauss_2(D.k[0], *p_mdc)
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
    plt.show()
    
def CROfig8(colmap=cm.ocean_r, print_fig=True):
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
    D1.norm(gold=48000)
    D2.norm(gold=48000)
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
    plt.show()
    
def CROfig9(colmap=cm.bone_r, print_fig=True):
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
    plt.show()
    
def CROfig10(colmap=cm.bone_r, print_fig=True):
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
    plt.show()
    
def CROfig11(print_fig=True):
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
    plt.show()
    
def CROfig12(colmap=cm.ocean_r, print_fig=True):
    """
    Constant energy maps oxygen band 
    """
    CRO_FS_plot(colmap, e=-5.2, v_min=.25, fignr=1012)
    if print_fig == True:
        plt.savefig(
                    '/Users/denyssutter/Documents/PhD/PhD_Denys/Figs/CROfig12.png', 
                    dpi = 300,bbox_inches="tight")
    plt.show()
        
def CROfig13(colmap=cm.ocean_r, print_fig=True):
    """
    Constant energy maps alpha band
    """
    CRO_FS_plot(colmap, e=-.5, v_min=.05, fignr=1013)
    if print_fig == True:
        plt.savefig(
                    '/Users/denyssutter/Documents/PhD/PhD_Denys/Figs/CROfig13.png', 
                    dpi = 300,bbox_inches="tight")
    plt.show()
    
def CROfig14(colmap=cm.ocean_r, print_fig=True):
    """
    Constant energy maps gamma band 
    """
    CRO_FS_plot(colmap, e=-2.4, v_min=.4, fignr=1014)
    if print_fig == True:
        plt.savefig(
                    '/Users/denyssutter/Documents/PhD/PhD_Denys/Figs/CROfig14.png', 
                    dpi = 300,bbox_inches="tight")
    plt.show()