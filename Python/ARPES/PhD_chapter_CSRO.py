#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 11:30:51 2018

@author: ilikecarbs

%%%%%%%%%%%%%%%%%%%%%
   PhD_chapter_CSRO
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
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.optimize import curve_fit
from scipy import integrate


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

 

def CSROfig1(colmap=cm.ocean_r, print_fig=True):
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
    bounds_bot = np.concatenate((p_mdc_i[0:-3] - np.inf, p_mdc_i[-3:] - delta))
    bounds_top = np.concatenate((p_mdc_i[0:-3] + np.inf, p_mdc_i[-3:] + delta))
    p_mdc_bounds = (bounds_bot, bounds_top)
    p_mdc, cov_mdc = curve_fit(
            utils.lor_8, A1.k[1], mdc, p_mdc_i, bounds=p_mdc_bounds)
    b_mdc = utils.poly_2(A1.k[1], p_mdc[-3], p_mdc[-2], p_mdc[-1])
    f_mdc = utils.lor_8(A1.k[1], *p_mdc) - b_mdc
    f_mdc[0] = 0
    f_mdc[-1] = 0
    plt.plot(A1.k[1], mdc, 'bo')
    plt.plot(A1.k[1], f_mdc)
    plt.plot(A1.k[1], b_mdc, 'k--')
        
    def CSROfig1a():
        ax = plt.subplot(1, 3, 1) 
        ax.set_position([.08, .3, .28, .35])
        plt.tick_params(direction='in', length=1.5, width=.5, colors='k')
        plt.contourf(A1.en_norm, A1.kys, A1.int_norm, 300, cmap=colmap,
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
            plt.plot((utils.lor(A1.k[1], p_mdc[i], p_mdc[i + 8], p_mdc[i + 16], 
                     p_mdc[-3], p_mdc[-2], p_mdc[-1]) - b_mdc) / 30 + .001, 
                     A1.k[1], linewidth=.5, color=cols[i])
            plt.text(p_mdc[i + 16] / 20 + corr[i], p_mdc[i]-.03, lbls[i], 
                     fontdict=font, fontsize=10, color=cols[i])
        plt.plot(f_mdc / 30 + .001, A1.k[1], color=c, linewidth=.5)
    
    def CSROfig1c():
        ax = plt.subplot(1, 3, 3) 
        ax.set_position([.66, .3, .217, .35])
        plt.tick_params(direction='in', length=1.5, width=.5, colors='k')
        plt.contourf(-np.transpose(np.fliplr(A2.en_norm)), np.transpose(A2.kys), 
                     np.transpose(np.fliplr(A2.int_norm)), 300, cmap=colmap,
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
        tb = utils.TB(a = np.pi, kbnd = 2, kpoints = 200)#Initialize 
        param = utils.paramCSRO20()  #Load parameters
        tb.CSRO(param)  #Calculate bandstructure
        plt.figure(2001)
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
                    dpi = 300,bbox_inches="tight")
    plt.show()
    
def CSROfig2(colmap=cm.ocean_r, print_fig=True):
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
            utils.lor_6, D.kx[0, :], mdc_d, p_mdc_d_i, bounds=p_mdc_d_bounds)
    b_mdc_d = utils.poly_2(D.kx[0, :], 
                          p_mdc_d[-3], p_mdc_d[-2], p_mdc_d[-1])
    f_mdc_d = utils.lor_6(D.kx[0, :], *p_mdc_d) - b_mdc_d
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
            utils.lor_4, D.ky[:, 0], mdc, p_mdc_i, bounds=p_mdc_bounds)
    b_mdc = utils.poly_2(D.ky[:, 0], p_mdc[-3], p_mdc[-2], p_mdc[-1])
    f_mdc = utils.lor_4(D.ky[:, 0], *p_mdc) - b_mdc
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
            plt.plot(D.kx[0, :], (utils.lor(D.kx[0, :], p_mdc_d[i], 
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
        D.FS_flatten(ang=False)
        ax = plt.subplot(1, 4, 3) 
        ax.set_position([.08, .2, .4, .4])
        plt.tick_params(direction='in', length=1.5, width=.5, colors='k')
        plt.contourf(D.kx, D.ky, D.map, 100, cmap=colmap,
                     vmin=.6 * np.max(D.map), vmax=1.0 * np.max(D.map))
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
        cbar.set_clim(np.min(D.map), np.max(D.map))
        ###Tight Binding Model###
#        ax = plt.subplot(1, 4, 3) 
#        ax.set_position([.08, .2, .4, .4])
#        tb = utils.TB(a = np.pi, kbnd = 2, kpoints = 200)#Initialize 
#        param = utils.paramCSRO20()  #Load parameters
#        tb.CSRO(param)  #Calculate bandstructure
#        bndstr = tb.bndstr  #Load bandstructure
#        coord = tb.coord  #Load coordinates
#        X = coord['X']; Y = coord['Y']   
#        Byz = bndstr['Byz']
#        C = plt.contour(X, Y, Byz, colors = 'black', linestyles = ':', 
#                        alpha=0, levels = -0.00)
#        p = C.collections[0].get_paths()
#        p = np.asarray(p)
#        byz = np.array([16, 17, 20, 21])
#        ind = byz; col = 'k'
#        v = p[18].vertices
#        plt.plot(v[:, 0], v[:, 1], linestyle = ':', color = 'm', 
#                 linewidth=1)
#        v = p[19].vertices
#        plt.plot(v[:, 0], v[:, 1], linestyle = ':', color = 'C1', 
#                 linewidth=1)
#        for j in ind:
#            v = p[j].vertices
#            plt.plot(v[:, 0], v[:, 1], linestyle = ':', color = col, 
#                     linewidth=1)
        
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
            plt.plot((utils.lor(D.ky[:, 0], p_mdc[i], p_mdc[i + 4], 
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
                    dpi = 300,bbox_inches="tight")
    plt.show()
    
def CSROfig3(colmap=cm.ocean_r, print_fig=True):
    """
    Polarization and orbital characters. Figure 3 in paper
    """
    ###Load and prepare calculated data###
    os.chdir('/Users/denyssutter/Documents/PhD/data')
    xz_data = np.loadtxt('DMFT_CSRO_xz.dat')
#    yz_data = np.loadtxt('DMFT_CSRO_yz.dat')
    xy_data = np.loadtxt('DMFT_CSRO_xy.dat')
    xz_lda = np.loadtxt('LDA_CSRO_xz.dat')
    yz_lda = np.loadtxt('LDA_CSRO_yz.dat')
    xy_lda = np.loadtxt('LDA_CSRO_xy.dat')
    os.chdir('/Users/denyssutter/Documents/library/Python/ARPES')
    m, n = 8000, 351 #dimensions energy, full k-path
    bot, top = 3000, 6000 #restrict energy window
    data = np.array([xz_lda + yz_lda + xy_lda, xz_data, xy_data]) #combine data
    spec = np.reshape(data[:, :, 2], (3, n, m)) #reshape into n,m
    spec = spec[:, :, bot:top] #restrict data to bot, top
    spec_en   = np.linspace(-8, 8, m) #define energy data
    spec_en   = spec_en[bot:top] #restrict energy data
    #[0, 56, 110, 187, 241, 266, 325, 350]  = [G,X,S,G,Y,T,G,Z]
    spec = np.transpose(spec, (0,2,1)) #transpose
    kB = 8.617e-5
    T = 39
    bkg = utils.FDsl(spec_en, p0=kB * T, p1=0, p2=1, p3=0, p4=0)
    bkg = bkg[:, None]
    ###Load and prepare experimental data###
    os.chdir('/Users/denyssutter/Documents/library/Python/ARPES')
    file = 25
    file_LH = 19
    file_LV = 20
    gold = 14
    mat = 'CSRO20'
    year = 2017
    sample = 'S1'
    D = ARPES.Bessy(file, mat, year, sample)
    LH = ARPES.Bessy(file_LH, mat, year, sample)
    LV = ARPES.Bessy(file_LV, mat, year, sample)
    D.norm(gold)
    LH.norm(gold)
    LV.norm(gold)
    D.restrict(bot=.7, top=.9, left=0, right=1)
    LH.restrict(bot=.55, top=.85, left=0, right=1)
    LV.restrict(bot=.55, top=.85, left=0, right=1)
#    D.bkg(norm=True)
#    LH.bkg(norm=True)
#    LV.bkg(norm=True)
    D.ang2k(D.ang, Ekin=40, lat_unit=True, a=5.5, b=5.5, c=11, 
              V0=0, thdg=2.7, tidg=0, phidg=42)
    LH.ang2k(LH.ang, Ekin=40, lat_unit=True, a=5.5, b=5.5, c=11, 
              V0=0, thdg=2.7, tidg=0, phidg=42)
    LV.ang2k(LV.ang, Ekin=40, lat_unit=True, a=5.5, b=5.5, c=11, 
              V0=0, thdg=2.7, tidg=0, phidg=42)
    c = (0, 238 / 256, 118 / 256)
    
    data = (D.int_norm, LH.int_norm, LV.int_norm)
    en = (D.en_norm - .008, LH.en_norm, LV.en_norm)
    ks = (D.kxs, LH.kxs, LV.kxs)
    k = (D.k[0], LH.k[0], LV.k[0])
    b_par = (np.array([.0037, .0002, .002]),
             np.array([.0037, .0002, .002]),
             np.array([.0037+.0005, .0002, .002]))
    
    def figCSROfig3abc():
        lbls = [r'(a) C$^+$-pol.', r'(b) $\bar{\pi}$-pol.', r'(c) $\bar{\sigma}$-pol.']
        for j in range(3): 
            plt.figure(20003)
            ax = plt.subplot(2, 3, j + 1) 
            ax.set_position([.08 + j * .26, .5, .25, .25])
            mdc_val = -.005
            mdcw_val = .015
            mdc = np.zeros(k[j].shape)
            for i in range(len(k[j])):
                val, _mdc = utils.find(en[j][i, :], mdc_val)
                val, _mdcw = utils.find(en[j][i, :], mdc_val - mdcw_val)
                mdc[i] = np.sum(data[j][i, _mdcw:_mdc])
            
            b_mdc = utils.poly_2(k[j], b_par[j][0], b_par[j][1], b_par[j][2])
        #    B_mdc = np.transpose(
        #            np.broadcast_to(b_mdc, (data[j].shape[1], data[j].shape[0])))
            plt.plot(k[j], mdc, 'bo')
            plt.plot(k[j], b_mdc, 'k--')
            plt.figure(2003)
            ax = plt.subplot(2, 3, j + 1) 
            ax.set_position([.08 + j * .26, .5, .25, .25])
            plt.tick_params(direction='in', length=1.5, width=.5, colors='k')
            if j == 0:
                plt.contourf(ks[j], en[j], data[j], 100, cmap=colmap,
                             vmin=.05 * np.max(data[j]), vmax=.35 * np.max(data[j]))
                mdc = mdc / np.max(mdc)
                plt.yticks(np.arange(-.1, .05, .02), ('-100', '-80', '-60', '-40', '-20',
                       '0', '20', '40'))
                plt.ylabel('$\omega\,(\mathrm{meV})$', fontdict = font)
            else:
                plt.contourf(ks[j], en[j], data[j], 100, cmap=colmap,
                             vmin=.3 * np.max(data[1]), vmax=.6 * np.max(data[1]))
                mdc = (mdc - b_mdc) / .005
                plt.yticks(np.arange(-.1, .05, .02), [])
            mdc[0] = 0
            mdc[-1] = 0
            plt.plot([np.min(ks[j]), np.max(ks[j])], [0, 0], 'k:')
            plt.plot([np.min(ks[j]), np.max(ks[j])], [mdc_val, mdc_val], 
                      linestyle='-.', color=c, linewidth=.5)
            plt.xticks(np.arange(-1, .5, .5), [])
            plt.xlim(xmax=np.max(ks[j]), xmin=np.min(ks[j]))   
            plt.ylim(ymax=.05, ymin=-.1)
            plt.plot(k[j], mdc / 30 + .001, 'o', markersize=1.5, color='C9')
            plt.fill(k[j], mdc / 30 + .001, alpha=.2, color='C9')
            plt.text(-1.2, .038, lbls[j], fontsize=12)
        pos = ax.get_position()
        cax = plt.axes([pos.x0+pos.width + 0.01 ,
                            pos.y0, 0.01, pos.height])
        cbar = plt.colorbar(cax = cax, ticks = None)
        cbar.set_ticks([])
        cbar.set_clim(np.min(LV.int_norm), np.max(LV.int_norm))
    
    def figCSROfig3def():
        plt.figure(2003)
        lbls = [r'(d) LDA $\Sigma_\mathrm{orb}$', r'(e) DMFT $d_{xz}$', r'(f) DMFT $d_{xy}$']
        for j in range(3):
            SG = spec[j, :, 110:187] * bkg
            GS = np.fliplr(SG)
            spec_full = np.concatenate((GS, SG, GS), axis=1)
            spec_k = np.linspace(-2, 1, spec_full.shape[1])
            ax = plt.subplot(2, 3, j + 4) 
            ax.set_position([.08 + j * .26, .24, .25, .25])
            plt.tick_params(direction='in', length=1.5, width=.5, colors='k')    
            plt.contourf(spec_k, spec_en, spec_full, 300, cmap = cm.bone_r,
                           vmin=.5, vmax=6)
            if j == 0:
                plt.yticks(np.arange(-.1, .05, .02), ('-100', '-80', '-60', '-40', '-20',
                       '0', '20', '40'))
                plt.ylabel('$\omega\,(\mathrm{meV})$', fontdict = font)
            else:
                plt.yticks(np.arange(-.1, .05, .02), [])
            plt.xticks(np.arange(-1, .5, .5), (r'S', r'', r'$\Gamma$', ''))
            plt.plot([np.min(spec_k), np.max(spec_k)], [0, 0], 'k:')
            plt.xlim(xmax=np.max(ks[0]), xmin=np.min(ks[0]))   
            plt.ylim(ymax=.05, ymin=-.1)
            plt.text(-1.2, .038, lbls[j], fontsize=12)
        pos = ax.get_position()
        cax = plt.axes([pos.x0+pos.width + 0.01 ,
                            pos.y0, 0.01, pos.height])
        cbar = plt.colorbar(cax = cax, ticks = None)
        cbar.set_ticks([])
        cbar.set_clim(np.min(spec_full), np.max(spec_full))
        
    ###Plotting###
    plt.figure(2003, figsize=(8, 8), clear=True)
    plt.figure(20003, figsize=(8, 8), clear=True)
    figCSROfig3abc()
    figCSROfig3def()
    if print_fig == True:
        plt.savefig(
                    '/Users/denyssutter/Documents/PhD/PhD_Denys/Figs/CSROfig3.png', 
                    dpi = 300,bbox_inches="tight")
    plt.show()
    
def CSROfig4(colmap=cm.ocean_r, print_fig=True):
    """
    Temperature dependence. Figure 4 in paper
    """  
    ###Plot FS map for k-scale
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
    ###Start Data loading for figure###
    os.chdir('/Users/denyssutter/Documents/library/Python/ARPES')
    files = [25, 26, 27, 28]
    gold = 14
    mat = 'CSRO20'
    year = 2017
    sample = 'S1'
    edc_e_val = -.9  #EDC espilon band
    edcw_e_val = .05
    edc_b_val = -.34  #EDC beta band
    edcw_b_val = .01
    top_e = .005; top_b = .005
    bot_e = -.015; bot_b = -.015
    left_e = -1.1; left_b = -.5
    right_e = -.7; right_b = -.2
    spec = ()
    en = ()
    k = ()
    int_e = np.zeros((4)) 
    int_b = np.zeros((4))
    eint_e = np.zeros((4))  #Error integrated epsilon band
    eint_b = np.zeros((4))  #Error integrated beta band
    T = np.array([1.3, 10., 20., 30.])
    EDC_e = () #EDC epsilon band
    EDC_b = () #EDC beta band
    eEDC_e = () #EDC error epsilon band
    eEDC_b = () #EDC error beta band
    Bkg_e = (); Bkg_b = ()
    _EDC_e = () #Index EDC epsilon band
    _EDC_b = () #Index EDC beta band
    _Top_e = (); _Top_b = ()
    _Bot_e = (); _Bot_b = ()
    _Left_e = (); _Left_b = ()
    _Right_e = (); _Right_b = ()
    
    for j in range(4): 
        D = ARPES.Bessy(files[j], mat, year, sample)
        D.norm(gold)
    #    D.restrict(bot=.7, top=.9, left=.33, right=.5)
    #    D.restrict(bot=.7, top=.9, left=.0, right=1)
        D.bkg(norm=True)
        if j == 0:
            D.ang2k(D.ang, Ekin=40, lat_unit=True, a=5.5, b=5.5, c=11, 
                      V0=0, thdg=2.5, tidg=0, phidg=42)
            int_norm = D.int_norm * 1.5
            eint_norm = D.eint_norm * 1.5
        else: 
            D.ang2k(D.ang, Ekin=40, lat_unit=True, a=5.5, b=5.5, c=11, 
                      V0=0, thdg=2.9, tidg=0, phidg=42)
            int_norm = D.int_norm
            eint_norm = D.eint_norm
            
        en_norm = D.en_norm - .008
        val, _edc_e = utils.find(D.kxs[:, 0], edc_e_val)
        val, _edcw_e = utils.find(D.kxs[:, 0], edc_e_val - edcw_e_val)
        val, _edc_b = utils.find(D.kxs[:, 0], edc_b_val)
        val, _edcw_b = utils.find(D.kxs[:, 0], edc_b_val - edcw_b_val)
        val, _top_e = utils.find(en_norm[0, :], top_e)
        val, _top_b = utils.find(en_norm[0, :], top_b)
        val, _bot_e = utils.find(en_norm[0, :], bot_e)
        val, _bot_b = utils.find(en_norm[0, :], bot_b)
        val, _left_e = utils.find(D.kxs[:, 0], left_e)
        val, _left_b = utils.find(D.kxs[:, 0], left_b)
        val, _right_e = utils.find(D.kxs[:, 0], right_e)
        val, _right_b = utils.find(D.kxs[:, 0], right_b)
        
        edc_e = np.sum(int_norm[_edcw_e:_edc_e, :], axis=0) / (_edc_e - _edcw_e + 1)
        eedc_e = np.sum(eint_norm[_edcw_e:_edc_e, :], axis=0) / (_edc_e - _edcw_e + 1)
        bkg_e = utils.Shirley(en_norm[_edc_e], edc_e)
        edc_b = np.sum(int_norm[_edcw_b:_edc_b, :], axis=0) / (_edc_b - _edcw_b + 1)
        eedc_b = np.sum(eint_norm[_edcw_b:_edc_b, :], axis=0) / (_edc_b - _edcw_b + 1)
        bkg_b = utils.Shirley(en_norm[_edc_b], edc_b)
        int_e[j] = np.sum(int_norm[_left_e:_right_e, _bot_e:_top_e])
        int_b[j] = np.sum(int_norm[_left_b:_right_b, _bot_b:_top_b])
        eint_e[j] = np.sum(eint_norm[_left_e:_right_e, _bot_e:_top_e])
        eint_b[j] = np.sum(eint_norm[_left_b:_right_b, _bot_b:_top_b])
        spec = spec + (int_norm,)
        en = en + (en_norm,)
        k = k + (D.kxs,)
        EDC_e = EDC_e + (edc_e,)
        EDC_b = EDC_b + (edc_b,)
        eEDC_e = eEDC_e + (eedc_e,)
        eEDC_b = eEDC_b + (eedc_b,)
        Bkg_e = Bkg_e + (bkg_e,)
        Bkg_b = Bkg_b + (bkg_b,)
        _EDC_e = _EDC_e + (_edc_e,)
        _EDC_b = _EDC_b + (_edc_b,)
        _Top_e = _Top_e + (_top_e,)
        _Top_b = _Top_b + (_top_b,)
        _Bot_e = _Bot_e + (_bot_e,)
        _Bot_b = _Bot_b + (_bot_b,)
        _Left_e = _Left_e + (_left_e,)
        _Left_b = _Left_b + (_left_b,)
        _Right_e = _Right_e + (_right_e,)
        _Right_b = _Right_b + (_right_b,)  
        
    eint_e = eint_e / int_e
    eint_b = eint_b / int_e
    int_e = int_e / int_e[0]
    int_b = int_b / int_b[0]
    def CSROfig4abcd():
        lbls = [r'(a) $T=1.3\,$K', r'(b) $T=10\,$K', r'(c) $T=20\,$K', r'(d) $T=30\,$K']
        for j in range(4): 
            ax = plt.subplot(2, 4, j + 1) 
            ax.set_position([.08 + j * .21, .5, .2, .2])
            plt.tick_params(direction='in', length=1.5, width=.5, colors='k')
    #        plt.contourf(k[j], en[j], spec[j], 200, cmap=colmap,
    #                         vmin=.02 * np.max(spec[j]), vmax=.165 * np.max(spec[j]))
            plt.contourf(k[j], en[j], spec[j], 200, cmap=colmap,
                             vmin=.01 * np.max(spec[0]), vmax=.28 * np.max(spec[0]))
            if j == 0:
                plt.yticks(np.arange(-.1, .03, .02), ('-100', '-80', '-60', '-40', '-20',
                       '0', '20'))
                plt.ylabel('$\omega\,(\mathrm{meV})$', fontdict = font)
                plt.plot([k[j][_EDC_e[j], 0], k[j][_EDC_e[j], 0]], [en[j][0, 0], en[j][0, -1]],
                     linestyle='-.', color='C9', linewidth=.5)
                plt.plot([k[j][_EDC_b[j], 0], k[j][_EDC_b[j], 0]], [en[j][0, 0], en[j][0, -1]],
                     linestyle='-.', color='C9', linewidth=.5)
                plt.text(-1.06, .007, r'$\bar{\epsilon}$-band', color='r')
                plt.text(-.5, .007, r'$\bar{\beta}$-band', color='m')
            elif j == 3:
                plt.plot([k[j][_EDC_e[j], 0], k[j][_EDC_e[j], 0]], [en[j][0, 0], en[j][0, -1]],
                     linestyle='-.', color='k', linewidth=.5)
                plt.plot([k[j][_EDC_b[j], 0], k[j][_EDC_b[j], 0]], [en[j][0, 0], en[j][0, -1]],
                     linestyle='-.', color='k', linewidth=.5)
                plt.yticks(np.arange(-.1, .05, .02), [])
            else: 
                plt.yticks(np.arange(-.1, .05, .02), [])
            plt.plot([np.min(k[j]), np.max(k[j])], [0, 0], 'k:')
            
            plt.plot([k[j][_Left_e[j], 0], k[j][_Left_e[j], 0]], 
                     [en[j][0, _Top_e[j]], en[j][0, _Bot_e[j]]],
                     linestyle='--', color='r', linewidth=.5)
            plt.plot([k[j][_Right_e[j], 0], k[j][_Right_e[j], 0]], 
                     [en[j][0, _Top_e[j]], en[j][0, _Bot_e[j]]],
                     linestyle='--', color='r', linewidth=.5)
            plt.plot([k[j][_Left_e[j], 0], k[j][_Right_e[j], 0]], 
                     [en[j][0, _Top_e[j]], en[j][0, _Top_e[j]]],
                     linestyle='--', color='r', linewidth=.5)
            plt.plot([k[j][_Left_e[j], 0], k[j][_Right_e[j], 0]], 
                     [en[j][0, _Bot_e[j]], en[j][0, _Bot_e[j]]],
                     linestyle='--', color='r', linewidth=.5)
            
            plt.plot([k[j][_Left_b[j], 0], k[j][_Left_b[j], 0]], 
                     [en[j][0, _Top_b[j]], en[j][0, _Bot_b[j]]],
                     linestyle='--', color='m', linewidth=.5)
            plt.plot([k[j][_Right_b[j], 0], k[j][_Right_b[j], 0]], 
                     [en[j][0, _Top_b[j]], en[j][0, _Bot_b[j]]],
                     linestyle='--', color='m', linewidth=.5)
            plt.plot([k[j][_Left_b[j], 0], k[j][_Right_b[j], 0]], 
                     [en[j][0, _Top_b[j]], en[j][0, _Top_b[j]]],
                     linestyle='--', color='m', linewidth=.5)
            plt.plot([k[j][_Left_b[j], 0], k[j][_Right_b[j], 0]], 
                     [en[j][0, _Bot_b[j]], en[j][0, _Bot_b[j]]],
                     linestyle='--', color='m', linewidth=.5)
            
            ax.xaxis.tick_top()
            plt.xticks(np.arange(-1, .5, 1.), [r'S', r'$\Gamma$'])
            plt.xlim(xmax=0.05, xmin=np.min(k[j]))   
            plt.ylim(ymax=.03, ymin=-.1)
            plt.text(-1.25, .018, lbls[j], fontsize=10)
            
        pos = ax.get_position()
        cax = plt.axes([pos.x0+pos.width + 0.01 ,
                            pos.y0, 0.01, pos.height])
        cbar = plt.colorbar(cax = cax, ticks = None)
        cbar.set_ticks([])
        cbar.set_clim(np.min(D.int_norm), np.max(D.int_norm))
    
    def CSROfig4efg():
        lbls = [r'(e) $\bar{\epsilon}$-band', r'(f) $\bar{\epsilon}$-band (zoom)', 
                r'(g) $\bar{\beta}$-band (zoom)']
        lbls_x = [-.77, -.093, -.093]
        lbls_y = [2.05, .99, .99]
        plt.figure(20004, figsize=(8, 8), clear=True)
        ax = plt.subplot(2, 2, 1) 
        ax.set_position([.08, .5, .3, .3])
        for j in range(4):
            plt.plot(en[j][_EDC_e[j]], EDC_e[j], 'o', markersize=1)
            plt.plot(en[j][_EDC_e[j]], Bkg_e[j], 'o', markersize=1)
        ax = plt.subplot(2, 2, 2) 
        ax.set_position([.08 + .31, .5, .3, .3])
        EDCn_e = () #normalized
        EDCn_b = () #normalized
        eEDCn_e = () #normalized
        eEDCn_b = () #normalized
        for j in range(4):
            tmp_e = EDC_e[j]-Bkg_e[j]
            tmp_b = EDC_b[j]-Bkg_b[j]
            tot_e = integrate.trapz(tmp_e, en[j][_EDC_e[j]])
            edcn_e = tmp_e / tot_e
            eedcn_e = eEDC_e[j] / tot_e
            edcn_b = tmp_b / tot_e
            eedcn_b = eEDC_b[j] / tot_e
            plt.plot(en[j][_EDC_e[j]], edcn_e, 'o', markersize=1)
            EDCn_e = EDCn_e + (edcn_e,)
            EDCn_b = EDCn_b + (edcn_b,)
            eEDCn_e = eEDCn_e + (eedcn_e,)
            eEDCn_b = eEDCn_b + (eedcn_b,)
        plt.figure(2004)
        for j in range(2):
            ax = plt.subplot(2, 4, j + 5) 
            ax.set_position([.08 + j * .21, .29, .2, .2])
            plt.tick_params(direction='in', length=1.5, width=.5, colors='k')
            plt.plot(en[0][_EDC_e[0]], EDCn_e[0], 'o', markersize=1, color='C9')
            plt.plot(en[3][_EDC_e[3]], EDCn_e[3], 'o', markersize=1, color='k', alpha = .8)
            plt.yticks([])
            if j == 0:
                y_max = 1.1; x_min = -.1; x_max = .05
                plt.plot([x_min, x_max], [y_max, y_max], 'k--', linewidth=.5)
                plt.plot([x_min, x_min], [0, y_max], 'k--', linewidth=.5)
                plt.plot([x_max, x_max], [0, y_max], 'k--', linewidth=.5)
                plt.xticks(np.arange(-.8, .2, .2))
                plt.ylabel(r'Intensity (a.u.)')
                plt.xlim(xmin=-.8, xmax=.1)
                plt.ylim(ymin=0, ymax=2.3)
                plt.xlabel(r'$\omega$ (eV)')
            else:
                plt.xticks(np.arange(-.08, .06, .04), ('-80', '-40', '0', '40'))
                plt.xlim(xmin=x_min, xmax=x_max)
                plt.ylim(ymin=0, ymax=y_max)
                plt.xlabel(r'$\omega$ (meV)')
                plt.text(.01, .7, r'$1.3\,$K', color='C9')
                plt.text(.01, .3, r'$30\,$K', color='k')
            plt.text(lbls_x[j], lbls_y[j], lbls[j])
        ax = plt.subplot(2, 4, 7) 
        ax.set_position([.08 + 2 * .21, .29, .2, .2])
        plt.tick_params(direction='in', length=1.5, width=.5, colors='k')
        plt.plot(en[0][_EDC_b[0]], EDCn_b[0], 'o', markersize=1, color='C9')
        plt.plot(en[3][_EDC_b[3]], EDCn_b[3], 'o', markersize=1, color='k', alpha = .8)
        plt.yticks([])
        plt.xticks(np.arange(-.08, .06, .04), ('-80', '-40', '0', '40'))
        plt.xlim(xmin=-.1, xmax=.05)
        plt.ylim(ymin=0, ymax=.005)
        plt.ylim(ymin=0, ymax=1.1)
        plt.xlabel(r'$\omega$ (meV)')
        plt.text(lbls_x[-1], lbls_y[-1], lbls[-1])
        return (en, EDCn_e, EDCn_b, EDC_e, EDC_b, Bkg_e, Bkg_b, _EDC_e, _EDC_b,
                eEDCn_e, eEDCn_b, eEDC_e, eEDC_b)
    
    def CSROfig4h():
        ax = plt.subplot(2, 4, 8) 
        ax.set_position([.08 + 3 * .21, .29, .2, .2])
        plt.tick_params(direction='in', length=1.5, width=.5, colors='k')
        plt.errorbar(T, int_e, yerr=eint_e, linewidth=.5,
                     capsize=2, color='red', fmt='o', ms=5)
        plt.errorbar(T, int_b, yerr=eint_b, linewidth=.5,
                     capsize=2, color='m', fmt='d', ms=5)
        plt.plot([1.3, 32], [1, .695], 'r--', linewidth=.5)
        plt.plot([1.3, 32], [1, 1], 'm--', linewidth=.5)
        plt.xticks(T)
        plt.yticks(np.arange(.7, 1.05, .1))
        plt.xlim(xmax=32, xmin=0)
        plt.ylim(ymax=1.07, ymin=.7)
        plt.grid(True, alpha=.2)
        plt.xlabel(r'$T$ (K)', fontdict = font)
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position('right')
        plt.ylabel(r'$\int_\boxdot \mathcal{A}(k, \omega, T) \, \slash \quad \int_\boxdot \mathcal{A}(k, \omega, 1.3\,\mathrm{K})$', 
                   fontdict = font, fontsize=8)
        plt.text(1.3, 1.032, r'(h)')
        plt.text(8, .83, r'$\bar{\epsilon}$-band', color='r')
        plt.text(15, .95, r'$\bar{\beta}$-band', color='m')
        
    plt.figure(2004, figsize=(8, 8), clear=True)
    CSROfig4abcd()
    (en, EDCn_e, EDCn_b, EDC_e, EDC_b, Bkg_e, Bkg_b, _EDC_e, _EDC_b,
     eEDCn_e, eEDCn_b, eEDC_e, eEDC_b) = CSROfig4efg();
    CSROfig4h()
    if print_fig == True:
        plt.savefig(
                '/Users/denyssutter/Documents/PhD/PhD_Denys/Figs/CSROfig4.png', 
                dpi = 300,bbox_inches="tight")
    dims = np.array([len(en), en[0].shape[0], en[0].shape[1]]);
    os.chdir('/Users/denyssutter/Documents/PhD/data')
    np.savetxt('Data_CSROfig4_en.dat', np.ravel(en));
    np.savetxt('Data_CSROfig4_EDCn_e.dat', np.ravel(EDCn_e));
    np.savetxt('Data_CSROfig4_EDCn_b.dat', np.ravel(EDCn_b));
    np.savetxt('Data_CSROfig4_EDC_e.dat', np.ravel(EDC_e));
    np.savetxt('Data_CSROfig4_EDC_b.dat', np.ravel(EDC_b));
    np.savetxt('Data_CSROfig4_Bkg_e.dat', np.ravel(Bkg_e));
    np.savetxt('Data_CSROfig4_Bkg_b.dat', np.ravel(Bkg_b));
    np.savetxt('Data_CSROfig4__EDC_e.dat', np.ravel(_EDC_e));
    np.savetxt('Data_CSROfig4__EDC_b.dat', np.ravel(_EDC_b));
    np.savetxt('Data_CSROfig4_eEDCn_e.dat', np.ravel(eEDCn_e));
    np.savetxt('Data_CSROfig4_eEDCn_b.dat', np.ravel(eEDCn_b));
    np.savetxt('Data_CSROfig4_eEDC_e.dat', np.ravel(eEDC_e));
    np.savetxt('Data_CSROfig4_eEDC_b.dat', np.ravel(eEDC_b));
    np.savetxt('Data_CSROfig4_dims.dat', dims);
    print('\n ~ Data saved (en, EDCs + normalized + indices + Bkgs)',
              '\n', '==========================================')  
    os.chdir('/Users/denyssutter/Documents/library/Python/ARPES')
    plt.show()
    return (en, EDCn_e, EDCn_b, EDC_e, EDC_b, Bkg_e, Bkg_b, _EDC_e, _EDC_b,
                eEDCn_e, eEDCn_b, eEDC_e, eEDC_b, dims);
    
def CSROfig5(print_fig = False, load=True):
    """
    Analysis Z of epsilon band
    """
    if load == True:
        os.chdir('/Users/denyssutter/Documents/PhD/data')
        en = np.loadtxt('Data_CSROfig4_en.dat');
        EDCn_e = np.loadtxt('Data_CSROfig4_EDCn_e.dat');
        EDCn_b = np.loadtxt('Data_CSROfig4_EDCn_b.dat');
        EDC_e = np.loadtxt('Data_CSROfig4_EDC_e.dat');
        EDC_b = np.loadtxt('Data_CSROfig4_EDC_b.dat');
        Bkg_e = np.loadtxt('Data_CSROfig4_Bkg_e.dat');
        Bkg_b = np.loadtxt('Data_CSROfig4_Bkg_b.dat');
        _EDC_e = np.loadtxt('Data_CSROfig4__EDC_e.dat', dtype=np.int32);
        _EDC_b = np.loadtxt('Data_CSROfig4__EDC_b.dat', dtype=np.int32);
        eEDCn_e = np.loadtxt('Data_CSROfig4_eEDCn_e.dat');
        eEDCn_b = np.loadtxt('Data_CSROfig4_eEDCn_b.dat');
        eEDC_e = np.loadtxt('Data_CSROfig4_eEDC_e.dat');
        eEDC_b = np.loadtxt('Data_CSROfig4_eEDC_b.dat');
        dims = np.loadtxt('Data_CSROfig4_dims.dat', dtype=np.int32);
        print('\n ~ Data loaded (en, EDCs + normalized + indices + Bkgs)',
              '\n', '==========================================')  
        os.chdir('/Users/denyssutter/Documents/library/Python/ARPES')
    else:     
        (en, EDCn_e, EDCn_b, EDC_e, EDC_b, Bkg_e, Bkg_b, _EDC_e, _EDC_b,
                eEDCn_e, eEDCn_b, eEDC_e, eEDC_b, dims) = CSROfig4()
    ###Reshape data###
    en = np.reshape(np.ravel(en), (dims[0], dims[1], dims[2]))
    EDCn_e = np.reshape(np.ravel(EDCn_e), (dims[0], dims[2]))
    EDCn_b = np.reshape(np.ravel(EDCn_b), (dims[0], dims[2]))
    EDC_e = np.reshape(np.ravel(EDC_e), (dims[0], dims[2]))
    EDC_b = np.reshape(np.ravel(EDC_b), (dims[0], dims[2]))
    Bkg_e = np.reshape(np.ravel(Bkg_e), (dims[0], dims[2]))
    Bkg_b = np.reshape(np.ravel(Bkg_b), (dims[0], dims[2]))
    _EDC_e = np.reshape(np.ravel(_EDC_e), (dims[0]))
    _EDC_b = np.reshape(np.ravel(_EDC_b), (dims[0]))
    eEDCn_e = np.reshape(np.ravel(eEDCn_e), (dims[0], dims[2]))
    eEDCn_b = np.reshape(np.ravel(eEDCn_b), (dims[0], dims[2]))
    eEDC_e = np.reshape(np.ravel(eEDC_e), (dims[0], dims[2]))
    eEDC_b = np.reshape(np.ravel(eEDC_b), (dims[0], dims[2]))
    d = 1e-6
    plt.figure(2005, figsize=(10, 10), clear=True)
    D = 1e6
    p_edc_i = np.array([6.9e-1, 7.3e-3, 4.6, 4.7e-3, 4.1e-2, 2.6e-3,
                        1e0, -.2, .3, 1, -.1, 1e-1])
    bounds_fl = ([p_edc_i[0] - D, p_edc_i[1] - d, p_edc_i[2] - d,
                  p_edc_i[3] - D, p_edc_i[4] - D, p_edc_i[5] - D],
                 [p_edc_i[0] + D, p_edc_i[1] + d, p_edc_i[2] + d, 
                  p_edc_i[3] + D, p_edc_i[4] + D, p_edc_i[5] + D])
    
    titles = [r'$T=1.3\,$K', r'$T=10\,$K', r'$T=20\,$K', r'$T=30\,$K']
    lbls = [r'(a)', r'(b)', r'(c)', r'(d)',
            r'(e)', r'(f)', r'(g)', r'(h)',
            r'(i)', r'(j)', r'(k)', r'(l)']
    cols = ([0, 1, 1], [0, .7, .7], [0, .4, .4], [0, 0, 0])
    cols_r = ([0, 0, 0], [0, .4, .4], [0, .7, .7], [0, 1, 1])
    xx = np.arange(-2, .5, .001)
    Z = np.ones((4))
    for j in range(4):
        ###First row###
        Bkg = Bkg_e[j]
        Bkg[0] = 0
        Bkg[-1] = 0
        ax = plt.subplot(5, 4, j + 1) 
        ax.set_position([.08 + j * .21, .61, .2, .2])
        plt.tick_params(direction='in', length=1.5, width=.5, colors='k')
        plt.plot(en[j][_EDC_e[j]], EDC_e[j], 'o', markersize=1, color=cols[j])
        plt.fill(en[j][_EDC_e[j]], Bkg, '--', linewidth=1, color='C8', alpha=.3)
        plt.yticks([])
        plt.xticks(np.arange(-.8, .2, .2), [])
        plt.xlim(xmin=-.8, xmax=.1)
        plt.ylim(ymin=0, ymax=.02)
        plt.text(-.77, .0183, lbls[j])
        plt.title(titles[j], fontsize=15)
        if j == 0:
            plt.text(-.77, .001, r'Background')
            plt.ylabel(r'Intensity (a.u.)')
        ###Third row#
        ax = plt.subplot(5, 4, j + 13) 
        ax.set_position([.08 + j * .21, .18, .2, .2])
        plt.tick_params(direction='in', length=1.5, width=.5, colors='k')
        plt.plot(en[j][_EDC_e[j]], EDCn_e[j], 'o', markersize=1, color=cols[j])
        
        p_fl, cov_fl = curve_fit(
                utils.FL_spectral_func, en[j][_EDC_e[j]][900:-1], 
                EDCn_e[j][900:-1], 
                p_edc_i[0: -6], bounds=bounds_fl)
        f_fl = utils.FL_spectral_func(xx, *p_fl)
            
        plt.yticks([])
        plt.xticks(np.arange(-.8, .2, .1))
        plt.xlim(xmin=-.1, xmax=.05)
        plt.ylim(ymin=0, ymax=1.1)
        plt.xlabel(r'$\omega$ (eV)')
        if j == 0:
            plt.ylabel(r'Intensity (a.u.)')
            plt.text(-.095, .2, 
                     r'$\int \, \, \mathcal{A}_\mathrm{coh.}(k\approx k_\mathrm{F}^{\bar\epsilon}, \omega) \, \mathrm{d}\omega$')
        plt.text(-.095, 1.01, lbls[j + 8])
        ###Second row###
        ax = plt.subplot(5, 4, j + 9) 
        ax.set_position([.08 + j * .21, .4, .2, .2])
        plt.tick_params(direction='in', length=1.5, width=.5, colors='k')
        plt.plot(en[j][_EDC_e[j]], EDCn_e[j], 'o', markersize=1, color=cols[j])
        
        bounds = (np.concatenate((p_fl - D, p_edc_i[6:] - D), axis=0),
                  np.concatenate((p_fl + D, p_edc_i[6:] + D), axis=0))
        bnd = 300
        p_edc, cov_edc = curve_fit(
                utils.Full_spectral_func, en[j][_EDC_e[j]][bnd:-1], EDCn_e[j][bnd:-1], 
                np.concatenate((p_fl, p_edc_i[-6:]), axis=0), bounds=bounds)
        f_edc = utils.Full_spectral_func(xx, *p_edc)
        plt.plot(xx, f_edc,'--', color=cols_r[j], linewidth=1.5)
        f_mod = utils.gauss_mod(xx, *p_edc[-6:])
        f_fl = utils.FL_spectral_func(xx, *p_edc[0:6]) 
        plt.fill(xx, f_mod, alpha=.3, color=cols[j])
        plt.yticks([])
        plt.xticks(np.arange(-.8, .2, .2))
        plt.xlim(xmin=-.8, xmax=.1)
        plt.ylim(ymin=0, ymax=2.2)
        if j == 0:
            plt.ylabel(r'Intensity (a.u.)')
            plt.text(-.68, .3, 
                     r'$\int \, \, \mathcal{A}_\mathrm{inc.}(k\approx k_\mathrm{F}^{\bar\epsilon}, \omega) \, \mathrm{d}\omega$')
        plt.text(-.77, 2.03, lbls[j + 4])  
        ###Third row###
        ax = plt.subplot(5, 4, j + 13) 
        plt.fill(xx, f_fl, alpha=.3, color=cols[j])
        p = plt.plot(xx, f_edc,'--', color=cols_r[j],  linewidth=2)
        plt.legend(p, [r'$\mathcal{A}_\mathrm{coh.} + \mathcal{A}_\mathrm{inc.}$'], frameon=False)
        ###Calculate Z###
        A_mod = integrate.trapz(f_mod, xx)
        A_fl = integrate.trapz(f_fl, xx)
        Z[j] = A_fl / A_mod
    if print_fig == True:
        plt.savefig(
                '/Users/denyssutter/Documents/PhD/PhD_Denys/Figs/CSROfig5.png', 
                dpi = 300,bbox_inches="tight")
    os.chdir('/Users/denyssutter/Documents/PhD/data')
    np.savetxt('Data_CSROfig5_Z_e.dat', Z);
    print('\n ~ Data saved (Z)',
              '\n', '==========================================')  
    os.chdir('/Users/denyssutter/Documents/library/Python/ARPES')
    return Z
    plt.show()
    
def CSROfig6(colmap=cm.ocean_r, print_fig=True, load=True):
    """
    Analysis MDC's beta band
    """
    if load == True:
        os.chdir('/Users/denyssutter/Documents/PhD/data')
        v_LDA_data = np.loadtxt('Data_CSROfig8_v_LDA.dat')
        v_LDA = v_LDA_data[0]
        ev_LDA = v_LDA_data[1]
        os.chdir('/Users/denyssutter/Documents/library/Python/ARPES')
    else:     
        v_LDA, ev_LDA = CSROfig8()
    os.chdir('/Users/denyssutter/Documents/library/Python/ARPES')
    files = [25, 26, 27, 28]
    gold = 14
    mat = 'CSRO20'
    year = 2017
    sample = 'S1'
    ###Create Placeholders###
    spec = () #ARPES spectra
    espec = () #Errors on signal
    en = () #energy scale
    k = () #momenta
    Z = () #quasiparticle residuum
    eZ = () #error
    Re = () #Real part of self energy
    Width = () #MDC Half width at half maximum
    eWidth = () #error
    Loc_en = () #Associated energy
    mdc_t_val = .001 #start energy of MDC analysis
    mdc_b_val = -.1 #end energy of MDC analysis
    n_spec = 4 #how many temperatures are analysed
    scale = 5e-5 #helper variable for plotting
    ###Colors###
    c = (0, 238 / 256, 118 / 256) 
    cols = ([0, 1, 1], [0, .7, .7], [0, .4, .4], [0, 0, 0])
    cols_r = ([0, 0, 0], [0, .4, .4], [0, .7, .7], [0, 1, 1])
    Re_cols = ['khaki', 'darkkhaki', 'goldenrod', 'darkgoldenrod']
    Re_cols_r = ['darkgoldenrod', 'goldenrod', 'darkkhaki', 'khaki']
    xx = np.arange(-.4, .25, .01) #helper variable for plotting
    for j in range(n_spec): 
        D = ARPES.Bessy(files[j], mat, year, sample) #Load Bessy data
        D.norm(gold) #noramlized
        D.restrict(bot=.7, top=.9, left=.31, right=.6) #restrict data set
        D.bkg(norm=True) #subtract background
        if j == 0:
            D.ang2k(D.ang, Ekin=40 - 4.5, lat_unit=False, a=5.5, b=5.5, c=11, 
                      V0=0, thdg=2.5, tidg=0, phidg=42) #plot as inverse Angstrom
            int_norm = D.int_norm * 1.5 #intensity adjustment from background comparison
            eint_norm = D.eint_norm * 1.5 #error adjustment
        else: 
            D.ang2k(D.ang, Ekin=40 - 4.5, lat_unit=False, a=5.5, b=5.5, c=11, 
                      V0=0, thdg=2.9, tidg=0, phidg=42)
            int_norm = D.int_norm
            eint_norm = D.eint_norm        
        en_norm = D.en_norm - .008
        spec = spec + (int_norm,)
        espec = espec + (eint_norm,)
        en = en + (en_norm,)
        k = k + (D.kxs * np.sqrt(2),) #D.kxs is only kx -> but we analyze along diagonal
        
    plt.figure('2006', figsize=(10, 10), clear=True)
    titles = [r'$T=1.3\,$K', r'$T=10\,$K', r'$T=20\,$K', r'$T=30\,$K']
    lbls = [r'(a)', r'(b)', r'(c)', r'(d)',
                r'(e)', r'(f)', r'(g)', r'(h)',
                r'(i)', r'(j)', r'(k)', r'(l)',
                r'(k)', r'(l)', r'(m)', r'(n)']
    for j in range(n_spec):
        val, _mdc_t = utils.find(en[j][0, :], mdc_t_val) #Get indices
        val, _mdc_b = utils.find(en[j][0, :], mdc_b_val) #Get indices
        mdc_seq = np.arange(_mdc_t,_mdc_b, -1) #range of indices
        loc = np.zeros((_mdc_t - _mdc_b)) #placeholders maximum position
        eloc = np.zeros((_mdc_t - _mdc_b)) #corresponding errors
        width = np.zeros((_mdc_t - _mdc_b)) #placheholder HWHM
        ewidth = np.zeros((_mdc_t - _mdc_b)) #corresponding error
        ###First row###
        ax = plt.subplot(4, 4, j + 1) 
        ax.set_position([.08 + j * .21, .76, .2, .2])
        plt.tick_params(direction='in', length=1.5, width=.5, colors='k')
        plt.contourf(k[j], en[j], spec[j], 200, cmap=colmap,
                         vmin=.0 * np.max(spec[0]), vmax=1 * np.max(spec[0]))
        plt.plot([-1, 0], [en[j][0, mdc_seq[2]], en[j][0, mdc_seq[2]]], '-.',
                 color=c, linewidth=.5)
        plt.plot([-1, 0], [en[j][0, mdc_seq[50]], en[j][0, mdc_seq[50]]], '-.',
                 color=c, linewidth=.5)
        plt.plot([-1, 0], [en[j][0, mdc_seq[100]], en[j][0, mdc_seq[100]]], '-.',
                 color=c, linewidth=.5)
        plt.plot([-1, 0], [0, 0], 'k:')
        if j == 0:
            plt.ylabel('$\omega\,(\mathrm{meV})$', fontdict = font)
            plt.yticks(np.arange(-.2, .1, .05), 
                       ('-200', '-150', '-100', '-50', '0', '50'))
            plt.text(-.43, .009, r'MDC maxima', color='C8')
            plt.text(-.24, .009, r'$\epsilon_\mathrm{LDA}(\mathbf{k})$', color='C4')
        else:
            plt.yticks(np.arange(-.2, .1, .05), [])
        plt.xticks(np.arange(-1, 0, .1), [])
        plt.xlim(xmax=-.05, xmin=-.45)
        plt.ylim(ymax=.05, ymin=-.15)
        plt.text(-.44, .035, lbls[j])
        plt.title(titles[j], fontsize=15)
        ###Second row###
        ax = plt.subplot(4, 4, j + 5) 
        ax.set_position([.08 + j * .21, .55, .2, .2])
        plt.tick_params(direction='in', length=1.5, width=.5, colors='k')
        n = 0
        p_mdc = []
        for i in mdc_seq:
            _sl1 = 10 #Index used for background endpoint slope
            _sl2 = 155 #other endpoint
            n += 1 #counter
            mdc_k = k[j][:, i] #current MDC k-axis
            mdc_int = spec[j][:, i] #current MDC
            mdc_eint = espec[j][:, i] #error
            if any(x==n for x in [1, 50, 100]):
                plt.errorbar(mdc_k, mdc_int - scale * n**1.15, mdc_eint, 
                             linewidth=.5, capsize=.1, color=cols[j], fmt='o', ms=.5)
            ###Fit MDC###
            d = 1e-2 #small boundary
            eps = 1e-8 #essentially fixed boundary
            D = 1e5 #essentially free boundary
            const_i = mdc_int[_sl2] #constant background estimation
            slope_i = (mdc_int[_sl1] - mdc_int[_sl2])/(mdc_k[_sl1] - mdc_k[_sl2]) #slope estimation
            p_mdc_i = np.array(
                        [-.27, 5e-2, 1e-3,
                         const_i, slope_i, .0]) #initial values
            if n > 70: #take fixed initial values until it reaches THIS iteration, then take last outcome as inital values
                p_mdc_i = p_mdc
                ###p0: position, p1: width, p2: amplitude###
                ###p3: constant bkg, p4: slope, p5: curvature###
                bounds_bot = np.array([
                                p_mdc_i[0] - d, p_mdc_i[1] - d, p_mdc_i[2] - D, 
                                p_mdc_i[3] - D, p_mdc_i[4] - eps, p_mdc_i[5] - eps])
                bounds_top = np.array([
                                p_mdc_i[0] + d, p_mdc_i[1] + d, p_mdc_i[2] + D, 
                                p_mdc_i[3] + D, p_mdc_i[4] + eps, p_mdc_i[5] + eps])
            else:
                bounds_bot = np.array([
                                p_mdc_i[0] - D, p_mdc_i[1] - D, p_mdc_i[2] - D, 
                                p_mdc_i[3] - D, p_mdc_i[4] - D, p_mdc_i[5] - eps])
                bounds_top = np.array([
                                p_mdc_i[0] + D, p_mdc_i[1] + D, p_mdc_i[2] + D, 
                                p_mdc_i[3] + D, p_mdc_i[4] + D, p_mdc_i[5] + eps])
            bounds = (bounds_bot, bounds_top) #boundaries
            p_mdc, c_mdc = curve_fit(
                utils.lorHWHM, mdc_k, mdc_int, p0=p_mdc_i, bounds=bounds) #fit curve
            err_mdc = np.sqrt(np.diag(c_mdc)) #errors estimation of parameters
            loc[n - 1] = p_mdc[0] #position of fit
            eloc[n - 1] = err_mdc[0] #error
            width[n - 1] = p_mdc[1] #HWHM of fit (2 times this value is FWHM)
            ewidth[n - 1] = err_mdc[1] #error
            b_mdc = utils.poly_2(mdc_k, *p_mdc[-3:]) #background
            f_mdc = utils.lor(mdc_k, *p_mdc) #fit
            ###Plot the fits###
            if any(x==n for x in [1, 50, 100]):
                plt.plot(mdc_k, f_mdc - scale * n**1.15, '--', color=cols_r[j])
                plt.plot(mdc_k, b_mdc - scale * n**1.15, 'C8-', linewidth=2, alpha=.3)
        if j == 0:
            plt.ylabel('Intensity (a.u.)', fontdict = font)
            plt.text(-.43, -.0092, r'Background', color='C8')
        plt.yticks([])
        plt.xticks(np.arange(-1, 0, .1))
        plt.xlim(xmax=-.05, xmin=-.45)
        plt.ylim(ymin=-.01, ymax = .003)
        plt.text(-.44, .0021, lbls[j + 4])
        plt.xlabel(r'$k_{\Gamma - \mathrm{S}}\,(\mathrm{\AA}^{-1})$', fontdict=font)
        ###Third row###
        loc_en = en[j][0, mdc_seq] #energies of Lorentzian fits
        ax = plt.subplot(4, 4, j + 9) 
        ax.set_position([.08 + j * .21, .29, .2, .2])
        plt.tick_params(direction='in', length=1.5, width=.5, colors='k')
        plt.errorbar(-loc_en, width, ewidth,
                     color=cols[j], linewidth=.5, capsize=2, fmt='o', ms=2)
        ###Fitting the width
        im_bot = np.array([1 - D, -.1 - d, 1 - D])
        im_top = np.array([1 + D, -.1 + d, 1 + D])
        im_bounds = (im_bot, im_top)
        p_im, c_im = curve_fit(
                utils.poly_2, -loc_en, width, bounds=im_bounds)
        plt.plot(-loc_en, utils.poly_2(-loc_en, *p_im),
                 '--', color=cols_r[j])
        if j == 0:
            plt.ylabel('HWHM $(\mathrm{\AA}^{-1})$', fontdict = font)
            plt.yticks(np.arange(0, 1, .05))
            plt.text(.005, .05, r'Quadratic fit', fontdict = font)
        else:
            plt.yticks(np.arange(0, 1, .05), [])
        plt.xticks(np.arange(0, .1, .02), [])
        plt.xlim(xmax=-loc_en[-1], xmin=-loc_en[0])
        plt.ylim(ymax=.13, ymin=0)
        plt.text(.0025, .12, lbls[j + 8])
        ###Fourth row###
        k_F = loc[0] #Position first fit
        p0 = -k_F * v_LDA #get constant from y=v_LDA*x + p0 (y=0, x=k_F)
        yy = p0 + xx * v_LDA #For plotting v_LDA
        en_LDA = p0 + loc * v_LDA #Energies
        re = loc_en - en_LDA #Real part of self energy
        ###Plotting###
        ax = plt.subplot(4, 4, j + 13) 
        ax.set_position([.08 + j * .21, .08, .2, .2])
        plt.tick_params(direction='in', length=1.5, width=.5, colors='k')
        plt.errorbar(-loc_en, re, ewidth * v_LDA,
                     color=Re_cols[j], linewidth=.5, capsize=2, fmt='o', ms=2)
        _bot = 0 #first index of fitting ReS
        _top = 20 #last index of fitting ReS
        re_bot = np.array([0 - eps, 1 - D]) #upper boundary
        re_top = np.array([0 + eps, 1 + D]) #bottom boundary
        re_bounds = (re_bot, re_top) #boundaries
        p_re, c_re = curve_fit(
                utils.poly_1, -loc_en[_bot:_top], re[_bot:_top], 
                bounds=re_bounds) #fit ReS
        dre = -p_re[1] #dReS / dw 
        edre = np.sqrt(np.diag(c_re))[1]
        plt.plot(-loc_en, utils.poly_1(-loc_en, *p_re),
                 '--', color=Re_cols_r[j])
        z = 1 / (1 - dre) #quasiparticle residue
        ez = np.abs(1 / (1 - dre)**2 * edre)
        if j == 0:
            plt.ylabel(r'$\mathfrak{Re}\Sigma$ (meV)', 
                       fontdict = font)
            plt.yticks(np.arange(0, .15, .05), ('0', '50', '100'))
            plt.text(.02, .03, r'Linear fit', fontsize=12, color=Re_cols[-1])
        else:
            plt.yticks(np.arange(0, .15, .05), [])
        plt.xticks(np.arange(0, .1, .02), ('0', '-20', '-40', '-60', '-80', '-100'))
        plt.xlabel(r'$\omega$ (meV)', fontdict = font)
        plt.xlim(xmax=-loc_en[-1], xmin=-loc_en[0])
        plt.ylim(ymax=.15, ymin=0)
        plt.text(.0025, .14, lbls[j + 12])
        ###First row again###
        ax = plt.subplot(4, 4, j + 1) 
        plt.plot(loc, loc_en, 'C8o', ms=.5)
        if j == 0:
            plt.plot(xx, yy, 'C4--', lw=1)
            ax.arrow(loc[20], -.05, 0, loc_en[20]+.04, 
                     head_width=0.01, head_length=0.01, fc='r', ec='r')
            ax.arrow(loc[20], -.05, 0, loc_en[20]+.005, 
                     head_width=0.01, head_length=0.01, fc='r', ec='r')
            plt.text(-.26, -.05, r'$\mathfrak{Re}\Sigma(\omega)$', color='r')
        if j == 3:
            pos = ax.get_position()
            cax = plt.axes([pos.x0+pos.width+0.01 ,
                                pos.y0, 0.01, pos.height])
            cbar = plt.colorbar(cax = cax, ticks = None)
            cbar.set_ticks([])
            cbar.set_clim(np.min(int_norm), np.max(int_norm))
        Z = Z + (z,);
        eZ = eZ + (ez,);
        Re = Re + (re,);
        Loc_en = Loc_en + (loc_en,);
        Width = Width + (width,);
        eWidth = eWidth + (ewidth,);
    print('Z=' + str(Z))
    if print_fig == True:
        plt.savefig(
                '/Users/denyssutter/Documents/PhD/PhD_Denys/Figs/CSROfig6.png', 
                dpi = 300,bbox_inches="tight")
    dims = np.array([len(Re), Re[0].shape[0]])
    os.chdir('/Users/denyssutter/Documents/PhD/data')
    np.savetxt('Data_CSROfig6_Z_b.dat', np.ravel(Z));
    np.savetxt('Data_CSROfig6_eZ_b.dat', np.ravel(eZ));
    np.savetxt('Data_CSROfig6_Re.dat', np.ravel(Re));
    np.savetxt('Data_CSROfig6_Loc_en.dat', np.ravel(Loc_en));
    np.savetxt('Data_CSROfig6_Width.dat', np.ravel(Width));
    np.savetxt('Data_CSROfig6_eWidth.dat', np.ravel(eWidth));
    np.savetxt('Data_CSROfig6_dims.dat', np.ravel(dims));
    print('\n ~ Data saved (Z, eZ, Re, Loc_en, Width, eWidth)',
              '\n', '==========================================')  
    os.chdir('/Users/denyssutter/Documents/library/Python/ARPES')
    plt.show()
    return Z, eZ, Re, Loc_en, Width, eWidth, dims

def CSROfig7(colmap=cm.ocean_r, print_fig=True):
    """
    Background subtraction
    """
    os.chdir('/Users/denyssutter/Documents/library/Python/ARPES')
    file = 25
    gold = 14
    mat = 'CSRO20'
    year = 2017
    sample = 'S1'
    plt.figure('2007', figsize=(8, 8), clear=True)
    lbls = [r'(a)', r'(b)', r'(c)']
    _bkg = [False, True]
    for i in range(2):
        D = ARPES.Bessy(file, mat, year, sample)
        D.norm(gold)
        D.bkg(norm=_bkg[i])
        
        if i == 0:
            edc_bkg = np.zeros((D.en.size))
            for ii in range(D.en.size):
                edc_bkg[ii] = np.min(D.int_norm[:, ii])
            edc_bkg[0] = 0
        D.ang2k(D.ang, Ekin=40, lat_unit=True, a=5.5, b=5.5, c=11, 
                  V0=0, thdg=2.5, tidg=0, phidg=42)
        int_norm = D.int_norm * 1.5
        en_norm = D.en_norm - .008
        
        ax = plt.subplot(1, 3, i + 1) 
        ax.set_position([.08 + i * .26, .3, .25, .5])
        plt.tick_params(direction='in', length=1.5, width=.5, colors='k')
        plt.contourf(D.kxs, en_norm, int_norm, 100, cmap=colmap,
                         vmin=.0, vmax=.05)
        plt.plot([np.min(D.kxs), np.max(D.kxs)], [0, 0], 'k:')
        if i == 0:
            plt.ylabel('$\omega\,(\mathrm{meV})$', fontdict=font)
            plt.yticks(np.arange(-.8, .2, .2))
        else:
            plt.yticks(np.arange(-.8, .2, .2), [])
        plt.xticks([-1, 0], ('S', r'$\Gamma$'))
        plt.ylim(ymax = .1, ymin=-.8)
        plt.text(-1.2, .06, lbls[i])
    
    ax = plt.subplot(1, 3, 3) 
    ax.set_position([.08 + .52, .3, .25, .5])
    plt.tick_params(direction='in', length=1.5, width=.5, colors='k')
    plt.plot(edc_bkg, en_norm[0], 'ko', ms=1)
    plt.fill(edc_bkg, en_norm[0], alpha=.2, color='C8')
    plt.plot([0, 1], [0, 0], 'k:')
    plt.xticks([])
    plt.yticks(np.arange(-.8, .2, .2), [])
    plt.ylim(ymax = .1, ymin=-.8)
    plt.xlim(xmax = np.max(edc_bkg) * 1.1, xmin=0)
    plt.text(.0025, -.45, 'Background EDC')
    plt.text(.0008, .06, lbls[2])
    plt.xlabel('Intensity (a.u.)', fontdict=font)
    pos = ax.get_position()
    cax = plt.axes([pos.x0+pos.width+0.01 ,
                        pos.y0, 0.01, pos.height])
    cbar = plt.colorbar(cax = cax, ticks = None)
    cbar.set_ticks([])
    cbar.set_clim(np.min(int_norm), np.max(int_norm))
    if print_fig == True:
        plt.savefig(
                '/Users/denyssutter/Documents/PhD/PhD_Denys/Figs/CSROfig7.png', 
                dpi = 300,bbox_inches="tight")
    plt.show()
    
def CSROfig8(colmap=cm.bone_r, print_fig=True):
    """
    Extraction LDA Fermi velocity
    """ 
    os.chdir('/Users/denyssutter/Documents/PhD/data')
    xz_lda = np.loadtxt('LDA_CSRO_xz.dat')
    os.chdir('/Users/denyssutter/Documents/library/Python/ARPES')
    #[0, 56, 110, 187, 241, 266, 325, 350]  = [G,X,S,G,Y,T,G,Z]
    m, n = 8000, 351 #dimensions energy, full k-path
    size = 187 - 110
    bot, top = 3840, 4055 #restrict energy window
    data = np.array([xz_lda]) #combine data
    spec = np.reshape(data[0, :, 2], (n, m)) #reshape into n,m
    spec = spec[110:187, bot:top] #restrict data to bot, top
    #spec = np.flipud(spec)
    spec_en = np.linspace(-8, 8, m) #define energy data
    spec_en = spec_en[bot:top] #restrict energy data
    spec_en = np.broadcast_to(spec_en, (spec.shape))
    spec_k = np.linspace(-np.sqrt(2) * np.pi / 5.5, 0, size)
    spec_k = np.transpose(
                np.broadcast_to(spec_k, (spec.shape[1], spec.shape[0])))
    max_pts = np.ones((size))
    for i in range(size):
        max_pts[i] = spec[i, :].argmax()
    ini = 40
    fin = 48
    max_pts = max_pts[ini:fin]
    
    max_k = spec_k[ini:fin, 0]
    max_en = spec_en[0, max_pts.astype(int)]
    p_max, c_max = curve_fit(
                    utils.poly_1, max_k, max_en)
    v_LDA = p_max[1]
    ev_LDA = np.sqrt(np.diag(c_max))[1]
#    k_F = -p_max[0] / p_max[1]
    xx = np.arange(-.43, -.27, .01)
    plt.figure(2008, figsize=(8, 8), clear=True)
    ax = plt.subplot(1, 3, 1) 
    ax.set_position([.2, .24, .5, .3])
    plt.tick_params(direction='in', length=1.5, width=.5, colors='k') 
    plt.contourf(spec_k, spec_en, spec, 200, cmap=colmap) 
    plt.plot(xx, utils.poly_1(xx, *p_max), 'C9--', linewidth=1)
    plt.plot(max_k, max_en, 'ro', ms=2)
    plt.plot([np.min(spec_k), 0], [0, 0], 'k:')
    plt.yticks(np.arange(-1, 1, .1))
    plt.ylim(ymax=.1, ymin=-.3)
    plt.ylabel(r'$\omega$ (eV)', fontdict=font)
    plt.xlabel(r'$k_{\Gamma - \mathrm{S}}\, (\mathrm{\AA}^{-1})$', fontdict=font)
    plt.text(-.3, -.15, '$v_\mathrm{LDA}=$' + str(np.round(v_LDA,2)) + ' eV$\,\mathrm{\AA}$')
    pos = ax.get_position()
    cax = plt.axes([pos.x0+pos.width + 0.01 ,
                        pos.y0, 0.01, pos.height])
    cbar = plt.colorbar(cax = cax, ticks = None)
    cbar.set_ticks([])
    cbar.set_clim(np.min(spec), np.max(spec))
    if print_fig == True:
        plt.savefig(
                '/Users/denyssutter/Documents/PhD/PhD_Denys/Figs/CSROfig8.png', 
                dpi = 300,bbox_inches="tight")
    os.chdir('/Users/denyssutter/Documents/PhD/data')
    np.savetxt('Data_CSROfig8_v_LDA.dat', [v_LDA, ev_LDA])
    os.chdir('/Users/denyssutter/Documents/library/Python/ARPES')
    return v_LDA, ev_LDA
    plt.show()
    
def CSROfig9(print_fig=True, load=True):
    """
    ReS vs ImS
    """ 
    if load == True:
        os.chdir('/Users/denyssutter/Documents/PhD/data')
        Z = np.loadtxt('Data_CSROfig6_Z_b.dat')
        eZ = np.loadtxt('Data_CSROfig6_eZ_b.dat')
        Re = np.loadtxt('Data_CSROfig6_Re.dat')
        Loc_en = np.loadtxt('Data_CSROfig6_Loc_en.dat')
        Width = np.loadtxt('Data_CSROfig6_Width.dat')
        eWidth = np.loadtxt('Data_CSROfig6_eWidth.dat')
        dims = np.loadtxt('Data_CSROfig6_dims.dat', dtype=np.int32)
        v_LDA_data = np.loadtxt('Data_CSROfig8_v_LDA.dat')
        v_LDA = v_LDA_data[0]
        ev_LDA = v_LDA_data[1]
        os.chdir('/Users/denyssutter/Documents/library/Python/ARPES')
    else:     
        Z, eZ, Re, Loc_en, Width, eWidth, dims = CSROfig6()
        v_LDA, ev_LDA = CSROfig8()
    
    Re = np.reshape(np.ravel(Re), (dims[0], dims[1]))
    Loc_en = np.reshape(np.ravel(Loc_en), (dims[0], dims[1]))
    Width = np.reshape(np.ravel(Width), (dims[0], dims[1]))
    eWidth = np.reshape(np.ravel(eWidth), (dims[0], dims[1]))
    plt.figure('2009', figsize=(8, 8), clear=True)
    lbls = [r'(a)  $T=1.3\,$K', r'(b)  $T=10\,$K', r'(c)  $T=20\,$K', 
            r'(d)  $T=30\,$K']
    Im_cols = np.array([[0, 1, 1], [0, .7, .7], [0, .4, .4], [0, 0, 0]])
    Re_cols = ['khaki', 'darkkhaki', 'goldenrod', 'darkgoldenrod']
    n_spec = 4
    position = ([.1, .55, .4, .4],
                [.1 + .41, .55, .4, .4],
                [.1, .55 - .41, .4, .4],
                [.1 + .41, .55 - .41, .4, .4])
    offset = [.048, .043, .04, .043]
    n = 0
    for j in range(n_spec):
        en = -Loc_en[j]
        width = Width[j]
        ewidth = eWidth[j]
        im = width * v_LDA - offset[j]
        eim = ewidth * v_LDA
        re = Re[j] / (1  - Z[j])
        ere = ewidth * v_LDA / (1 - Z[j])
        
        ax = plt.subplot(2, 2, j + 1) 
        ax.set_position(position[j])
        plt.tick_params(direction='in', length=1.5, width=.5, colors='k')
        plt.errorbar(en, im, eim,
                     color=Im_cols[j], linewidth=.5, capsize=2, fmt='d', ms=2)
        plt.errorbar(en, re, ere,
                     color=Re_cols[j], linewidth=.5, capsize=2, fmt='o', ms=2)
        plt.text(.002, .235, lbls[j], fontdict=font)
        if j==0:
            plt.text(.005, .14, r'$\mathfrak{Re}\Sigma \, (1-Z)^{-1}$',
                     fontsize=15, color=Re_cols[3])
            plt.text(.06, .014, r'$\mathfrak{Im}\Sigma$',
                     fontsize=15, color=Im_cols[2])
        if any(x==j for x in [0, 2]):
            plt.ylabel( 'Self energy (meV)', fontdict=font)
            plt.yticks(np.arange(0, .25, .05), ('0', '50', '100', '150', '200'))
            plt.xticks(np.arange(0, .1, .02), [])
        else:
            plt.yticks(np.arange(0, .25, .05), [])
        if any(x==j for x in [2, 3]):    
            plt.xticks(np.arange(0, .1, .02), 
                       ('0', '-20', '-40', '-60', '-80', '-100'))
            plt.xlabel('$\omega\,(\mathrm{meV})$', fontdict=font)
        else:
            plt.xticks(np.arange(0, .1, .02), ())
        n += 1
        plt.xlim(xmax=.1, xmin=0)
        plt.ylim(ymax=.25, ymin=0)
        plt.grid(True, alpha=.2)
    os.chdir('/Users/denyssutter/Documents/PhD/data')
    np.savetxt('Data_CSROfig9_re.dat', re);
    np.savetxt('Data_CSROfig9_ere.dat', ere);
    np.savetxt('Data_CSROfig9_im.dat', im);
    np.savetxt('Data_CSROfig9_eim.dat', eim);
    print('\n ~ Data saved (re, ere, im, eim)',
              '\n', '==========================================')  
    os.chdir('/Users/denyssutter/Documents/library/Python/ARPES')
    if print_fig == True:
        plt.savefig(
                '/Users/denyssutter/Documents/PhD/PhD_Denys/Figs/CSROfig9.png', 
                dpi = 300,bbox_inches="tight")
    plt.show()
    
def CSROfig10(print_fig=True):
    """
    Quasiparticle Z
    """ 
    os.chdir('/Users/denyssutter/Documents/PhD/data')
    Z_e = np.loadtxt('Data_CSROfig5_Z_e.dat');
    Z_b = np.loadtxt('Data_CSROfig6_Z_b.dat');
    eZ_b = np.loadtxt('Data_CSROfig6_eZ_b.dat');
    v_LDA_data = np.loadtxt('Data_CSROfig8_v_LDA.dat')
    v_LDA = v_LDA_data[0]
    C_B = np.genfromtxt('Data_C_Braden.csv', delimiter=',')
    C_M = np.genfromtxt('Data_C_Maeno.csv', delimiter=',')
    R_1 = np.genfromtxt('Data_R_1.csv', delimiter=',')
    R_2 = np.genfromtxt('Data_R_2.csv', delimiter=',')
    print('\n ~ Data loaded (Zs, specific heat, transport data)',
          '\n', '==========================================')  
    os.chdir('/Users/denyssutter/Documents/library/Python/ARPES')
    
    plt.figure('2010', figsize=(10, 10), clear=True)
    T = np.array([1.3, 10, 20, 30])
    hbar = 1.0545717e-34
    NA = 6.022141e23
    kB = 1.38065e-23
    a = 5.33e-10
    m_e = 9.109383e-31
    m_LDA = 1.6032
    gamma = (np.pi * NA * kB ** 2 * a ** 2 / (3 * hbar ** 2)) * m_LDA * m_e
    Z_B = gamma / C_B[:, 1] 
    Z_M = gamma / C_M[:, 1] * 1e3
    xx = np.array([1e-3, 1e4])
    yy = 2.3 * xx ** 2
    ax = plt.subplot(1, 2, 1) 
    ax.set_position([.2, .3, .3, .3])
    plt.tick_params(direction='in', length=1.5, width=.5, colors='k')
    plt.errorbar(T, Z_b, eZ_b * v_LDA,
                 color='m', linewidth=.5, capsize=2, fmt='o', ms=2)
    plt.fill_between([0, 50], .24, .33, alpha=.1, color='m')
    plt.plot(39, .229, 'm*')
    #plt.plot(39, .326, 'C1+')
    plt.arrow(28, .16, 8.5, .06, head_width=0.0, head_length=0, fc='k', ec='k')
    plt.arrow(28, .125, 8.5, -.06, head_width=0.0, head_length=0, fc='k', ec='k')
    plt.errorbar(T, Z_e, Z_e / v_LDA,
                 color='r', linewidth=.5, capsize=2, fmt='d', ms=2)
    plt.fill_between([0, 50], 0.01, .07, alpha=.1, color='r')
    plt.plot(39, .052, 'r*')
    #plt.plot(39, .175, 'r+')
    plt.plot(C_B[:, 0], Z_B, 'o', ms=1, color='cadetblue')
    plt.plot(C_M[:, 0], Z_M, 'o', ms=1, color='slateblue')
    ax.set_xscale("log", nonposx='clip')
    plt.yticks(np.arange(0, .5, .1))
    plt.xlim(xmax=44, xmin=1)
    plt.ylim(ymax=.4, ymin=0)
    plt.xlabel(r'$T$ (K)')
    plt.ylabel(r'$Z$')
    plt.text(1.2, .37, 'S. Nakatsuji $\mathit{et\, \,al.}$', color='slateblue')
    plt.text(1.2, .34, 'J. Baier $\mathit{et\, \,al.}$', color='cadetblue')
    plt.text(2.5e0, .28, r'$\bar{\beta}$-band', color='m')
    plt.text(2.5e0, .04, r'$\bar{\epsilon}$-band', color='r')
    plt.text(20, .135, 'DMFT')
    ###Inset###
    ax = plt.subplot(1, 2, 2) 
    ax.set_position([.28, .38, .13, .08])
    plt.tick_params(direction='in', length=1.5, width=.5, colors='k')
    plt.loglog(np.sqrt(R_1[:, 0]), R_1[:, 1], 'o', ms=1, color='slateblue')
    plt.loglog(np.sqrt(R_2[:, 0]), R_2[:, 1], 'o', ms=1, color='slateblue')
    plt.loglog(xx, yy, 'k--', lw=1)
    plt.ylabel(r'$\rho\,(\mu \Omega \mathrm{cm})$')
    plt.xlim(xmax=1e1, xmin=1e-1)
    plt.ylim(ymax=1e4, ymin=1e-2)
    plt.text(2e-1, 1e1, r'$\propto T^2$')
    plt.show()
    if print_fig == True:
        plt.savefig(
                '/Users/denyssutter/Documents/PhD/PhD_Denys/Figs/CSROfig10.png', 
                dpi = 600,bbox_inches="tight")
        
def CSROfig11(print_fig=True):
    """
    Tight binding model CSRO
    """    
    os.chdir('/Users/denyssutter/Documents/library/Python/ARPES')
    kbnd = 2
    tb = utils.TB(a = np.pi, kbnd = kbnd, kpoints = 500)  #Initialize tight binding model
    param = utils.paramCSRO20()  
    tb.CSRO(param=param, e0=0, vert=True, proj=True) 
    bndstr = tb.bndstr
    coord = tb.coord   
    X = coord['X']; Y = coord['Y']   
    Axz = bndstr['Axz']; Ayz = bndstr['Ayz']; Axy = bndstr['Axy']
    Bxz = bndstr['Bxz']; Byz = bndstr['Byz']; Bxy = bndstr['Bxy']
    en = (Axz, Ayz, Axy, Bxz, Byz, Bxy)
    v_bnd = np.max(tb.FS)
    plt.figure(2011, figsize=(8, 8), clear=True)
    ax = plt.subplot(122)
    ax.set_position([.1, .3, .4 , .4])
    plt.tick_params(direction='in', length=1.5, width=.5, colors='k')
    for i in en:
        plt.contour(X, Y, i, colors = 'black', linestyles = '-', levels = 0,
                    alpha=.2)
    plt.contourf(tb.kx, tb.ky, tb.FS, 200, cmap='PuOr', vmin=-v_bnd, vmax=v_bnd)
    plt.text(-.1, -.1, r'$\Gamma$', fontsize=18, color='r')
    plt.text(.9, .9, 'S', fontsize=18, color='r')
    plt.text(-.1, .9, 'X', fontsize=18, color='r')
    plt.plot([-1, 1], [1, 1], 'k-', lw=2)
    plt.plot([-1, 1], [-1, -1], 'k-', lw=2)
    plt.plot([1, 1], [-1, 1], 'k-', lw=2)
    plt.plot([-1, -1], [-1, 1], 'k-', lw=2)
    plt.xticks(np.arange(-kbnd - 1, kbnd + 1, 1))
    plt.yticks(np.arange(-kbnd - 1, kbnd + 1, 1))
    plt.xlim(xmax=kbnd, xmin=-kbnd)
    plt.ylim(ymax=kbnd, ymin=-kbnd)
    plt.xlabel(r'$k_x \, (\pi/a)$', fontdict=font)
    plt.ylabel(r'$k_y \, (\pi/b)$', fontdict=font)
    pos = ax.get_position()
    cax = plt.axes([pos.x0+pos.width + 0.01 ,
                        pos.y0, 0.01, pos.height])
    cbar = plt.colorbar(cax = cax, ticks = None)
    cbar.set_ticks([])
    cbar.set_clim(np.min(tb.FS), np.max(tb.FS))
    if print_fig == True:
        plt.savefig(
                '/Users/denyssutter/Documents/PhD/PhD_Denys/Figs/CSROfig11.png', 
                dpi = 600,bbox_inches="tight")
        
def CSROfig12(print_fig=True):
    """
    Tight binding model SRO
    """    
    os.chdir('/Users/denyssutter/Documents/library/Python/ARPES')
    kbnd = 2
    tb = utils.TB(a = np.pi, kbnd = kbnd, kpoints = 500)  #Initialize tight binding model
    param = utils.paramSRO()  
    tb.SRO(param=param, e0=0, vert=True, proj=True) 
    bndstr = tb.bndstr
    coord = tb.coord   
    X = coord['X']; Y = coord['Y']   
    xz = bndstr['xz']; yz = bndstr['yz']; xy = bndstr['xy']
    en = (xz, yz, xy)
    v_bnd = np.max(tb.FS)
    plt.figure(2012, figsize=(8, 8), clear=True)
    ax = plt.subplot(122)
    ax.set_position([.1, .3, .4 , .4])
    plt.tick_params(direction='in', length=1.5, width=.5, colors='k')
    for i in en:
        plt.contour(X, Y, i, colors = 'black', linestyles = '-', levels = 0,
                    alpha=.2)
    plt.contourf(tb.kx, tb.ky, tb.FS, 200, cmap='PuOr', vmin=-v_bnd, vmax=v_bnd)
    plt.plot([-1, 1], [1, 1], 'k-', lw=2)
    plt.plot([-1, 1], [-1, -1], 'k-', lw=2)
    plt.plot([1, 1], [-1, 1], 'k-', lw=2)
    plt.plot([-1, -1], [-1, 1], 'k-', lw=2)
    plt.plot([-1, 0], [0, 1], 'k--', lw=1)
    plt.plot([-1, 0], [0, -1], 'k--', lw=1)
    plt.plot([0, 1], [1, 0], 'k--', lw=1)
    plt.plot([0, 1], [-1, 0], 'k--', lw=1)
    plt.xticks(np.arange(-kbnd - 1, kbnd + 1, 1))
    plt.yticks(np.arange(-kbnd - 1, kbnd + 1, 1))
    plt.xlim(xmax=kbnd, xmin=-kbnd)
    plt.ylim(ymax=kbnd, ymin=-kbnd)
    plt.xlabel(r'$k_x \, (\pi/a)$', fontdict=font)
    plt.ylabel(r'$k_y \, (\pi/b)$', fontdict=font)
    pos = ax.get_position()
    cax = plt.axes([pos.x0+pos.width + 0.01 ,
                        pos.y0, 0.01, pos.height])
    cbar = plt.colorbar(cax = cax, ticks = None)
    cbar.set_ticks([])
    cbar.set_clim(np.min(tb.FS), np.max(tb.FS))
    if print_fig == True:
        plt.savefig(
                '/Users/denyssutter/Documents/PhD/PhD_Denys/Figs/CSROfig12.png', 
                dpi = 600,bbox_inches="tight")
        
def CSROfig13(print_fig=True):
    """
    TB along high symmetry directions, orbitally resolved
    """    
    k_pts = 300
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
        en, spec, bndstr = utils.CSRO_eval(x[i], y[i])
        k = np.sqrt(x[i] ** 2 + y[i] ** 2)
        v_bnd = .1
        if i != 0:
            ax = plt.subplot(2, 3, i + 1)
            ax.set_position([.1 + i * .15 + .15 * (np.sqrt(2)-1), .2, .15 , .4])
            plt.tick_params(direction='in', length=1.5, width=.5, colors='k') 
            k = -k
        else:
            ax = plt.subplot(2, 3, i + 1)
            ax.set_position([.1 + i * .15 , .2, .15 * np.sqrt(2) , .4])
            plt.tick_params(direction='in', length=1.5, width=.5, colors='k') 
        plt.contourf(k, en, spec, 200, 
                     cmap='PuOr', vmin=-v_bnd, vmax=v_bnd)
        for j in range(len(bndstr)):
            plt.plot(k, bndstr[j], 'k-', alpha=.2)
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
        plt.ylim(ymax=np.max(en), ymin=np.min(en))
    pos = ax.get_position()
    cax = plt.axes([pos.x0+pos.width+0.01 ,
                            pos.y0, 0.01, pos.height])
    cbar = plt.colorbar(cax = cax, ticks = None)
    cbar.set_ticks([])
    cbar.set_clim(np.min(spec), np.max(spec))
    plt.show()
    if print_fig == True:
        plt.savefig(
                '/Users/denyssutter/Documents/PhD/PhD_Denys/Figs/CSROfig13.png', 
                dpi = 600,bbox_inches="tight")

def CSROfig14(print_fig=True):
    """
    TB and density of states
    """    
    os.chdir('/Users/denyssutter/Documents/PhD/data')
    Axz_dos = np.loadtxt('Data_Axz_kpts_5000.dat')
    Ayz_dos = np.loadtxt('Data_Ayz_kpts_5000.dat')
    Axy_dos = np.loadtxt('Data_Axy_kpts_5000.dat')
    Bxz_dos = np.loadtxt('Data_Bxz_kpts_5000.dat')
    Byz_dos = np.loadtxt('Data_Byz_kpts_5000.dat')
    Bxy_dos = np.loadtxt('Data_Bxy_kpts_5000.dat')
    os.chdir('/Users/denyssutter/Documents/library/Python/ARPES')
    bands = (Ayz_dos, Axz_dos, Axy_dos, Byz_dos, Bxz_dos, Bxy_dos)
    
    En = ()
    _EF = ()
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
        ef, _ef = utils.find(en, 0.003)
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
        _EF = _EF + (_ef,)
        DOS = DOS + (dos,)
    N = np.sum(N_bands)
    k_pts = 200
    x_GS = np.linspace(0, 1, k_pts)
    y_GS = np.linspace(0, 1, k_pts)
    x_SX = np.linspace(1, 0, k_pts)
    y_SX = np.ones(k_pts)
    x_XG = np.zeros(k_pts)
    y_XG = np.linspace(1, 0, k_pts)
    x = (x_GS, x_SX, x_XG)
    y = (y_GS, y_SX, y_XG)
    cols = ['C1', 'C0', 'm', 'C8', 'C9', 'C3']
    plt.figure('TB_eval', figsize=(8, 8), clear=True)
    for i in range(len(x)):
        en, spec, bndstr = utils.CSRO_eval(x[i], y[i])
        k = np.sqrt(x[i] ** 2 + y[i] ** 2)
        if i != 0:
            ax = plt.subplot(3, 3, i + 1)
            ax.set_position([.1 + i * .15 + .15 * (np.sqrt(2) - 1), .2, .15 , .4])
            plt.tick_params(direction='in', length=1.5, width=.5, colors='k') 
            k = -k
        else:
            ax = plt.subplot(3, 3, i + 1)
            ax.set_position([.1 + i * .15 , .2, .15 * np.sqrt(2) , .4])
            plt.tick_params(direction='in', length=1.5, width=.5, colors='k') 
        for j in range(len(bndstr)):
            plt.plot(k, bndstr[j], color=cols[j])
        if i == 0:
            plt.xticks([0, np.sqrt(2)], ('$\Gamma$', 'S'))
            plt.yticks(np.arange(-1, 1, .2))
            plt.text(.05, .25, '(a)', fontdict=font)
            plt.ylabel(r'$\omega$ (eV)', fontdict=font)
        elif i==1:
            plt.xticks([-np.sqrt(2), -1], ('', 'X'))
            plt.yticks(np.arange(-1, 1, .2), [])
        elif i==2:
            plt.xticks([-1, 0], ('', '$\Gamma$'))
            plt.yticks(np.arange(-1, 1, .2), [])
        plt.plot([k[0], k[-1]], [0, 0], 'k:')    
        plt.xlim(xmin=k[0], xmax=k[-1])
        plt.ylim(ymax=np.max(en), ymin=np.min(en))
    n = 0
    for band in bands:
        dos = DOS[n]
        dos[0] = 0
        dos[-1] = 0
        ax = plt.subplot(3, 3, 9)
        ax.set_position([.1 + 3 * .15 + .01 + .15 * (np.sqrt(2) - 1), .2, .15 * np.sqrt(2) , .4])
        plt.tick_params(direction='in', length=1.5, width=.5, colors='k') 
        ef, _ef = utils.find(En[n], 0.00)
        plt.plot(DOS[n], En[n], color = 'k', lw=.5)
        plt.fill_betweenx(En[n][:_ef], 0, DOS[n][:_ef], color=cols[n], alpha=.7)
        plt.yticks(np.arange(-1, 1, .2), [])
        plt.xticks([])
        plt.ylim(ymax=np.max(en), ymin=np.min(en))
        plt.xlim(xmax=35, xmin=0)
        n += 1
    plt.text(1, .25, '(b)', fontdict=font)
    plt.xlabel('DOS $\partial_\omega\Omega_2(\omega)$', fontdict=font)
    print(N)
    plt.show()
    if print_fig == True:
        plt.savefig(
                '/Users/denyssutter/Documents/PhD/PhD_Denys/Figs/CSROfig14.png', 
                dpi = 600,bbox_inches="tight")
    os.chdir('/Users/denyssutter/Documents/PhD/data')
    np.savetxt('Data_TB_DOS_Ayz.dat', DOS[0]);
    np.savetxt('Data_TB_DOS_Axz.dat', DOS[1]);
    np.savetxt('Data_TB_DOS_Axy.dat', DOS[2]);
    np.savetxt('Data_TB_DOS_Byz.dat', DOS[3]);
    np.savetxt('Data_TB_DOS_Bxz.dat', DOS[4]);
    np.savetxt('Data_TB_DOS_Bxy.dat', DOS[5]);
    np.savetxt('Data_TB_DOS_en_Ayz.dat', En[0]);
    np.savetxt('Data_TB_DOS_en_Axz.dat', En[1]);
    np.savetxt('Data_TB_DOS_en_Axy.dat', En[2]);
    np.savetxt('Data_TB_DOS_en_Byz.dat', En[3]);
    np.savetxt('Data_TB_DOS_en_Bxz.dat', En[4]);
    np.savetxt('Data_TB_DOS_en_Bxy.dat', En[5]);
    print('\n ~ Data saved Density of states',
              '\n', '==========================================')  
    os.chdir('/Users/denyssutter/Documents/library/Python/ARPES')
    
def CSROfig15(print_fig=True):
    """
    DMFT FS
    """   
    os.chdir('/Users/denyssutter/Documents/PhD/data')
    FS_DMFT_xy_data = np.loadtxt('FS_DMFT_xy.dat')
    FS_DMFT_xz_data = np.loadtxt('FS_DMFT_xz.dat')
    FS_DMFT_yz_data = np.loadtxt('FS_DMFT_yz.dat')
    FS_LDA_xy_data = np.loadtxt('FS_LDA_xy.dat')
    FS_LDA_xz_data = np.loadtxt('FS_LDA_xz.dat')
    FS_LDA_yz_data = np.loadtxt('FS_LDA_yz.dat')
    os.chdir('/Users/denyssutter/Documents/library/Python/ARPES')
    m = 51
    ###Reshape###
    FS_DMFT_xy = np.reshape(FS_DMFT_xy_data, (m, m, 3))
    FS_DMFT_xz = np.reshape(FS_DMFT_xz_data, (m, m, 3))
    FS_DMFT_yz = np.reshape(FS_DMFT_yz_data, (m, m, 3))
    FS_LDA_xy = np.reshape(FS_LDA_xy_data, (m, m, 3))
    FS_LDA_xz = np.reshape(FS_LDA_xz_data, (m, m, 3))
    FS_LDA_yz = np.reshape(FS_LDA_yz_data, (m, m, 3))
    ###Normalize###
    FS_DMFT_xy = FS_DMFT_xy / np.max(FS_DMFT_xy[:, :, 2])
    FS_DMFT_xz = FS_DMFT_xz / np.max(FS_DMFT_xz[:, :, 2])
    FS_DMFT_yz = FS_DMFT_yz / np.max(FS_DMFT_yz[:, :, 2])
    FS_LDA_xy = FS_LDA_xy / np.max(FS_LDA_xy[:, :, 2])
    FS_LDA_xz = FS_LDA_xz / np.max(FS_LDA_xz[:, :, 2])
    FS_LDA_yz = FS_LDA_yz / np.max(FS_LDA_yz[:, :, 2])
    ###Weight distribution###
    FS_DMFT = (FS_DMFT_xz[:, :, 2] + FS_DMFT_yz[:, :, 2]) / 2 - FS_DMFT_xy[:, :, 2]
    FS_LDA = (FS_LDA_xz[:, :, 2] + FS_LDA_yz[:, :, 2]) / 2 - FS_LDA_xy[:, :, 2]
    ###Flip data###
    d1 = FS_DMFT
    d2 = np.fliplr(d1)
    d3 = np.flipud(d2)
    d4 = np.flipud(d1)
    l1 = FS_LDA
    l2 = np.fliplr(l1)
    l3 = np.flipud(l2)
    l4 = np.flipud(l1)
    ####Placeholders
    #DMFT = np.zeros((4 * m, 4 * m))
    #LDA = np.zeros((4 * m, 4 * m))
    #kx = np.linspace(-2, 2, 4 * m)
    #ky = np.linspace(-2, 2, 4 * m)
    ##Build data###
    #for i in range(m):
    #    for j in range(m):
    #        DMFT[i, j] = d1[i, j]
    #        DMFT[i, j + m] = d2[i, j]
    #        DMFT[i, j + 2 * m] = d1[i, j]
    #        DMFT[i, j + 3 * m] = d2[i, j]
    #        DMFT[i + m, j] = d4[i, j]
    #        DMFT[i + m, j + m] = d3[i, j]
    #        DMFT[i + m, j + 2 * m] = d4[i, j]
    #        DMFT[i + m, j + 3 * m] = d3[i, j]
    #        DMFT[i + 2 * m, j] = d1[i, j]
    #        DMFT[i + 2 * m, j + m] = d2[i, j]
    #        DMFT[i + 2 * m, j + 2 * m] = d1[i, j]
    #        DMFT[i + 2 * m, j + 3 * m] = d2[i, j]
    #        DMFT[i + 3 * m, j] = d4[i, j]
    #        DMFT[i + 3 * m, j + m] = d3[i, j]
    #        DMFT[i + 3 * m, j + 2 * m] = d4[i, j]
    #        DMFT[i + 3 * m, j + 3 * m] = d3[i, j]
    #        LDA[i, j] = l1[i, j]
    #        LDA[i, j + m] = l2[i, j]
    #        LDA[i, j + 2 * m] = l1[i, j]
    #        LDA[i, j + 3 * m] = l2[i, j]
    #        LDA[i + m, j] = l4[i, j]
    #        LDA[i + m, j + m] = l3[i, j]
    #        LDA[i + m, j + 2 * m] = l4[i, j]
    #        LDA[i + m, j + 3 * m] = l3[i, j]
    #        LDA[i + 2 * m, j] = l1[i, j]
    #        LDA[i + 2 * m, j + m] = l2[i, j]
    #        LDA[i + 2 * m, j + 2 * m] = l1[i, j]
    #        LDA[i + 2 * m, j + 3 * m] = l2[i, j]
    #        LDA[i + 3 * m, j] = l4[i, j]
    #        LDA[i + 3 * m, j + m] = l3[i, j]
    #        LDA[i + 3 * m, j + 2 * m] = l4[i, j]
    #        LDA[i + 3 * m, j + 3 * m] = l3[i, j]    
    ###Placeholders###
    DMFT = np.zeros((2 * m, 2 * m))
    LDA = np.zeros((2 * m, 2 * m))
    kx = np.linspace(-1, 1, 2 * m)
    ky = np.linspace(-1, 1, 2 * m)
    ###Build data###
    for i in range(m):
        for j in range(m):
            DMFT[i, j] = d3[i, j]
            DMFT[i, j + m] = d4[i, j]
            DMFT[i + m, j] = d2[i, j]
            DMFT[i + m, j + m] = d1[i, j]
            LDA[i, j] = l3[i, j]
            LDA[i, j + m] = l4[i, j]
            LDA[i + m, j] = l2[i, j]
            LDA[i + m, j + m] = l1[i, j]
    ###Plot data###
    v_d = np.max(DMFT)
    plt.figure(2015, figsize=(8, 8), clear=True)
    ax = plt.subplot(121)
    ax.set_position([.1, .3, .4, .4])
    plt.tick_params(direction='in', length=1.5, width=.5, colors='k') 
    plt.contourf(kx, ky, DMFT, 300, cmap='PuOr', vmin=-v_d, vmax=v_d)
    plt.text(-.05, -.05, r'$\Gamma$', fontsize=18, color='r')
    plt.text(.95, .95, 'S', fontsize=18, color='r')
    plt.text(-.06, .95, 'X', fontsize=18, color='r')
    plt.xticks(np.arange(-2, 2, 1), [])
    plt.yticks(np.arange(-2, 2, 1), [])
    plt.ylim(ymax=1, ymin=-1)
    plt.xlim(xmax=1, xmin=-1)
    plt.xlabel(r'$k_x \, (\pi / a)$', fontdict=font)
    plt.ylabel(r'$k_y \, (\pi / b)$', fontdict=font)
    pos = ax.get_position()
    cax = plt.axes([pos.x0+pos.width + 0.01 ,
                        pos.y0, 0.01, pos.height])
    cbar = plt.colorbar(cax = cax, ticks = None)
    cbar.set_ticks([])
    cbar.set_clim(np.min(DMFT), np.max(DMFT))
    plt.show()
    if print_fig == True:
        plt.savefig(
                '/Users/denyssutter/Documents/PhD/PhD_Denys/Figs/CSROfig15.png', 
                dpi = 600,bbox_inches="tight")
        
def CSROfig16(print_fig=True):
    """
    DMFT bandstructure calculation
    """    
    os.chdir('/Users/denyssutter/Documents/PhD/data')
    xz_data = np.loadtxt('DMFT_CSRO_xz.dat')
    yz_data = np.loadtxt('DMFT_CSRO_yz.dat')
    xy_data = np.loadtxt('DMFT_CSRO_xy.dat')
    os.chdir('/Users/denyssutter/Documents/library/Python/ARPES')
    #[0, 56, 110, 187, 241, 266, 325, 350]  = [G,X,S,G,Y,T,G,Z]
    xz = np.reshape(xz_data[:, 2], (351, 8000))
    yz = np.reshape(yz_data[:, 2], (351, 8000))
    xy = np.reshape(xy_data[:, 2], (351, 8000))
    DMFT = np.transpose(xz + yz - 2 * xy)
    en = np.linspace(-8, 8, 8000)
    k = np.linspace(0, 350, 351)
    GX = DMFT[:, :56]
    k_GX = k[:56]
    XS = DMFT[:, 56:110]
    k_XS = k[56:110]
    SG = DMFT[:, 110:187]
    k_SG = k[110:187]
    v_bnd = np.max(DMFT)
    k = (np.flipud(k_SG), np.flipud(k_XS), np.flipud(k_GX))
    bndstr = (SG, XS, GX)
    plt.figure('2016', figsize=(8, 8), clear=True)
    for i in range(len(bndstr)):
        if i != 0:
            ax = plt.subplot(3, 3, i + 1)
            ax.set_position([.1 + i * .15 + .15 * (np.sqrt(2) - 1), .2, .15 , .4])
            plt.tick_params(direction='in', length=1.5, width=.5, colors='k') 
        else:
            ax = plt.subplot(3, 3, i + 1)
            ax.set_position([.1 + i * .15 , .2, .15 * np.sqrt(2) , .4])
            plt.tick_params(direction='in', length=1.5, width=.5, colors='k') 
        plt.contourf(k[i], en, bndstr[i], 200, cmap='PuOr', vmax=v_bnd, vmin=-v_bnd)
        if i == 0:
            plt.xticks([110, 186], ('$\Gamma$', 'S')) 
            plt.yticks(np.arange(-1, 1, .2))
            plt.ylabel(r'$\omega$ (eV)', fontdict=font)
        elif i==1:
            plt.xticks([56, 109], ('', 'X'))
            plt.yticks(np.arange(-1, 1, .2), [])
        elif i==2:
            plt.xticks([0, 55], ('', '$\Gamma$'))
            plt.yticks(np.arange(-1, 1, .2), [])
        plt.plot([k[i][0], k[i][-1]], [0, 0], 'k:')    
        plt.ylim(ymax = .25, ymin=-.65)
    plt.show()
    if print_fig == True:
        plt.savefig(
                '/Users/denyssutter/Documents/PhD/PhD_Denys/Figs/CSROfig16.png', 
                dpi = 600,bbox_inches="tight")
        
def CSROfig17(print_fig=True):
    """
    LDA bandstructure calculation
    """    
    os.chdir('/Users/denyssutter/Documents/PhD/data')
    xz_data = np.loadtxt('LDA_CSRO_xz.dat')
    yz_data = np.loadtxt('LDA_CSRO_yz.dat')
    xy_data = np.loadtxt('LDA_CSRO_xy.dat')
    os.chdir('/Users/denyssutter/Documents/library/Python/ARPES')
    #[0, 56, 110, 187, 241, 266, 325, 350]  = [G,X,S,G,Y,T,G,Z]
    xz = np.reshape(xz_data[:, 2], (351, 8000))
    yz = np.reshape(yz_data[:, 2], (351, 8000))
    xy = np.reshape(xy_data[:, 2], (351, 8000))
    LDA = np.transpose(xz + yz - 2 * xy)
    en = np.linspace(-8, 8, 8000)
    k = np.linspace(0, 350, 351)
    GX = LDA[:, :56]
    k_GX = k[:56]
    XS = LDA[:, 56:110]
    k_XS = k[56:110]
    SG = LDA[:, 110:187]
    k_SG = k[110:187]
    v_bnd = np.max(LDA)
    k = (np.flipud(k_SG), np.flipud(k_XS), np.flipud(k_GX))
    bndstr = (SG, XS, GX)
    plt.figure(2017, figsize=(8, 8), clear=True)
    for i in range(len(bndstr)):
        if i != 0:
            ax = plt.subplot(3, 3, i + 1)
            ax.set_position([.1 + i * .15 + .15 * (np.sqrt(2) - 1), .2, .15 , .4])
            plt.tick_params(direction='in', length=1.5, width=.5, colors='k') 
        else:
            ax = plt.subplot(3, 3, i + 1)
            ax.set_position([.1 + i * .15 , .2, .15 * np.sqrt(2) , .4])
            plt.tick_params(direction='in', length=1.5, width=.5, colors='k') 
        plt.contourf(k[i], en, bndstr[i], 200, cmap='PuOr', vmax=v_bnd, vmin=-v_bnd)
        if i == 0:
            plt.xticks([110, 186], ('$\Gamma$', 'S')) 
            plt.yticks(np.arange(-5, 2, .5))
            plt.ylabel(r'$\omega$ (eV)', fontdict=font)
        elif i==1:
            plt.xticks([56, 109], ('', 'X'))
            plt.yticks(np.arange(-5, 2, .5), [])
        elif i==2:
            plt.xticks([0, 55], ('', '$\Gamma$'))
            plt.yticks(np.arange(-5, 2, .5), [])
        plt.plot([k[i][0], k[i][-1]], [0, 0], 'k:')    
        plt.ylim(ymax = 1, ymin=-2.5)
    plt.show()
    if print_fig == True:
        plt.savefig(
                '/Users/denyssutter/Documents/PhD/PhD_Denys/Figs/CSROfig17.png', 
                dpi = 600,bbox_inches="tight")
        
def CSROfig18(print_fig=True):
    """
    CSRO30 experimental band structure
    """     
    colmap = cm.ocean_r
    os.chdir('/Users/denyssutter/Documents/library/Python/ARPES')
    file = 62421
    gold = 62440
    mat = 'CSRO30'
    year = 2017
    sample = 'S13'
    D = ARPES.DLS(file, mat, year, sample)
    D.norm(gold)
    D.restrict(bot=0, top=1, left=.12, right=.9)
    D.ang2kFS(D.ang, Ekin=22-4.5, lat_unit=True, a=5.33, b=5.55, c=11, 
              V0=0, thdg=-9, tidg=9, phidg=-90)
    D.FS(e=0.005, ew=.02, norm=True)
    D.FS_flatten(ang=True)
    file = 62444
    gold = 62455
    A1 = ARPES.DLS(file, mat, year, sample)
    A1.norm(gold)
    A1.ang2k(A1.ang, Ekin=22-4.5, lat_unit=True, a=5.33, b=5.55, c=11, 
              V0=0, thdg=-8.3, tidg=0, phidg=-90)
    file = 62449
    gold = 62455
    A2 = ARPES.DLS(file, mat, year, sample)
    A2.norm(gold)
    A2.ang2k(A1.ang, Ekin=22-4.5, lat_unit=True, a=5.33, b=5.55, c=11, 
              V0=0, thdg=-9, tidg=16, phidg=-90)
    c = (0, 238 / 256, 118 / 256)
    def CSROfig18a():
        ax = plt.subplot(4, 4, 1) 
        ax.set_position([.08, .3, .2, .35])
        plt.tick_params(direction='in', length=1.5, width=.5, colors='k')
        plt.contourf(A1.en_norm, A1.kys, A1.int_norm, 300, cmap=colmap,
                     vmin=0 * np.max(A1.int_norm), vmax=.8 * np.max(A1.int_norm))
        plt.plot([0, 0], [np.min(A1.kys), np.max(A1.kys)], 'k:')
        plt.plot([-.005, -.005], [np.min(A1.kys), np.max(A1.kys)], linestyle='-.',
                 color=c, linewidth=.5) 
        plt.xticks(np.arange(-.08, .0, .02), ('-80', '-60', '-40', '-20', '0'))
        plt.yticks(np.arange(-5, 5, .5))
        plt.xlim(xmax=.005, xmin=-.08)
        plt.ylim(ymax=np.max(D.ky), ymin=np.min(D.ky))  
        plt.xlabel('$\omega\,(\mathrm{meV})$', fontdict = font)
        plt.ylabel('$k_x \,(\pi/a)$', fontdict = font)
        plt.text(-.075, .48, r'(a)', fontsize=12, color='C1')
        plt.text(-.002, -.03, r'$\Gamma$', fontsize=12, color='r')
        plt.text(-.002, -1.03, r'Y', fontsize=12, color='r')
    
    def CSROfig18c():
        ax = plt.subplot(4, 4, 15) 
        ax.set_position([.3 + .35 * (np.max(D.kx) - np.min(D.kx)) / (np.max(D.ky) - np.min(D.ky)), .3, .2, .35])
        plt.tick_params(direction='in', length=1.5, width=.5, colors='k')
        plt.contourf(-np.transpose(np.fliplr(A2.en_norm))-.002, np.transpose(A2.kys), 
                     np.transpose(np.fliplr(A2.int_norm)), 300, cmap=colmap,
                     vmin=0 * np.max(A2.int_norm), vmax=.8 * np.max(A2.int_norm))
        plt.plot([0, 0], [np.min(A2.kys), np.max(A2.kys)], 'k:')
        plt.xticks(np.arange(0, .1, .02), ('0', '-20', '-40', '-60', '-80'))
        plt.yticks(np.arange(-5, 5, .5), [])
        plt.xlim(xmin=-.005, xmax=.08)
        plt.ylim(ymax=np.max(D.ky), ymin=np.min(D.ky))  
        plt.xlabel('$\omega\,(\mathrm{meV})$', fontdict = font)
        plt.text(0.0, .48, r'(c)', fontdict = font)
        plt.text(-.002, -.03, r'X', fontsize=12, color='r')
        plt.text(-.002, -1.03, r'S', fontsize=12, color='r')
        pos = ax.get_position()
        cax = plt.axes([pos.x0+pos.width+0.01 ,
                            pos.y0, 0.01, pos.height])
        cbar = plt.colorbar(cax = cax, ticks = None)
        cbar.set_ticks([])
        cbar.set_clim(np.min(A2.int_norm), np.max(A2.int_norm))
    
    def CSROfig18b():
        ax = plt.subplot(4, 4, 2) 
        ax.set_position([.29, .3,
                         .35 * (np.max(D.kx) - np.min(D.kx)) / (np.max(D.ky) - np.min(D.ky)), .35])
        plt.tick_params(direction='in', length=1.2, width=.5, colors='k')  
        plt.contourf(D.kx, D.ky, np.flipud(D.map), 300, 
                     vmax=.9 * np.max(D.map), vmin=.1 * np.max(D.map), cmap=colmap)
        plt.xlabel('$k_y \,(\pi/a)$', fontdict = font)
        plt.text(-1.8, .48, r'(b)', fontdict=font)
        plt.text(-.05, -.03, r'$\Gamma$', fontsize=12, color='r')
        plt.text(-.05, -1.03, r'Y', fontsize=12, color='r')
        plt.text(.95, -.03, r'X', fontsize=12, color='r')
        plt.text(.95, -1.03, r'S', fontsize=12, color='r')
        plt.plot(A1.k[0], A1.k[1], linestyle='-.', color=c, linewidth=1)
        plt.plot(A2.k[0], A2.k[1], linestyle='-.', color=c, linewidth=1)
        plt.xticks(np.arange(-5, 5, .5))
        plt.yticks(np.arange(-5, 5, .5), [])
        plt.xlim(xmax=np.max(D.kx), xmin=np.min(D.kx))
        plt.ylim(ymax=np.max(D.ky), ymin=np.min(D.ky))     
        
    plt.figure(2018, figsize=(8, 8), clear=True)
    CSROfig18a()
    CSROfig18b()
    CSROfig18c()
    plt.show()
    if print_fig == True:
        plt.savefig(
                '/Users/denyssutter/Documents/PhD/PhD_Denys/Figs/CSROfig18.png', 
                dpi = 400,bbox_inches="tight")
        
def CSROfig19(print_fig=True):
    """
    CSRO30 Gamma - S cut epsilon pocket
    """    
    os.chdir('/Users/denyssutter/Documents/library/Python/ARPES')
    mat = 'CSRO30'
    year = 2017
    sample = 'S13'
    file = 62488
    gold = 62492
    
    D = ARPES.DLS(file, mat, year, sample)
    D.norm(gold)
    D.bkg(norm=True)
    D.ang2k(D.ang, Ekin=72-4.5, lat_unit=True, a=5.33, b=5.55, c=11, 
                  V0=0, thdg=-10, tidg=0, phidg=45)
    mdc_val = .00
    mdcw_val = .005
    mdc = np.zeros(D.ang.shape)
    for i in range(len(D.ang)):
        val, _mdc = utils.find(D.en_norm[i, :], mdc_val)
        val, _mdcw = utils.find(D.en_norm[i, :], mdc_val - mdcw_val)
        mdc[i] = np.sum(D.int_norm[i, _mdcw:_mdc])
    mdc = mdc / np.max(mdc)
    plt.figure(20019, figsize=(7, 5), clear=True)
    ###Fit MDC###
    delta = 1e5
    p_mdc_i = np.array(
                [-.4, .4, .65, .9, 1.25, 1.4, 1.7,
                 .05, .05, .05, .1, .1, .05, .05,
                 .4, .8, .3, .4, .5, .5, 1,
                 .06, -0.02, .02])
    bounds_bot = np.concatenate((p_mdc_i[0:-3] - np.inf, p_mdc_i[-3:] - delta))
    bounds_top = np.concatenate((p_mdc_i[0:-3] + np.inf, p_mdc_i[-3:] + delta))
    p_mdc_bounds = (bounds_bot, bounds_top)
    p_mdc, cov_mdc = curve_fit(
            utils.lor_7, D.k[1], mdc, p_mdc_i, bounds=p_mdc_bounds)
    b_mdc = utils.poly_2(D.k[1], p_mdc[-3], p_mdc[-2], p_mdc[-1])
    f_mdc = utils.lor_7(D.k[1], *p_mdc) - b_mdc
    f_mdc[0] = -.05
    f_mdc[-1] = -.05
    plt.plot(D.k[1], mdc, 'bo')
    plt.plot(D.k[1], f_mdc)
    plt.plot(D.k[1], b_mdc, 'k--')
    c = (0, 238 / 256, 118 / 256)
    plt.figure(2019, figsize=(6, 6), clear=True)
    ax = plt.subplot(1, 2, 1) 
    ax.set_position([.2, .3, .5, .5])
    plt.tick_params(direction='in', length=1.5, width=.5, colors='k')
    plt.contourf(D.kxs, D.en_norm+.002, D.int_norm, 300, cmap=cm.ocean_r,
                 vmin=0 * np.max(D.int_norm), vmax=.8 * np.max(D.int_norm))
    plt.plot([np.min(D.kxs), np.max(D.kxs)], [0, 0], 'k:')
    plt.plot(D.k[1], (mdc - b_mdc) / 25 + .005, 'o', markersize=1.5, color='C9')
    plt.fill(D.k[1], (f_mdc) / 25 + .005, alpha=.2, color='C9')
    plt.plot(D.k[1], (f_mdc) / 25 + .005, color=c, linewidth=.5)
    cols = ['m', 'm', 'k', 'r', 'r', 'k', 'm', 'C1']
    alphas = [0.5, 0.5, .5, 1, 1, .5, 0.5, 0.5, 0.5]
    lws = [0.5, 0.5, .5, 1, 1, .5, 0.5, 0.5, 0.5]
    p_mdc[6 + 16] *= 1.5
    for i in range(7):
        plt.plot(D.k[1], (utils.lor(D.k[1], p_mdc[i], p_mdc[i + 7], p_mdc[i + 14], 
                 p_mdc[-3]*0, p_mdc[-2]*0, p_mdc[-1]*0) - 0*b_mdc) / 25 + .002, 
                 lw=lws[i], color=cols[i], alpha=alphas[i])
    plt.yticks(np.arange(-.15, .1, .05), ('-150', '-100', '-50', '0', '50'))
    plt.xticks(np.arange(0, 3, 1), (r'$\Gamma$', 'S', r'$\Gamma$'))
    plt.ylim(ymax=.05, ymin=-.15)
    plt.xlim(xmax=np.max(D.kxs), xmin=np.min(D.kxs))  
    plt.ylabel('$\omega\,(\mathrm{meV})$', fontdict=font)
    plt.xlabel('$k_x \,(\pi/a)$', fontdict=font)
    plt.text(.87, .027, r'$\bar{\epsilon}$', fontsize=12, color='r')
    plt.text(1.2, .032, r'$\bar{\epsilon}$', fontsize=12, color='r')
    plt.show()
    if print_fig == True:
        plt.savefig(
                '/Users/denyssutter/Documents/PhD/PhD_Denys/Figs/CSROfig19.png', 
                dpi = 400,bbox_inches="tight")