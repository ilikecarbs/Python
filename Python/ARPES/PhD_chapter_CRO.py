#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 11:30:51 2018

@author: ilikecarbs

%%%%%%%%%%%%%%%%%%%%%
   PhD_chapter_CRO
%%%%%%%%%%%%%%%%%%%%%

**Thesis figures Ca2RuO4**

.. note::
        To-Do:
            -
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import linalg as la
import matplotlib.cm as cm
from scipy.stats import exponnorm
from scipy.optimize import curve_fit

import ARPES_header as ARPES
import ARPES_utils as utils


# Set standard fonts
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.serif'] = ['Computer Modern Roman']
plt.rc('font', **{'family': 'serif', 'serif': ['STIXGeneral']})

# Dictionaries
font = {'family': 'serif',
        'style': 'normal',
        'color': 'black',
        'weight': 'ultralight',
        'size': 12,
        }

kwargs_ex = {'cmap': cm.ocean_r}  # Experimental plots
kwargs_th = {'cmap': cm.bone_r}  # Theory plots
kwargs_ticks = {'bottom': True,
                'top': True,
                'left': True,
                'right': True,
                'direction': 'in',
                'length': 1.5,
                'width': .5,
                'colors': 'black'}
kwargs_cut = {'linestyle': '-.',
              'color': 'red',
              'lw': .5}
kwargs_ef = {'linestyle': ':',
             'color': 'k',
             'lw': 1}

# Directory paths
save_dir = '/Users/denyssutter/Documents/PhD/PhD_Denys/Figs/'
data_dir = '/Users/denyssutter/Documents/PhD/data/'
home_dir = '/Users/denyssutter/Documents/library/Python/ARPES'


def CRO_theory_plot(k_pts, data_en, data, v_max, figname):
    """Plots theory data for figs 1-4

    **Used to plot DFT data with different self energies**

    Args
    ----
    :kpts:      k points as tuples
    :data_en:   energy axis of data
    :data:      DFT data
    :v_max:     contrast 0..1 of contourf
    :fignr:     figure number

    Return
    ------
    Plots DFT data

    """
    scale = .02  # to scale k-axis

    fig = plt.figure(figname, figsize=(10, 10), clear=True)

    # labels
    lbls = (['S', r'$\Gamma$', 'S'], ['', 'X', 'S'], ['', r'$\Gamma$'],
            ['', 'X', r'$\Gamma$', 'X'])
    # looping over segments of k-path
    for k in range(len(data)):
        c = len(data[k])
        m, n = data[k][0].shape

        # Placeholders
        data_kpath = np.zeros((1, c * n))
        data_spec = np.zeros((m, c * n))

        # Used to mark k-points along in k-path
        k_seg = [0]
        for i in range(c):
            # Distances in k-space
            diff = abs(np.subtract(k_pts[k][i], k_pts[k][i+1]))

            # extending list cummulative and building data
            k_seg.append(k_seg[i] + la.norm(diff))
            data_spec[:, n*i:n*(i+1)] = data[k][i]

        # Full k-path for plotting
        data_kpath = np.linspace(0, k_seg[-1], c * n)

        # Subplot sensitive formatting
        if k == 0:
            ax = fig.add_subplot(1, len(data), k+1)
            ax.set_position([.1, .3, k_seg[-1] * scale, .3])

            # Position for next iteration
            pos = ax.get_position()
            k_prev = k_seg[-1]

        else:
            ax = fig.add_subplot(1, len(data), k+1)
            ax.set_position([pos.x0 + k_prev * scale, pos.y0,
                             k_seg[-1] * scale, pos.height])

        # Plot data
        ax.tick_params(**kwargs_ticks)
        c0 = ax.contourf(data_kpath, data_en, data_spec, 300, **kwargs_th,
                         vmin=0, vmax=v_max*np.max(data_spec), zorder=.1)
        ax.set_rasterization_zorder(0.2)
        ax.set_ylim(-2.5, 0)

        # For formatting subplot axis
        k_prev = k_seg[-1]
        pos = ax.get_position()

        # Labels
        ax.set_xticks(k_seg)
        ax.set_xticklabels(lbls[k], fontdict=font)
        if k == 0:
            ax.set_ylabel(r'$\omega$ (eV)', fontdict=font)
        else:
            ax.set_yticklabels([])

    # colorbar
    cax = plt.axes([pos.x0 + k_prev * scale + 0.01,
                    pos.y0, 0.01, pos.height])
    cbar = plt.colorbar(c0, cax=cax, ticks=None)
    cbar.set_ticks([])
    cbar.set_clim(np.min(data_spec), np.max(data_spec))


def CRO_FS_plot(e, v_min, figname):
    """Plots constant energy maps for figs. 12-14

    **Used to plot DFT data with different self energies**

    Args
    ----
    :e:         energy for constant energy map
    :v_min:     change contrast 0..1
    :figname:   figure name

    Return
    ------
    Plots constant energy maps

    """

    # data files used
    p65 = '0618_00113'
    s65 = '0618_00114'
    p120 = '0618_00115'
    s120 = '0618_00116'
    mat = 'Ca2RuO4'
    year = '2016'
    sample = 'data'
    files = [p120, s120, p65, s65]

    # labels
    lbls1 = ['(a)', '(b)', '(c)', '(d)']
    lbls2 = [r'$120\,\mathrm{eV}$', r'$120\,\mathrm{eV}$',
             r'$65\,\mathrm{eV}$', r'$65\,\mathrm{eV}$']
    lbls3 = [r'$\bar{\pi}$-pol.', r'$\bar{\sigma}$-pol.',
             r'$\bar{\pi}$-pol.', r'$\bar{\sigma}$-pol.']

    # Creating figure
    fig = plt.figure(figname, figsize=(10, 10), clear=True)
    for i in range(4):
        D = ARPES.ALS(files[i], mat, year, sample)
        D.ang2kFS(D.ang, Ekin=D.hv-4.5+e, lat_unit=True, a=4.8, b=5.7, c=11,
                  V0=0, thdg=25, tidg=-.5, phidg=-25)

        # generate FS map, energy off set (Fermi level not specified)
        D.FS(e=e+2.1, ew=0.1)
        FSmap = D.map

        # set subplots
        ax = fig.add_subplot(1, 5, i+2)
        ax.set_position([.06+(i*.23), .3, .22, .3])
        if i == 2:
            ax.set_position([.06 + (i * .23), .3, .16, .3])
        elif i == 3:
            ax.set_position([.06 + (2 * .23) + .17, .3, .16, .3])
        ax.tick_params(**kwargs_ticks)

        # Plot constant energy map
        c0 = ax.contourf(D.kx, D.ky, D.map, 300, **kwargs_ex,
                         vmin=v_min * np.max(D.map), zorder=.1,
                         vmax=.95 * np.max(D.map))
        ax.set_rasterization_zorder(.2)
        ax.grid(alpha=0.5)
        ax.set_xticks(np.arange(-10, 10, 2))
        ax.set_xlabel(r'$k_x$ ($\pi/a$)', fontdict=font)

        # kwarg dictionaries
        ortho = {'linestyle': '-', 'color': 'black', 'lw': 1}
        tetra = {'linestyle': '--', 'color': 'black', 'lw': .5}

        # Plot Brillouin zones
        ax.plot([-1, -1], [-1, 1], **ortho)
        ax.plot([1, 1], [-1, 1], **ortho)
        ax.plot([-1, 1], [1, 1], **ortho)
        ax.plot([-1, 1], [-1, -1], **ortho)
        ax.plot([-2, 0], [0, 2], **tetra)
        ax.plot([-2, 0], [0, -2], **tetra)
        ax.plot([2, 0], [0, 2], **tetra)
        ax.plot([2, 0], [0, -2], **tetra)

        # labels and plot ARPES cut path
        if i == 0:
            ax.set_ylabel(r'$k_y$ ($\pi/a$)', fontdict=font)
            ax.set_yticks(np.arange(-10, 10, 2))
#            ax.plot([-1, 1], [-1, 1], **kwargs_cut)
#            ax.plot([-1, 1], [1, 1], **kwargs_cut)
#            ax.plot([-1, 0], [1, 2], **kwargs_cut)
#            ax.plot([0, 0], [2, -1], **kwargs_cut)
#            ax.arrow(-1, -1, .3, .3, head_width=0.3, head_length=0.3,
#                     fc='turquoise', ec='k')
#            ax.arrow(0, -.4, 0, -.3, head_width=0.3, head_length=0.3,
#                     fc='turquoise', ec='k')
        else:
            ax.set_yticks(np.arange(-10, 10, 2))
            ax.set_yticklabels([])

        # some additional text
        if any(x == i for x in [0, 1]):
            x_pos = -2.7
        else:
            x_pos = -1.9
        ax.text(x_pos, 5.6, lbls1[i], fontsize=12)
        ax.text(x_pos, 5.0, lbls2[i], fontsize=10)
        ax.text(x_pos, 4.4, lbls3[i], fontsize=10)
        ax.text(-0.2, -0.15, r'$\Gamma$',
                fontsize=12, color='r')
        ax.text(-0.2, 1.85, r'$\Gamma$',
                fontsize=12, color='r')
        ax.text(.85, .85, r'S',
                fontsize=12, color='r')
        ax.text(-0.2, .9, r'X',
                fontsize=12, color='r')
        ax.set_xlim(xmin=-3, xmax=4)
        if any(x == i for x in [2, 3]):
            ax.set_xlim(xmin=-2.2, xmax=2.9)
        ax.set_ylim(ymin=-3.3, ymax=6.2)

    # colorbar
    pos = ax.get_position()
    cax = plt.axes([pos.x0+pos.width+0.01,
                    pos.y0, 0.01, pos.height])
    cbar = plt.colorbar(c0, cax=cax, ticks=None)
    cbar.set_ticks([])
    cbar.set_clim(np.min(FSmap), np.max(FSmap))


def fig1(print_fig=True):
    """figure 1

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    DFT plot: figure 3 of Nature Comm.
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    """

    figname = 'CROfig1'

    os.chdir(data_dir)
    GS = pd.read_csv('DFT_CRO_GS_final.dat').values
    SG = np.fliplr(GS)
    GX = pd.read_csv('DFT_CRO_GX_final.dat').values
    XG = np.fliplr(GX)
    XS = pd.read_csv('DFT_CRO_YS_final.dat').values
    SX = np.fliplr(XS)
    os.chdir(home_dir)

    # k-points
    G = (0, 0)
    X = (np.pi, 0)
    S = (np.pi, np.pi)

    # Data along path in k-space
    k_pts = np.array([[S, G, S], [S, X, S], [S, G], [G, X, G, X]])
    DFT = np.array([[SG, GS], [SX, XS], [SG], [GX, XG, GX]])
    DFT_en = np.linspace(-2.5, 0, 500)

    # Plot data
    CRO_theory_plot(k_pts, DFT_en, DFT, v_max=1, figname=figname)
    plt.show()

    # Save figure
    if print_fig:
        plt.savefig(save_dir + figname + '.pdf', dpi=100,
                    bbox_inches="tight", rasterized=True)


def fig2(print_fig=True):
    """figure 2

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    DMFT plot: figure 3 of Nature Comm.
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    """

    figname = 'CROfig2'

    os.chdir(data_dir)
    xz_data = np.loadtxt('DMFT_CRO_xz.dat')
    yz_data = np.loadtxt('DMFT_CRO_yz.dat')
    xy_data = np.loadtxt('DMFT_CRO_xy.dat')
    os.chdir(home_dir)

    m, n = 8000, 351  # Dimensions energy, full k-path
    bot, top = 2500, 5000  # restrict energy window
    DMFT_data = np.array([xz_data, yz_data, xy_data])  # combine data
    DMFT_spec = np.reshape(DMFT_data[:, :, 2], (3, n, m))  # reshape into n,m
    DMFT_spec = DMFT_spec[:, :, bot:top]  # restrict data to bot, top
    DMFT_en = np.linspace(-8, 8, m)  # define energy data
    DMFT_en = DMFT_en[bot:top]  # restrict energy data

    # Full k-path:
    # [0, 56, 110, 187, 241, 266, 325, 350]  = [G, X, S, G, Y, T, G, Z]
    DMFT_spec = np.transpose(DMFT_spec, (0, 2, 1))  # transpose
    DMFT_spec = np.sum(DMFT_spec, axis=0)  # sum up over orbitals

    # Data used for plot:
    GX = DMFT_spec[:, 0:56]
    XG = np.fliplr(GX)
    XS = DMFT_spec[:, 56:110]
    SX = np.fliplr(XS)
    SG = DMFT_spec[:, 110:187]
    GS = np.fliplr(SG)

    # k-points
    G = (0, 0)
    X = (np.pi, 0)
    S = (np.pi, np.pi)

    # Data along path in k-space
    k_pts = np.array([[S, G, S], [S, X, S], [S, G], [G, X, G, X]])
    DMFT = np.array([[SG, GS], [SX, XS], [SG], [GX, XG, GX]])

    # Plot data
    CRO_theory_plot(k_pts, DMFT_en, DMFT, v_max=.5, figname=figname)
    plt.show()

    # Save figure
    if print_fig:
        plt.savefig(save_dir + figname + '.pdf', dpi=100,
                    bbox_inches="tight", rasterized=True)


def fig3(print_fig=True):
    """figure 3

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    DFT plot: orbitally selective Mott scenario
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    """

    figname = 'CROfig3'

    os.chdir(data_dir)
    GS = pd.read_csv('DFT_CRO_GS_OSMT.dat').values
    SG = np.fliplr(GS)
    GX = pd.read_csv('DFT_CRO_GX_OSMT.dat').values
    XG = np.fliplr(GX)
    XS = pd.read_csv('DFT_CRO_YS_OSMT.dat').values
    SX = np.fliplr(XS)
    os.chdir(home_dir)

    # k-points
    G = (0, 0)
    X = (np.pi, 0)
    S = (np.pi, np.pi)

    # Data along path in k-space
    k_pts = np.array([[S, G, S], [S, X, S], [S, G], [G, X, G, X]])
    DFT = np.array([[SG, GS], [SX, XS], [SG], [GX, XG, GX]])
    DFT_en = np.linspace(-2.5, 0, 500)

    # Plot data
    CRO_theory_plot(k_pts, DFT_en, DFT, v_max=1, figname=figname)
    plt.show()

    # Save figure
    if print_fig:
        plt.savefig(save_dir + figname + '.pdf', dpi=100,
                    bbox_inches="tight", rasterized=True)


def fig4(print_fig=True):
    """figure 4

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    DFT plot: uniform gap scnenario
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    """

    figname = 'CROfig4'

    os.chdir(data_dir)
    GS = pd.read_csv('DFT_CRO_GS_uni.dat').values
    SG = np.fliplr(GS)
    GX = pd.read_csv('DFT_CRO_GX_uni.dat').values
    XG = np.fliplr(GX)
    XS = pd.read_csv('DFT_CRO_YS_uni.dat').values
    SX = np.fliplr(XS)
    os.chdir(home_dir)

    # k-points
    G = (0, 0)
    X = (np.pi, 0)
    S = (np.pi, np.pi)

    # Data along path in k-space
    k_pts = np.array([[S, G, S], [S, X, S], [S, G], [G, X, G, X]])
    DFT = np.array([[SG, GS], [SX, XS], [SG], [GX, XG, GX]])
    DFT_en = np.linspace(-2.5, 0, 500)

    # Plot data
    CRO_theory_plot(k_pts, DFT_en, DFT, v_max=1, figname=figname)
    plt.show()

    # Save figure
    if print_fig:
        plt.savefig(save_dir + figname + '.pdf', dpi=100,
                    bbox_inches="tight", rasterized=True)


def fig5(print_fig=True):
    """figure 5

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Experimental Data of Nature Comm.
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    """

    figname = 'CROfig5'

    mat = 'Ca2RuO4'
    year = '2016'
    sample = 'T10'
    files = np.array([47974, 48048, 47993, 48028])
    gold = 48000

    # Setting which axes should be ticked and labelled
    scale = .02
    v_scale = 1.3
    k_seg_1 = np.array([0, np.pi*np.sqrt(2), 2*np.pi*np.sqrt(2)])
    k_seg_2 = np.array([0, np.pi, 2*np.pi])
    k_seg_3 = np.array([0, np.pi*np.sqrt(2)])
    k_seg_4 = np.array([0, np.pi, 2*np.pi, 3*np.pi])
    k_segs = (k_seg_1, k_seg_2, k_seg_3, k_seg_4)

    fig = plt.figure(figname, figsize=(10, 10), clear=True)

    # labels and ticks
    lbls = (['S', r'$\Gamma$', 'S'], ['', 'X', 'S'],
            ['', r'$\Gamma$'], ['', 'X', r'$\Gamma$', 'X'])
    x_lims = ([-1, 1], [-1, 0], [0, 1], [0, 1.5])
    x_ticks = (1, .5, 1, .5)

    # Manipulator angles for k-space transformation
    th = (-4, -7.5, 5, -9.5)
    ti = (0, 8.5, 12.5, 0)
    phi = (0, 45, 0, 45)

    n = 0  # counter
    for file in files:
        # Load data
        D = ARPES.DLS(file, mat, year, sample)
        D.norm(gold)
        D.restrict(bot=.6, top=1, left=0, right=1)
        D.flatten()
        D.ang2k(D.ang, Ekin=D.hv-4.5, lat_unit=True, a=3.89, b=3.89, c=11,
                V0=0, thdg=th[n], tidg=ti[n], phidg=phi[n])

        # Create figure
        ax = fig.add_subplot(1, 4, n+1)
        ax.tick_params(**kwargs_ticks)

        # axes positions and labels
        if n == 0:
            ax.set_position([.1, .3, k_segs[n][-1]*scale, .3])
            ax.set_ylabel(r'$\omega$ (eV)', fontdict=font)
            pos = ax.get_position()  # pos for next iteration
        else:
            ax.set_position([pos.x0 + k_segs[n-1][-1] * scale, pos.y0,
                             k_segs[n][-1] * scale, pos.height])
            ax.set_yticklabels([])

        # Plot data
        if n == 0:
            ax.contourf(D.kxs, D.en_norm+.1, D.int_norm, 300,
                        **kwargs_ex, zorder=.1,
                        vmin=v_scale*0.01*np.max(D.int_norm),
                        vmax=v_scale*0.5*np.max(D.int_norm))
            ax.text(-.93, -.16, '(a)', color='k', fontsize=12)

        elif n == 1:
            ax.contourf(D.kxs, D.en_norm+.1, D.int_norm, 300,
                        **kwargs_ex, zorder=.1,
                        vmin=v_scale*0.0*np.max(D.int_norm),
                        vmax=v_scale*0.54*np.max(D.int_norm))

        elif n == 2:
            ax.contourf(D.kxs, D.en_norm+.1, np.flipud(D.int_norm), 300,
                        **kwargs_ex, zorder=.1,
                        vmin=v_scale * 0.01 * np.max(D.int_norm),
                        vmax=v_scale * 0.7 * np.max(D.int_norm))

        elif n == 3:
            c0 = ax.contourf(D.kxs, D.en_norm+.1, np.flipud(D.int_norm), 300,
                             **kwargs_ex, zorder=.1,
                             vmin=v_scale * 0.01 * np.max(D.int_norm),
                             vmax=v_scale * 0.53 * np.max(D.int_norm))
        ax.set_rasterization_zorder(0.2)
        ax.plot([np.min(D.kxs), np.max(D.kxs)], [-2.4, -2.4], **kwargs_cut)
        # decorate axes
        ax.set_xticks(np.arange(x_lims[n][0], x_lims[n][1] + .5,
                                x_ticks[n]))
        ax.set_xticklabels(lbls[n], fontdict=font)
        ax.set_xlim(x_lims[n])
        pos = ax.get_position()
        ax.set_yticks(np.arange(-2.5, 0.5, .5), [])
        ax.set_ylim(-2.5, 0)
        n += 1  # counter

    # colorbar
    cax = plt.axes([pos.x0+pos.width+0.01,
                    pos.y0, 0.01, pos.height])

    cbar = plt.colorbar(c0, cax=cax, ticks=None)
    cbar.set_ticks([])
    cbar.set_clim(np.min(D.int_norm), np.max(D.int_norm))

    file1 = '0619_00161'
    file2 = '0619_00162'
    mat = 'Ca2RuO4'
    year = '2016'
    sample = 'data'
    th = 20
    ti = -2
    phi = 21
    a = 5.5

    D1 = ARPES.ALS(file1, mat, year, sample)  # frist scan
    D2 = ARPES.ALS(file2, mat, year, sample)  # second scan
    D1.ang2kFS(D1.ang, Ekin=D1.hv-4.5-4.7, lat_unit=True, a=a, b=a, c=11,
               V0=0, thdg=th, tidg=ti, phidg=phi)
    D2.ang2kFS(D2.ang, Ekin=D2.hv-4.5-4.7, lat_unit=True, a=a, b=a, c=11,
               V0=0, thdg=th, tidg=ti, phidg=phi)

    # Combining two scans
    data = np.concatenate((D1.int, D2.int), axis=0)
    kx = np.concatenate((D1.kx, D2.kx), axis=0)
    ky = np.concatenate((D1.ky, D2.ky), axis=0)

    # energy off set (Fermi level not specified)
    en = D1.en - 2.3
    e = -2.2
    ew = 0.2
    e_val, e_idx = utils.find(en, e)
    ew_val, ew_idx = utils.find(en, e-ew)
    FS = np.sum(data[:, :, ew_idx:e_idx], axis=2)

    ax = fig.add_axes([.68+.04, .4375, .5/4, .65/4])
    ax.tick_params(**kwargs_ticks)

    # Plot data
    ax.contourf(kx, ky, FS, 300, **kwargs_ex, zorder=.1,
                vmin=.55*np.max(FS), vmax=.95*np.max(FS))
    ax.set_rasterization_zorder(0.2)

    # Labels
    ax.grid(alpha=0.3)
    ax.set_xticks(np.arange(-10, 10, 1))
    ax.set_yticks(np.arange(-10, 10, 1))
    ax.set_xlim(-1.2, 2.2)
    ax.set_ylim(-1.2, 3.2)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    pos = ax.get_position()

    ax.text(-1.05, 2.7, '(b)', color='w', fontsize=12)
    # kwarg dictionaries
    ortho = {'linestyle': '-', 'color': 'black', 'lw': 1}

    # Plot Brillouin zone
    ax.plot([-1, -1], [-1, 1], **ortho, zorder=1)
    ax.plot([1, 1], [-1, 1], **ortho, zorder=1)
    ax.plot([-1, 1], [1, 1], **ortho, zorder=1)
    ax.plot([-1, 1], [-1, -1], **ortho, zorder=1)

    # Plot ARPES cut path
    ax.plot([-1, 1], [-1, 1], 'r-', zorder=2)
    ax.plot([-1, 1], [1, 1], 'r-', zorder=2)
    ax.plot([-1, 0], [1, 2], 'r-', zorder=2)
    ax.plot([0, 0], [2, -1], 'r-', zorder=2)
    ax.arrow(-.8, -.8, .001, .001, head_width=0.17, head_length=0.2,
             fc='k', ec='k', zorder=3)
    ax.arrow(0, -.7, 0, -.001, head_width=0.17, head_length=0.2,
             fc='k', ec='k', zorder=3)

    # High symmetry points
    font_HS = {'fontsize': 12, 'color': 'w'}
    ax.text(-0.1, -0.1, r'$\Gamma$', fontdict=font_HS)
    ax.text(-0.1, 1.9, r'$\Gamma$', fontdict=font_HS)
    ax.text(.9, .9, r'S', fontdict=font_HS)
    ax.text(-0.1, .9, r'X', fontdict=font_HS)

    # Save figure
    if print_fig:
        plt.savefig(save_dir + figname + '.pdf', dpi=100,
                    bbox_inches="tight", rasterized=True)


def fig6(print_fig=True):
    """figure 6

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Constant energy map CaRuO4 of alpha branch
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    """

    figname = 'CROfig6'

    file1 = '0619_00161'
    file2 = '0619_00162'
    mat = 'Ca2RuO4'
    year = '2016'
    sample = 'data'
    th = 20
    ti = -2
    phi = 21
    a = 5.5

    D1 = ARPES.ALS(file1, mat, year, sample)  # frist scan
    D2 = ARPES.ALS(file2, mat, year, sample)  # second scan
    D1.ang2kFS(D1.ang, Ekin=D1.hv-4.5-4.7, lat_unit=True, a=a, b=a, c=11,
               V0=0, thdg=th, tidg=ti, phidg=phi)
    D2.ang2kFS(D2.ang, Ekin=D2.hv-4.5-4.7, lat_unit=True, a=a, b=a, c=11,
               V0=0, thdg=th, tidg=ti, phidg=phi)

    # Combining two scans
    data = np.concatenate((D1.int, D2.int), axis=0)
    kx = np.concatenate((D1.kx, D2.kx), axis=0)
    ky = np.concatenate((D1.ky, D2.ky), axis=0)

    # energy off set (Fermi level not specified)
    en = D1.en - 2.3
    e = -2.2
    ew = 0.2
    e_val, e_idx = utils.find(en, e)
    ew_val, ew_idx = utils.find(en, e-ew)
    FS = np.sum(data[:, :, ew_idx:e_idx], axis=2)

    fig = plt.figure(figname, figsize=(3, 3), clear=True)
    ax = fig.add_axes([.2, .2, .5, .65])
    ax.tick_params(**kwargs_ticks)

    # Plot data
    c0 = ax.contourf(kx, ky, FS, 300, **kwargs_ex, zorder=.1,
                     vmin=.55*np.max(FS), vmax=.95*np.max(FS))
    ax.set_rasterization_zorder(.2)

    # Labels
    ax.set_xlabel(r'$k_x$', fontdict=font)
    ax.set_ylabel(r'$k_y$', fontdict=font)
    ax.grid(alpha=0.3)
    ax.set_xticks(np.arange(-10, 10, 1))
    ax.set_yticks(np.arange(-10, 10, 1))
    ax.set_xlim(-1.2, 2.2)
    ax.set_ylim(-1.2, 3.2)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    pos = ax.get_position()

    # kwarg dictionaries
    ortho = {'linestyle': '-', 'color': 'black', 'lw': 1}

    # Plot Brillouin zone
    ax.plot([-1, -1], [-1, 1], **ortho, zorder=1)
    ax.plot([1, 1], [-1, 1], **ortho, zorder=1)
    ax.plot([-1, 1], [1, 1], **ortho, zorder=1)
    ax.plot([-1, 1], [-1, -1], **ortho, zorder=1)

    # Plot ARPES cut path
    ax.plot([-1, 1], [-1, 1], 'r-', zorder=2)
    ax.plot([-1, 1], [1, 1], 'r-', zorder=2)
    ax.plot([-1, 0], [1, 2], 'r-', zorder=2)
    ax.plot([0, 0], [2, -1], 'r-', zorder=2)
    ax.arrow(-.8, -.8, .001, .001, head_width=0.17, head_length=0.2,
             fc='k', ec='k', zorder=3)
    ax.arrow(0, -.7, 0, -.001, head_width=0.17, head_length=0.2,
             fc='k', ec='k', zorder=3)

    # High symmetry points
    font_HS = {'fontsize': 12, 'color': 'w'}
    ax.text(-0.1, -0.1, r'$\Gamma$', fontdict=font_HS)
    ax.text(-0.1, 1.9, r'$\Gamma$', fontdict=font_HS)
    ax.text(.9, .9, r'S', fontdict=font_HS)
    ax.text(-0.1, .9, r'X', fontdict=font_HS)

    # colorbar
    cax = plt.axes([pos.x0+pos.width+0.02,
                    pos.y0, 0.02, pos.height])
    cbar = plt.colorbar(c0, cax=cax, ticks=None)
    cbar.set_ticks([])
    cbar.set_clim(np.min(FS), np.max(FS))

    plt.show()

    # Save figure
    if print_fig:
        plt.savefig(save_dir + figname + '.pdf', dpi=100,
                    bbox_inches="tight", rasterized=True)


def fig7(print_fig=True):
    """figure 7

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Photon energy dependence Ca2RuO4: figure 2 of Nature Comm.
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    """

    figname = 'CROfig7'

    file = 'CRO_SIS_0048'
    mat = 'Ca2RuO4'
    year = '2015'
    sample = 'data'
    D = ARPES.SIS(file, mat, year, sample)
    D.ang2k(D.ang, Ekin=65-4.5, lat_unit=True, a=5.5, b=5.5, c=11,
            V0=0, thdg=-4, tidg=0, phidg=45)

    # Data at particular photon energies
    int1 = D.int[11, :, :]
    # Flux of beamline decreases, normalized to high energy tail
    int2 = D.int[16, :, :] * 3.9

    edc_ = 1
    mdc_ = -2.2
    mdcw_ = .1
    edc_val, edc_idx = utils.find(D.k[0], edc_)
    mdc_val, mdc_idx = utils.find(D.en, mdc_)
    mdcw_val, mdcw_idx = utils.find(D.en, mdc_ - mdcw_)
    edc1 = int1[edc_idx, :]
    edc2 = int2[edc_idx, :]
    mdc = np.sum(int1[:, mdcw_idx:mdc_idx], axis=1)
    mdc = mdc / np.max(mdc)

    plt.figure('MDC', figsize=(4, 4), clear=True)

    # Fit MDC
    delta = 1e-5
    p_mdc_i = [-.3, .35, .1, .1, 1, 1, .66, 0.02, -.0]
    p_mdc_bounds = ([-.3, .2, 0, 0, 0, 0,
                     p_mdc_i[-3]-delta, p_mdc_i[-2]-delta, p_mdc_i[-1]-delta],
                    [-.2, .5, .12, .12, np.inf, np.inf,
                     p_mdc_i[-3]+delta, p_mdc_i[-2]+delta, p_mdc_i[-1]+delta])
    p_mdc, c_mdc = curve_fit(
            utils.gauss_2, D.k[0], mdc, p_mdc_i, bounds=p_mdc_bounds)
    b_mdc = utils.poly_2(D.k[0], *p_mdc[-3:])
    f_mdc = utils.gauss_2(D.k[0], *p_mdc)

    plt.plot(D.k[0], mdc, 'o')
    plt.plot(D.k[0], f_mdc)
    plt.plot(D.k[0], b_mdc, 'k--')

    # Figure panels
    def fig7a():
        ax = fig.add_subplot(131)
        ax.set_position([.1, .3, .2, .3])
        ax.tick_params(**kwargs_ticks)

        # Plot data
        ax.contourf(D.k[0], D.en, np.transpose(int1), 300, **kwargs_ex,
                    vmin=0, vmax=1.4e4, zorder=.1)
        ax.set_rasterization_zorder(.2)
        ax.plot([D.k[0][0], D.k[0][-1]], [0, 0], **kwargs_ef)

        # Plot distribution cuts
        ax.plot([D.k[0][0], D.k[0][-1]],
                [mdc_-mdcw_/2, mdc_-mdcw_/2], **kwargs_cut)
        ax.plot([edc_, edc_], [-2.5, .5], **kwargs_cut)

        # Plot MDC
        ax.plot(D.k[0], (mdc - b_mdc) * 1.1, 'o', ms=1, c='C9')
        ax.fill(D.k[0], (f_mdc - b_mdc) * 1., alpha=.2, c='C9')

        # decorate axes
        ax.set_ylabel(r'$\omega$ (eV)', fontdict=font)
        ax.set_xticks([-1, 0, 1])
        ax.set_xticklabels(['S', r'$\Gamma$', 'S'])
        ax.set_yticks(np.arange(-2.5, 1, .5))
        # ax.set_xlim(-1, 1.66)
        ax.set_ylim(-2.5, .5)

        # add text
        ax.text(-1.02, 0.33, r'(a) $63\,$eV', fontdict=font)
        ax.text(.22, .1, r'$\mathcal{C}$', fontsize=15)

    def fig7b():
        ax = fig.add_subplot(132)
        ax.set_position([.31, .3, .2, .3])
        ax.tick_params(**kwargs_ticks)

        # Plot data
        c0 = ax.contourf(D.k[0], D.en+.07, np.transpose(int2), 300,
                         **kwargs_ex, vmin=0, vmax=1.4e4, zorder=.1)
        ax.set_rasterization_zorder(.2)

        # Plot distribution cuts
        ax.plot([D.k[0][0], D.k[0][-1]], [0, 0], **kwargs_ef)
        ax.plot([edc_, edc_], [-2.5, .5], **kwargs_cut)

        # decorate axes
        ax.set_xticks([-1, 0, 1])
        ax.set_xticklabels(['S', r'$\Gamma$', 'S'])
        ax.set_yticks(np.arange(-2.5, 1, .5))
        ax.set_yticklabels([])
        # ax.set_xlim(-1, 1.66)
        ax.set_ylim(-2.5, .5)

        # add text
        ax.text(-1.02, 0.33, r'(b) $78\,$eV', fontdict=font)

        # colorbar
        pos = ax.get_position()
        cax = plt.axes([pos.x0+pos.width+0.01,
                        pos.y0, 0.01, pos.height])
        cbar = plt.colorbar(c0, cax=cax, ticks=None)
        cbar.set_ticks([])
        cbar.set_clim(np.min(D.int), np.max(D.int))

    def fig7c():
        ax = plt.subplot(133)
        ax.set_position([.55, .3, .2, .3])
        ax.tick_params(**kwargs_ticks)

        # Plot EDC's
        xx = np.linspace(1, -5, 200)
        ax.plot(edc1, D.en, 'o', ms=3, c=(0, 0, .8))
        ax.plot(edc2, D.en, 'd', ms=3, c='C0')
        ax.fill(7.4e3 * exponnorm.pdf(-xx, K=2, loc=.63, scale=.2), xx,
                alpha=.2, fc=(0, 0, .8), zorder=.1)
        ax.fill(1.3e4 * exponnorm.pdf(-xx, K=2, loc=1.34, scale=.28), xx,
                alpha=.2, fc='C0', zorder=.1)

        # Plot Mott gap estimate
        ax.fill_between([0, 1.5e4], 0, -.2, color='C3', alpha=0.2, zorder=.1)
        ax.plot([0, 1.5e4], [0, 0], **kwargs_ef)
        ax.plot([0, 1.5e4], [-.2, -.2], 'k:', linewidth=.2)
        ax.set_rasterization_zorder(.2)

        # decorate axes
        ax.set_xticks([])
        ax.set_yticks(np.arange(-2.5, 1, .5))
        ax.set_yticklabels([])
        # ax.legend((r'63$\,$eV', r'78$\,$eV'),
        # frameon=False, loc='lower right')
        ax.plot(8.2e3, -.35, 'o', ms=3, c=(0, 0, .8))
        ax.plot(8.2e3, -.5, 'd', ms=3, c='C0')
        ax.text(8.7e3, -.4, r'63$\,$eV', color=(0, 0, .8))
        ax.text(8.7e3, -.55, r'78$\,$eV', color='C0')
        ax.set_xlabel('Intensity (a.u)', fontdict=font)
        ax.set_xlim(0, 1.2e4)
        ax.set_ylim(-2.5, 0.5)

        # Add text
        ax.text(1e3, -0.15, r'$\Delta$', fontsize=12)
        ax.text(5e2, 0.33, r'(c)', fontdict=font)
        ax.text(6e3, -.9, r'$\mathcal{A}$', fontsize=15)
        ax.text(6e3, -1.75, r'$\mathcal{B}$', fontsize=15)

        axi = fig.add_axes([0.69, .545, .05, .05])
        axi.plot(D.k[0], D.k[1], 'r-')
        axi.plot([-1, 1], [-1, -1], 'k-')
        axi.plot([-1, 1], [1, 1], 'k-')
        axi.plot([-1, -1], [1, -1], 'k-')
        axi.plot([1, 1], [-1, 1], 'k-')
        axi.set_xlim(-2, 2)
        axi.set_ylim(-2, 2)
        # axi.text(-.3, -.3, r'$\Gamma$', fontsize=10)
        # axi.text(-1.8, -1.6, 'S', fontsize=10)
        # axi.text(1.1, .9, 'S', fontsize=10)
        axi.axis('off')

    # Plot panels
    fig = plt.figure(figname, figsize=(10, 10), clear=True)
    fig7a()
    fig7b()
    fig7c()
    plt.show()

    # Save figure
    if print_fig:
        plt.savefig(save_dir + figname + '.pdf', dpi=100,
                    bbox_inches="tight", rasterized=True)


def fig8(print_fig=True):
    """figure 8

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Polarization dependence Ca2RuO4: figure 2 of Nature Comm.
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    """

    figname = 'CROfig8'

    file1 = '47991'
    file2 = '47992'
    mat = 'Ca2RuO4'
    year = '2016'
    sample = 'T10'
    gold = '48000'

    # load data
    D1 = ARPES.DLS(file1, mat, year, sample)
    D2 = ARPES.DLS(file2, mat, year, sample)
    D1.norm(gold=gold)
    D2.norm(gold=gold)
    D1.restrict(bot=.6, top=1, left=0, right=1)
    D2.restrict(bot=.6, top=1, left=0, right=1)
    D1.flatten()
    D2.flatten()
    D1.ang2k(D1.ang, Ekin=60-4.5, lat_unit=True, a=5.4, b=5.4, c=11,
             V0=0, thdg=3.5, tidg=12, phidg=-45)
    D2.ang2k(D2.ang, Ekin=60-4.5, lat_unit=True, a=5.4, b=5.4, c=11,
             V0=0, thdg=3.5, tidg=12, phidg=-45)

    edc_ = 1.35
    edc_val, edc_idx = utils.find(np.flipud(D1.k[0]), edc_)
    edc1 = D1.int_norm[edc_idx, :]
    edc2 = D2.int_norm[edc_idx, :]

    # Figure panels
    def fig8a():
        ax = fig.add_subplot(131)
        ax.set_position([.1, .3, .2, .3])
        ax.tick_params(**kwargs_ticks)

        # Plot data
        ax.contourf(D1.kxs, D1.en_norm+.1, 1.2*np.flipud(D1.int_norm), 300,
                    **kwargs_ex, vmin=0, vmax=.008, zorder=.1)
        ax.set_rasterization_zorder(.2)

        # Plot distribution cuts
        ax.plot([np.min(D1.kxs), np.max(D1.kxs)], [0, 0], **kwargs_ef)
        ax.plot([edc_val, edc_val], [-2.5, .5], **kwargs_cut)
        ax.arrow(-1, -1, 0, -.3, head_width=0.2, head_length=0.2,
                 fc='g', ec='k')

        # decorate axes
        ax.set_ylabel(r'$\omega$ (eV)', fontdict=font)
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels([r'$\Gamma$', 'S', r'$\Gamma$'])
        ax.set_yticks(np.arange(-2.5, 1, .5))
        # ax.set_xlim(0, 1)
        ax.set_ylim(-2.5, .5)

        # add text
        ax.text(-.55, 0.33, r'(a) $\bar{\sigma}$-pol.', fontdict=font)

    def fig8b():
        ax = plt.subplot(132)
        ax.set_position([.31, .3, .2, .3])
        ax.tick_params(**kwargs_ticks)

        # Plot data
        c0 = ax.contourf(D2.kxs, D2.en_norm+.1, np.flipud(D2.int_norm), 300,
                         **kwargs_ex, vmin=0, vmax=.008, zorder=.1)
        ax.plot([np.min(D2.kxs), np.max(D2.kxs)], [0, 0], **kwargs_ef)
        ax.set_rasterization_zorder(.2)

        # Plot EDC
        ax.plot([edc_, edc_], [-2.5, .5], **kwargs_cut)

        # decorate axes
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels([r'$\Gamma$', 'S', r'$\Gamma$'])
        ax.set_yticks(np.arange(-2.5, 1, .5), ())
        ax.set_yticklabels([])
        # ax.set_xlim(0, 1)
        ax.set_ylim(-2.5, .5)

        # add text
        ax.text(-.55, 0.33, r'(b) $\bar{\pi}$-pol.', fontdict=font)

        # colorbar
        pos = ax.get_position()
        cax = plt.axes([pos.x0+pos.width+0.01,
                        pos.y0, 0.01, pos.height])
        cbar = plt.colorbar(c0, cax=cax, ticks=None)
        cbar.set_ticks([])
        cbar.set_clim(np.min(D2.int_norm), np.max(D2.int_norm))

    def fig8c():
        ax = fig.add_subplot(1, 3, 3)
        ax.set_position([.55, .3, .2, .3])
        ax.tick_params(**kwargs_ticks)

        xx = np.linspace(1, -5, 200)
        ax.plot(edc1, D1.en_norm[edc_idx, :]+.1, 'o', ms=3, color=(0, 0, .8))
        ax.plot(edc2 * .8, D2.en_norm[edc_idx, :]+.1, 'd', ms=3, color='C0')
        ax.fill(5.5e-3 * exponnorm.pdf(-xx, K=2, loc=.6, scale=.2), xx,
                alpha=.2, fc=(0, 0, .8), zorder=.1)
        ax.fill(5.5e-3 * exponnorm.pdf(-xx, K=2, loc=1.45, scale=.25), xx,
                alpha=.2, fc='C0', zorder=.1)

        # Plot Mott gap estimate
        ax.fill_between([0, 1e-2], 0, -.2, color='C3', alpha=0.2, zorder=.1)
        ax.set_rasterization_zorder(.2)
        ax.plot([np.min(D1.kxs), np.max(D1.kxs)], [0, 0], **kwargs_ef)
        ax.plot([0, 1e-2], [-.2, -.2], 'k:', linewidth=.2)

        # decorate axes
        ax.set_xticks([])
        ax.set_yticks(np.arange(-2.5, 1, .5))
        ax.set_yticklabels([])
        # ax.legend((r'$\sigma$-pol.', r'$\pi$-pol.'), frameon=False)
        ax.set_xlabel('Intensity (a.u)', fontdict=font)
        ax.set_xlim(0, .007)
        ax.set_ylim(-2.5, .5)

        # Add text
        ax.plot(4.5e-3, -.3, 'o', ms=3, c=(0, 0, .8))
        ax.plot(4.5e-3, -.45, 'd', ms=3, c='C0')
        ax.text(5e-3, -.35, r'$\bar{\sigma}$-pol.', color=(0, 0, .8))
        ax.text(5e-3, -.5, r'$\bar{\pi}$-pol.', color='C0')
        ax.text(7e-4, -0.15, r'$\Delta$', fontsize=12)
        ax.text(5e-4, 0.33, r'(c)', fontdict=font)
        ax.text(3.3e-3, -.9, r'$\mathcal{A}$', fontsize=15)
        ax.text(3.3e-3, -1.75, r'$\mathcal{B}$', fontsize=15)

        axi = fig.add_axes([0.69, .548, .05, .05])
        axi.plot(D1.k[0], D1.k[1], 'r-')
        axi.plot([-1, 1], [-1, -1], 'k-')
        axi.plot([-1, 1], [1, 1], 'k-')
        axi.plot([-1, -1], [1, -1], 'k-')
        axi.plot([1, 1], [-1, 1], 'k-')
        axi.set_xlim(-2., 2.)
        axi.set_ylim(-1.5, 2.5)
        axi.axis('off')

    fig = plt.figure(figname, figsize=(10, 10), clear=True)
    fig8a()
    fig8b()
    fig8c()
    plt.show()

    # Save figure
    if print_fig:
        plt.savefig(save_dir + figname + '.pdf', dpi=100,
                    bbox_inches="tight", rasterized=True)


def fig9(print_fig=True):
    """figure 9

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    plot dxy/dxz,yz: figure 4 of Nature Comm.
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    """

    figname = 'CROfig9'

    os.chdir(data_dir)
    xz_data = np.loadtxt('DMFT_CRO_xz.dat')
    yz_data = np.loadtxt('DMFT_CRO_yz.dat')
    xy_data = np.loadtxt('DMFT_CRO_xy.dat')
    os.chdir(home_dir)

    m, n = 8000, 351  # dimensions energy, full k-path
    bot, top = 2500, 5000  # restrict energy window
    DMFT_data = np.array([xz_data, yz_data, xy_data])  # combine data
    DMFT_spec = np.reshape(DMFT_data[:, :, 2], (3, n, m))  # reshape into n,m
    DMFT_spec = DMFT_spec[:, :, bot:top]  # restrict data to bot, top
    DMFT_en = np.linspace(-8, 8, m)  # define energy data
    DMFT_en = DMFT_en[bot:top]  # restrict energy data

    # full k-path
    # [0, 56, 110, 187, 241, 266, 325, 350] = [G, X, S, G, Y, T, G, Z]
    DMFT_spec = np.transpose(DMFT_spec, (0, 2, 1))  # transpose
    DMFT_k = np.arange(0, 351, 1)

    # Plot figure
    fig = plt.figure(figname, figsize=(8, 8), clear=True)
    for i in range(2):
        ax = fig.add_subplot(1, 2, i + 1)
        ax.set_position([.1 + (i * .38), .3, .35, .35])
        ax.tick_params(**kwargs_ticks)

        # Plot data
        c0 = ax.contourf(DMFT_k, DMFT_en, DMFT_spec[i+1, :, :], 300,
                         **kwargs_th, vmin=0, vmax=.3, zorder=.1)
        ax.set_rasterization_zorder(.2)
        ax.plot([0, 350], [0, 0], **kwargs_ef)
        ax.set_xlim(0, 350)
        ax.set_ylim(-3, 1.5)
        ax.set_xticks([0, 56, 110, 187, 241, 266, 325, 350])
        ax.set_xticklabels([r'$\Gamma$', 'X', 'S', r'$\Gamma$', 'Y', 'T',
                            r'$\Gamma$', 'Z'])

        if i == 0:
            # decorate axes
            ax.arrow(188, 0, 0, .7, head_width=8, head_length=0.2,
                     fc='g', ec='g')
            ax.arrow(188, 0, 0, -1.7, head_width=8, head_length=0.2,
                     fc='g', ec='g')
            ax.set_yticks(np.arange(-3, 2, 1.))
            ax.set_ylabel(r'$\omega$ (eV)', fontdict=font)

            # add text
            ax.text(10, -2.8, r'(a) $d_{\gamma z}$', fontdict=font)
            ax.text(198, -.65, r'$U+J_\mathrm{H}$', fontdict=font)

        elif i == 1:
            # decorate axes
            ax.arrow(253, -.8, 0, .22, head_width=8, head_length=0.2,
                     fc='g', ec='g')
            ax.arrow(253, -.8, 0, -.5, head_width=8, head_length=0.2,
                     fc='g', ec='g')
            ax.set_yticks(np.arange(-3, 2, 1.))
            ax.set_yticklabels([])

            # add text
            ax.text(10, -2.8, r'(b) $d_{xy}$', fontsize=12)
            ax.text(263, -1, r'$3J_\mathrm{H}$', fontsize=12)

    # colorbar
    pos = ax.get_position()
    cax = plt.axes([pos.x0+pos.width+0.01,
                    pos.y0, 0.01, pos.height])
    cbar = plt.colorbar(c0, cax=cax, ticks=None)
    cbar.set_ticks([])
    cbar.set_clim(np.min(DMFT_spec), np.max(DMFT_spec) / 2)

    plt.show()

    # Save figure
    if print_fig:
        plt.savefig(save_dir + figname + '.pdf', dpi=100,
                    bbox_inches="tight", rasterized=True)


def fig10(print_fig=True):
    """figure 10

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    DFT plot: spaghetti and spectral representation
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    """

    figname = 'CROfig10'

    # Load DFT spaghetti Plot
    os.chdir(data_dir)
    DFT_data = pd.read_table('DFT_CRO.dat', sep='\t')
    DFT_data = DFT_data.replace({'{': '', '}': ''}, regex=True)
    DFT_data = DFT_data.values
    os.chdir(home_dir)

    # Build k-axis segments
    G = (0, 0, 0)
    X = (np.pi, 0, 0)
    Y = (0, np.pi, 0)
    Z = (0, 0, np.pi)
    T = (0, np.pi, np.pi)
    S = (np.pi, np.pi, 0)

    # Data along path in k-space
    k_pts = np.array([G, X, S, G, Y, T, G, Z])
    k_seg = [0]
    for k in range(len(k_pts)-1):
        diff = abs(np.subtract(k_pts[k], k_pts[k + 1]))
        k_seg.append(k_seg[k] + la.norm(diff))  # extending list cummulative

    # Spaceholders DFT spaghetti plot
    (M, N) = DFT_data.shape
    data = np.zeros((M, N, 3))
    en = np.zeros((M, N))
    xz = np.zeros((M, N))
    k = np.linspace(0, 350, M)

    # Load Data spectral representation
    os.chdir(data_dir)
    DFT_spec = pd.read_csv('DFT_CRO_all.dat').values
    os.chdir(home_dir)

    # spaceholders spectral plot
    (m, n) = DFT_spec.shape
    DFT_en = np.linspace(-3, 1.5, m)
    DFT_k = np.linspace(0, 350, n)

    def fig10a():
        ax = fig.add_subplot(121)
        ax.set_position([.1, .3, .35, .35])
        ax.tick_params(**kwargs_ticks)

        # plot points for legend
        ax.plot(0, 3, 'bo')
        ax.plot(50, 3, 'ro')

        # plot DFT eigenenergies
        for m in range(M):
            for n in range(N):
                data[m][n][:] = np.asfarray(DFT_data[m][n].split(','))
                en[m][n] = data[m][n][1]
                xz[m][n] = data[m][n][2]
                ax.plot(k[m], en[m, n], 'o', ms=3, zorder=.1,
                        color=(xz[m, n], 0, (1-xz[m, n])))  # orbital plot
        ax.plot([0, 350], [0, 0], **kwargs_ef)
        ax.set_rasterization_zorder(.2)

        # decorate axes
        ax.set_xlim(0, 350)
        ax.set_ylim(-3, 1.5)
        ax.set_xticks(k_seg / k_seg[-1] * 350)
        ax.set_xticklabels([r'$\Gamma$', 'X', 'S', r'$\Gamma$', 'Y', 'T',
                            r'$\Gamma$', 'Z'])
        ax.set_yticks(np.arange(-3, 2, 1.))
        ax.set_ylabel(r'$\omega$ (eV)', fontdict=font)
        ax.legend((r'$d_{xy}$', r'$d_{\gamma z}$'), frameon=False)

        # add text
        ax.text(10, 1.15, r'(a)', fontdict=font)

    def fig10b():
        ax = fig.add_subplot(122)
        ax.set_position([.1 + .38, .3, .35, .35])
        ax.tick_params(**kwargs_ticks)

        # plot data
        c0 = ax.contourf(DFT_k, DFT_en, DFT_spec, 300, **kwargs_th,
                         vmin=0, vmax=25, zorder=.1)
        ax.set_rasterization_zorder(.2)
        ax.plot([0, 350], [0, 0], **kwargs_ef)

        # decorate axes
        ax.set_xticks(k_seg / k_seg[-1] * 350)
        ax.set_xticklabels([r'$\Gamma$', 'X', 'S', r'$\Gamma$', 'Y', 'T',
                            r'$\Gamma$', 'Z'])
        ax.set_yticks(np.arange(-3, 2, 1.))
        ax.set_yticklabels([])
        ax.set_xlim(0, 350)
        ax.set_ylim(-3, 1.5)

        # add text
        ax.text(10, 1.15, r'(b)', fontdict=font)

        # colorbar
        pos = ax.get_position()
        cax = plt.axes([pos.x0+pos.width+0.01,
                        pos.y0, 0.01, pos.height])
        cbar = plt.colorbar(c0, cax=cax, ticks=None)
        cbar.set_ticks([])
        cbar.set_clim(np.min(DFT_spec), np.max(DFT_spec))

    fig = plt.figure(figname, figsize=(8, 8), clear=True)
    fig10a()
    fig10b()
    plt.show()

    # Save figure
    if print_fig:
        plt.savefig(save_dir + figname + '.pdf', dpi=150,
                    bbox_inches="tight", rasterized=True)


def fig11(print_fig=True):
    """figure 11

    %%%%%%%%%%%%%%%%%%%%%%%%%%
    Multiplet analysis Ca2RuO4
    %%%%%%%%%%%%%%%%%%%%%%%%%%
    """

    figname = 'CROfig11'

    # Create figure
    fig = plt.figure(figname, figsize=(8, 8), clear=True)
    ax = fig.add_subplot(111)
    ax.set_position([.1, .3, .52, .15])
    ax.tick_params(**kwargs_ticks)

    off = [0, 3, 5, 8, 9.25, 11.25, 12.5]  # helpful for the plot
    n = 0

    # plot lines
    for i in off:
        n += 1
        ax.plot([0 + i, 1 + i], [0, 0], 'k-')
        ax.plot([0 + i, 1 + i], [-.5, -.5], 'k-')
        ax.plot([0 + i, 1 + i], [-2.5, -2.5], 'k-')

        if any(x == n for x in [1, 2, 3, 4, 6, 7]):
            ax.arrow(.33 + i, -.5, 0, 1, head_width=0.2, head_length=0.4,
                     linewidth=1.5, fc='r', ec='r')
        if any(x == n for x in [6]):
            ax.arrow(.1 + i, .8, 0, -1, head_width=0.2, head_length=0.4,
                     linewidth=1.5, fc='r', ec='r')
        if any(x == n for x in [1, 2, 3, 5, 6, 7]):
            ax.arrow(.66 + i, -1, 0, 1, head_width=0.2, head_length=0.4,
                     linewidth=1.5, fc='r', ec='r')
        if any(x == n for x in [7]):
            ax.arrow(.9 + i, .3, 0, -1, head_width=0.2, head_length=0.4,
                     linewidth=1.5, fc='r', ec='r')
        if any(x == n for x in [1, 2, 4, 5, 6, 7]):
            ax.arrow(.33 + i, -3, 0, 1, head_width=0.2, head_length=0.4,
                     linewidth=1.5, fc='r', ec='r')
        if any(x == n for x in [1, 3, 4, 5, 6, 7]):
            ax.arrow(.66 + i, -1.7, 0, -1, head_width=0.2, head_length=0.4,
                     linewidth=1.5, fc='r', ec='r')

    # some background
    ax.fill_between([2, 7], 4, -4, color='C0', alpha=0.2, zorder=.1)
    ax.fill_between([7, 14.3], 4, -4, color=(0, 0, .8), alpha=0.2, zorder=.1)
    ax.set_rasterization_zorder(.2)

    # add text
    ax.text(-1.7, -2.7, r'$d_{xy}$', fontsize=12)
    ax.text(-1.7, -.3, r'$d_{\gamma z}$', fontsize=12)
    ax.text(4., 1.5, r'$3J_\mathrm{H}$', fontsize=12)
    ax.text(9.9, 1.5, r'$U+J_\mathrm{H}$', fontsize=12)
    ax.text(-1.7, 3, r'$| d^4; S=1, L=1\rangle$', fontsize=8)
    ax.text(2.6, 3, r'$| d^3; \frac{3}{2}, 0 \rangle$', fontsize=8)
    ax.text(4.6, 3, r'$| d^3; \frac{1}{2}, 2 \rangle$', fontsize=8)
    ax.text(8.4, 3, r'$| d^3; \frac{1}{2}, 2 \rangle$', fontsize=8)
    ax.text(11.5, 3, r'$| d^5; \frac{1}{2}, 1 \rangle$', fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-2, 14.3)
    ax.set_ylim(-4, 4)
    plt.show()

    # Save figure
    if print_fig:
        plt.savefig(save_dir + figname + '.pdf', dpi=100,
                    bbox_inches="tight", rasterized=True)


def fig12(print_fig=True):
    """figure 12

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Constant energy maps oxygen band -5.2eV
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    """

    figname = 'CROfig12'
    CRO_FS_plot(e=-5.2, v_min=.25, figname=figname)
    # Save figure
    if print_fig:
        plt.savefig(save_dir + figname + '.pdf', dpi=100,
                    bbox_inches="tight", rasterized=True)


def fig13(print_fig=True):
    """figure 13

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Constant energy maps alpha band -.5eV
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    """

    figname = 'CROfig13'
    CRO_FS_plot(e=-.5, v_min=.05, figname=figname)
    # Save figure
    if print_fig:
        plt.savefig(save_dir + figname + '.pdf', dpi=100,
                    bbox_inches="tight", rasterized=True)


def fig14(print_fig=True):
    """figure 14

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Constant energy maps gamma band -2.4eV
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    """

    figname = 'CROfig14'
    CRO_FS_plot(e=-2.4, v_min=.4, figname=figname)
    # Save figure
    if print_fig:
        plt.savefig(save_dir + figname + '.pdf', dpi=100,
                    bbox_inches="tight", rasterized=True)


def fig15(print_fig=True):
    """figure 15

    %%%%%%%%%%%%%%%%%%
    Schematic DOS OSMP
    %%%%%%%%%%%%%%%%%%
    """

    figname = 'CROfig15'

    fig = plt.figure(figname, figsize=(6, 4), clear=True)

    ax = fig.add_axes([.2, .2, .6, .6])
    ax.tick_params(**kwargs_ticks)

    def DOS(R, w):
        # semi circle
        th = np.linspace(-np.pi/2, np.pi/2, 200)
        r = R / np.sqrt(1 - (w * np.cos(th)) ** 2)
        x = r * np.cos(th)
        y = r * np.sin(th)
        return x, y

    # coordinates
    x, y = DOS(.3, .95)
    x_z, y_z = DOS(.2, .98)
    x_xy, y_xy = DOS(.35, .8)
    xs_z, ys_z = DOS(.12, .98)
    xl_z, yl_z = DOS(.17, .98)
    xs_xy, ys_xy = DOS(.1, .8)
    xl_xy, yl_xy = DOS(.3, .8)

    # filling indices
    fill1 = 108
    fill2 = 103
    fill3 = 128

    # coloumn 1
    ax.plot(x, y, color='k', lw=1)
    ax.arrow(0, -.75, 0, 1.5, head_width=0.05, head_length=0.05,
             fc='k', ec='k')
    ax.arrow(0.3, .5, 1.2, 0, head_width=0.02, head_length=0.05,
             fc='k', ec='k')
    ax.plot([0, .9], [.1, .1], **kwargs_ef)
    ax.fill_betweenx(y[:fill1], 0, x[:fill1], color='k', alpha=.3)
    ax.text(.3, .56, r'$+\,\mathrm{crystal\,\,field}$', fontsize=8)
    ax.text(-.1, .85, r'$\omega$', fontdict=font)
    ax.text(.1, -.6, r'DOS($\omega$)', fontdict=font)

    # coloumn 2
    ax.plot(x_z+2, y_z+.06, color='b', lw=1)
    ax.plot(x_xy+2, y_xy-.12, color='C1', lw=1)
    ax.arrow(2, -.75, 0, 1.5, head_width=0.05, head_length=0.05,
             fc='k', ec='k')
    ax.arrow(2.3, .5, 1.2, 0, head_width=0.02, head_length=0.05,
             fc='k', ec='k')
    ax.plot([0+2, .97+2], [.1, .1], **kwargs_ef)
    ax.fill_betweenx(y_z[:fill2]+.06, 0+2,
                     x_z[:fill2]+2, color='b', alpha=.3)
    ax.fill_betweenx(y_xy[:fill3]-.12, 0+2,
                     x_xy[:fill3]+2, color='C1', alpha=.3)
    ax.text(3.1, 0, r'$d_{\gamma z}$', color='b')
    ax.text(2.7, -.22, r'$d_{xy}$', color='C1')
    ax.text(2.3, .58, r'$+\,U,$ $+\,J_\mathrm{H}$')
    ax.text(1.9, .85, r'$\omega$', fontdict=font)

    # coloumn 3
    ax.plot(xs_z+4, ys_z+.5, color='b', lw=1)
    ax.plot(xl_z+4, yl_z-.3, color='b', lw=1)
    ax.plot(xs_xy+4, ys_xy+.35, color='C1', lw=1)
    ax.plot(xl_xy+4, yl_xy-.25, color='C1', lw=1)
    ax.arrow(4, -.75, 0, 1.5, head_width=0.05, head_length=0.05,
             fc='k', ec='k')
    ax.arrow(5, .1, 0, .2, head_width=0.05, head_length=0.05,
             fc='C1', ec='C1')
    ax.arrow(5, .1, 0, -.32, head_width=0.05, head_length=0.05,
             fc='C1', ec='C1')
    ax.arrow(5.6, .1, 0, .35, head_width=0.05, head_length=0.05,
             fc='b', ec='b')
    ax.arrow(5.6, .1, 0, -.4, head_width=0.05, head_length=0.05,
             fc='b', ec='b')
    ax.plot([0+4, .9+4], [.1, .1], **kwargs_ef)
    ax.fill_betweenx(yl_z-.3, 0+4, xl_z+4, color='b', alpha=.3)
    ax.fill_betweenx(yl_xy-.25, 0+4, xl_xy+4, color='C1', alpha=.3)
    ax.text(3.9, .85, r'$\omega$', fontdict=font)
    ax.text(5.05, 0.02, r'$\Delta_{xy}$', color='C1')
    ax.text(5.65, 0.02, r'$\Delta_{\gamma z}$', color='b')

    ax.set_xlim(-.3, 6.2)
    ax.set_ylim(-.9, 1)
    plt.axis('off')
    plt.show()

    # Save figure
    if print_fig:
        plt.savefig(save_dir + figname + '.png', dpi=300,
                    bbox_inches="tight")


def fig16(print_fig=True):
    """figure 16

    %%%%%%%%%%%%%%%%%%%%%%%%%
    Schematic DOS band / Mott
    %%%%%%%%%%%%%%%%%%%%%%%%%
    """

    figname = 'CROfig16'

    fig = plt.figure(figname, figsize=(6, 4), clear=True)

    ax = fig.add_axes([.2, .2, .6, .6])
    ax.tick_params(**kwargs_ticks)

    def DOS(R, w):
        # semi circle
        th = np.linspace(-np.pi/2, np.pi/2, 200)
        r = R / np.sqrt(1 - (w * np.cos(th)) ** 2)
        x = r * np.cos(th)
        y = r * np.sin(th)
        return x, y

    # coordinates
    x, y = DOS(.3, .95)
    x_z, y_z = DOS(.2, .98)
    x_xy, y_xy = DOS(.35, .8)
    xs_z, ys_z = DOS(.15, .98)
    xl_z, yl_z = DOS(.15, .98)

    # filling indices
    fill1 = 108
    fill2 = 100

    # coloumn 1
    ax.plot(x, y, color='k', lw=1)
    ax.arrow(0, -.75, 0, 1.5, head_width=0.05, head_length=0.05,
             fc='k', ec='k')
    ax.arrow(0.3, .5, 1.2, 0, head_width=0.02, head_length=0.05,
             fc='k', ec='k')
    ax.plot([0, .9], [.1, .1], **kwargs_ef)
    ax.fill_betweenx(y[:fill1], 0, x[:fill1], color='k', alpha=.3)
    ax.text(.3, .56, r'$+\,\mathrm{crystal\,\,field}$', fontsize=8)
    # ax.text(.3, .39, r'$+\,\mathrm{interactions}$', fontsize=8)
    ax.text(-.1, .85, r'$\omega$', fontdict=font)
    ax.text(.1, -.6, r'DOS($\omega$)', fontdict=font)

    # coloumn 2
    ax.plot(x_z+2, y_z+.105, color='b', lw=1)
    ax.plot(x_xy+2, y_xy-.3, color='C1', lw=1)
    ax.arrow(2, -.75, 0, 1.5, head_width=0.05, head_length=0.05,
             fc='k', ec='k')
    ax.arrow(2.3, .5, 1.2, 0, head_width=0.02, head_length=0.05,
             fc='k', ec='k')
    ax.plot([0+2, .97+2], [.1, .1], **kwargs_ef)
    ax.fill_betweenx(y_z[:fill2]+.105, 0+2, x_z[:fill2]+2, color='b', alpha=.3)
    ax.fill_betweenx(y_xy-.3, 0+2, x_xy+2, color='C1', alpha=.3)
    ax.text(3.1, 0, r'$d_{\gamma z}$', color='b')
    ax.text(2.7, -.22, r'$d_{xy}$', color='C1')
    ax.text(2.3, .58, r'$+\,U,$ $+\,J_\mathrm{H}$')
    ax.text(1.9, .85, r'$\omega$', fontdict=font)

    # coloumn 3
    ax.plot(xs_z+4, ys_z+.5, color='b', lw=1)
    ax.plot(xl_z+4, yl_z-.3, color='b', lw=1)
    ax.plot(x_xy+4, y_xy-.3, color='C1', lw=1)
    ax.arrow(4, -.75, 0, 1.5, head_width=0.05, head_length=0.05,
             fc='k', ec='k')
    ax.arrow(5, .1, 0, .35, head_width=0.05, head_length=0.05,
             fc='b', ec='b')
    ax.arrow(5, .1, 0, -.35, head_width=0.05, head_length=0.05,
             fc='b', ec='b')
    ax.plot([0+4, .9+4], [.1, .1], **kwargs_ef)
    ax.fill_betweenx(yl_z-.3, 0+4, xl_z+4, color='b', alpha=.3)
    ax.fill_betweenx(y_xy-.3, 0+4, x_xy+4, color='C1', alpha=.3)
    ax.text(3.9, .85, r'$\omega$', fontdict=font)
    ax.text(5.05, 0.02, r'$\Delta_{\gamma z}$', color='b')

    ax.set_xlim(-.3, 6.2)
    ax.set_ylim(-.9, 1)
    plt.axis('off')
    plt.show()

    # Save figure
    if print_fig:
        plt.savefig(save_dir + figname + '.png', dpi=300,
                    bbox_inches="tight")


def fig17(print_fig=True):
    """figure 17

    %%%%%%%%%%%%%%%%%%%%%%%%%
    Schematic DOS uniform gap
    %%%%%%%%%%%%%%%%%%%%%%%%%
    """

    figname = 'CROfig17'

    fig = plt.figure(figname, figsize=(6, 4), clear=True)

    ax = fig.add_axes([.2, .2, .6, .6])
    ax.tick_params(**kwargs_ticks)

    def DOS(R, w):
        # semi circle
        th = np.linspace(-np.pi/2, np.pi/2, 200)
        r = R / np.sqrt(1 - (w * np.cos(th)) ** 2)
        x = r * np.cos(th)
        y = r * np.sin(th)
        return x, y

    # coordinates
    x, y = DOS(.3, .95)
    x_z, y_z = DOS(.2, .98)
    x_xy, y_xy = DOS(.35, .8)
    xs_z, ys_z = DOS(.12, .98)
    xl_z, yl_z = DOS(.17, .98)
    xs_xy, ys_xy = DOS(.1, .8)
    xl_xy, yl_xy = DOS(.3, .8)

    # filling indices
    fill1 = 108
    fill2 = 103
    fill3 = 128

    # coloumn 1
    ax.plot(x, y, color='k', lw=1)
    ax.arrow(0, -.75, 0, 1.5, head_width=0.05, head_length=0.05,
             fc='k', ec='k')
    ax.arrow(0.3, .5, 1.2, 0, head_width=0.02, head_length=0.05,
             fc='k', ec='k')
    ax.plot([0, .9], [.1, .1], **kwargs_ef)
    ax.fill_betweenx(y[:fill1], 0, x[:fill1], color='k', alpha=.3)
    ax.text(.3, .56, r'$+\,\mathrm{crystal\,\,field}$', fontsize=8)
    ax.text(-.1, .85, r'$\omega$', fontdict=font)
    ax.text(.1, -.6, r'DOS($\omega$)', fontdict=font)

    # coloumn 2
    ax.plot(x_z+2, y_z+.06, color='b', lw=1)
    ax.plot(x_xy+2, y_xy-.12, color='C1', lw=1)
    ax.arrow(2, -.75, 0, 1.5, head_width=0.05, head_length=0.05,
             fc='k', ec='k')
    ax.arrow(2.3, .5, 1.2, 0, head_width=0.02, head_length=0.05,
             fc='k', ec='k')
    ax.plot([0+2, .97+2], [.1, .1], **kwargs_ef)
    ax.fill_betweenx(y_z[:fill2]+.06, 0+2,
                     x_z[:fill2]+2, color='b', alpha=.3)
    ax.fill_betweenx(y_xy[:fill3]-.12, 0+2,
                     x_xy[:fill3]+2, color='C1', alpha=.3)
    ax.text(3.1, 0, r'$d_{\gamma z}$', color='b')
    ax.text(2.7, -.22, r'$d_{xy}$', color='C1')
    ax.text(2.3, .56, r'$+\,\mathrm{uniform\,\,gap}$', fontsize=8)
    ax.text(1.9, .85, r'$\omega$', fontdict=font)

    # coloumn 3
    ax.plot(xs_z+4, ys_z+.3, color='b', lw=1)
    ax.plot(xl_z+4, yl_z-.3, color='b', lw=1)
    ax.plot(xs_xy+4, ys_xy+.3, color='C1', lw=1)
    ax.plot(xl_xy+4, yl_xy-.3, color='C1', lw=1)
    ax.arrow(4, -.75, 0, 1.5, head_width=0.05, head_length=0.05,
             fc='k', ec='k')
    ax.arrow(5, .1, 0, .15, head_width=0.05, head_length=0.05,
             fc='k', ec='k')
    ax.arrow(5, .1, 0, -.35, head_width=0.05, head_length=0.05,
             fc='k', ec='k')
    ax.plot([0+4, .9+4], [.1, .1], **kwargs_ef)
    ax.fill_betweenx(yl_z-.3, 0+4, xl_z+4, color='b', alpha=.3)
    ax.fill_betweenx(yl_xy-.3, 0+4, xl_xy+4, color='C1', alpha=.3)
    ax.text(3.9, .85, r'$\omega$', fontdict=font)
    ax.text(5.05, -0.03, r'$\Delta$', color='k')

    ax.set_xlim(-.3, 6.2)
    ax.set_ylim(-.9, 1)
    plt.axis('off')
    plt.show()

    # Save figure
    if print_fig:
        plt.savefig(save_dir + figname + '.png', dpi=300,
                    bbox_inches="tight")


def fig18(print_fig=True):
    """figure 18

    %%%%%%%%%%%%%%%%%%
    Oxygen bands + DFT
    %%%%%%%%%%%%%%%%%%
    """

    figname = 'CROfig18'

    # load data
    os.chdir(data_dir)
    GX = np.loadtxt('DFT_CRO_GX_ALL.dat')
    XG = np.fliplr(GX)
    XS = np.loadtxt('DFT_CRO_XS_ALL.dat')
    SX = np.fliplr(XS)
    os.chdir(home_dir)

    GX_data = np.concatenate((GX, XG, GX, XG, GX), axis=1)
    GX_en = np.linspace(-7.5, 0, GX_data.shape[0])
    GX_k = np.linspace(-2, 3, GX_data.shape[1])

    XS_data = np.concatenate((XS, SX, XS, SX, XS), axis=1)
    XS_en = np.linspace(-7.5, 0, XS_data.shape[0])
    XS_k = np.linspace(-2, 3, XS_data.shape[1])

    mat = 'Ca2RuO4'
    year = '2016'
    sample = 'T10'
    files = np.array([48026, 48046])
    gold = 48000

    fig = plt.figure(figname, figsize=(6, 6), clear=True)

    # Manipulator angles for k-space transformation
    th = np.array([-6.5, -1.5])
    ti = np.array([0, 8.5])
    phi = np.array([0, 0])
    lbls = ('(a)', '(b)', '(c)', '(d)', '(e)')
    lbls_x = (-1, -1.6)
    n = 0
    for file in files:
        D = ARPES.DLS(file, mat, year, sample)
        D.norm(gold)
        D.flatten()
        D.ang2k(D.ang, Ekin=D.hv-4.5, lat_unit=True, a=5.4, b=5.4, c=11,
                V0=0, thdg=th[n], tidg=ti[n], phidg=phi[n])
        n += 1
        # Create figure
        ax = fig.add_subplot(2, 3, n)
        ax.set_position([.1+(n-1)*.26, .46, .25, .35])
        ax.tick_params(**kwargs_ticks)
        c0 = ax.contourf(D.kxs, D.en_norm, D.int_norm, 50, **kwargs_ex,
                         zorder=.1)
        ax.set_rasterization_zorder(.2)

        ax.plot([np.min(D.kxs), np.max(D.kxs)], [-5.2, -5.2],
                **kwargs_cut)

        # decorate axes
        if n == 1:
            k1 = D.k
            ax.set_ylabel(r'$\omega$ (eV)', fontdict=font)
        else:
            k2 = D.k
            ax.set_yticklabels([])
        ax.set_ylim(-7.5, 0)
        ax.set_xticklabels([])

        # add text
        ax.text(lbls_x[n-1], -.6, lbls[n-1], fontdict=font)
    # colorbar
    pos = ax.get_position()
    cax = plt.axes([pos.x0+pos.width+0.01,
                    pos.y0, 0.01, pos.height])
    cbar = plt.colorbar(c0, cax=cax, ticks=None)
    cbar.set_ticks([])
    cbar.set_clim(np.min(D.int_norm), np.max(D.int_norm))

    ax_GX = fig.add_subplot(233)
    ax_GX.tick_params(**kwargs_ticks)
    ax_GX.set_position([.1, .1, .25, .35])
    ax_GX.contourf(GX_k, GX_en, GX_data, 50, **kwargs_th, zorder=.1)
    ax_GX.set_rasterization_zorder(.2)
#    ax_GX.plot([GX_k[0], GX_k[-1]], [-4.1, -4.1], 'w--')

    # decorate axes
    ax_GX.set_ylabel(r'$\omega$ (eV)', fontdict=font)
    ax_GX.set_xticks(np.arange(-1, 4, 1))
    ax_GX.set_xticklabels(('X', r'$\Gamma$', 'X', r'$\Gamma$'))
    ax_GX.set_xlim(k1[0][0], k1[0][-1])

    # add text
    ax_GX.text(lbls_x[0], -.6, lbls[-2], fontdict=font)

    ax_XS = fig.add_subplot(234)
    ax_XS.tick_params(**kwargs_ticks)
    ax_XS.set_position([.1+.26, .1, .25, .35])
    c0 = ax_XS.contourf(XS_k, XS_en, XS_data, 50, **kwargs_th, zorder=.1)
    ax_XS.set_rasterization_zorder(.2)
#    ax_XS.plot([XS_k[0], XS_k[-1]], [-4.1, -4.1], 'w--')

    # decorate axes
    ax_XS.set_xticks(np.arange(-2, 4, 1))
    ax_XS.set_xticklabels(('X', 'S', 'X', 'S', 'X'))
    ax_XS.set_yticklabels([])
    ax_XS.set_xlim(k2[0][0], k2[0][-1])

    # add text
    ax_XS.text(lbls_x[1], -.6, lbls[-1], fontdict=font)

    # colorbar
    pos = ax_XS.get_position()
    cax = plt.axes([pos.x0+pos.width+0.01,
                    pos.y0, 0.01, pos.height])
    cbar = plt.colorbar(c0, cax=cax, ticks=None)
    cbar.set_ticks([])
    cbar.set_clim(np.min(XS_data), np.max(XS_data))

    file1 = '0619_00161'
    file2 = '0619_00162'
    mat = 'Ca2RuO4'
    year = '2016'
    sample = 'data'
    th = 20
    ti = -2
    phi = 21
    a = 5.4
    b = 5.5

    D1 = ARPES.ALS(file1, mat, year, sample)  # frist scan
    D2 = ARPES.ALS(file2, mat, year, sample)  # second scan
    D1.ang2kFS(D1.ang, Ekin=D1.hv-4.5-4.7, lat_unit=True, a=a, b=b, c=11,
               V0=0, thdg=th, tidg=ti, phidg=phi)
    D2.ang2kFS(D2.ang, Ekin=D2.hv-4.5-4.7, lat_unit=True, a=a, b=b, c=11,
               V0=0, thdg=th, tidg=ti, phidg=phi)

    # Combining two scans
    data = np.concatenate((D1.int, D2.int), axis=0)
    kx = np.concatenate((D1.kx, D2.kx), axis=0)
    ky = np.concatenate((D1.ky, D2.ky), axis=0)

    # energy off set (Fermi level not specified)
    en = D1.en - 2.3
    e = -5
    ew = 0.4
    e_val, e_idx = utils.find(en, e)
    ew_val, ew_idx = utils.find(en, e-ew)
    FS = np.sum(data[:, :, ew_idx:e_idx], axis=2)

    ax = fig.add_subplot(236)
    ax.tick_params(**kwargs_ticks)
    ax.set_position([.1+.26+.28, .46, .25, .35])

    # Plot data
    c0 = ax.contourf(kx, ky, FS, 300, **kwargs_ex,
                     vmin=.4*np.max(FS), vmax=.95*np.max(FS), zorder=.1)
    ax.set_rasterization_zorder(.2)

    ax.plot(k1[0], k1[1], 'r-')
    ax.plot(k2[0], k2[1], 'r-')

    # Labels
    ax.grid(alpha=0.3)
    ax.set_xticks(np.arange(-10, 10, 1))
    ax.set_yticks(np.arange(-10, 10, 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xlim(-2.8, 2.914)
    ax.set_ylim(-2, 6)
    pos = ax.get_position()

    # kwarg dictionaries
    ortho = {'linestyle': '-', 'color': 'black', 'lw': 1}

    # Plot Brillouin zone
    ax.plot([-1, -1], [-1, 1], **ortho)
    ax.plot([1, 1], [-1, 1], **ortho)
    ax.plot([-1, 1], [1, 1], **ortho)
    ax.plot([-1, 1], [-1, -1], **ortho)

    # High symmetry points
    font_HS = {'fontsize': 12, 'color': 'w'}
    ax.text(-0.2, -0.2, r'$\Gamma$', fontdict=font_HS)
    ax.text(.8, .8, r'S', fontdict=font_HS)
    ax.text(-0.2, .8, r'X', fontdict=font_HS)
    ax.text(.8, -.2, r'X', fontdict=font_HS)

    # add text
    ax.text(-2.6, 5.35, lbls[2], fontdict=font)

    # label plot
    ax = fig.add_subplot(235)
    ax.tick_params(**kwargs_ticks)
    ax.set_position([.1+.26+.28, .1, .25, .35])

    ax.arrow(-.8, -4, 0, 1.6, head_width=0.06, head_length=0.3,
             fc='b', ec='b')
    ax.arrow(-.8, -4, 0, -3.1, head_width=0.06, head_length=0.3,
             fc='b', ec='b')
    ax.arrow(-.8, -1, 0, .6, head_width=0.06, head_length=0.3,
             fc='C8', ec='C8')
    ax.arrow(-.8, -1, 0, -.6, head_width=0.06, head_length=0.3,
             fc='C8', ec='C8')

    # decorate axes
    ax.set_xlim(-1, 1)
    ax.set_ylim(-7.5, 0)

    ax.axis('off')
    # add text
    ax.text(-.65, -1.2, 'Ruthenium bands', fontdict=font)
    ax.text(-.65, -5, 'Oxygen bands', fontdict=font)
    plt.show()

    # Save figure
    if print_fig:
        plt.savefig(save_dir + figname + '.pdf', dpi=100,
                    bbox_inches="tight", rasterized=True)


def fig19(print_fig=True):
    """figure 19

    %%%%%%%%%%%%%%%%%%%%%%%%%%
    Schematic DOS Mott-Hubbard
    %%%%%%%%%%%%%%%%%%%%%%%%%%
    """

    figname = 'CROfig19'

    fig = plt.figure(figname, figsize=(6, 4), clear=True)

    ax = fig.add_axes([.2, .2, .6, .6])
    ax.tick_params(**kwargs_ticks)

    def DOS(R, w):
        # semi circle
        th = np.linspace(-np.pi/2, np.pi/2, 200)
        r = R / np.sqrt(1 - (w * np.cos(th)) ** 2)
        x = r * np.cos(th)
        y = r * np.sin(th)
        return x, y

    # coordinates
    x, y = DOS(.3, .95)
    x_UHB, y_UHB = DOS(.1, .996)
    x_LHB, y_LHB = DOS(.1, .996)

    # filling indices
    fill1 = 100
    fill2 = 0
    fill3 = 200

    # coloumn 1
    ax.plot(x, y+.1, color='k', lw=1)
    ax.arrow(0, -.75, 0, 1.5, head_width=0.05, head_length=0.05,
             fc='k', ec='k')
    ax.arrow(0.3, .5, 1.2, 0, head_width=0.02, head_length=0.05,
             fc='k', ec='k')
    ax.plot([0, 1], [.1, .1], **kwargs_ef)
    ax.fill_betweenx(y[:fill1]+.1, 0, x[:fill1], color='k', alpha=.3)
    ax.text(.7, .56, r'$+U$', fontsize=10)
    ax.text(-.1, .85, r'$\omega$', fontdict=font)
    ax.text(.1, -.6, r'DOS($\omega$)', fontdict=font)

    # coloumn 2
    ax.plot(x_UHB+2, y_UHB+.4, color='k', lw=1)
    ax.plot(x_LHB+2, y_LHB-.2, color='k', lw=1)
    ax.arrow(2, -.75, 0, 1.5, head_width=0.05, head_length=0.05,
             fc='k', ec='k')

    ax.plot([0+2, .97+2], [.1, .1], **kwargs_ef)
    ax.fill_betweenx(y_UHB[:fill2]+.4, 0+2,
                     x_UHB[:fill2]+2, color='k', alpha=.3)
    ax.fill_betweenx(y_LHB[:fill3]-.2, 0+2,
                     x_LHB[:fill3]+2, color='k', alpha=.3)
    ax.arrow(3.3, .18, 0, .15, head_width=0.05, head_length=0.05,
             fc='k', ec='k')
    ax.arrow(3.3, .18, 0, -.35, head_width=0.05, head_length=0.05,
             fc='k', ec='k')
    ax.text(2.1, .37, r'$\mathrm{UHB}$', color='k')
    ax.text(2.1, -.23, r'$\mathrm{LHB}$', color='k')
    ax.text(1.9, .85, r'$\omega$', fontdict=font)

    ax.text(3.4, 0.06, r'$\Delta$', color='k')

    ax.set_xlim(-.3, 6.2)
    ax.set_ylim(-.9, 1)
    plt.axis('off')
    plt.show()

    # Save figure
    if print_fig:
        plt.savefig(save_dir + figname + '.png', dpi=400,
                    bbox_inches="tight")
