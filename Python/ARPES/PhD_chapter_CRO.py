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
              'color': 'turquoise',
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
                         vmin=0, vmax=v_max*np.max(data_spec))
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
                         vmin=v_min * np.max(D.map),
                         vmax=.95 * np.max(D.map))
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
            ax.plot([-1, 1], [-1, 1], **kwargs_cut)
            ax.plot([-1, 1], [1, 1], **kwargs_cut)
            ax.plot([-1, 0], [1, 2], **kwargs_cut)
            ax.plot([0, 0], [2, -1], **kwargs_cut)
            ax.arrow(-1, -1, .3, .3, head_width=0.3, head_length=0.3,
                     fc='turquoise', ec='k')
            ax.arrow(0, -.4, 0, -.3, head_width=0.3, head_length=0.3,
                     fc='turquoise', ec='k')
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
        plt.savefig(save_dir + figname + '.png', dpi=600, bbox_inches="tight")


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
        plt.savefig(save_dir + figname + '.png', dpi=600, bbox_inches="tight")


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

    # Save data
    if print_fig:
        plt.savefig(save_dir + figname + '.png', dpi=600, bbox_inches="tight")


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
        plt.savefig(save_dir + figname + '.png', dpi=600, bbox_inches="tight")


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
                        **kwargs_ex,
                        vmin=v_scale*0.01*np.max(D.int_norm),
                        vmax=v_scale*0.5*np.max(D.int_norm))

        elif n == 1:
            ax.contourf(D.kxs, D.en_norm+.1, D.int_norm, 300,
                        **kwargs_ex,
                        vmin=v_scale*0.0*np.max(D.int_norm),
                        vmax=v_scale*0.54*np.max(D.int_norm))

        elif n == 2:
            ax.contourf(D.kxs, D.en_norm+.1, np.flipud(D.int_norm), 300,
                        **kwargs_ex,
                        vmin=v_scale * 0.01 * np.max(D.int_norm),
                        vmax=v_scale * 0.7 * np.max(D.int_norm))

        elif n == 3:
            c0 = ax.contourf(D.kxs, D.en_norm+.1, np.flipud(D.int_norm), 300,
                             **kwargs_ex,
                             vmin=v_scale * 0.01 * np.max(D.int_norm),
                             vmax=v_scale * 0.53 * np.max(D.int_norm))

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
    plt.show()

    # Save figure
    if print_fig:
        plt.savefig(save_dir + figname + '.png', dpi=600, bbox_inches="tight")


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

    fig = plt.figure(figname, figsize=(6, 6), clear=True)
    ax = fig.add_axes([.2, .2, .5, .65])
    ax.tick_params(**kwargs_ticks)

    # Plot data
    c0 = ax.contourf(kx, ky, FS, 300, **kwargs_ex,
                     vmin=.55*np.max(FS), vmax=.95*np.max(FS))

    # Labels
    ax.set_xlabel(r'$k_x$ ($\pi/a$)', fontdict=font)
    ax.set_ylabel(r'$k_y$ ($\pi/b$)', fontdict=font)
    ax.grid(alpha=0.3)
    ax.set_xticks(np.arange(-10, 10, 1))
    ax.set_yticks(np.arange(-10, 10, 1))
    ax.set_xlim(-1.2, 2.2)
    ax.set_ylim(-1.2, 3.2)
    pos = ax.get_position()

    # kwarg dictionaries
    ortho = {'linestyle': '-', 'color': 'black', 'lw': 1}

    # Plot Brillouin zone
    ax.plot([-1, -1], [-1, 1], **ortho)
    ax.plot([1, 1], [-1, 1], **ortho)
    ax.plot([-1, 1], [1, 1], **ortho)
    ax.plot([-1, 1], [-1, -1], **ortho)

    # Plot ARPES cut path
    ax.plot([-1, 1], [-1, 1], **kwargs_cut)
    ax.plot([-1, 1], [1, 1], **kwargs_cut)
    ax.plot([-1, 0], [1, 2], **kwargs_cut)
    ax.plot([0, 0], [2, -1], **kwargs_cut)
    ax.arrow(-1, -1, .3, .3, head_width=0.2, head_length=0.2,
             fc='turquoise', ec='k')
    ax.arrow(0, -.5, 0, -.3, head_width=0.2, head_length=0.2,
             fc='turquoise', ec='k')

    # High symmetry points
    font_HS = {'fontsize': 20, 'color': 'r'}
    ax.text(-0.1, -0.1, r'$\Gamma$', fontdict=font_HS)
    ax.text(-0.1, 1.9, r'$\Gamma$', fontdict=font_HS)
    ax.text(.9, .9, r'S', fontdict=font_HS)
    ax.text(-0.1, .9, r'X', fontdict=font_HS)

    # colorbar
    cax = plt.axes([pos.x0+pos.width+0.01,
                    pos.y0, 0.02, pos.height])
    cbar = plt.colorbar(c0, cax=cax, ticks=None)
    cbar.set_ticks([])
    cbar.set_clim(np.min(FS), np.max(FS))

    plt.show()

    # Save figure
    if print_fig:
        plt.savefig(save_dir + figname + '.png', dpi=600, bbox_inches="tight")


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
    D.ang2k(D.ang, Ekin=65-4.5, lat_unit=True, a=3.89, b=3.89, c=11,
            V0=0, thdg=-4, tidg=0, phidg=0)

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
        ax.set_position([.1, .3, .2, .6])
        ax.tick_params(**kwargs_ticks)

        # Plot data
        ax.contourf(D.k[0], D.en, np.transpose(int1), 300, **kwargs_ex,
                    vmin=0, vmax=1.4e4)
        ax.plot([-1, 1.66], [0, 0], **kwargs_ef)

        # Plot distribution cuts
        ax.plot([-1, 1.66], [mdc_-mdcw_/2, mdc_-mdcw_/2], **kwargs_cut)
        ax.plot([edc_, edc_], [-2.5, .5], **kwargs_cut)

        # Plot MDC
        ax.plot(D.k[0], (mdc - b_mdc) * 1.3, 'o', ms=1, c='C9')
        ax.fill(D.k[0], (f_mdc - b_mdc) * 1.3, alpha=.2, c='C9')

        # decorate axes
        ax.set_ylabel(r'$\omega$ (eV)', fontdict=font)
        ax.set_xticks([-1, 0, 1])
        ax.set_xticklabels(['S', r'$\Gamma$', 'S'])
        ax.set_yticks(np.arange(-2.5, 1, .5))
        ax.set_xlim(-1, 1.66)
        ax.set_ylim(-2.5, .5)

        # add text
        ax.text(-.9, 0.35, r'(a)', fontdict=font)
        ax.text(.22, .1, r'$\mathcal{C}$', fontsize=15)

    def fig7b():
        ax = fig.add_subplot(132)
        ax.set_position([.32, .3, .2, .6])
        ax.tick_params(**kwargs_ticks)

        # Plot data
        c0 = ax.contourf(D.k[0], D.en+.07, np.transpose(int2), 300,
                         **kwargs_ex, vmin=0, vmax=1.4e4)

        # Plot distribution cuts
        ax.plot([-1, 1.66], [0, 0], **kwargs_ef)
        ax.plot([edc_, edc_], [-2.5, .5], **kwargs_cut)

        # decorate axes
        ax.set_xticks([-1, 0, 1])
        ax.set_xticklabels(['S', r'$\Gamma$', 'S'])
        ax.set_yticks(np.arange(-2.5, 1, .5))
        ax.set_yticklabels([])
        ax.set_xlim(-1, 1.66)
        ax.set_ylim(-2.5, .5)

        # add text
        ax.text(-.9, 0.35, r'(b)', fontdict=font)

        # colorbar
        pos = ax.get_position()
        cax = plt.axes([pos.x0+pos.width+0.01,
                        pos.y0, 0.01, pos.height])
        cbar = plt.colorbar(c0, cax=cax, ticks=None)
        cbar.set_ticks([])
        cbar.set_clim(np.min(D.int), np.max(D.int))

    def fig7c():
        ax = plt.subplot(133)
        ax.set_position([.57, .3, .2, .6])
        ax.tick_params(**kwargs_ticks)

        # Plot EDC's
        xx = np.linspace(1, -5, 200)
        ax.plot(edc1, D.en, 'o', ms=3, c=(0, 0, .8))
        ax.plot(edc2, D.en, 'd', ms=3, c='C0')
        ax.fill(7.4e3 * exponnorm.pdf(-xx, K=2, loc=.63, scale=.2), xx,
                alpha=.2, fc=(0, 0, .8))
        ax.fill(1.3e4 * exponnorm.pdf(-xx, K=2, loc=1.34, scale=.28), xx,
                alpha=.2, fc='C0')

        # Plot Mott gap estimate
        ax.fill_between([0, 1.5e4], 0, -.2, color='C3', alpha=0.2)
        ax.plot([0, 1.5e4], [0, 0], **kwargs_ef)
        ax.plot([0, 1.5e4], [-.2, -.2], 'k:', linewidth=.2)

        # decorate axes
        ax.set_xticks([])
        ax.set_yticks(np.arange(-2.5, 1, .5))
        ax.set_yticklabels([])
        ax.legend((r'63$\,$eV', r'78$\,$eV'), frameon=False)
        ax.set_xlabel('Intensity (a.u)', fontdict=font)
        ax.set_xlim(0, 1.2e4)
        ax.set_ylim(-2.5, 0.5)

        # Add text
        ax.text(1e3, -0.15, r'$\Delta$', fontsize=12)
        ax.text(7e2, 0.35, r'(c)', fontdict=font)
        ax.text(6e3, -.9, r'$\mathcal{A}$', fontsize=15)
        ax.text(6e3, -1.75, r'$\mathcal{B}$', fontsize=15)

    # Plot panels
    fig = plt.figure(figname, figsize=(8, 6), clear=True)
    fig7a()
    fig7b()
    fig7c()
    plt.show()

    # Save figure
    if print_fig:
        plt.savefig(save_dir + figname + '.png', dpi=600, bbox_inches="tight")


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
    D1.ang2k(D1.ang, Ekin=65-4.5, lat_unit=True, a=3.89, b=3.89, c=11,
             V0=0, thdg=5, tidg=12.5, phidg=0)
    D2.ang2k(D2.ang, Ekin=65-4.5, lat_unit=True, a=3.89, b=3.89, c=11,
             V0=0, thdg=5, tidg=12.5, phidg=0)

    edc_ = .35
    edc_val, edc_idx = utils.find(np.flipud(D1.k[0]), edc_)
    edc1 = D1.int_norm[edc_idx, :]
    edc2 = D2.int_norm[edc_idx, :]

    # Figure panels
    def fig8a():
        ax = fig.add_subplot(131)
        ax.set_position([.1, .3, .2, .6])
        ax.tick_params(**kwargs_ticks)

        # Plot data
        ax.contourf(D1.kxs, D1.en_norm+.1, np.flipud(D1.int_norm), 300,
                    **kwargs_ex, vmin=0, vmax=.007)

        # Plot distribution cuts
        ax.plot([-1, 1.66], [0, 0], **kwargs_ef)
        ax.plot([edc_val, edc_val], [-2.5, .5], **kwargs_cut)
        ax.arrow(-1, -1, 0, -.3, head_width=0.2, head_length=0.2,
                 fc='g', ec='k')

        # decorate axes
        ax.set_ylabel(r'$\omega$ (eV)', fontdict=font)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['S', r'$\Gamma$'])
        ax.set_yticks(np.arange(-2.5, 1, .5))
        ax.set_xlim(0, 1)
        ax.set_ylim(-2.5, .5)

        # add text
        ax.text(.05, 0.35, r'(a)', fontdict=font)

    def fig8b():
        ax = plt.subplot(132)
        ax.set_position([.32, .3, .2, .6])
        ax.tick_params(**kwargs_ticks)

        # Plot data
        c0 = ax.contourf(D2.kxs, D2.en_norm+.1, np.flipud(D2.int_norm), 300,
                         **kwargs_ex, vmin=0, vmax=.007)
        ax.plot([-1, 1.66], [0, 0], **kwargs_ef)

        # Plot EDC
        ax.plot([edc_, edc_], [-2.5, .5], **kwargs_cut)

        # decorate axes
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['S', r'$\Gamma$'])
        ax.set_yticks(np.arange(-2.5, 1, .5), ())
        ax.set_yticklabels([])
        ax.set_xlim(0, 1)
        ax.set_ylim(-2.5, .5)

        # add text
        ax.text(.05, 0.35, r'(b)', fontdict=font)

        # colorbar
        pos = ax.get_position()
        cax = plt.axes([pos.x0+pos.width+0.01,
                        pos.y0, 0.01, pos.height])
        cbar = plt.colorbar(c0, cax=cax, ticks=None)
        cbar.set_ticks([])
        cbar.set_clim(np.min(D2.int_norm), np.max(D2.int_norm))

    def fig8c():
        ax = fig.add_subplot(1, 3, 3)
        ax.set_position([.57, .3, .2, .6])
        ax.tick_params(**kwargs_ticks)

        xx = np.linspace(1, -5, 200)
        ax.plot(edc1, D1.en_norm[edc_idx, :]+.1, 'o', ms=3, color=(0, 0, .8))
        ax.plot(edc2 * .8, D2.en_norm[edc_idx, :]+.1, 'd', ms=3, color='C0')
        ax.fill(5.5e-3 * exponnorm.pdf(-xx, K=2, loc=.6, scale=.2), xx,
                alpha=.2, fc=(0, 0, .8))
        ax.fill(5.5e-3 * exponnorm.pdf(-xx, K=2, loc=1.45, scale=.25), xx,
                alpha=.2, fc='C0')

        # Plot Mott gap estimate
        ax.fill_between([0, 1e-2], 0, -.2, color='C3', alpha=0.2)
        ax.plot([0, 1e-2], [0, 0], **kwargs_ef)
        ax.plot([0, 1e-2], [-.2, -.2], 'k:', linewidth=.2)

        # decorate axes
        ax.set_xticks([])
        ax.set_yticks(np.arange(-2.5, 1, .5))
        ax.set_yticklabels([])
        ax.legend((r'$\sigma$-pol.', r'$\pi$-pol.'), frameon=False)
        ax.set_xlabel('Intensity (a.u)', fontdict=font)
        ax.set_xlim(0, .007)
        ax.set_ylim(-2.5, .5)

        # Add text
        ax.text(7e-4, -0.15, r'$\Delta$', fontsize=12)
        ax.text(5e-4, 0.35, r'(c)', fontdict=font)
        ax.text(3.3e-3, -.9, r'$\mathcal{A}$', fontsize=15)
        ax.text(3.3e-3, -1.75, r'$\mathcal{B}$', fontsize=15)

    fig = plt.figure(figname, figsize=(8, 6), clear=True)
    fig8a()
    fig8b()
    fig8c()
    plt.show()

    # Save figure
    if print_fig:
        plt.savefig(save_dir + figname + '.png', dpi=600, bbox_inches="tight")


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
                         **kwargs_th, vmin=0, vmax=.3)
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
        plt.savefig(save_dir + figname + '.png', dpi=600, bbox_inches="tight")


def fig10(print_fig=True):
    """figure 10

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    DFT plot: spaghetti and spectral representation
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    """

    figname = 'CRO10'

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
                ax.plot(k[m], en[m, n], 'o', ms=3,
                        color=(xz[m, n], 0, (1-xz[m, n])))  # orbital plot
        ax.plot([0, 350], [0, 0], **kwargs_ef)

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
                         vmin=0, vmax=25)
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
        plt.savefig(save_dir + figname + '.png', dpi=600, bbox_inches="tight")


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
    ax.fill_between([2, 7], 4, -4, color='C0', alpha=0.2)
    ax.fill_between([7, 14.3], 4, -4, color=(0, 0, .8), alpha=0.2)

    # add text
    ax.text(-1.7, -2.7, r'$d_{xy}$', fontsize=12)
    ax.text(-1.7, -.3, r'$d_{\gamma z}$', fontsize=12)
    ax.text(4., 1.5, r'$3J_\mathrm{H}$', fontsize=12)
    ax.text(9.9, 1.5, r'$U+J_\mathrm{H}$', fontsize=12)
    ax.text(-1.7, 3, r'$| d_4; S=1,\alpha = xy\rangle$', fontsize=8)
    ax.text(2.6, 3, r'$| d_3; \frac{3}{2},\gamma z\rangle$', fontsize=8)
    ax.text(4.6, 3, r'$| d_3; \frac{1}{2},\gamma z\rangle$', fontsize=8)
    ax.text(8.4, 3, r'$| d_3; \frac{1}{2}, xy\rangle$', fontsize=8)
    ax.text(11.5, 3, r'$| d_5; \frac{1}{2}, xy\rangle$', fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-2, 14.3)
    ax.set_ylim(-4, 4)
    plt.show()

    # Save figure
    if print_fig:
        plt.savefig(save_dir + figname + '.png', dpi=600, bbox_inches="tight")


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
        plt.savefig(save_dir + figname + '.png', dpi=600, bbox_inches="tight")


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
        plt.savefig(save_dir + figname + '.png', dpi=600, bbox_inches="tight")


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
        plt.savefig(save_dir + figname + '.png', dpi=600, bbox_inches="tight")
