#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 11:30:51 2018

@author: ilikecarbs

%%%%%%%%%%%%%%%%%
   PhD_defense
%%%%%%%%%%%%%%%%%

**Defense figures**

.. note::
        To-Do:
            -
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.optimize import curve_fit
from scipy import integrate
from scipy.special import sph_harm
from scipy.signal import hilbert, chirp
from scipy.interpolate import interp1d
from scipy.stats import exponnorm

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
font_small = {'family': 'serif',
              'style': 'normal',
              'color': 'black',
              'weight': 'ultralight',
              'size': 8,
              }

kwargs_ex = {'cmap': cm.afmhot_r}  # Experimental plots
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
              'color': 'b',
              'lw': 1}
kwargs_ef = {'linestyle': ':',
             'color': 'k',
             'lw': 1}

# Directory paths
save_dir = '/Users/denyssutter/Documents/2_physics/PhD/PhD_Denys/Figs/'
data_dir = '/Users/denyssutter/Documents/2_physics/PhD/data/'
home_dir = '/Users/denyssutter/Documents/3_library/Python/ARPES'


def fig1(print_fig=True):
    """figure 1

    """

    figname = 'DEFfig1'

    mat = 'CSRO20'
    year = '2017'
    sample = 'S6'

    # load data for FS map
    file = '62087'
    gold = '62081'

    # load data for cut Gamma-X
    file = '62090'
    gold = '62091'
    A1 = ARPES.DLS(file, mat, year, sample)
    A1.norm(gold)
    A1.ang2k(A1.ang, Ekin=22-4.5, lat_unit=True, a=5.2, b=5.55, c=11,
             V0=0, thdg=9.2, tidg=0, phidg=90)

    # MDC
    mdc_ = -.004
    mdcw_ = .002
    mdc = np.zeros(A1.ang.shape)  # placeholder

    # build MDC
    for i in range(len(A1.ang)):
        mdc_val, _mdc = utils.find(A1.en_norm[i, :], mdc_)
        mdcw_val, _mdcw = utils.find(A1.en_norm[i, :], mdc_ - mdcw_)
        mdc[i] = np.sum(A1.int_norm[i, _mdcw:_mdc])
    mdc = mdc / np.max(mdc)  # normalize

    # start MDC fitting
    plt.figure('MDC', figsize=(4, 4), clear=True)
    d = 1e-5

    # initial guess
    p_mdc_i = np.array([-1.4, -1.3, -1.1, -.9, -.7, -.6, -.3, .3,
                        .05, .05, .05, .05, .05, .05, .1, .1,
                        .3, .3, .4, .4, .5, .5, .1, .1,
                        .29, 0.02, .02])

    # fit boundaries
    bounds_bot = np.concatenate((p_mdc_i[0:-3] - np.inf, p_mdc_i[-3:] - d))
    bounds_top = np.concatenate((p_mdc_i[0:-3] + np.inf, p_mdc_i[-3:] + d))
    p_mdc_bounds = (bounds_bot, bounds_top)

    # fit MDC
    p_mdc, cov_mdc = curve_fit(
            utils.lor_8, A1.k[1], mdc, p_mdc_i, bounds=p_mdc_bounds)

    # plot fit and background
    b_mdc = utils.poly_2(A1.k[1], *p_mdc[-3:])
    f_mdc = utils.lor_8(A1.k[1], *p_mdc) - b_mdc
    f_mdc[0] = 0  # for the filling plot to have nice edges
    f_mdc[-1] = 0
    plt.plot(A1.k[1], mdc, 'bo')
    plt.plot(A1.k[1], f_mdc)
    plt.plot(A1.k[1], b_mdc, 'k--')

    def fig1():
        ax = fig.add_subplot(141)
        # ax.set_position([.08, .3, .22, .28])
        ax.set_position([.277, .3, .28, .28])
        ax.tick_params(**kwargs_ticks)

        # plot data
        c0 = ax.contourf(A1.kys+.13, A1.en_norm+.002,
                         np.flipud(A1.int_norm) *
                         utils.poly_2(A1.kys, 1.2, .8, 0),
                         300, **kwargs_ex,
                         vmin=.03*np.max(A1.int_norm),
                         vmax=.7*np.max(A1.int_norm)*.8,
                         zorder=.1)
        ax.set_rasterization_zorder(.2)
        ax.plot([np.min(A1.kys), np.max(A1.kys)*1.25], [0, 0], **kwargs_ef)
#        ax.plot([np.min(A1.kys), np.max(A1.kys)], [-.005, -.005], 'b-.', lw=1)

        # decorate axes
        ax.set_ylim(-.06, .02)
        ax.set_xlim(np.min(A1.kys)*.7, np.max(A1.kys)*1.25)
        ax.set_yticks(np.arange(-.06, .03, .02))
        ax.set_yticklabels(['-60', '-40', '-20', '0', '20'])
        ax.set_xticks(np.arange(-1, 1, 0.5))
        ax.set_xticklabels(['0.0', '0.5', '1.0', '1.5'])
        ax.set_ylabel(r'$\omega\,(\mathrm{meV})$', fontdict=font)
        ax.set_xlabel(r'$k_x (\pi/a), \quad k_y=0$', fontdict=font)
#        ax.plot(A1.k[1], (mdc - b_mdc) / 30 + .001, 'o', ms=1.5, color='C9')
#        ax.fill(A1.k[1], f_mdc / 30 + .001, alpha=.2, color='C9')

        # add text
#        ax.text(-.05, .015, r'$\Gamma$', fontsize=12, color='k')
#        ax.text(-1.05, .015, 'Y', fontsize=12, color='k')

        # labels
#        cols = ['c', 'm', 'b', 'b', 'm', 'c', 'C1', 'C1']
#        lbls = [r'$\alpha$', r'$\gamma$', r'$\beta$',
#                r'$\beta$', r'$\gamma$',
#                r'$\alpha$', r'$\delta$', r'$\delta$']
#
#        # coordinate corrections to label positions
#        corr = np.array([.007, .003, .007, .004, .001, .009, -.003, -.004])
#        p_mdc[6 + 16] *= 1.5

        # plot MDC fits
#        for i in range(8):
#            ax.plot(A1.k[1], (utils.lor(A1.k[1], p_mdc[i], p_mdc[i+8],
#                    p_mdc[i+16],
#                    p_mdc[-3], p_mdc[-2], p_mdc[-1]) - b_mdc) / 30 + .001,
#                    lw=.5, color=cols[i])
#            ax.text(p_mdc[i+16]/5+corr[i], p_mdc[i]-.03, lbls[i],
#                    fontsize=10, color=cols[i])
#        ax.plot(A1.k[1], f_mdc / 30 + .001, color='k', lw=.5)

#        pos = ax.get_position()
#        cax = plt.axes([pos.x0+pos.width+0.01,
#                        pos.y0, 0.01, pos.height])
#        cbar = plt.colorbar(c0, cax=cax, ticks=None)
#        cbar.set_ticks([])
#        cbar.set_clim(np.min(A1.int_norm), np.max(A1.int_norm))

    fig = plt.figure(figname, figsize=(10, 10), clear=True)
    fig1()
    fig.show()

    # Save figure
    if print_fig:
        fig.savefig(save_dir + figname + '.pdf', dpi=100,
                    bbox_inches="tight", rasterized=True)


def fig2(print_fig=True):
    """figure 2

    """

    figname = 'DEFfig2'

    mat = 'CSRO20'
    year = '2017'
    sample = 'S6'

    # load data for cut X-S
    file = '62097'
    gold = '62091'
    A2 = ARPES.DLS(file, mat, year, sample)
    A2.norm(gold)
    A2.ang2k(A2.ang, Ekin=22-4.5, lat_unit=True, a=5.2, b=5.7, c=11,
             V0=0, thdg=9.2-3.5, tidg=-16, phidg=90)

    def fig2():
        ax = fig.add_subplot(143)
        ax.set_position([.3, .3, .28, .28])
        ax.tick_params(**kwargs_ticks)

        # plot data
        c0 = ax.contourf(A2.kys-.25, A2.en_norm+.003,
                         np.flipud(A2.int_norm), 300,
                         **kwargs_ex, zorder=.1,
                         vmin=.1*np.max(A2.int_norm),
                         vmax=.7*np.max(A2.int_norm)*.8)
        ax.set_rasterization_zorder(.2)
        ax.plot([np.min(A2.kys), np.max(A2.kys)], [0, 0], **kwargs_ef)

        # decorate axes
        ax.set_yticks(np.arange(-.06, .04, .02))
        ax.set_yticklabels([])
        ax.set_xticks(np.arange(-1, 1, .5))
        ax.set_xticklabels(['0.0', '0.5', '1.0', '1.5'])
        ax.set_xlabel(r'$k_x (\pi/a), \quad k_y=\pi/b$', fontdict=font)
#        ax.set_ylabel(r'$\omega\,(\mathrm{meV})$', fontdict=font)
        ax.set_ylim(-.06, .02)
        ax.set_xlim(np.min(A2.kys), 0.5)

        # add text
#        ax.text(-.05, .015, 'X', fontsize=12, color='k')
#        ax.text(-1.05, .015, 'S', fontsize=12, color='k')

    fig = plt.figure(figname, figsize=(10, 10), clear=True)
    fig2()
    fig.show()

    # Save figure
    if print_fig:
        fig.savefig(save_dir + figname + '.pdf', dpi=100,
                    bbox_inches="tight", rasterized=True)


def fig3(print_fig=True):
    """figure 3

    """

    figname = 'DEFfig3'

    mat = 'CSRO20'
    year = '2017'
    sample = 'S6'

    # load data for FS map
    file = '62087'
    gold = '62081'

    e = .01  # start from above EF
    ew = .015  # integration window (5meV below EF)
    D = ARPES.DLS(file, mat, year, sample)
    D.norm(gold=gold)
    D.restrict(bot=0, top=1, left=.12, right=.9)
    D.FS(e=e, ew=ew)
    D.FS_flatten()

    # distortion of spectrum corrected with a/b
    D.ang2kFS(D.ang, Ekin=22-4.5, lat_unit=True, a=5.2, b=5.7, c=11,
              V0=0, thdg=8.7, tidg=4, phidg=88)

    # useful for panels
    ratio = (np.max(D.ky) - np.min(D.ky))/(np.max(D.kx) - np.min(D.kx))
    print(ratio)

    # load data for cut Gamma-X
    file = '62090'
    gold = '62091'
    A1 = ARPES.DLS(file, mat, year, sample)
    A1.norm(gold)
    A1.ang2k(A1.ang, Ekin=22-4.5, lat_unit=True, a=5.2, b=5.55, c=11,
             V0=0, thdg=9.2, tidg=0, phidg=90)

    # load data for cut X-S
    file = '62097'
    gold = '62091'
    A2 = ARPES.DLS(file, mat, year, sample)
    A2.norm(gold)
    A2.ang2k(A2.ang, Ekin=22-4.5, lat_unit=True, a=5.2, b=5.7, c=11,
             V0=0, thdg=9.2-3.5, tidg=-16, phidg=90)

    # TB
    param = utils.paramCSRO20_opt()  # Load parameters

    # MDC
    mdc_ = -.004
    mdcw_ = .002
    mdc = np.zeros(A1.ang.shape)  # placeholder

    # build MDC
    for i in range(len(A1.ang)):
        mdc_val, _mdc = utils.find(A1.en_norm[i, :], mdc_)
        mdcw_val, _mdcw = utils.find(A1.en_norm[i, :], mdc_ - mdcw_)
        mdc[i] = np.sum(A1.int_norm[i, _mdcw:_mdc])
    mdc = mdc / np.max(mdc)  # normalize

    # start MDC fitting
    plt.figure('MDC', figsize=(4, 4), clear=True)
    d = 1e-5

    # initial guess
    p_mdc_i = np.array([-1.4, -1.3, -1.1, -.9, -.7, -.6, -.3, .3,
                        .05, .05, .05, .05, .05, .05, .1, .1,
                        .3, .3, .4, .4, .5, .5, .1, .1,
                        .29, 0.02, .02])

    # fit boundaries
    bounds_bot = np.concatenate((p_mdc_i[0:-3] - np.inf, p_mdc_i[-3:] - d))
    bounds_top = np.concatenate((p_mdc_i[0:-3] + np.inf, p_mdc_i[-3:] + d))
    p_mdc_bounds = (bounds_bot, bounds_top)

    # fit MDC
    p_mdc, cov_mdc = curve_fit(
            utils.lor_8, A1.k[1], mdc, p_mdc_i, bounds=p_mdc_bounds)

    # plot fit and background
    b_mdc = utils.poly_2(A1.k[1], *p_mdc[-3:])
    f_mdc = utils.lor_8(A1.k[1], *p_mdc) - b_mdc
    f_mdc[0] = 0  # for the filling plot to have nice edges
    f_mdc[-1] = 0
    plt.plot(A1.k[1], mdc, 'bo')
    plt.plot(A1.k[1], f_mdc)
    plt.plot(A1.k[1], b_mdc, 'k--')

    def fig3():
        ax = fig.add_subplot(142)
        # ax.set_position([.31, .3, .28/ratio, .28])
        ax.set_position([.277+.01+.22, .3, .28/ratio, .28])
        ax.tick_params(**kwargs_ticks)

        # plot data
        ax.contourf(D.kx, D.ky, np.flipud(D.map), 300, **kwargs_ex, zorder=.1,
                    vmax=.9 * np.max(D.map)*.8, vmin=.3 * np.max(D.map))
        ax.set_rasterization_zorder(.2)
#        ax.plot(A1.k[0], A1.k[1], 'b-.', lw=1)
#        ax.plot(A2.k[0], A2.k[1], 'b-.', lw=1)

        # decorate axes
        ax.set_xlabel(r'$k_y \,(\pi/b)$', fontdict=font)
        ax.set_ylabel(r'$k_x \,(\pi/a)$', fontdict=font)
#        ax.set_xticklabels([])

        # add text
        ax.text(-.05, -.03, r'$\Gamma$', fontsize=12, color='k')
        ax.text(-.05, -1.03, r'Y', fontsize=12, color='w')
        ax.text(.95, -.03, r'X', fontsize=12, color='w')
        ax.text(.95, -1.03, r'S', fontsize=12, color='k')

        # Tight Binding Model
        tb = utils.TB(a=np.pi, kbnd=2, kpoints=200)  # Initialize
        tb.CSRO(param)  # Calculate bandstructure

        plt.figure(figname)
        bndstr = tb.bndstr  # Load bandstructure
        coord = tb.coord  # Load coordinates

        # read dictionaries
        X = coord['X']
        Y = coord['Y']
        Axy = bndstr['Axy']
        Bxz = bndstr['Bxz']
        Byz = bndstr['Byz']
        bands = (Axy, Bxz, Byz)

#         loop over bands
#        n = 0  # counter
#        for band in bands:
#            n += 1
#            ax.contour(X, Y, band, colors='w', linestyles=':', levels=0,
#                       linewidths=1.5)

        ax.set_xticks([-.5, 0, .5, 1])
        ax.set_yticks([-1.5, -1, -.5, 0, .5])
#        ax.set_yticklabels([])
        ax.set_xlim(np.min(D.kx), np.max(D.kx))
        ax.set_ylim(np.min(D.ky), np.max(D.ky))

    fig = plt.figure(figname, figsize=(10, 10), clear=True)
#    fig1a()
    fig3()
    fig.show()

    # Save figure
    if print_fig:
        fig.savefig(save_dir + figname + '.pdf', dpi=100,
                    bbox_inches="tight", rasterized=True)


def fig4(print_fig=True):
    """figure 4

    """

    figname = 'DEFfig4'

    # Load and prepare experimental data
    file = 25
    gold = 14
    mat = 'CSRO20'
    year = 2017
    sample = 'S1'

    D = ARPES.Bessy(file, mat, year, sample)

    D.norm(gold)

    D.restrict(bot=.7, top=.9, left=0, right=1)

    D.ang2k(D.ang, Ekin=48, lat_unit=True, a=5.5, b=5.5, c=11,
            V0=0, thdg=2.4, tidg=0, phidg=45)

    data = (D.int_norm, )
    en = (D.en_norm - .008, )
    ks = (D.kxs, )
    k = (D.k[0], )
    b_par = (np.array([.0037, .0002, .002]),
             np.array([.0037, .0002, .002]),
             np.array([.0037+.0005, .0002, .002]))

    def fig4():
        j = 0
        plt.figure('MDC')
        ax = plt.subplot(2, 3, j+1)
        ax.set_position([.3, .3, .35, .35])

        # build MDC
        mdc_ = -.005
        mdcw_ = .015
        mdc = np.zeros(k[j].shape)
        for i in range(len(k[j])):
            mdc_val, mdc_idx = utils.find(en[j][i, :], mdc_)
            mdcw_val, mdcw_idx = utils.find(en[j][i, :], mdc_ - mdcw_)
            mdc[i] = np.sum(data[j][i, mdcw_idx:mdc_idx])

        # background
        b_mdc = utils.poly_2(k[j], b_par[j][0], b_par[j][1], b_par[j][2])
        ax.plot(k[j], mdc, 'bo')
        ax.plot(k[j], b_mdc, 'k--')

        # create figure
        ax = plt.figure(figname)
        ax = plt.subplot(2, 4, j+1)
        ax.set_position([.08 + j * .21, .5, .35, .35])
        ax.tick_params(**kwargs_ticks)

        ax.contourf(ks[j]+.37, en[j]-.001, np.flipud(data[j]), 300, **kwargs_ex,
                    vmin=.05*np.max(data[j]), vmax=.35*np.max(data[j])*.8,
                    zorder=.1)
        ax.set_rasterization_zorder(.2)
#                for bands in range(6):
#                    TB_D[bands][TB_D[bands] > 0] = 10
#                    ax.plot(k[j], TB_D[bands], 'wo', ms=.5, alpha=.2)
        mdc = mdc / np.max(mdc) / 2

        # decorate axes
        ax.set_yticks(np.arange(-.1, .05, .02))
        ax.set_yticklabels([])
        ax.set_xticks(np.arange(0, 1.5, 0.5))
#        ax.set_ylabel(r'$\omega\,(\mathrm{meV})$', fontdict=font)
        ax.set_xlabel(r'$k_x (\pi/a), \quad k_y=k_x$', fontdict=font)

        ax.plot([np.min(ks[j]), np.max(ks[j])*1.5], [0, 0], **kwargs_ef)
#        ax.plot([np.min(ks[j]), np.max(ks[j])], [mdc_val, mdc_val],
#                'b-.', lw=.5)

        # decorate axes
        ax.set_xlim(-.1*np.min(ks[0]), np.max(ks[0])*1.3)
        ax.set_ylim(-.06, .02)
#        ax.text(-1.05, .015, 'S', fontsize=12, color='k')
#        ax.text(-.05, .015, r'$\Gamma$', fontsize=12, color='k')

        # plot MDC
#        mdc[0] = 0
#        mdc[-1] = 0
#        ax.plot(k[j], mdc / 30 + .002, 'o', ms=1, color='C9')
#        ax.fill(k[j], mdc / 30 + .002, alpha=.2, color='C9')

    plt.figure('MDC', figsize=(8, 8), clear=True)
    plt.figure(figname, figsize=(8, 8), clear=True)
    fig4()
    plt.show()

    # Save figure
    if print_fig:
        plt.savefig(save_dir + figname + '.pdf', dpi=100,
                    bbox_inches="tight", rasterized=True)


def fig5(print_fig=True):
    """figure 5

    """

    figname = 'DEFfig5'

    # Load calculations
    os.chdir(data_dir)
    xz_data = np.loadtxt('DMFT_CSRO_xz.dat')
    xy_data = np.loadtxt('DMFT_CSRO_xy.dat')
    xz_lda = np.loadtxt('LDA_CSRO_xz.dat')
    yz_lda = np.loadtxt('LDA_CSRO_yz.dat')
    xy_lda = np.loadtxt('LDA_CSRO_xy.dat')
    os.chdir(home_dir)

    # Prepare data
    m, n = 8000, 351  # dimensions energy, full k-path
    bot, top = 3000, 6000  # restrict energy window
    data = np.array([xz_lda + yz_lda + xy_lda, xz_data, xy_data])
    spec = np.reshape(data[:, :, 2], (3, n, m))  # reshape into n,m
    spec = spec[:, :, bot:top]  # restrict data to bot, top
    spec_en = np.linspace(-8, 8, m)  # define energy data
    spec_en = spec_en[bot:top]  # restrict energy data

    # Full k-path
    # [0, 56, 110, 187, 241, 266, 325, 350]  = [G, X, S, G, Y, T, G, Z]
    spec = np.transpose(spec, (0, 2, 1))
    kB = 8.617e-5  # Boltzmann constant
    T = 39  # temperature

    p_FDsl = [kB * T, 0, 1, 0, 0]
    bkg = utils.FDsl(spec_en, *p_FDsl)
    bkg = bkg[:, None]

    # Load and prepare experimental data
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

    D.ang2k(D.ang, Ekin=48, lat_unit=True, a=5.5, b=5.5, c=11,
            V0=0, thdg=2.4, tidg=0, phidg=45)
    LH.ang2k(LH.ang, Ekin=48, lat_unit=True, a=5.5, b=5.5, c=11,
             V0=0, thdg=2.4, tidg=0, phidg=45)
    LV.ang2k(LV.ang, Ekin=48, lat_unit=True, a=5.5, b=5.5, c=11,
             V0=0, thdg=2.4, tidg=0, phidg=45)

    # TB
    # TB_D = utils.CSRO_eval(D.k[0], D.k[1])

    # Collect data
    data = (D.int_norm, LH.int_norm, LV.int_norm)
    en = (D.en_norm - .008, LH.en_norm, LV.en_norm)
    ks = (D.kxs, LH.kxs, LV.kxs)
    k = (D.k[0], LH.k[0], LV.k[0])
    b_par = (np.array([.0037, .0002, .002]),
             np.array([.0037, .0002, .002]),
             np.array([.0037+.0005, .0002, .002]))

    def fig5abc():
        lbls = [r'C$^+$-pol.', r'$\bar{\pi}$-pol.',
                r'$\bar{\sigma}$-pol.']

        for j in range(3):
            plt.figure('MDC')
            ax = plt.subplot(2, 3, j+1)
            ax.set_position([.08+j*.26, .5, .25, .25])

            # build MDC
            mdc_ = -.005
            mdcw_ = .015
            mdc = np.zeros(k[j].shape)
            for i in range(len(k[j])):
                mdc_val, mdc_idx = utils.find(en[j][i, :], mdc_)
                mdcw_val, mdcw_idx = utils.find(en[j][i, :], mdc_ - mdcw_)
                mdc[i] = np.sum(data[j][i, mdcw_idx:mdc_idx])

            # background
            b_mdc = utils.poly_2(k[j], b_par[j][0], b_par[j][1], b_par[j][2])
            ax.plot(k[j], mdc, 'bo')
            ax.plot(k[j], b_mdc, 'k--')

            # create figure
            ax = plt.figure(figname)
            ax = plt.subplot(2, 4, j+1)
            ax.set_position([.08 + j * .21, .5, .2, .2])
            ax.tick_params(**kwargs_ticks)

            # plot data
            if j == 0:
                ax.contourf(ks[j], en[j], data[j], 300, **kwargs_ex,
                            vmin=.05*np.max(data[j]),
                            vmax=.35*np.max(data[j])*.85,
                            zorder=.1)
                ax.set_rasterization_zorder(.2)
#                for bands in range(6):
#                    TB_D[bands][TB_D[bands] > 0] = 10
#                    ax.plot(k[j], TB_D[bands], 'wo', ms=.5, alpha=.2)
                mdc = mdc / np.max(mdc)

                # decorate axes
                ax.set_yticks(np.arange(-.08, .05, .04))
                ax.set_yticklabels(['-80', '-40', '0', '40'])
                ax.set_ylabel(r'$\omega\,(\mathrm{meV})$', fontdict=font)
            else:
                c0 = ax.contourf(ks[j], en[j], data[j], 300, **kwargs_ex,
                                 vmin=.3*np.max(data[1]),
                                 vmax=.6*np.max(data[1])*.9, zorder=.1)
                ax.set_rasterization_zorder(.2)
                mdc = (mdc - b_mdc) / .007

                # decorate axes
                ax.set_yticks(np.arange(-.1, .05, .02))
                ax.set_yticklabels([])

            ax.plot([np.min(ks[j]), np.max(ks[j])], [0, 0], **kwargs_ef)
            ax.plot([np.min(ks[j]), np.max(ks[j])], [mdc_val, mdc_val],
                    'b-.', lw=1)

            # decorate axes
            ax.set_xticks(np.arange(-1, 1, .5))
            ax.set_xticklabels([])
            ax.set_xlim(np.min(ks[0]), np.max(ks[0]))
            ax.set_ylim(-.1, .05)

            # plot MDC
            mdc[0] = 0
            mdc[-1] = 0
            ax.plot(k[j], mdc / 30 + .002, 'o', ms=1, color='C9')
            ax.fill(k[j], mdc / 30 + .002, alpha=.2, color='C9')

            # add text
            ax.text(-1.28, .038, lbls[j])

        # colorbar
        pos = ax.get_position()
        cax = plt.axes([pos.x0+pos.width + 0.01,
                        pos.y0, 0.01, pos.height])
        cbar = plt.colorbar(c0, cax=cax, ticks=None)
        cbar.set_ticks([])
        cbar.set_clim(np.min(LV.int_norm), np.max(LV.int_norm))

    def fig5efg():
        fig = plt.figure(figname)
        lbls = [r'LDA $\Sigma_\mathrm{orb}$', r'DMFT $d_{xz}$',
                r'DMFT $d_{xy}$']

        for j in range(3):
            SG = spec[j, :, 110:187] * bkg  # add Fermi Dirac
            GS = np.fliplr(SG)
            spec_full = np.concatenate((GS, SG, GS), axis=1)
            spec_k = np.linspace(-2, 1, spec_full.shape[1])

            # Plot DMFT
            ax = fig.add_subplot(2, 4, j + 4)
            ax.set_position([.08+j*.21, .29, .2, .2])
            ax.tick_params(**kwargs_ticks)
            c0 = ax.contourf(spec_k, spec_en, spec_full, 300, **kwargs_th,
                             vmin=.5, vmax=6, zorder=.1)
            ax.set_rasterization_zorder(.2)

            n_0 = .3
            n_w = .2
            n_h = .03
            n_lbls = [r'$xz$', r'$yz$', r'$xy$']

            if j == 0:
                for nn in range(3):
                    ax.plot([n_0+nn*n_w+.05, n_0+nn*n_w+.05], [.005, .005+n_h],
                            'k-', lw=.5)
                    ax.plot([n_0+nn*n_w+.15, n_0+nn*n_w+.15], [.005, .005+n_h],
                            'k-', lw=.5)
                    ax.plot([n_0+nn*n_w+.05, n_0+nn*n_w+.15], [.005, .005],
                            'k-', lw=.5)
                    ax.plot([n_0+nn*n_w+.05, n_0+nn*n_w+.15],
                            [.005+n_h, .005+n_h],
                            'k-', lw=.5)
                    ax.plot([n_0+nn*n_w+.05, n_0+nn*n_w+.15],
                            [.005+.66*n_h, .005+.66*n_h],
                            'k:', lw=.5)
                    ax.fill_between([n_0+nn*n_w+.05, n_0+nn*n_w+.15],
                                    .005, .005+.66*n_h,
                                    color='C8', alpha=.5)
                    ax.text(n_0+nn*n_w+.02, .038, n_lbls[nn], fontsize=8)

            elif j == 1:
                for nn in range(3):
                    ax.plot([n_0+nn*n_w+.05, n_0+nn*n_w+.05], [.005, .005+n_h],
                            'k-', lw=.5)
                    ax.plot([n_0+nn*n_w+.15, n_0+nn*n_w+.15], [.005, .005+n_h],
                            'k-', lw=.5)
                    ax.plot([n_0+nn*n_w+.05, n_0+nn*n_w+.15], [.005, .005],
                            'k-', lw=.5)
                    ax.plot([n_0+nn*n_w+.05, n_0+nn*n_w+.15],
                            [.005+n_h, .005+n_h],
                            'k-', lw=.5)
                    if nn == 2:
                        ax.plot([n_0+nn*n_w+.05, n_0+nn*n_w+.15],
                                [.005+.5*n_h, .005+.5*n_h],
                                'k:', lw=.5)
                        ax.fill_between([n_0+nn*n_w+.05, n_0+nn*n_w+.15],
                                        .005, .005+.5*n_h,
                                        color='darkgoldenrod', alpha=.5)
                    else:
                        ax.plot([n_0+nn*n_w+.05, n_0+nn*n_w+.15],
                                [.005+.75*n_h, .005+.75*n_h],
                                'k:', lw=.5)
                        ax.fill_between([n_0+nn*n_w+.05, n_0+nn*n_w+.15],
                                        .005, .005+.75*n_h,
                                        color='khaki', alpha=.5)
                    ax.text(n_0+nn*n_w+.02, .038, n_lbls[nn], fontsize=8)
            # decorate axes
            if j == 0:
                ax.set_yticks(np.arange(-.08, .05, .04))
                ax.set_yticklabels(['-80', '-40',
                                    '0', '40'])
                ax.set_ylabel(r'$\omega\,(\mathrm{meV})$', fontdict=font)
            else:
                ax.set_yticks(np.arange(-.1, .05, .02))
                ax.set_yticklabels([])
            ax.set_xticks(np.arange(-1, 1, .5))
            ax.set_xticklabels([r'S', r'', r'$\Gamma$', ''])
            ax.plot([np.min(spec_k), np.max(spec_k)], [0, 0], **kwargs_ef)
            ax.set_xlim(np.min(ks[0]), np.max(ks[0]))
            ax.set_ylim(-.1, .05)
            ax.text(-1.28, .036, lbls[j])

        # colorbar
        pos = ax.get_position()
        cax = plt.axes([pos.x0+pos.width + 0.01,
                        pos.y0, 0.01, pos.height])
        cbar = plt.colorbar(c0, cax=cax, ticks=None)
        cbar.set_ticks([])
        cbar.set_clim(np.min(spec_full), np.max(spec_full))

    def fig5dh():
        fig = plt.figure(figname)

        ax_d = fig.add_subplot(247, projection='3d')
        ax_h = fig.add_subplot(248, projection='3d')
        ax_d.set_position([.73, .5, .2, .2])
        ax_h.set_position([.73, .29, .2, .2])
        ax_d.tick_params(**kwargs_ticks)
        ax_h.tick_params(**kwargs_ticks)

        ax_d_b = fig.add_axes([.73, .5, .2, .2])
        ax_h_b = fig.add_axes([.73, .29, .2, .2])
        ax_d_b.tick_params(**kwargs_ticks)
        ax_h_b.tick_params(**kwargs_ticks)
        ax_d_b.patch.set_alpha(0)
        ax_h_b.patch.set_alpha(0)
        ax_d_b.set_xticks([])
        ax_h_b.set_xticks([])
        ax_d_b.set_yticks([])
        ax_h_b.set_yticks([])

        # Create a sphere
        k_i = 3

        k_phi = np.pi/2
        k_th = np.pi/4

        theta_1d = np.linspace(0, np.pi, 300)
        phi_1d = np.linspace(0, 2*np.pi, 300)

        theta_2d, phi_2d = np.meshgrid(theta_1d, phi_1d)
        xyz_2d = np.array([np.sin(theta_2d) * np.sin(phi_2d),
                          np.sin(theta_2d) * np.cos(phi_2d),
                          np.cos(theta_2d)])

        colormap = cm.ScalarMappable(cmap=plt.get_cmap("PRGn"))
        colormap.set_clim(-.45, .45)

        l_ = 2
        dxy = (1j * (sph_harm(-2, l_, phi_2d, theta_2d)
               - sph_harm(2, l_, phi_2d, theta_2d))
               / np.sqrt(2))
        dxz = ((sph_harm(-1, l_, phi_2d, theta_2d)
               - sph_harm(1, l_, phi_2d, theta_2d))
               / np.sqrt(2))

        dxy_r = np.abs(dxy.real)*xyz_2d
        dxz_r = np.abs(dxz.real)*xyz_2d

        # Cylinder
        r = 3
        x_cyl = np.linspace(-r, r, 100)
        z_cyl = np.linspace(-1, 0, 100)
        X_cyl, Z_cyl = np.meshgrid(x_cyl, z_cyl)
        Y_cyl = np.sqrt(r**2 - X_cyl**2)

        x_cir = r * np.cos(np.linspace(0, 2*np.pi, 360))
        y_cir = r * np.sin(np.linspace(0, 2*np.pi, 360))

        R, Phi = np.meshgrid(np.linspace(0, r, 100),
                             np.linspace(0, 2*np.pi, 100))

        X_cir = R * np.cos(Phi)
        Y_cir = R * np.sin(Phi)
        Z_ceil = np.zeros((100, 100))
        Z_floor = -np.ones((100, 100))

        ax_d.plot_surface(dxz_r[0]*.0, dxz_r[1]*3,
                          dxz_r[2]*3, alpha=.1,
                          facecolors=colormap.to_rgba(dxz.real),
                          rstride=2, cstride=2, zorder=.1)

        ax_h.plot_surface(dxy_r[0]*3, dxy_r[1]*3,
                          dxy_r[2]*.0, alpha=.1,
                          facecolors=colormap.to_rgba(dxy.real),
                          rstride=2, cstride=2, zorder=.1)

        X = np.zeros((2, 2))
        z = [0, 3.5]
        y = [-3, 3]
        Y, Z = np.meshgrid(y, z)
        ax_d.plot_surface(X, Y, Z, alpha=.2, color='C8', zorder=.1)
        ax_h.plot_surface(X, Y, Z, alpha=.2, color='C8', zorder=.1)
        ax_d.set_rasterization_zorder(.2)
        ax_h.set_rasterization_zorder(.2)

        # angdg = np.linspace(-15, 15, 100)
        # tidg = 0
        # k_1 = utils.det_angle(4, angdg, -40, tidg, 90)
        # k_2 = utils.det_angle(4, angdg, 0, 40, 0)
        y_hv = np.linspace(-2, -.25, 100)
        x_hv = .2*np.sin(y_hv*25)
        z_hv = -y_hv

        kx_i = k_i * np.sin(k_th) * np.cos(k_phi)
        ky_i = k_i * np.sin(k_th) * np.sin(k_phi)
        kz_i = k_i * np.cos(k_th)

        kwargs_cyl = {'alpha': .05, 'color': 'k'}  # keywords cylinder

        for ax in [ax_d, ax_h]:
            # draw cylinder
            ax.plot_surface(X_cyl, Y_cyl, Z_cyl, **kwargs_cyl, zorder=.1)
            ax.plot_surface(X_cyl, -Y_cyl, Z_cyl, **kwargs_cyl, zorder=.1)
            ax.plot_surface(X_cir, Y_cir, Z_floor, **kwargs_cyl, zorder=.1)
            ax.plot_surface(X_cir, Y_cir, Z_ceil, **kwargs_cyl, zorder=.1)
            ax.plot(x_cir, y_cir, 'k-', alpha=.1, lw=.5, zorder=.1)
            ax.plot(x_cir, y_cir, -1, 'k--', alpha=.1, lw=.5, zorder=.1)
            ax.set_rasterization_zorder(.2)
            ax.axis('off')
            ax.view_init(elev=20, azim=50)

            ax.quiver(0, 0, 0, kx_i-.1, ky_i-.1, kz_i-.1,
                      arrow_length_ratio=.08,
                      lw=1, color='r', zorder=.1)
            ax.quiver(x_hv[-1], y_hv[-1], z_hv[-1], .1, .3, -.2,
                      arrow_length_ratio=.6, color='c', zorder=.1)
            ax.plot([0, 0], [0, 0], [0, 0], 'o', mec='k', mfc='w', ms=3,
                    zorder=.1)
            ax.plot([kx_i, kx_i], [ky_i, ky_i], [kz_i, kz_i],
                    'o', mec='k', mfc='k', ms=3, zorder=.1)
            ax.plot(x_hv, y_hv, z_hv, 'c', lw=1, zorder=.1)
            ax.set_xlim([-2, 2])
            ax.set_ylim([-2, 2])
            ax.set_zlim([-1, 4])
            ax.set_aspect("equal")

            # add text
            ax.text(0, -2.5, 2, r'$hv$', fontdict=font_small)
            ax.text(0, 2.3, 2., r'$e^-$', fontdict=font_small)
            ax.text(2.8, 2.2, -0.33, 'SAMPLE', fontdict=font_small)
            ax.text(0, 0, 2.2, 'Mirror plane', fontdict=font_small)
        ax_d.quiver(0, -1.5, 1.5, 0, 1, 1, arrow_length_ratio=.2,
                    color='b', lw=1, zorder=.1)
        ax_h.quiver(0, -1.5, 1.5, 1, 0, 0, arrow_length_ratio=.2,
                    color='b', lw=1, zorder=.1)
        ax_d.text(.8, 0, 2.1, r'$\bar{\pi}$', fontdict=font_small)
        ax_h.text(1.9, 0, 1.8, r'$\bar{\sigma}$', fontdict=font_small)
        ax_d.text(0, -6, 3.5, r'$d_{xz}$')
        ax_h.text(0, -6, 3.5, r'$d_{xy}$')

    # Plotting
    plt.figure('MDC', figsize=(8, 8), clear=True)
    plt.figure(figname, figsize=(8, 8), clear=True)
    fig5abc()
    fig5efg()
    fig5dh()
    plt.show()

    # Save figure
    if print_fig:
        plt.savefig(save_dir + figname + '.png', dpi=400,
                    bbox_inches="tight", rasterized=False)


def fig6(print_fig=True, load=True):
    """figure 6

    """

    figname = 'DEFfig6'

    if load:
        os.chdir(data_dir)
        v_LDA_data = np.loadtxt('Data_CSROfig8_v_LDA.dat')
        v_LDA = v_LDA_data[0]
        ev_LDA = v_LDA_data[1]
        os.chdir(home_dir)
    else:
        v_LDA, ev_LDA = fig8()

    files = [25, 26, 27, 28]
    gold = 14
    mat = 'CSRO20'
    year = 2017
    sample = 'S1'

    # Create Placeholders
    spec = ()  # ARPES spectra
    espec = ()  # Errors on signal
    en = ()  # energy scale
    k = ()  # momenta
    Z = ()  # quasiparticle residuum
    eZ = ()  # error
    Re = ()  # Real part of self energy
    Width = ()  # MDC Half width at half maximum
    eWidth = ()  # error
    Loc_en = ()  # Associated energy
    mdc_t_val = .001  # start energy of MDC analysis
    mdc_b_val = -.1  # end energy of MDC analysis
    n_spec = 4  # how many temperatures are analysed
    scale = 5e-5  # helper variable for plotting

    # Colors
    cols = ([0, 0, 1], [0, .7, .7], [0, .4, .4], [0, 0, 0])
    cols_r = ([0, 0, 0], [0, .4, .4], [0, .7, .7], [0, 1, 1])
    Re_cols = ['khaki', 'darkkhaki', 'goldenrod', 'darkgoldenrod']
    Re_cols_r = ['darkgoldenrod', 'goldenrod', 'darkkhaki', 'khaki']
    xx = np.arange(-.4, .25, .01)  # helper variable for plotting

    for j in range(n_spec):

        # Load Bessy data
        D = ARPES.Bessy(files[j], mat, year, sample)
        if j == 0:
            D.int_amp(1.52)  # renoramlize intensity for this spectrum
        D.norm(gold=gold)
        D.bkg()
        D.restrict(bot=.7, top=.9, left=.31, right=.56)
        if j == 0:
            D.ang2k(D.ang, Ekin=48, lat_unit=False, a=5.5, b=5.5, c=11,
                    V0=0, thdg=2.4, tidg=0, phidg=45)
        else:
            D.ang2k(D.ang, Ekin=48, lat_unit=False, a=5.5, b=5.5, c=11,
                    V0=0, thdg=2.8, tidg=0, phidg=45)
        int_norm = D.int_norm
        eint_norm = D.eint_norm
        en_norm = D.en_norm - .008  # Fermi level adjustment

        # collect data
        spec = spec + (int_norm,)
        espec = espec + (eint_norm,)
        en = en + (en_norm,)

        # D.kxs is only kx -> but we analyze along diagonal
        k = k + (D.kxs * np.sqrt(2),)

    fig = plt.figure(figname, figsize=(10, 10), clear=True)

    # labels
    titles = [r'$T=1.3\,$K', r'$T=10\,$K', r'$T=20\,$K', r'$T=30\,$K']
    lbls = [r'(a)', r'(b)', r'(c)', r'(d)',
            r'(e)', r'(f)', r'(g)', r'(h)',
            r'(i)', r'(j)', r'(k)', r'(l)',
            r'(k)', r'(l)', r'(m)', r'(n)']
    for j in range(n_spec):
        val, _mdc_t = utils.find(en[j][0, :], mdc_t_val)  # Get indices
        val, _mdc_b = utils.find(en[j][0, :], mdc_b_val)  # Get indices
        mdc_seq = np.arange(_mdc_t, _mdc_b, -1)  # range of indices
        loc = np.zeros((_mdc_t - _mdc_b))  # placeholders maximum position
        eloc = np.zeros((_mdc_t - _mdc_b))  # corresponding errors
        width = np.zeros((_mdc_t - _mdc_b))  # placheholder HWHM
        ewidth = np.zeros((_mdc_t - _mdc_b))  # corresponding error

        # first row
        ax = fig.add_subplot(4, 4, j+1)
        ax.set_position([.08+j*.21, .76, .2, .2])
        ax.tick_params(**kwargs_ticks)

        # plot spectra
        c0 = ax.contourf(k[j], en[j], spec[j], 300, **kwargs_ex,
                         vmin=.05 * np.max(spec[0]), vmax=.75 * np.max(spec[0]),
                         zorder=.1)
        ax.set_rasterization_zorder(.2)
        ax.plot([-1, 0], [en[j][0, mdc_seq[2]], en[j][0, mdc_seq[2]]],
                **kwargs_cut)
        ax.plot([-1, 0], [en[j][0, mdc_seq[50]], en[j][0, mdc_seq[50]]],
                **kwargs_cut)
        ax.plot([-1, 0], [en[j][0, mdc_seq[100]], en[j][0, mdc_seq[100]]],
                **kwargs_cut)
        ax.plot([-1, 0], [0, 0], **kwargs_ef)

        # decorate axes
        if j == 0:
            ax.set_ylabel(r'$\omega\,(\mathrm{meV})$', fontdict=font)
            ax.set_yticks(np.arange(-.2, .1, .05))
            ax.set_yticklabels(['-200', '-150', '-100', '-50', '0', '50'])

            # add text
            ax.text(-.48, .009, r'MDC maxima', color='c', fontsize=12)
            ax.text(-.26, .009, r'$\epsilon_\mathbf{k}^b$',
                    color='C4', fontsize=12)
        else:
            ax.set_yticks(np.arange(-.2, .1, .05))
            ax.set_yticklabels([])
        ax.set_xticks(np.arange(-1, 0, .1))
        ax.set_xticklabels([])
        ax.set_xlim(-.5, -.1)
        ax.set_ylim(-.15, .05)

        # add labels
#        ax.text(-.49, .035, lbls[j])
#        ax.set_title(titles[j], fontsize=15)

        # second row
        ax = plt.subplot(4, 4, j+5)
        ax.set_position([.08+j*.21, .55, .2, .2])
        ax.tick_params(**kwargs_ticks)

        n = 0  # counter

        # Extract MDC's and fit
        p_mdc = []
        for i in mdc_seq:
            _sl1 = 100  # Index used for background endpoint slope
            _sl2 = 155  # other endpoint
            n += 1
            mdc_k = k[j][:, i]  # current MDC k-axis
            mdc_int = spec[j][:, i]  # current MDC
            mdc_eint = espec[j][:, i]  # error
            if any(x == n for x in [1, 50, 100]):
                plt.errorbar(mdc_k, mdc_int-scale * n**1.15, mdc_eint,
                             lw=.5, capsize=.1, color='b', fmt='o', ms=.5)

            # Fit MDC
            d = 1e-2  # small boundary
            eps = 1e-8  # essentially fixed boundary
            Delta = 1e5  # essentially free boundary

            const_i = mdc_int[-1]  # constant background estimation
            slope_i = ((mdc_int[_sl1] - mdc_int[_sl2]) /
                       (mdc_k[_sl1] - mdc_k[_sl2]))  # slope estimation

            # initial values
            p_mdc_i = np.array([-.27, 5e-2, 1e-3, const_i, slope_i, .0])

            # take fixed initial values until it reaches THIS iteration,
            # then take last outcome as inital values
            if n > 70:
                p_mdc_i = p_mdc

                # p0: position, p1: width, p2: amplitude
                # p3: constant bkg, p4: slope, p5: curvature
                bounds_bot = np.array([p_mdc_i[0] - d, p_mdc_i[1] - d,
                                       p_mdc_i[2] - Delta, p_mdc_i[3] - Delta,
                                       p_mdc_i[4] - eps, p_mdc_i[5] - eps])
                bounds_top = np.array([p_mdc_i[0] + d, p_mdc_i[1] + d,
                                       p_mdc_i[2] + Delta, p_mdc_i[3] + Delta,
                                       p_mdc_i[4] + eps, p_mdc_i[5] + eps])
            else:
                bounds_bot = np.array([p_mdc_i[0] - Delta, p_mdc_i[1] - Delta,
                                       p_mdc_i[2] - Delta, p_mdc_i[3] - Delta,
                                       p_mdc_i[4] - Delta, p_mdc_i[5] - eps])
                bounds_top = np.array([p_mdc_i[0] + Delta, p_mdc_i[1] + Delta,
                                       p_mdc_i[2] + Delta, p_mdc_i[3] + Delta,
                                       p_mdc_i[4] + Delta, p_mdc_i[5] + eps])

            # boundaries
            bounds = (bounds_bot, bounds_top)

            # fit MDC
            p_mdc, c_mdc = curve_fit(utils.lor, mdc_k, mdc_int,
                                     p0=p_mdc_i, bounds=bounds)

            # errors estimation of parameters
            err_mdc = np.sqrt(np.diag(c_mdc))
            loc[n-1] = p_mdc[0]  # position of fit
            eloc[n-1] = err_mdc[0]  # error
            width[n-1] = p_mdc[1]  # HWHM of fit (2 times this value is FWHM)
            ewidth[n-1] = err_mdc[1]  # error

            # Plot Background and fit
            b_mdc = utils.poly_2(mdc_k, *p_mdc[-3:])  # background
            f_mdc = utils.lor(mdc_k, *p_mdc)  # fit

            # Plot the fits
            if any(x == n for x in [1, 50, 100]):
                ax.plot(mdc_k, f_mdc - scale * n**1.15, '--', color='r')
                ax.plot(mdc_k, b_mdc - scale * n**1.15, 'C8-', lw=2, alpha=.3)

        # decorate axes
        if j == 0:
            ax.set_ylabel('Intensity (a.u.)', fontdict=font)
            ax.text(-.48, -.0092, r'Background', color='C8')
        ax.set_yticks([])
        ax.set_xticks(np.arange(-1, -.1, .1))
        ax.set_xlim(-.5, -.1)
        ax.set_ylim(-.01, .003)
#        ax.text(-.49, .0021, lbls[j + 4])
        ax.set_xlabel(r'$k_{\Gamma - \mathrm{S}}\,(\mathrm{\AA}^{-1})$',
                      fontdict=font)

        # Third row
        loc_en = en[j][0, mdc_seq]  # energies of Lorentzian fits
        ax = fig.add_subplot(4, 4, j+9)
        ax.set_position([.08+j*.21, .29, .2, .2])
        ax.tick_params(**kwargs_ticks)
        ax.errorbar(-loc_en, width, ewidth,
                    color=cols[j], lw=.5, capsize=2, fmt='o', ms=2)

        # initial parameters
        p_im_i = np.array([0, -.1, 1])

        # Fitting the width
        im_bot = np.array([0 - Delta, -.1 - d, 1 - Delta])
        im_top = np.array([0 + Delta, -.1 + d, 1 + Delta])
        im_bounds = (im_bot, im_top)  # boundaries

        # fit data
        p_im, c_im = curve_fit(utils.poly_2, -loc_en, width, p0=p_im_i,
                               bounds=im_bounds)

        ax.plot(-loc_en, utils.poly_2(-loc_en, *p_im), '--', color='r')
#        print(np.sqrt(np.diag(c_im)))
        print('const='+str(p_im[0]))

        # decorate axes
        if j == 0:
            ax.set_ylabel(r'HWHM $(\mathrm{\AA}^{-1})$', fontdict=font)
            ax.set_yticks(np.arange(0, 1, .05))
#            ax.text(.005, .05, r'Quadratic fit', fontdict=font)
        else:
            ax.set_yticks(np.arange(0, 1, .05))
            ax.set_yticklabels([])
        ax.set_xticks(np.arange(0, .1, .02))
        ax.set_xticklabels([])
        ax.set_xlim(-loc_en[0], -loc_en[-1])
        ax.set_ylim(0, .13)
#        ax.text(.0025, .12, lbls[j+8])

        # Fourth row
        k_F = loc[0]  # Position first fit
#        print('kF='+str(k_F))
        p0 = -k_F * v_LDA  # get constant from y=v_LDA*x + p0 (y=0, x=k_F)
        yy = p0 + xx * v_LDA  # For plotting v_LDA
        en_LDA = p0 + loc * v_LDA  # Energies
        re = loc_en - en_LDA  # Real part of self energy

        # Plotting
        ax = fig.add_subplot(4, 4, j+13)
        ax.set_position([.08+j*.21, .08, .2, .2])
        ax.tick_params(**kwargs_ticks)
        ax.errorbar(-loc_en, re, ewidth * v_LDA,
                    color=Re_cols[j], lw=.5, capsize=2, fmt='o', ms=2)
        _bot = 7  # first index of fitting ReS
        _top = 25  # last index of fitting ReS
        if j == 1:
            _bot = 5
            _top = 15
        if j == 2:
            _bot = 7
            _top = 20
#        ax.errorbar(-loc_en[_bot], re[_bot], ewidth[_bot] * v_LDA,
#                    color='r', lw=.5, capsize=2, fmt='o', ms=2)
#        ax.errorbar(-loc_en[_top], re[_top], ewidth[_top] * v_LDA,
#                    color='r', lw=.5, capsize=2, fmt='o', ms=2)
        # Boundaries
        re_bot = np.array([0 - eps, 1 - Delta])  # upper boundary
        re_top = np.array([0 + eps, 1 + Delta])  # bottom boundary
        re_bounds = (re_bot, re_top)

        # initial values
        p_re_i = np.array([0, 0])

        # fit Real part
        p_re, c_re = curve_fit(utils.poly_1, -loc_en[_bot:_top], re[_bot:_top],
                               p0=p_re_i, bounds=re_bounds)
        dre = -p_re[1]  # dReS / dw
        edre = np.sqrt(np.diag(c_re))[1]
        ax.plot(-loc_en, utils.poly_1(-loc_en, *p_re),
                '--', color='r')

        # quasiparticle residue
        z = 1 / (1 - dre)
        ez = np.abs(1 / (1 - dre)**2 * edre)  # error

        # decorate axes
        if j == 0:
            ax.set_ylabel(r'$\Re \Sigma$ (meV)', fontdict=font)
            ax.set_yticks(np.arange(0, .15, .05))
            ax.set_yticklabels(['0', '50', '100'])
#            ax.text(.02, .03, 'Linear fit', fontsize=12, color=Re_cols[-1])
        else:
            ax.set_yticks(np.arange(0, .15, .05))
            ax.set_yticklabels([])
        ax.set_xticks(np.arange(0, .1, .02))
        ax.set_xticklabels(['0', '-20', '-40', '-60', '-80', '-100'])
        ax.set_xlabel(r'$\omega$ (meV)', fontdict=font)
        ax.set_xlim(-loc_en[0], -loc_en[-1])
        ax.set_ylim(0, .15)
#        ax.text(.0025, .14, lbls[j+12])

        # First row again
        ax = fig.add_subplot(4, 4, j+1)
        ax.plot(loc, loc_en, 'co', ms=1)
        ax.plot(xx, yy, 'C4--', lw=2)

        # decorate axes
        if j == 0:
            ax.arrow(loc[20], -.05, 0, loc_en[20]+.038,
                     head_width=0.01, head_length=0.01, fc='k', ec='k')
            ax.arrow(loc[20], -.05, 0, loc_en[20]+.005,
                     head_width=0.01, head_length=0.01, fc='k', ec='k')
            plt.text(-.28, -.048, r'$\Re \Sigma(\omega)$', color='k', fontsize=12)
        if j == 3:

            # colorbar
            pos = ax.get_position()
            cax = plt.axes([pos.x0+pos.width+0.01, pos.y0, 0.01, pos.height])
            cbar = plt.colorbar(c0, cax=cax, ticks=None)
            cbar.set_ticks([])
            cbar.set_clim(np.min(int_norm), np.max(int_norm))

        # collect the data
        Z = Z + (z,)
        eZ = eZ + (ez,)
        Re = Re + (re,)
        Loc_en = Loc_en + (loc_en,)
        Width = Width + (width,)
        eWidth = eWidth + (ewidth,)
    print('Z=' + str(Z))
    plt.show()

    # Save figure
    if print_fig:
        plt.savefig(save_dir + figname + '.pdf', dpi=100,
                    bbox_inches="tight", rasterized=True)

    fig_width = plt.figure('width', figsize=(10, 10), clear=True)
    ax = fig_width.add_subplot(221)
#    print(len(Width))
    ax.plot(1.3, Width[0][10], 'ko')
    ax.plot(10, Width[1][10], 'ko')
    ax.plot(20, Width[2][10], 'ko')
    ax.plot(30, Width[3][10], 'ko')
    ax.set_xlim(0, 32)

#    dims = np.array([len(Re), Re[0].shape[0]])
#    os.chdir(data_dir)
#    np.savetxt('Data_CSROfig6_Z_b.dat', np.ravel(Z))
#    np.savetxt('Data_CSROfig6_eZ_b.dat', np.ravel(eZ))
#    np.savetxt('Data_CSROfig6_Re.dat', np.ravel(Re))
#    np.savetxt('Data_CSROfig6_Loc_en.dat', np.ravel(Loc_en))
#    np.savetxt('Data_CSROfig6_Width.dat', np.ravel(Width))
#    np.savetxt('Data_CSROfig6_eWidth.dat', np.ravel(eWidth))
#    np.savetxt('Data_CSROfig6_dims.dat', np.ravel(dims))
#    print('\n ~ Data saved (Z, eZ, Re, Loc_en, Width, eWidth)',
#          '\n', '==========================================')
#    os.chdir(home_dir)

#    return Z, eZ, Re, Loc_en, Width, eWidth, dims


def fig7(print_fig=True):
    """figure 7

    """

    figname = 'DEFfig7'

    file = 8
    gold = 14
    mat = 'CSRO20'
    year = 2017
    sample = 'S1'

    # load and plot Fermi surface
    D = ARPES.Bessy(file, mat, year, sample)
    D.norm(gold)
    D.FS(e=0.07, ew=.02)
    D.ang2kFS(D.ang, Ekin=36, lat_unit=True, a=5.5, b=5.5, c=11,
              V0=0, thdg=2.7, tidg=-1.5, phidg=42)
    D.FS_flatten()
    D.plt_FS()

    # data loading for figure
    files = [25, 26, 27, 28]
    gold = 14
    mat = 'CSRO20'
    year = 2017
    sample = 'S1'

    # EDC values
    edc_e_val = -1  # EDC espilon band
    edcw_e_val = .05
    edc_b_val = -.36  # EDC beta band
    edcw_b_val = .01

    # boundaries of fit
    top_e = .005
    top_b = .005
    bot_e = -.015
    bot_b = -.015
    left_e = -1.2
    left_b = -.5
    right_e = -.8
    right_b = -.2

    # placeholders for spectra
    spec = ()
    en = ()
    k = ()
    int_e = np.zeros((4))
    int_b = np.zeros((4))
    eint_e = np.zeros((4))  # Error integrated epsilon band
    eint_b = np.zeros((4))  # Error integrated beta band

    # temperature
    T = np.array([1.3, 10., 20., 30.])

    # placeholders EDC and other useful stuff
    EDC_e = ()  # EDC epsilon band
    EDC_b = ()  # EDC beta band
    eEDC_e = ()  # EDC error epsilon band
    eEDC_b = ()  # EDC error beta band
    Bkg_e = ()  # Background epsilon band
    Bkg_b = ()  # Background beta band
    _EDC_e = ()  # Index EDC epsilon band
    _EDC_b = ()  # Index EDC beta band
    _Top_e = ()
    _Top_b = ()
    _Bot_e = ()
    _Bot_b = ()
    _Left_e = ()
    _Left_b = ()
    _Right_e = ()
    _Right_b = ()

    for j in range(4):
        D = ARPES.Bessy(files[j], mat, year, sample)
        if j == 0:
            D.int_amp(1.52)  # renoramlize intensity for this spectrum
        D.norm(gold=gold)
        D.bkg()
    #    D.restrict(bot=.7, top=.9, left=.33, right=.5)
    #    D.restrict(bot=.7, top=.9, left=.0, right=1)

        # Transform data
        if j == 0:
            D.ang2k(D.ang, Ekin=48, lat_unit=True, a=5.5, b=5.5, c=11,
                    V0=0, thdg=2.4, tidg=0, phidg=45)
        else:
            D.ang2k(D.ang, Ekin=48, lat_unit=True, a=5.5, b=5.5, c=11,
                    V0=0, thdg=2.8, tidg=0, phidg=45)
        int_norm = D.int_norm
        eint_norm = D.eint_norm

        # The data set appears to have an offset in energy
        # This has been corrected by comparing to the other spectra (e.g. LH)
        en_norm = D.en_norm - .008

        # Find all indices
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

        # Build EDC's with errors and Shirley backgrounds
        edc_e = (np.sum(int_norm[_edcw_e:_edc_e, :], axis=0) /
                 (_edc_e - _edcw_e + 1))
        eedc_e = (np.sum(eint_norm[_edcw_e:_edc_e, :], axis=0) /
                  (_edc_e - _edcw_e + 1))
        bkg_e = utils.Shirley(edc_e)
        edc_b = (np.sum(int_norm[_edcw_b:_edc_b, :], axis=0) /
                 (_edc_b - _edcw_b + 1))
        eedc_b = (np.sum(eint_norm[_edcw_b:_edc_b, :], axis=0) /
                  (_edc_b - _edcw_b + 1))
        bkg_b = utils.Shirley(edc_b)

        # integrates around Ef
        int_e[j] = np.sum(int_norm[_left_e:_right_e, _bot_e:_top_e])
        int_b[j] = np.sum(int_norm[_left_b:_right_b, _bot_b:_top_b])
        eint_e[j] = np.sum(eint_norm[_left_e:_right_e, _bot_e:_top_e])
        eint_b[j] = np.sum(eint_norm[_left_b:_right_b, _bot_b:_top_b])

        # collect data
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

    eint_e = eint_e / int_e * 2
    eint_b = eint_b / int_b
    int_e = int_e / int_e[0]
    int_b = int_b / int_b[0]

#    v_F = 2.34
#    k_F = -.36
#    k_lda = np.linspace(-1, .0, 2)
#    lda = (k_lda-k_F) * v_F

    # Figure panels
    def fig7abcd():
        lbls = [r'$T=1.3\,$K', r'$T=10\,$K', r'$T=20\,$K',
                r'$T=30\,$K']
        for j in range(4):
            ax = fig.add_subplot(2, 4, j + 1)
            ax.set_position([.08 + j * .21, .5, .2, .2])
            ax.tick_params(**kwargs_ticks)

            # Plot data
            c0 = ax.contourf(k[j], en[j], spec[j], 300, **kwargs_ex,
                             vmin=.0*np.max(spec[0]),
                             vmax=.25*np.max(spec[0]), zorder=.1)
            ax.set_rasterization_zorder(.2)
            ax.plot([np.min(k[j]), np.max(k[j])], [0, 0], **kwargs_ef)

            if j == 0:
                # Plot cut of EDC's low temp
#                ax.plot([k[j][_EDC_e[j], 0], k[j][_EDC_e[j], 0]],
#                        [-1, .005],
#                        ls='-.', color='b', lw=1)
                ax.plot([-.35, -.35],
                        [-1, .005],
                        ls='-.', color='b', lw=1)
                ax.plot([-.35, -.35],
                        [.015, .04],
                        ls='-.', color='b', lw=1)
#                ax.plot(k_lda, lda, 'w--')
#                ax.text(-.4, -.06, r'$\varepsilon_k^b$', color='w')
                # decorate axes
                ax.set_yticks(np.arange(-.1, .03, .02))
                ax.set_yticklabels(['-100', '-80', '-60', '-40', '-20',
                                    '0', '20'])
                ax.set_ylabel(r'$\omega\,(\mathrm{meV})$', fontdict=font)

                # add text
#                ax.text(-1.18, .007, r'$\gamma$-band', color='m')
                ax.text(-.52, .007, r'$\alpha$-band', color='g')

            elif j == 1:
                ax.set_yticks(np.arange(-.1, .05, .02))
                ax.set_yticklabels([])
            elif j == 3:
                # Plot cut of EDC's high temp
#                ax.plot([k[j][_EDC_e[j], 0], k[j][_EDC_e[j], 0]],
#                        [-1, .015],
#                        ls='-.', color='b', lw=1)
                ax.plot([-.35, -.35],
                        [en[j][0, 0], en[j][0, -1]],
                        ls='-.', color='b', lw=1)

                # decorate axes
                ax.set_yticks(np.arange(-.1, .05, .02))
                ax.set_yticklabels([])
            else:
                ax.set_yticks(np.arange(-.1, .05, .02))
                ax.set_yticklabels([])

            # Draw boxes of intensity integration
            rbox = {'ls': '--', 'color': 'm', 'lw': .5}
            bbox = {'ls': '--', 'color': 'g', 'lw': .5}
#            ax.plot([k[j][_Left_e[j], 0], k[j][_Left_e[j], 0]],
#                    [en[j][0, _Top_e[j]], en[j][0, _Bot_e[j]]], **rbox)
#            ax.plot([k[j][_Right_e[j], 0], k[j][_Right_e[j], 0]],
#                    [en[j][0, _Top_e[j]], en[j][0, _Bot_e[j]]], **rbox)
#            ax.plot([k[j][_Left_e[j], 0], k[j][_Right_e[j], 0]],
#                    [en[j][0, _Top_e[j]], en[j][0, _Top_e[j]]], **rbox)
#            ax.plot([k[j][_Left_e[j], 0], k[j][_Right_e[j], 0]],
#                    [en[j][0, _Bot_e[j]], en[j][0, _Bot_e[j]]], **rbox)
#
#            ax.plot([k[j][_Left_b[j], 0], k[j][_Left_b[j], 0]],
#                    [en[j][0, _Top_b[j]], en[j][0, _Bot_b[j]]], **bbox)
#            ax.plot([k[j][_Right_b[j], 0], k[j][_Right_b[j], 0]],
#                    [en[j][0, _Top_b[j]], en[j][0, _Bot_b[j]]], **bbox)
#            ax.plot([k[j][_Left_b[j], 0], k[j][_Right_b[j], 0]],
#                    [en[j][0, _Top_b[j]], en[j][0, _Top_b[j]]], **bbox)
#            ax.plot([k[j][_Left_b[j], 0], k[j][_Right_b[j], 0]],
#                    [en[j][0, _Bot_b[j]], en[j][0, _Bot_b[j]]], **bbox)

            # decorate axes
            ax.xaxis.tick_top()
            ax.set_xticks(np.arange(-1, .5, 1.))
            ax.set_xticklabels([r'S', r'$\Gamma$'])
            ax.set_xlim(np.min(k[j]), 0.05)
            print('min=', str(np.min(k[j])))
            ax.set_ylim(-.1, .03)
            ax.text(-1.28, .018, lbls[j], fontsize=10)

        # colorbar
        pos = ax.get_position()
        cax = plt.axes([pos.x0+pos.width + 0.01,
                        pos.y0, 0.01, pos.height])
        cbar = plt.colorbar(c0, cax=cax, ticks=None)
        cbar.set_ticks([])
        cbar.set_clim(np.min(D.int_norm), np.max(D.int_norm))

    def fig7efg():
        # labels and position
        lbls = [r'$\gamma$-band, @$\,$S',
                r'$\gamma$-band (zoom), @$\,$S',
                r'$\alpha$-band (zoom), @$\,k_\mathrm{F}$']
        lbls_x = [-.77, -.093, -.093]
        lbls_y = [2.05, 1.08, 1.08]

        # Visualize EDCs and Background
        fig1 = plt.figure('EDC', figsize=(8, 8), clear=True)
        ax1 = fig1.add_subplot(221)
        ax1.set_position([.08, .5, .3, .3])
        for j in range(4):
            ax1.plot(en[j][_EDC_e[j]], EDC_e[j], 'o', ms=1)
            ax1.plot(en[j][_EDC_e[j]], Bkg_e[j], 'o', ms=1)

        ax2 = fig1.add_subplot(222)
        ax2.set_position([.08 + .31, .5, .3, .3])

        # Placeholders for normalized data
        EDCn_e = ()
        EDCn_b = ()
        eEDCn_e = ()
        eEDCn_b = ()
        xx = np.arange(-2, .5, .001)
#        f_edc = np.zeros((2, len(xx)))
#        f_mod = np.zeros((2, len(xx)))
#        f_fl = np.zeros((2, len(xx)))

        # Loop over EDC's
        for j in range(4):
            tmp_e = EDC_e[j] - Bkg_e[j]
            tmp_b = EDC_b[j] - Bkg_b[j]

            # total area
            tot_e = integrate.trapz(tmp_e, en[j][_EDC_e[j]])

            # normalize
            edcn_e = tmp_e / tot_e
            eedcn_e = eEDC_e[j] / tot_e
            edcn_b = tmp_b / tot_e
            eedcn_b = eEDC_b[j] / tot_e

            # Plot EDC's
            plt.plot(en[j][_EDC_e[j]], edcn_e, 'o', ms=1)

            # Collect
            EDCn_e = EDCn_e + (edcn_e,)
            EDCn_b = EDCn_b + (edcn_b,)
            eEDCn_e = eEDCn_e + (eedcn_e,)
            eEDCn_b = eEDCn_b + (eedcn_b,)

            if j == 0:
                # initial guess
                p_edc_i = np.array([6.9e-1, 7.3e-3, 4.6, 4.7e-3, 4e-2, 2.6e-3,
                                    1e0, -.2, .3, 1, -.1, 1e-1])

                d = 1e-6
                D = 1e6

                # boundaries for fit
                bounds_fl = ([p_edc_i[0] - D, p_edc_i[1] - d, p_edc_i[2] - d,
                              p_edc_i[3] - D, p_edc_i[4] - D, p_edc_i[5] - D],
                             [p_edc_i[0] + D, p_edc_i[1] + d, p_edc_i[2] + d,
                              p_edc_i[3] + D, p_edc_i[4] + D, p_edc_i[5] + D])

                # fit data
                p_fl, cov_fl = curve_fit(utils.FL_spectral_func,
                                         en[j][_EDC_e[j]][900:-1],
                                         EDCn_e[j][900:-1],
                                         p_edc_i[:6], bounds=bounds_fl)

                f_fl = utils.FL_spectral_func(xx, *p_fl)

                # boundary for fit
                bounds = (np.concatenate((p_fl - D, p_edc_i[6:] - D), axis=0),
                          np.concatenate((p_fl + D, p_edc_i[6:] + D), axis=0))
                bnd = 300  # range to fit the data

                # fit data
                p_edc, c_edc = curve_fit(utils.Full_spectral_func,
                                         en[j][_EDC_e[j]][bnd:-1],
                                         EDCn_e[j][bnd:-1],
                                         np.concatenate((p_fl, p_edc_i[-6:]),
                                                        axis=0),
                                         bounds=bounds)

                # plot spectral function
                f_edc = utils.Full_spectral_func(xx, *p_edc)

                # plot coherent and incoherent weight
                f_mod = utils.gauss_mod(xx, *p_edc[-6:])
                f_fl = utils.FL_spectral_func(xx, *p_edc[0:6])

        # Create figure
        plt.figure(figname)
        for j in range(2):
            ax = fig.add_subplot(2, 4, j+5)
            ax.set_position([.08+j*.21, .29, .2, .2])
            ax.tick_params(**kwargs_ticks)

            # Plot EDC's
            ax.plot(en[0][_EDC_e[0]], EDCn_e[0], 'C9o', ms=1)
            ax.plot(en[3][_EDC_e[3]], EDCn_e[3], 'ko', ms=1, alpha=.8)
            ax.plot(xx, f_edc, '--', color='C8', lw=1)
            ax.fill(xx, f_mod, alpha=.3, color='C0')
            ax.fill(xx, f_fl, alpha=.3, color='b')
            ax.set_yticks([])

            # Plot zoom box
            if j == 0:
                y_max = 1.2
                x_min = -.1
                x_max = .05
                ax.plot([x_min, x_max], [y_max, y_max], 'k--', lw=.5)
                ax.plot([x_min, x_min], [0, y_max], 'k--', lw=.5)
                ax.plot([x_max, x_max], [0, y_max], 'k--', lw=.5)

                # decorate axis
                ax.set_xticks(np.arange(-.8, .2, .2))
                ax.set_xlabel(r'$\omega$ (eV)')
                ax.set_ylabel('Intensity (a.u.)')
                ax.set_xlim(-.8, .1)
                ax.set_ylim(0, 2.3)
            else:
                # decorate axis
                ax.set_xticks(np.arange(-.08, .06, .04))
                ax.set_xticklabels(['-80', '-40', '0', '40'])
                ax.set_xlim(x_min, x_max)
                ax.set_ylim(0, y_max)
                ax.set_xlabel(r'$\omega$ (meV)')

                # add text
                ax.text(.01, .7, r'$1.3\,$K', color='C9')
                ax.text(.01, .3, r'$30\,$K', color='k')
            ax.text(lbls_x[j], lbls_y[j], lbls[j])

        # plot next panel
        ax = fig.add_subplot(247)
        ax.set_position([.08 + 2 * .21, .29, .2, .2])
        ax.tick_params(**kwargs_ticks)
        ax.plot(en[0][_EDC_b[0]], EDCn_b[0], 'o', ms=1, color='C9')
        ax.plot(en[3][_EDC_b[3]], EDCn_b[3], 'o', ms=1, color='k', alpha=.8)

        # decorate axes
        ax.set_yticks([])
        ax.set_xticks(np.arange(-.08, .06, .04))
        ax.set_xticklabels(['-80', '-40', '0', '40'])
        ax.set_xlim(-.1, .05)
        ax.set_ylim(0, .005)
        ax.set_ylim(0, y_max)
        ax.set_xlabel(r'$\omega$ (meV)')

        # add text
        ax.text(lbls_x[-1], lbls_y[-1], lbls[-1])

        return (en, EDCn_e, EDCn_b, EDC_e, EDC_b, Bkg_e, Bkg_b, _EDC_e, _EDC_b,
                eEDCn_e, eEDCn_b, eEDC_e, eEDC_b)

    def fig7h():
        ax = fig.add_subplot(248)
        ax.set_position([.08+3*.21, .29, .2, .2])
        ax.tick_params(**kwargs_ticks)

        # Plot integrated intensity
        ax.errorbar(T, int_e, yerr=eint_e, lw=.5,
                    capsize=2, color='m', fmt='o', ms=5)
        ax.errorbar(T, int_b, yerr=eint_b, lw=.5,
                    capsize=2, color='g', fmt='d', ms=5)
        ax.plot([1.3, 32], [1, .64], 'm--', lw=.5)
        ax.plot([1.3, 32], [.99, .99], 'g--', lw=.5)

        # decorate axes
        ax.set_xticks(T)
        ax.set_yticks(np.arange(.7, 1.05, .1))
        ax.set_xlim(0, 32)
        ax.set_ylim(.65, 1.08)
        ax.grid(True, alpha=.2)
        ax.set_xlabel(r'$T$ (K)', fontdict=font)
        ax.tick_params(labelleft='off', labelright='on')
        ax.yaxis.set_label_position('right')
#        ax.set_ylabel((r'$\int_\boxdot \mathcal{A}(k, \omega, T)$' +
#                       r'$\, \slash \quad \int_\boxdot \mathcal{A}$' +
#                       r'$(k, \omega, 1.3\,\mathrm{K})$'),
#                      fontdict=font, fontsize=8)
        ax.set_ylabel((r'$\int\,\,\boxdot(T)\,\mathrm{d}k\,\mathrm{d}\omega\,$'
                       + r'$\,\slash \quad \int\,\,\boxdot(1.3\,\mathrm{K})\,$'
                       + r'$\mathrm{d}k \, \mathrm{d}\omega$'),
                      fontdict=font, fontsize=8)
        # add text
        ax.text(8, .79, r'$\gamma$-band', color='m')
        ax.text(17, .92, r'$\alpha$-band', color='g')

    fig = plt.figure(figname, figsize=(8, 8), clear=True)
    fig7abcd()
    (en, EDCn_e, EDCn_b, EDC_e, EDC_b, Bkg_e, Bkg_b, _EDC_e, _EDC_b,
     eEDCn_e, eEDCn_b, eEDC_e, eEDC_b) = fig7efg()
#    fig7h()
    plt.show()

    # Save figure
    if print_fig:
        plt.savefig(save_dir + figname + '.pdf', dpi=100,
                    bbox_inches="tight", rasterized=True)

    dims = np.array([len(en), en[0].shape[0], en[0].shape[1]])
    os.chdir(data_dir)
    np.savetxt('Data_CSROfig4_en.dat', np.ravel(en))
    np.savetxt('Data_CSROfig4_EDCn_e.dat', np.ravel(EDCn_e))
    np.savetxt('Data_CSROfig4_EDCn_b.dat', np.ravel(EDCn_b))
    np.savetxt('Data_CSROfig4_EDC_e.dat', np.ravel(EDC_e))
    np.savetxt('Data_CSROfig4_EDC_b.dat', np.ravel(EDC_b))
    np.savetxt('Data_CSROfig4_Bkg_e.dat', np.ravel(Bkg_e))
    np.savetxt('Data_CSROfig4_Bkg_b.dat', np.ravel(Bkg_b))
    np.savetxt('Data_CSROfig4__EDC_e.dat', np.ravel(_EDC_e))
    np.savetxt('Data_CSROfig4__EDC_b.dat', np.ravel(_EDC_b))
    np.savetxt('Data_CSROfig4_eEDCn_e.dat', np.ravel(eEDCn_e))
    np.savetxt('Data_CSROfig4_eEDCn_b.dat', np.ravel(eEDCn_b))
    np.savetxt('Data_CSROfig4_eEDC_e.dat', np.ravel(eEDC_e))
    np.savetxt('Data_CSROfig4_eEDC_b.dat', np.ravel(eEDC_b))
    np.savetxt('Data_CSROfig4_dims.dat', dims)
    print('\n ~ Data saved (en, EDCs + normalized + indices + Bkgs)',
          '\n', '==========================================')
    os.chdir(home_dir)

    return (en, EDCn_e, EDCn_b, EDC_e, EDC_b, Bkg_e, Bkg_b, _EDC_e, _EDC_b,
            eEDCn_e, eEDCn_b, eEDC_e, eEDC_b, dims)


def fig8(print_fig=True):
    """figure 8

    """

    figname = 'DEFfig8'

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
    def fig8a():
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
#        ax.plot([edc_, edc_], [-2.5, .5], **kwargs_cut)

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
        ax.text(-1.02, 0.33, r'$63\,$eV', fontdict=font)
        ax.text(.22, .1, r'$\mathcal{C}$', fontsize=15)

    def fig8b():
        ax = fig.add_subplot(132)
        ax.set_position([.31, .3, .2, .3])
        ax.tick_params(**kwargs_ticks)

        # Plot data
        c0 = ax.contourf(D.k[0], D.en+.07, np.transpose(int2), 300,
                         **kwargs_ex, vmin=0, vmax=1.4e4, zorder=.1)
        ax.set_rasterization_zorder(.2)

        # Plot distribution cuts
        ax.plot([D.k[0][0], D.k[0][-1]], [0, 0], **kwargs_ef)
#        ax.plot([edc_, edc_], [-2.5, .5], **kwargs_cut)

        # decorate axes
        ax.set_xticks([-1, 0, 1])
        ax.set_xticklabels(['S', r'$\Gamma$', 'S'])
        ax.set_yticks(np.arange(-2.5, 1, .5))
        ax.set_yticklabels([])
        # ax.set_xlim(-1, 1.66)
        ax.set_ylim(-2.5, .5)

        # add text
        ax.text(-1.02, 0.33, r'$78\,$eV', fontdict=font)

        # colorbar
        pos = ax.get_position()
        cax = plt.axes([pos.x0+pos.width+0.01,
                        pos.y0, 0.01, pos.height])
        cbar = plt.colorbar(c0, cax=cax, ticks=None)
        cbar.set_ticks([])
        cbar.set_clim(np.min(D.int), np.max(D.int))

    def fig8c():
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
#        ax.text(5e2, 0.33, r'(c)', fontdict=font)
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

    """

    figname = 'DEFfig9'

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
    def fig9a():
        ax = fig.add_subplot(131)
        ax.set_position([.1, .3, .2, .3])
        ax.tick_params(**kwargs_ticks)

        # Plot data
        ax.contourf(D1.kxs, D1.en_norm+.1, 1.2*np.flipud(D1.int_norm), 300,
                    **kwargs_ex, vmin=0, vmax=.008, zorder=.1)
        ax.set_rasterization_zorder(.2)

        # Plot distribution cuts
        ax.plot([np.min(D1.kxs), np.max(D1.kxs)], [0, 0], **kwargs_ef)
#        ax.plot([edc_val, edc_val], [-2.5, .5], **kwargs_cut)
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
        ax.text(-.55, 0.33, r'$\bar{\sigma}$-pol.', fontdict=font)

    def fig9b():
        ax = plt.subplot(132)
        ax.set_position([.31, .3, .2, .3])
        ax.tick_params(**kwargs_ticks)

        # Plot data
        c0 = ax.contourf(D2.kxs, D2.en_norm+.1, np.flipud(D2.int_norm), 300,
                         **kwargs_ex, vmin=0, vmax=.008, zorder=.1)
        ax.plot([np.min(D2.kxs), np.max(D2.kxs)], [0, 0], **kwargs_ef)
        ax.set_rasterization_zorder(.2)

        # Plot EDC
#        ax.plot([edc_, edc_], [-2.5, .5], **kwargs_cut)

        # decorate axes
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels([r'$\Gamma$', 'S', r'$\Gamma$'])
        ax.set_yticks(np.arange(-2.5, 1, .5), ())
        ax.set_yticklabels([])
        # ax.set_xlim(0, 1)
        ax.set_ylim(-2.5, .5)

        # add text
        ax.text(-.55, 0.33, r'$\bar{\pi}$-pol.', fontdict=font)

        # colorbar
        pos = ax.get_position()
        cax = plt.axes([pos.x0+pos.width+0.01,
                        pos.y0, 0.01, pos.height])
        cbar = plt.colorbar(c0, cax=cax, ticks=None)
        cbar.set_ticks([])
        cbar.set_clim(np.min(D2.int_norm), np.max(D2.int_norm))

    def fig9c():
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
#        ax.text(5e-4, 0.33, r'(c)', fontdict=font)
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
    fig9a()
    fig9b()
    fig9c()
    plt.show()

    # Save figure
    if print_fig:
        plt.savefig(save_dir + figname + '.pdf', dpi=100,
                    bbox_inches="tight", rasterized=True)


def fig10(print_fig=True):
    """figure 10

    """

    figname = 'DEFfig10'

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
    lbls = ('', '', '', '', '')
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
                         vmax = .8*np.max(D.int_norm), zorder=.1)
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
                     vmin=.4*np.max(FS), vmax=.8*np.max(FS), zorder=.1)
    ax.set_rasterization_zorder(.2)

    ax.plot(k1[0], k1[1], 'b-')
    ax.plot(k2[0], k2[1], 'b-')

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
             fc='r', ec='r')
    ax.arrow(-.8, -4, 0, -3.1, head_width=0.06, head_length=0.3,
             fc='r', ec='r')
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


def fig11(print_fig=True):
    """figure 11

    """

    figname = 'DEFfig11'

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
                        vmax=v_scale*0.5*np.max(D.int_norm)*.9)
#            ax.text(-.93, -.16, '(a)', color='k', fontsize=12)

        elif n == 1:
            ax.contourf(D.kxs, D.en_norm+.1, D.int_norm, 300,
                        **kwargs_ex, zorder=.1,
                        vmin=v_scale*0.0*np.max(D.int_norm),
                        vmax=v_scale*0.54*np.max(D.int_norm)*.9)

        elif n == 2:
            ax.contourf(D.kxs, D.en_norm+.1, np.flipud(D.int_norm), 300,
                        **kwargs_ex, zorder=.1,
                        vmin=v_scale * 0.01 * np.max(D.int_norm),
                        vmax=v_scale * 0.7 * np.max(D.int_norm)*.9)

        elif n == 3:
            c0 = ax.contourf(D.kxs, D.en_norm+.1, np.flipud(D.int_norm), 300,
                             **kwargs_ex, zorder=.1,
                             vmin=v_scale * 0.01 * np.max(D.int_norm),
                             vmax=v_scale * 0.53 * np.max(D.int_norm)*.9)
        ax.set_rasterization_zorder(0.2)
#        ax.plot([np.min(D.kxs), np.max(D.kxs)], [-2.4, -2.4], **kwargs_cut)
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

    # Save figure
    if print_fig:
        plt.savefig(save_dir + figname + '.pdf', dpi=100,
                    bbox_inches="tight", rasterized=True)


def fig12(print_fig=True):
    """figure 12

    """

    figname = 'DEFfig12'

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
#            ax.text(10, -2.8, r'(a) $d_{\gamma z}$', fontdict=font)
            ax.text(198, -.65, r'$U+J_\mathrm{H}$', fontdict=font)

        elif i == 1:
            # decorate axes
#            ax.arrow(253, -.8, 0, .22, head_width=8, head_length=0.2,
#                     fc='g', ec='g')
#            ax.arrow(253, -.8, 0, -.5, head_width=8, head_length=0.2,
#                     fc='g', ec='g')
            ax.set_yticks(np.arange(-3, 2, 1.))
            ax.set_yticklabels([])

            # add text
#            ax.text(10, -2.8, r'(b) $d_{xy}$', fontsize=12)
#            ax.text(263, -1, r'$3J_\mathrm{H}$', fontsize=12)

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


def fig13(print_fig=True):
    """figure 13

    """

    figname = 'DEFfig13'

    file = 'CSRO_P1_0032'
    mat = 'CSRO20'
    year = 2017
    sample = 'data'
    D = ARPES.SIS(file, mat, year, sample)
    D.FS(e=19.3, ew=.01)
    D.ang2kFS(D.ang, Ekin=D.hv-4.5, lat_unit=True, a=5.5, b=5.5, c=11,
              V0=0, thdg=-1.2, tidg=1, phidg=0)

    # MDC's
    mdc_d = np.zeros(D.ang.size)  # placeholder
    mdc = np.zeros(D.pol.size)  # placeholder

    # build diagonal MDC
    for i in range(D.ang.size):
        mdc_d_val, mdc_d_idx = utils.find(D.ky[:, i], D.kx[0, i])
        mdc_d[i] = D.map[mdc_d_idx, i]

    # build MDC
    mdc_val, mdc_idx = utils.find(D.kx[0, :], .02)
    mdc_val, mdcw_idx = utils.find(D.kx[0, :], -.02)
    mdc = np.sum(D.map[:, mdcw_idx:mdc_idx], axis=1)

    # normalize
    mdc_d = mdc_d / np.max(mdc_d)
    mdc = mdc / np.max(mdc) / 1.03

    # Fit diagonal MDC
    plt.figure('MDCs', figsize=(4, 4), clear=True)
    d = 1e-5

    # initial guess
    p_mdc_d_i = np.array([-.6, -.4, -.2, .2, .4, .6,
                          .05, .05, .05, .05, .05, .05,
                          .3, .3, .4, .4, .5, .5,
                          .59, -0.2, .0])

    # fit boundaries
    bounds_bot = np.concatenate(
                        (p_mdc_d_i[0:-3] - np.inf,
                         p_mdc_d_i[-3:21] - d))
    bounds_top = np.concatenate(
                        (p_mdc_d_i[0:-3] + np.inf,
                         p_mdc_d_i[-3:21] + d))
    p_mdc_d_bounds = (bounds_bot, bounds_top)

    # fitting data
    p_mdc_d, cov_mdc = curve_fit(
            utils.lor_6, D.kx[0, :], mdc_d, p_mdc_d_i, bounds=p_mdc_d_bounds)

    # fit and background
    b_mdc_d = utils.poly_2(D.kx[0, :], *p_mdc_d[-3:])
    f_mdc_d = utils.lor_6(D.kx[0, :], *p_mdc_d) - b_mdc_d
    f_mdc_d[0] = 0  # for nice edges of the fill-plots
    f_mdc_d[-1] = 0

    # plot diagonal MDC and fits
    plt.subplot(211)
    plt.plot(D.kx[0, :], mdc_d, 'bo')
    plt.plot(D.kx[0, :], f_mdc_d + b_mdc_d)
    plt.plot(D.kx[0, :], b_mdc_d, 'k--')

    # Fit MDC
    d = 5e-2

    # initial guess
    p_mdc_i = np.array([-.6,  -.2, .2, .6,
                        .05, .05, .05, .05,
                        .3, .3, .4, .4,
                        .6, -0.15, 0])

    # fit boundaries
    bounds_bot = np.concatenate((p_mdc_i[0:-3] - np.inf,
                                 p_mdc_i[-3:15] - d))
    bounds_top = np.concatenate((p_mdc_i[0:-3] + np.inf,
                                 p_mdc_i[-3:15] + d))
    p_mdc_bounds = (bounds_bot, bounds_top)

    # fit MDC
    p_mdc, cov_mdc = curve_fit(
            utils.lor_4, D.ky[:, 0], mdc, p_mdc_i, bounds=p_mdc_bounds)

    # fit and background
    b_mdc = utils.poly_2(D.ky[:, 0], *p_mdc[-3:])
    f_mdc = utils.lor_4(D.ky[:, 0], *p_mdc) - b_mdc
    f_mdc[0] = 0
    f_mdc[-1] = 0

    # plot MDC
    plt.subplot(212)
    plt.plot(D.ky[:, 0], mdc, 'bo')
    plt.plot(D.ky[:, 0], f_mdc + b_mdc)
    plt.plot(D.ky[:, 0], b_mdc, 'k--')

    # boundary of the subplots
    bnd = .72

    # Figure panels
    def fig13a():
        ax = fig.add_axes([.1, .35, .25, .4])
        ax.tick_params(**kwargs_ticks)

        intsum = np.sum(D.int[:, 85:88, :], 1)

        xx = np.linspace(-bnd, bnd, D.pol.size)
        ef = 19.315 + xx * -.008
        ef = np.transpose(np.broadcast_to(ef, (D.en.size, D.pol.size)))
        ax.contourf(D.ens[:, 86, :]-ef, D.kys[:, 86, :]*1.12-.02,
                    intsum, 300, **kwargs_ex,
                    vmin=0.02*np.max(intsum), vmax=.32*np.max(intsum)*.9,
                    zorder=.1)
        ax.set_rasterization_zorder(.2)
        ax.plot([0, 0], [-bnd, bnd], **kwargs_ef)

        ax.set_yticks(np.arange(-10, 10, .5))
        ax.set_xlim([-.1, .035])
        ax.set_ylim(-bnd, bnd)
        ax.set_xlabel(r'$\omega$ (eV)')
        ax.set_ylabel(r'$k_y \, (\pi/a)$')

    def fig13b():
        ax = fig.add_subplot(141)
        ax.set_position([.08+.3, .755, .4, .15])
        ax.plot(D.kx[0, :], mdc_d - b_mdc_d + .01, 'o', ms=1.5, color='C9')
        ax.plot(D.kx[0, :], f_mdc_d + .01, color='k', lw=.5)
        ax.fill(D.kx[0, :], f_mdc_d + .01, alpha=.2, color='C9')

        # labels and positions
        corr = np.array([.13, .26, .11, .14, .145, .13])
        cols = ['b', 'c', 'C1', 'C1', 'c', 'b']
        lbls = [r'$\beta$', r'$\alpha$', r'$\delta$',
                r'$\delta$', r'$\alpha$', r'$\beta$']

        # plot Lorentzians and labels
        for i in range(6):
            ax.plot(D.kx[0, :], (utils.lor(D.kx[0, :], p_mdc_d[i],
                    p_mdc_d[i+6], p_mdc_d[i+12], p_mdc_d[-3], p_mdc_d[-2],
                    p_mdc_d[-1]) - b_mdc_d) + .01, lw=.5, color=cols[i])
            ax.text(p_mdc_d[i] - .02, p_mdc_d[i+12]*2+corr[i], lbls[i],
                    fontsize=10, color=cols[i])
        ax.tick_params(**kwargs_ticks)

        # decorate axes
        ax.xaxis.set_label_position('top')
        ax.tick_params(labelbottom='off', labeltop='on')
        ax.set_xticks(np.arange(-10, 10, .5))
        ax.set_yticks([])
        ax.set_xlim(-bnd, bnd)
        ax.set_ylim(0, .42)
        ax.set_xlabel(r'$k_x = k_y \, (\pi/a)$')
        ax.set_ylabel(r'Intensity (a.u.)')

    def fig13c():
        D.FS_flatten(ang=False)
        ax = fig.add_subplot(143)
        ax.set_position([.08+.3, .35, .4, .4])
        ax.tick_params(**kwargs_ticks)

        # plot data
        c0 = ax.contourf(D.kx, D.ky, D.map, 100, **kwargs_ex,
                         vmin=.58*np.max(D.map), vmax=.95*np.max(D.map),
                         zorder=.1)
        ax.set_rasterization_zorder(.2)
        ax.plot([-bnd, bnd], [-bnd, bnd], **kwargs_cut)
        ax.plot([0, 0], [-bnd, bnd], **kwargs_cut)
#
#        # decorate axes
        ax.arrow(.55, .55, 0, .13, head_width=0.03,
                 head_length=0.03, fc='b', ec='b')
        ax.set_xticks(np.arange(-10, 10, .5))
        ax.set_yticks(np.arange(-10, 10, .5))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_ylabel(r'$k_x \, (\pi/a)$')
        ax.set_xlabel(r'$k_y \, (\pi/b)$')
        ax.set_xlim(-bnd, bnd)
        ax.set_ylim(-bnd, bnd)

        pos = ax.get_position()
        cax = plt.axes([0.94,
                        pos.y0, 0.01, pos.height])
        cbar = plt.colorbar(c0, cax=cax, ticks=None)
        cbar.set_ticks([])
        cbar.set_clim(np.min(D.map), np.max(D.map))

    def fig13d():
        ax = fig.add_subplot(144)
#        ax.set_position([.485, .35, .15, .4])
        ax.set_position([.485+.3, .35, .15, .4])
        ax.tick_params(**kwargs_ticks)
        ax.plot(mdc - b_mdc, D.ky[:, 0], 'o', markersize=1.5, color='C9')
        ax.plot(f_mdc + .01, D.ky[:, 0], color='k', lw=.5)
        ax.fill(f_mdc + .01, D.ky[:, 0], alpha=.2, color='C9')

        # labels and positions
        corr = np.array([.24, .13, .13, .19])
        cols = ['c', 'C1', 'C1', 'c']
        lbls = [r'$\alpha$', r'$\delta$', r'$\delta$',
                r'$\alpha$']

        # plot Lorentzians
        for i in range(4):
            ax.plot((utils.lor(D.ky[:, 0], p_mdc[i], p_mdc[i + 4],
                               p_mdc[i + 8], p_mdc[-3], p_mdc[-2],
                               p_mdc[-1]) - b_mdc) + .01,
                    D.ky[:, 0], lw=.5, color=cols[i])
            ax.text(p_mdc[i + 8] + corr[i], p_mdc[i] - .02, lbls[i],
                    fontsize=10, color=cols[i])

        # decorate axes
        ax.set_yticks(np.arange(-10, 10, .5))
        ax.set_yticklabels([])
        ax.set_xticks([])
        ax.set_ylim(ymax=bnd, ymin=-bnd)
        ax.set_xlim(xmax=.42, xmin=0)
        ax.tick_params(labelleft='off', labelright='on')
        ax.yaxis.set_label_position('right')
#        ax.set_ylabel(r'$k_y \, (\pi/b)$')
        ax.set_xlabel(r'Intensity (a.u.)')


#    def fig2d():
#        ax = fig.add_axes([.08, .08, .4, .25])
#        ax.tick_params(**kwargs_ticks)
#
#        intsum = np.sum(D.int[:, 85:88, :], 1)
#
#        xx = np.linspace(-bnd, bnd, D.pol.size)
#        ef = 19.315 + xx * -.008
#        ef = np.transpose(np.broadcast_to(ef, (D.en.size, D.pol.size)))
#        ax.contourf(D.kys[:, 86, :]*1.12-.02, D.ens[:, 86, :]-ef,
#                    intsum, 300, **kwargs_ex,
#                    vmin=0.02*np.max(intsum), vmax=.32*np.max(intsum),
#                    zorder=.1)
#        ax.set_rasterization_zorder(.2)
#        ax.plot([-bnd, bnd], [0, 0], **kwargs_ef)
#        ax.set_ylim([-.1, .035])
#        ax.set_xlim(-bnd, bnd)
#        ax.set_xlabel(r'$k_x \, (\pi/a)$')
#        ax.text(-.7, .02, r'(d)', fontsize=12)

    # Create figure
    fig = plt.figure(figname, figsize=(6, 6), clear=True)
#    fig13a()
    fig13b()
    fig13c()
    fig13d()
    plt.show()

    # Save figure
    if print_fig:
        plt.savefig(save_dir + figname + '.pdf', dpi=100,
                    bbox_inches="tight", rasterized=True)