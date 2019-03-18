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
save_dir = '/Users/denyssutter/Documents/PhD/PhD_Denys/Figs/'
data_dir = '/Users/denyssutter/Documents/PhD/data/'
home_dir = '/Users/denyssutter/Documents/library/Python/ARPES'


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

    def fig1():
        ax = fig.add_subplot(141)
        # ax.set_position([.08, .3, .22, .28])
        ax.set_position([.277, .3, .28, .28])
        ax.tick_params(**kwargs_ticks)

        # plot data
        c0 = ax.contourf(A1.kys, A1.en_norm, A1.int_norm, 300, **kwargs_ex,
                         vmin=.1*np.max(A1.int_norm),
                         vmax=.7*np.max(A1.int_norm)*.8,
                         zorder=.1)
        ax.set_rasterization_zorder(.2)
        ax.plot([np.min(A1.kys), np.max(A1.kys)], [0, 0], **kwargs_ef)
        ax.plot([np.min(A1.kys), np.max(A1.kys)], [-.005, -.005], 'b-.', lw=1)

        # decorate axes
        ax.set_ylim(-.06, .03)
        ax.set_xlim(np.min(A1.kys), np.max(A1.kys))
        ax.set_yticks(np.arange(-.06, .03, .02))
        ax.set_yticklabels(['-60', '-40', '-20', '0', '20'])
        ax.set_xticks([-1.5, -1, -.5, 0, .5])
        ax.set_ylabel(r'$\omega\,(\mathrm{meV})$', fontdict=font)
        ax.set_xlabel(r'$k_x \,(\pi/a)\, \quad k_y=0$', fontdict=font)
        ax.plot(A1.k[1], (mdc - b_mdc) / 30 + .001, 'o', ms=1.5, color='C9')
        ax.fill(A1.k[1], f_mdc / 30 + .001, alpha=.2, color='C9')

        # add text
        ax.text(-.05, .024, r'$\Gamma$', fontsize=12, color='k')
        ax.text(-1.05, .024, 'Y', fontsize=12, color='k')

        # labels
        cols = ['c', 'm', 'b', 'b', 'm', 'c', 'C1', 'C1']
        lbls = [r'$\alpha$', r'$\gamma$', r'$\beta$',
                r'$\beta$', r'$\gamma$',
                r'$\alpha$', r'$\delta$', r'$\delta$']

        # coordinate corrections to label positions
        corr = np.array([.007, .003, .007, .004, .001, .009, -.003, -.004])
        p_mdc[6 + 16] *= 1.5

        # plot MDC fits
        for i in range(8):
            ax.plot(A1.k[1], (utils.lor(A1.k[1], p_mdc[i], p_mdc[i+8],
                    p_mdc[i+16],
                    p_mdc[-3], p_mdc[-2], p_mdc[-1]) - b_mdc) / 30 + .001,
                    lw=.5, color=cols[i])
#            ax.text(p_mdc[i+16]/5+corr[i], p_mdc[i]-.03, lbls[i],
#                    fontsize=10, color=cols[i])
        ax.plot(A1.k[1], f_mdc / 30 + .001, color='k', lw=.5)

        pos = ax.get_position()
        cax = plt.axes([pos.x0+pos.width+0.01,
                        pos.y0, 0.01, pos.height])
        cbar = plt.colorbar(c0, cax=cax, ticks=None)
        cbar.set_ticks([])
        cbar.set_clim(np.min(A1.int_norm), np.max(A1.int_norm))

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

    def fig2():
        ax = fig.add_subplot(143)
        ax.set_position([.3, .3, .28, .28])
        ax.tick_params(**kwargs_ticks)

        # plot data
        c0 = ax.contourf(A2.kys, A2.en_norm, A2.int_norm, 300,
                         **kwargs_ex, zorder=.1,
                         vmin=.1*np.max(A2.int_norm),
                         vmax=.7*np.max(A2.int_norm)*.8)
        ax.set_rasterization_zorder(.2)
        ax.plot([np.min(A2.kys), np.max(A2.kys)], [0, 0], **kwargs_ef)

        # decorate axes
        ax.set_yticks(np.arange(-.06, .04, .02))
        ax.set_yticklabels([])
        ax.set_xticks([-1.5, -1, -.5, 0, .5])
        ax.set_xlabel(r'$k_x \,(\pi/a), \quad k_y=\pi/b$', fontdict=font)
#        ax.set_ylabel(r'$\omega\,(\mathrm{meV})$', fontdict=font)
        ax.set_ylim(-.06, .03)
        ax.set_xlim(np.min(A2.kys), np.max(A2.kys))

        # add text
        ax.text(-.05, .024, 'X', fontsize=12, color='k')
        ax.text(-1.05, .024, 'S', fontsize=12, color='k')

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

        ax.contourf(ks[j], en[j], data[j], 300, **kwargs_ex,
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
#        ax.set_ylabel(r'$\omega\,(\mathrm{meV})$', fontdict=font)
        ax.set_xlabel(r'$k_x \,(\pi/a), \quad k_y=k_x$', fontdict=font)

        ax.plot([np.min(ks[j]), np.max(ks[j])], [0, 0], **kwargs_ef)
        ax.plot([np.min(ks[j]), np.max(ks[j])], [mdc_val, mdc_val],
                'b-.', lw=.5)

        # decorate axes
        ax.set_xticks(np.arange(-1, 1, .5))
        ax.set_xlim(np.min(ks[0]), np.max(ks[0]))
        ax.set_ylim(-.06, .03)
        ax.text(-1.05, .024, 'S', fontsize=12, color='k')
        ax.text(-.05, .024, r'$\Gamma$', fontsize=12, color='k')

        # plot MDC
        mdc[0] = 0
        mdc[-1] = 0
        ax.plot(k[j], mdc / 30 + .002, 'o', ms=1, color='C9')
        ax.fill(k[j], mdc / 30 + .002, alpha=.2, color='C9')

    plt.figure('MDC', figsize=(8, 8), clear=True)
    plt.figure(figname, figsize=(8, 8), clear=True)
    fig4()
    plt.show()

    # Save figure
    if print_fig:
        plt.savefig(save_dir + figname + '.png', dpi=400,
                    bbox_inches="tight", rasterized=False)


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
        lbls = [r'(a) C$^+$-pol.', r'(b) $\bar{\pi}$-pol.',
                r'(c) $\bar{\sigma}$-pol.']

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
                            vmin=.05*np.max(data[j]), vmax=.35*np.max(data[j]),
                            zorder=.1)
                ax.set_rasterization_zorder(.2)
#                for bands in range(6):
#                    TB_D[bands][TB_D[bands] > 0] = 10
#                    ax.plot(k[j], TB_D[bands], 'wo', ms=.5, alpha=.2)
                mdc = mdc / np.max(mdc)

                # decorate axes
                ax.set_yticks(np.arange(-.1, .05, .02))
                ax.set_yticklabels(['-100', '-80', '-60', '-40', '-20',
                                    '0', '20', '40'])
                ax.set_ylabel(r'$\omega\,(\mathrm{meV})$', fontdict=font)
            else:
                c0 = ax.contourf(ks[j], en[j], data[j], 300, **kwargs_ex,
                                 vmin=.3*np.max(data[1]),
                                 vmax=.6*np.max(data[1]), zorder=.1)
                ax.set_rasterization_zorder(.2)
                mdc = (mdc - b_mdc) / .007

                # decorate axes
                ax.set_yticks(np.arange(-.1, .05, .02))
                ax.set_yticklabels([])

            ax.plot([np.min(ks[j]), np.max(ks[j])], [0, 0], **kwargs_ef)
            ax.plot([np.min(ks[j]), np.max(ks[j])], [mdc_val, mdc_val],
                    'r-.', lw=.5)

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
        lbls = [r'(e) LDA $\Sigma_\mathrm{orb}$', r'(f) DMFT $d_{xz}$',
                r'(g) DMFT $d_{xy}$']

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
                ax.set_yticks(np.arange(-.1, .05, .02))
                ax.set_yticklabels(['-100', '-80', '-60', '-40', '-20',
                                    '0', '20', '40'])
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
        ax_d.text(0, -6, 3.5, r'(d) $d_{xz}$')
        ax_h.text(0, -6, 3.5, r'(h) $d_{xy}$')

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
