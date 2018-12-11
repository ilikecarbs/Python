#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 11:30:51 2018

@author: ilikecarbs

%%%%%%%%%%%%%%%%%%%%%
   PhD_chapter_CSRO
%%%%%%%%%%%%%%%%%%%%%

**Thesis figures Ca1.8Sr0.2RuO4**

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


def fig1(print_fig=True):
    """figure 1

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Experimental data: Figure 1 CSRO20 paper
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    """

    figname = 'CSROfig1'

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

    # load data for cut Gamma-X
    file = '62090'
    gold = '62091'
    A1 = ARPES.DLS(file, mat, year, sample)
    A1.norm(gold)
    A1.ang2k(A1.ang, Ekin=22-4.5, lat_unit=True, a=5.2, b=5.55, c=11,
             V0=0, thdg=9.2, tidg=0, phidg=90)

#    # load data for intermediate
#    file = '62092'
#    gold = '62091'
#    A2 = ARPES.DLS(file, mat, year, sample)
#    A2.norm(gold)
#    A2.ang2k(A2.ang, Ekin=22-4.5, lat_unit=True, a=5.2, b=5.55, c=11,
#             V0=0, thdg=9.2, tidg=-4.2, phidg=90)

    # load data for cut X-S
    file = '62097'
    gold = '62091'
    A2 = ARPES.DLS(file, mat, year, sample)
    A2.norm(gold)
    A2.ang2k(A2.ang, Ekin=22-4.5, lat_unit=True, a=5.2, b=5.7, c=11,
             V0=0, thdg=9.2-3.5, tidg=-15.7, phidg=90)

    # TB
    param = utils.paramCSRO20_opt()  # Load parameters
    TB_A1 = utils.CSRO_eval(A1.k[0], A1.k[1], param)
    TB_A2 = utils.CSRO_eval(A2.k[0], A2.k[1], param)

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
                        .33, 0.02, .02])

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

    # Figure panels
    def fig1a():
        ax = fig.add_subplot(131)
        ax.set_position([.08, .3, .28, .35])
        ax.tick_params(**kwargs_ticks)

        # plot data
        ax.contourf(A1.en_norm, A1.kys, A1.int_norm, 300, **kwargs_ex,
                    vmin=.1*np.max(A1.int_norm), vmax=.7*np.max(A1.int_norm),
                    zorder=.1)
        ax.set_rasterization_zorder(.2)
        ax.plot([0, 0], [np.min(A1.kys), np.max(A1.kys)], **kwargs_ef)
        ax.plot([-.005, -.005], [np.min(A1.kys), np.max(A1.kys)],
                **kwargs_cut)
#        for i in range(6):
#            TB_A1[i][TB_A1[i] > 0] = 10
#            ax.plot(TB_A1[i], A1.k[1], 'wo', ms=.5, alpha=.2)

        # decorate axes
        ax.set_xlim(-.06, .03)
        ax.set_ylim(np.min(D.ky), np.max(D.ky))
        ax.set_xticks(np.arange(-.06, .03, .02))
        ax.set_xticklabels(['-60', '-40', '-20', '0', '20'])
        ax.set_yticks([-1.5, -1, -.5, 0, .5])
        ax.set_xlabel(r'$\omega\,(\mathrm{meV})$', fontdict=font)
        ax.set_ylabel(r'$k_x \,(\pi/a)$', fontdict=font)
        ax.plot((mdc - b_mdc) / 30 + .001, A1.k[1], 'o', ms=1.5, color='C9')
        ax.fill(f_mdc / 30 + .001, A1.k[1], alpha=.2, color='C9')

        # add text
        ax.text(-.058, .57, '(a)', fontsize=15)
        ax.text(.024, -.03, r'$\Gamma$', fontsize=12, color='k')
        ax.text(.024, -1.03, 'Y', fontsize=12, color='k')

#        # labels
#        cols = ['m', 'b', 'b', 'b', 'b', 'm', 'C1', 'C1']
#        lbls = [r'$\bar{\beta}$', r'$\bar{\gamma}$', r'$\bar{\gamma}$',
#                r'$\bar{\gamma}$', r'$\bar{\gamma}$',
#                r'$\bar{\beta}$', r'$\bar{\alpha}$', r'$\bar{\alpha}$']
#
#        # coordinate corrections to label positions
#        corr = np.array([.012, .004, .007, .004, .002, .008, .002, .0015])
#        p_mdc[6 + 16] *= 1.5
#
#        # plot MDC fits
#        for i in range(8):
#            plt.plot((utils.lor(A1.k[1], p_mdc[i], p_mdc[i+8], p_mdc[i+16],
#                     p_mdc[-3], p_mdc[-2], p_mdc[-1]) - b_mdc) / 30 + .001,
#                     A1.k[1], lw=.5, color=cols[i])
#            plt.text(p_mdc[i+16]/5+corr[i], p_mdc[i]-.03, lbls[i],
#                     fontsize=10, color=cols[i])
#        plt.plot(f_mdc / 30 + .001, A1.k[1], color='k', lw=.5)

    def fig1c():
        ax = fig.add_subplot(133)
        ax.set_position([.37+.35/ratio+.01, .3, .217, .35])
        ax.tick_params(**kwargs_ticks)

        # plot data
        c0 = ax.contourf(-np.transpose(np.fliplr(A2.en_norm)),
                         np.transpose(A2.kys),
                         np.transpose(np.fliplr(A2.int_norm)), 300,
                         **kwargs_ex,
                         vmin=.1*np.max(A2.int_norm),
                         vmax=.7*np.max(A2.int_norm), zorder=.1)
        ax.set_rasterization_zorder(.2)
        ax.plot([0, 0], [np.min(A2.kys), np.max(A2.kys)], **kwargs_ef)
#        for i in range(6):
#            TB_A2[i][TB_A2[i] > 0] = 10
#            ax.plot(-TB_A2[i], A2.k[1],
#                    'wo', ms=.5, alpha=.1)

        # decorate axes
        ax.set_xticks(np.arange(0, .08, .02))
        ax.set_xticklabels(['0', '-20', '-40', '-60'])
        ax.set_yticks([-1.5, -1, -.5, 0, .5])
        ax.set_yticklabels([])
        ax.set_xlabel(r'$\omega\,(\mathrm{meV})$', fontdict=font)
        ax.set_xlim(-.01, .06)
        ax.set_ylim(np.min(D.ky), np.max(D.ky))

        # add text
        ax.text(-.0085, .56, '(c)', fontsize=15)
        ax.text(-.008, -.03, 'X', fontsize=12, color='k')
        ax.text(-.008, -1.03, 'S', fontsize=12, color='k')

        pos = ax.get_position()
        cax = plt.axes([pos.x0+pos.width+0.01,
                        pos.y0, 0.01, pos.height])
        cbar = plt.colorbar(c0, cax=cax, ticks=None)
        cbar.set_ticks([])
        cbar.set_clim(np.min(A2.int_norm), np.max(A2.int_norm))

    def fig1b():
        ax = fig.add_subplot(132)
        ax.set_position([.37, .3, .35/ratio, .35])
        ax.tick_params(**kwargs_ticks)

        # plot data
        ax.contourf(D.kx, D.ky, np.flipud(D.map), 300, **kwargs_ex,
                    vmax=.9 * np.max(D.map), vmin=.3 * np.max(D.map),
                    zorder=.1)
        ax.set_rasterization_zorder(.2)

        # decorate axes
        ax.set_xlabel(r'$k_y \,(\pi/b)$', fontdict=font)

        # add text
        ax.text(-.65, .56, r'(b)', fontsize=15, color='w')
        ax.text(-.05, -.03, r'$\Gamma$', fontsize=15, color='w')
        ax.text(-.05, -1.03, r'Y', fontsize=15, color='w')
        ax.text(.95, -.03, r'X', fontsize=15, color='w')
        ax.text(.95, -1.03, r'S', fontsize=15, color='w')

#        # labels
#        lblmap = [r'$\bar{\alpha}$', r'$\bar{\beta}$', r'$\bar{\gamma}$',
#                  r'$\bar{\delta}$', r'$\bar{\epsilon}$']
#
#        # label position
#        lblx = np.array([.15, .4, .22, .66, .8])
#        lbly = np.array([-.18, -.42, -.68, -.71, -.8])
#
#        # label colors
#        lblc = ['C1', 'm', 'b', 'r', 'k']
#
#        # add lables
#        for k in range(4):
#            ax.text(lblx[k], lbly[k], lblmap[k], fontsize=15, color=lblc[k])
        ax.plot(A1.k[0], A1.k[1], **kwargs_cut)
        ax.plot(A2.k[0], A2.k[1], **kwargs_cut)

#        # Tight Binding Model
#        tb = utils.TB(a=np.pi, kbnd=2, kpoints=200)  # Initialize
#        tb.CSRO(param)  # Calculate bandstructure
#
#        plt.figure(figname)
#        bndstr = tb.bndstr  # Load bandstructure
#        coord = tb.coord  # Load coordinates
#
#        # read dictionaries
#        X = coord['X']
#        Y = coord['Y']
#        Axy = bndstr['Axy']
#        Bxz = bndstr['Bxz']
#        Byz = bndstr['Byz']
#        bands = (Axy, Bxz, Byz)
#
#        # loop over bands
#        n = 0  # counter
#        for band in bands:
#            n += 1
#            C = plt.contour(X, Y, band, alpha=0, levels=0)
#
#            # get paths
#            p = C.collections[0].get_paths()
#            p = np.asarray(p)
#
#            al = np.array([25])
#            be = np.array([2, 6, 9, 13, 24])
#            ga = np.arange(16, 24, 1)
#            de = np.concatenate((
#                    np.arange(26, 34, 1),
#                    np.arange(16, 24, 1)))
#
#            if n == 3:
#                for j in al:
#                    v = p[j].vertices
#                    ax.plot(v[:, 0], v[:, 1], ls=':', color='C1', lw=1.5)
#                for j in be:
#                    v = p[j].vertices
#                    ax.plot(v[:, 0], v[:, 1], ls=':', color='m', lw=1.5)
#                for j in de:
#                    v = p[j].vertices
#                    ax.plot(v[:, 0], v[:, 1], ls=':', color='r', lw=1.5)
#            elif n == 2:
#                for j in ga:
#                    v = p[j].vertices
#                    ax.plot(v[:, 0], v[:, 1], ls=':', color='b', lw=1.5)

        ax.set_xticks([-.5, 0, .5, 1])
        ax.set_yticks([-1.5, -1, -.5, 0, .5])
        ax.set_yticklabels([])
        ax.set_xlim(np.min(D.kx), np.max(D.kx))
        ax.set_ylim(np.min(D.ky), np.max(D.ky))

    fig = plt.figure(figname, figsize=(10, 10), clear=True)
    fig1a()
    fig1b()
    fig1c()
    plt.show()

    # Save figure
    if print_fig:
        plt.savefig(save_dir + figname + '.pdf', dpi=100,
                    bbox_inches="tight", rasterized=True)


def fig2(print_fig=True):
    """figure 2

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Experimental PSI data: Figure 2 CSCRO20 paper
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    """

    figname = 'CSROfig2'

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
    def fig2a():
        ax = fig.add_axes([.1, .35, .25, .4])
        ax.tick_params(**kwargs_ticks)

        intsum = np.sum(D.int[:, 85:88, :], 1)

        xx = np.linspace(-bnd, bnd, D.pol.size)
        ef = 19.315 + xx * -.008
        ef = np.transpose(np.broadcast_to(ef, (D.en.size, D.pol.size)))
        ax.contourf(D.ens[:, 86, :]-ef, D.kys[:, 86, :]*1.12-.02,
                    intsum, 300, **kwargs_ex,
                    vmin=0.02*np.max(intsum), vmax=.32*np.max(intsum),
                    zorder=.1)
        ax.set_rasterization_zorder(.2)
        ax.plot([0, 0], [-bnd, bnd], **kwargs_ef)

        ax.set_yticks(np.arange(-10, 10, .5))
        ax.set_xlim([-.1, .035])
        ax.set_ylim(-bnd, bnd)
        ax.set_xlabel(r'$\omega$ (eV)')
        ax.set_ylabel(r'$k_y \, (\pi/a)$')
        ax.text(-.095, .62, r'(a)', color='w', fontsize=12)

    def fig2b():
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

        # add text
        ax.text(-.7, .35, '(b)', fontdict=font)

    def fig2c():
        D.FS_flatten(ang=False)
        ax = fig.add_subplot(143)
        ax.set_position([.08+.3, .35, .4, .4])
        ax.tick_params(**kwargs_ticks)

        # plot data
        c0 = ax.contourf(D.kx, D.ky, D.map, 100, **kwargs_ex,
                         vmin=.6*np.max(D.map), vmax=1.0*np.max(D.map),
                         zorder=.1)
        ax.set_rasterization_zorder(.2)
        ax.plot([-bnd, bnd], [-bnd, bnd], **kwargs_cut)
        ax.plot([0, 0], [-bnd, bnd], **kwargs_cut)

        # decorate axes
        ax.arrow(.55, .55, 0, .13, head_width=0.03,
                 head_length=0.03, fc='r', ec='r')
        ax.set_xticks(np.arange(-10, 10, .5))
        ax.set_yticks(np.arange(-10, 10, .5))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xlabel(r'$k_x \, (\pi/a)$')
        ax.set_xlim(-bnd, bnd)
        ax.set_ylim(-bnd, bnd)

        # add label
        ax.text(-.7, .62, r'(c)', fontdict=font, color='w')

        pos = ax.get_position()
        cax = plt.axes([pos.x0 - .02,
                        pos.y0, 0.01, pos.height])
        cbar = plt.colorbar(c0, cax=cax, ticks=None)
        cbar.set_ticks([])
        cbar.set_clim(np.min(D.map), np.max(D.map))

    def fig2d():
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

        # add text
        ax.text(.32, .63, r'(d)', fontsize=12)

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
    fig2a()
    fig2b()
    fig2c()
    fig2d()
    plt.show()

    # Save figure
    if print_fig:
        plt.savefig(save_dir + figname + '.pdf', dpi=100,
                    bbox_inches="tight", rasterized=True)


def fig3(print_fig=True):
    """figure 3

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Polarization and orbital characters. Figure 3 in paper
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    """

    figname = 'CSROfig3'

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

    def fig3abc():
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

    def fig3efg():
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

    def fig3dh():
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
    fig3abc()
    fig3efg()
    fig3dh()
    plt.show()

    # Save figure
    if print_fig:
        plt.savefig(save_dir + figname + '.png', dpi=400,
                    bbox_inches="tight", rasterized=False)


def fig4(print_fig=True):
    """figure 4

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Temperature dependence. Figure 4 in paper
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    """

    figname = 'CSROfig4'

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
    def fig4abcd():
        lbls = [r'(a) $T=1.3\,$K', r'(b) $T=10\,$K', r'(c) $T=20\,$K',
                r'(d) $T=30\,$K']
        for j in range(4):
            ax = fig.add_subplot(2, 4, j + 1)
            ax.set_position([.08 + j * .21, .5, .2, .2])
            ax.tick_params(**kwargs_ticks)

            # Plot data
            c0 = ax.contourf(k[j], en[j], spec[j], 300, **kwargs_ex,
                             vmin=.01*np.max(spec[0]),
                             vmax=.28*np.max(spec[0]), zorder=.1)
            ax.set_rasterization_zorder(.2)
            ax.plot([np.min(k[j]), np.max(k[j])], [0, 0], **kwargs_ef)

            if j == 0:
                # Plot cut of EDC's low temp
                ax.plot([k[j][_EDC_e[j], 0], k[j][_EDC_e[j], 0]],
                        [-1, .005],
                        ls='-.', color='r', lw=.5)
                ax.plot([-.35, -.35],
                        [-1, .005],
                        ls='-.', color='r', lw=.5)
                ax.plot([-.35, -.35],
                        [.015, .04],
                        ls='-.', color='r', lw=.5)
#                ax.plot(k_lda, lda, 'w--')
#                ax.text(-.4, -.06, r'$\varepsilon_k^b$', color='w')
                # decorate axes
                ax.set_yticks(np.arange(-.1, .03, .02))
                ax.set_yticklabels(['-100', '-80', '-60', '-40', '-20',
                                    '0', '20'])
                ax.set_ylabel(r'$\omega\,(\mathrm{meV})$', fontdict=font)

                # add text
                ax.text(-1.18, .007, r'$\gamma$-band', color='m')
                ax.text(-.52, .007, r'$\alpha$-band', color='b')

            elif j == 1:
                os.chdir(data_dir)
                v_LDA_data = np.loadtxt('Data_CSROfig8_v_LDA.dat')
                v_LDA = v_LDA_data[0]

                k_F = -.35
                v_LDA = 2.34
                xx = np.arange(-.4, .25, .01)  # helper variable for plotting
                ax.text(-.29, .01, r'$\varepsilon_k^b$',
                        color='C4', fontsize=12)
                p0 = -k_F * v_LDA
                yy = p0 + xx * v_LDA  # For plotting v_LDA
                ax.plot(xx, yy, 'C4--', lw=1.5)
                ax.set_yticks(np.arange(-.1, .05, .02))
                ax.set_yticklabels([])
            elif j == 3:
                # Plot cut of EDC's high temp
                ax.plot([k[j][_EDC_e[j], 0], k[j][_EDC_e[j], 0]],
                        [-1, .015],
                        ls='-.', color='r', lw=.5)
                ax.plot([-.35, -.35],
                        [en[j][0, 0], en[j][0, -1]],
                        ls='-.', color='r', lw=.5)

                # decorate axes
                ax.set_yticks(np.arange(-.1, .05, .02))
                ax.set_yticklabels([])
            else:
                ax.set_yticks(np.arange(-.1, .05, .02))
                ax.set_yticklabels([])

            # Draw boxes of intensity integration
            rbox = {'ls': '--', 'color': 'm', 'lw': .5}
            bbox = {'ls': '--', 'color': 'b', 'lw': .5}
            ax.plot([k[j][_Left_e[j], 0], k[j][_Left_e[j], 0]],
                    [en[j][0, _Top_e[j]], en[j][0, _Bot_e[j]]], **rbox)
            ax.plot([k[j][_Right_e[j], 0], k[j][_Right_e[j], 0]],
                    [en[j][0, _Top_e[j]], en[j][0, _Bot_e[j]]], **rbox)
            ax.plot([k[j][_Left_e[j], 0], k[j][_Right_e[j], 0]],
                    [en[j][0, _Top_e[j]], en[j][0, _Top_e[j]]], **rbox)
            ax.plot([k[j][_Left_e[j], 0], k[j][_Right_e[j], 0]],
                    [en[j][0, _Bot_e[j]], en[j][0, _Bot_e[j]]], **rbox)

            ax.plot([k[j][_Left_b[j], 0], k[j][_Left_b[j], 0]],
                    [en[j][0, _Top_b[j]], en[j][0, _Bot_b[j]]], **bbox)
            ax.plot([k[j][_Right_b[j], 0], k[j][_Right_b[j], 0]],
                    [en[j][0, _Top_b[j]], en[j][0, _Bot_b[j]]], **bbox)
            ax.plot([k[j][_Left_b[j], 0], k[j][_Right_b[j], 0]],
                    [en[j][0, _Top_b[j]], en[j][0, _Top_b[j]]], **bbox)
            ax.plot([k[j][_Left_b[j], 0], k[j][_Right_b[j], 0]],
                    [en[j][0, _Bot_b[j]], en[j][0, _Bot_b[j]]], **bbox)

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

    def fig4efg():
        # labels and position
        lbls = [r'(e) $\gamma$-band, @$\,$S',
                r'(f) $\gamma$-band (zoom), @$\,$S',
                r'(g) $\alpha$-band (zoom), @$\,k_\mathrm{F}$']
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
            ax.plot(xx, f_edc, '--', color='g', lw=1)
            ax.fill(xx, f_mod, alpha=.3, color='C8')
            ax.fill(xx, f_fl, alpha=.3, color='C8')
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

    def fig4h():
        ax = fig.add_subplot(248)
        ax.set_position([.08+3*.21, .29, .2, .2])
        ax.tick_params(**kwargs_ticks)

        # Plot integrated intensity
        ax.errorbar(T, int_e, yerr=eint_e, lw=.5,
                    capsize=2, color='m', fmt='o', ms=5)
        ax.errorbar(T, int_b, yerr=eint_b, lw=.5,
                    capsize=2, color='b', fmt='d', ms=5)
        ax.plot([1.3, 32], [1, .64], 'm--', lw=.5)
        ax.plot([1.3, 32], [.99, .99], 'b--', lw=.5)

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
        ax.text(1.3, 1.038, r'(h)')
        ax.text(8, .79, r'$\gamma$-band', color='m')
        ax.text(17, .92, r'$\alpha$-band', color='b')

    fig = plt.figure(figname, figsize=(8, 8), clear=True)
    fig4abcd()
    (en, EDCn_e, EDCn_b, EDC_e, EDC_b, Bkg_e, Bkg_b, _EDC_e, _EDC_b,
     eEDCn_e, eEDCn_b, eEDC_e, eEDC_b) = fig4efg()
    fig4h()
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


def fig5(print_fig=False, load=True):
    """figure 5

    %%%%%%%%%%%%%%%%%%%%%%%
    Analysis Z epsilon band
    %%%%%%%%%%%%%%%%%%%%%%%
    """

    figname = 'CSROfig5'

    # Load the data used
    if load:
        os.chdir(data_dir)
        en = np.loadtxt('Data_CSROfig4_en.dat')
        EDCn_e = np.loadtxt('Data_CSROfig4_EDCn_e.dat')
        EDCn_b = np.loadtxt('Data_CSROfig4_EDCn_b.dat')
        EDC_e = np.loadtxt('Data_CSROfig4_EDC_e.dat')
        EDC_b = np.loadtxt('Data_CSROfig4_EDC_b.dat')
        Bkg_e = np.loadtxt('Data_CSROfig4_Bkg_e.dat')
        Bkg_b = np.loadtxt('Data_CSROfig4_Bkg_b.dat')
        _EDC_e = np.loadtxt('Data_CSROfig4__EDC_e.dat', dtype=np.int32)
        _EDC_b = np.loadtxt('Data_CSROfig4__EDC_b.dat', dtype=np.int32)
        eEDCn_e = np.loadtxt('Data_CSROfig4_eEDCn_e.dat')
        eEDCn_b = np.loadtxt('Data_CSROfig4_eEDCn_b.dat')
        eEDC_e = np.loadtxt('Data_CSROfig4_eEDC_e.dat')
        eEDC_b = np.loadtxt('Data_CSROfig4_eEDC_b.dat')
        dims = np.loadtxt('Data_CSROfig4_dims.dat', dtype=np.int32)
        print('\n ~ Data loaded (en, EDCs + normalized + indices + Bkgs)',
              '\n', '==========================================')
        os.chdir(home_dir)
    else:
        (en, EDCn_e, EDCn_b, EDC_e, EDC_b, Bkg_e, Bkg_b, _EDC_e, _EDC_b,
         eEDCn_e, eEDCn_b, eEDC_e, eEDC_b, dims) = fig4()

    # Reshape data into right format
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

    # lables
    titles = [r'$T=1.3\,$K', r'$T=10\,$K', r'$T=20\,$K', r'$T=30\,$K']
    lbls = [r'(a)', r'(b)', r'(c)', r'(d)',
            r'(e)', r'(f)', r'(g)', r'(h)',
            r'(i)', r'(j)', r'(k)', r'(l)']

    # colors
    cols = ([0, 1, 1], [0, .7, .7], [0, .4, .4], [0, 0, 0])
    cols_r = ([0, 0, 0], [0, .4, .4], [0, .7, .7], [0, 1, 1])
    xx = np.arange(-2, .5, .001)
    Z = np.ones((4))

    # Create figure
    fig = plt.figure(figname, figsize=(10, 10), clear=True)
    for j in range(4):

        # first row
        Bkg = Bkg_e[j]
        Bkg[0] = 0
        Bkg[-1] = 0
        ax = fig.add_subplot(5, 4, j+1)
        ax.set_position([.08+j*.21, .61, .2, .2])
        ax.tick_params(**kwargs_ticks)

        # plot data
        ax.plot(en[j][_EDC_e[j]], EDC_e[j], 'o', ms=1, color=cols[j])
        ax.fill(en[j][_EDC_e[j]], Bkg, '--', lw=1, color='C8', alpha=.3)
        ax.set_yticks([])
        ax.set_xticks(np.arange(-.8, .2, .2))
        ax.set_xticklabels([])
        ax.set_xlim(-.8, .1)
        ax.set_ylim(0, .018)

        # add text
        ax.text(-.77, .0163, lbls[j])
        ax.set_title(titles[j], fontsize=15)
        if j == 0:
            ax.text(-.77, .001, r'Background')
            ax.set_ylabel(r'Intensity (a.u.)')

        # third row
        ax = plt.subplot(5, 4, j+13)
        ax.set_position([.08+j*.21, .18, .2, .2])
        ax.tick_params(**kwargs_ticks)
        ax.plot(en[j][_EDC_e[j]], EDCn_e[j], 'o', ms=1, color=cols[j])

        # initial guess
        p_edc_i = np.array([6.9e-1, 7.3e-3, 4.6, 4.7e-3, 4.1e-2, 2.6e-3,
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

        # decorate axes
        ax.set_yticks([])
        ax.set_xticks(np.arange(-.8, .2, .1))
        ax.set_xlim(-.1, .05)
        ax.set_ylim(0, 1.3)
        ax.set_xlabel(r'$\omega$ (eV)')
        if j == 0:
            ax.set_ylabel(r'Intensity (a.u.)')

            # add text
            ax.text(-.09, .2, r'$\int \, \, \mathcal{A}_\mathrm{coh.}$' +
                    r'$(k=\mathrm{S}, \omega)$' +
                    r'$\, \mathrm{d}\omega$')

        ax.text(-.095, 1.21, lbls[j + 8])

        # second row
        ax = fig.add_subplot(5, 4, j+9)
        ax.set_position([.08+j*.21, .4, .2, .2])
        ax.tick_params(**kwargs_ticks)

        # plot data
        ax.plot(en[j][_EDC_e[j]], EDCn_e[j], 'o', ms=1, color=cols[j])

        # boundary for fit
        bounds = (np.concatenate((p_fl - D, p_edc_i[6:] - D), axis=0),
                  np.concatenate((p_fl + D, p_edc_i[6:] + D), axis=0))
        bnd = 300  # range to fit the data

        # fit data
        p_edc, c_edc = curve_fit(utils.Full_spectral_func,
                                 en[j][_EDC_e[j]][bnd:-1],
                                 EDCn_e[j][bnd:-1],
                                 np.concatenate((p_fl, p_edc_i[-6:]), axis=0),
                                 bounds=bounds)

        # plot spectral function
        f_edc = utils.Full_spectral_func(xx, *p_edc)
        ax.plot(xx, f_edc, '--', color=cols_r[j], lw=1.5)

        # plot coherent and incoherent weight
        f_mod = utils.gauss_mod(xx, *p_edc[-6:])
        f_fl = utils.FL_spectral_func(xx, *p_edc[0:6])
        ax.fill(xx, f_mod, alpha=.3, color=cols[j])

        # decorate axes
        ax.set_yticks([])
        ax.set_xticks(np.arange(-.8, .2, .2))
        ax.set_xlim(-.8, .1)
        ax.set_ylim(0, 2.2)
        if j == 0:
            ax.set_ylabel(r'Intensity (a.u.)')
            ax.text(-.63, .3, r'$\int \, \, \mathcal{A}_\mathrm{inc.}$' +
                    r'$(k=\mathrm{S}, \omega)$' +
                    r'$\, \mathrm{d}\omega$')
        ax.text(-.77, 2.03, lbls[j+4])

        # third row
        ax = plt.subplot(5, 4, j+13)
        ax.fill(xx, f_fl, alpha=.3, color=cols[j])
        p = ax.plot(xx, f_edc, '--', color=cols_r[j],  linewidth=2)
        ax.legend(p, [r'$\mathcal{A}_\mathrm{coh.} +$' +
                      r'$\mathcal{A}_\mathrm{inc.}$'], frameon=False)

        # Calculate Z
        A_mod = integrate.trapz(f_mod, xx)
        A_fl = integrate.trapz(f_fl, xx)
        Z[j] = A_fl / A_mod

    os.chdir(data_dir)
    np.savetxt('Data_CSROfig5_Z_e.dat', Z)
    print('\n ~ Data saved (Z)',
          '\n', '==========================================')
    os.chdir(home_dir)
    plt.show()

    # Save figure
    if print_fig:
        plt.savefig(save_dir + figname + '.pdf', dpi=100,
                    bbox_inches="tight", rasterized=True)

    return Z


def fig6(print_fig=True, load=True):
    """figure 6

    %%%%%%%%%%%%%%%%%%%%%%%%
    Analysis MDC's beta band
    %%%%%%%%%%%%%%%%%%%%%%%%
    """

    figname = 'CSROfig6'

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
    cols = ([0, 1, 1], [0, .7, .7], [0, .4, .4], [0, 0, 0])
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
                         vmin=.0 * np.max(spec[0]), vmax=1 * np.max(spec[0]),
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
            ax.text(-.46, .009, r'MDC maxima', color='C8')
            ax.text(-.26, .009, r'$\epsilon_\mathbf{k}^b$',
                    color='C4')
        else:
            ax.set_yticks(np.arange(-.2, .1, .05))
            ax.set_yticklabels([])
        ax.set_xticks(np.arange(-1, 0, .1))
        ax.set_xticklabels([])
        ax.set_xlim(-.5, -.1)
        ax.set_ylim(-.15, .05)

        # add labels
        ax.text(-.49, .035, lbls[j])
        ax.set_title(titles[j], fontsize=15)

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
                             lw=.5, capsize=.1, color=cols[j], fmt='o', ms=.5)

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
                ax.plot(mdc_k, f_mdc - scale * n**1.15, '--', color=cols_r[j])
                ax.plot(mdc_k, b_mdc - scale * n**1.15, 'C8-', lw=2, alpha=.3)

        # decorate axes
        if j == 0:
            ax.set_ylabel('Intensity (a.u.)', fontdict=font)
            ax.text(-.48, -.0092, r'Background', color='C8')
        ax.set_yticks([])
        ax.set_xticks(np.arange(-1, -.1, .1))
        ax.set_xlim(-.5, -.1)
        ax.set_ylim(-.01, .003)
        ax.text(-.49, .0021, lbls[j + 4])
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

        ax.plot(-loc_en, utils.poly_2(-loc_en, *p_im), '--', color=cols_r[j])
        print(np.sqrt(np.diag(c_im)))

        # decorate axes
        if j == 0:
            ax.set_ylabel(r'HWHM $(\mathrm{\AA}^{-1})$', fontdict=font)
            ax.set_yticks(np.arange(0, 1, .05))
            ax.text(.005, .05, r'Quadratic fit', fontdict=font)
        else:
            ax.set_yticks(np.arange(0, 1, .05))
            ax.set_yticklabels([])
        ax.set_xticks(np.arange(0, .1, .02))
        ax.set_xticklabels([])
        ax.set_xlim(-loc_en[0], -loc_en[-1])
        ax.set_ylim(0, .13)
        ax.text(.0025, .12, lbls[j+8])

        # Fourth row
        k_F = loc[0]  # Position first fit
        print('kF='+str(k_F))
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
                '--', color=Re_cols_r[j])

        # quasiparticle residue
        z = 1 / (1 - dre)
        ez = np.abs(1 / (1 - dre)**2 * edre)  # error

        # decorate axes
        if j == 0:
            ax.set_ylabel(r'$\Re \Sigma$ (meV)', fontdict=font)
            ax.set_yticks(np.arange(0, .15, .05))
            ax.set_yticklabels(['0', '50', '100'])
            ax.text(.02, .03, 'Linear fit', fontsize=12, color=Re_cols[-1])
        else:
            ax.set_yticks(np.arange(0, .15, .05))
            ax.set_yticklabels([])
        ax.set_xticks(np.arange(0, .1, .02))
        ax.set_xticklabels(['0', '-20', '-40', '-60', '-80', '-100'])
        ax.set_xlabel(r'$\omega$ (meV)', fontdict=font)
        ax.set_xlim(-loc_en[0], -loc_en[-1])
        ax.set_ylim(0, .15)
        ax.text(.0025, .14, lbls[j+12])

        # First row again
        ax = fig.add_subplot(4, 4, j+1)
        ax.plot(loc, loc_en, 'C8o', ms=.5)
        ax.plot(xx, yy, 'C4--', lw=1)

        # decorate axes
        if j == 0:
            ax.arrow(loc[20], -.05, 0, loc_en[20]+.04,
                     head_width=0.01, head_length=0.01, fc='k', ec='k')
            ax.arrow(loc[20], -.05, 0, loc_en[20]+.005,
                     head_width=0.01, head_length=0.01, fc='k', ec='k')
            plt.text(-.28, -.048, r'$\Re \Sigma(\omega)$', color='k')
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

    dims = np.array([len(Re), Re[0].shape[0]])
    os.chdir(data_dir)
    np.savetxt('Data_CSROfig6_Z_b.dat', np.ravel(Z))
    np.savetxt('Data_CSROfig6_eZ_b.dat', np.ravel(eZ))
    np.savetxt('Data_CSROfig6_Re.dat', np.ravel(Re))
    np.savetxt('Data_CSROfig6_Loc_en.dat', np.ravel(Loc_en))
    np.savetxt('Data_CSROfig6_Width.dat', np.ravel(Width))
    np.savetxt('Data_CSROfig6_eWidth.dat', np.ravel(eWidth))
    np.savetxt('Data_CSROfig6_dims.dat', np.ravel(dims))
    print('\n ~ Data saved (Z, eZ, Re, Loc_en, Width, eWidth)',
          '\n', '==========================================')
    os.chdir(home_dir)

    return Z, eZ, Re, Loc_en, Width, eWidth, dims


def fig7(print_fig=True):
    """figure 7

    %%%%%%%%%%%%%%%%%%%%%%
    Background subtraction
    %%%%%%%%%%%%%%%%%%%%%%
    """

    figname = 'CSROfig7'

    file = 25
    gold = 14
    mat = 'CSRO20'
    year = 2017
    sample = 'S1'

    # labels and background on/off
    lbls = [r'(a)', r'(b)', r'(c)']

    fig = plt.figure(figname, figsize=(8, 8), clear=True)
    for i in range(2):

        # load data
        D = ARPES.Bessy(file, mat, year, sample)
        D.norm(gold)

        # background subtraction
        if i == 1:
            D.bkg()

        D.ang2k(D.ang, Ekin=40, lat_unit=True, a=5.5, b=5.5, c=11,
                V0=0, thdg=2.5, tidg=0, phidg=42)

        # create background EDC
        if i == 0:
            edc_bkg = np.zeros((D.en.size))
            for j in range(D.en.size):
                edc_bkg[j] = np.min(D.int_norm[:, j])
            edc_bkg[0] = 0

        ax = fig.add_subplot(1, 3, i+1)
        ax.set_position([.08+i*.26, .3, .25, .5])
        ax.tick_params(**kwargs_ticks)

        # plot data
        c0 = ax.contourf(D.kxs, D.en_norm - .008, D.int_norm, 300,
                         **kwargs_ex, vmin=.0, vmax=.03, zorder=.1)
        ax.set_rasterization_zorder(.2)
        ax.plot([np.min(D.kxs), np.max(D.kxs)], [0, 0], **kwargs_ef)

        # decorate axes
        if i == 0:
            ax.set_ylabel(r'$\omega\,(\mathrm{meV})$', fontdict=font)
            ax.set_yticks(np.arange(-.8, .4, .2))
        else:
            ax.set_yticks(np.arange(-.8, .2, .2))
            ax.set_yticklabels([])
        ax.set_xticks([-1, 0])
        ax.set_xticklabels(['S', r'$\Gamma$'])
        ax.set_ylim(-.8, .1)

        # add text
        ax.text(-1.2, .06, lbls[i])

    # EDC panel
    ax = fig.add_subplot(133)
    ax.set_position([.08 + .52, .3, .25, .5])
    ax.tick_params(**kwargs_ticks)

    # Plot background EDC
    ax.plot(edc_bkg, D.en_norm[0] - .008, 'ko', ms=1)
    ax.fill(edc_bkg, D.en_norm[0] - .008, alpha=.2, color='C8')
    ax.plot([0, 1], [0, 0], **kwargs_ef)

    # decorate axes
    ax.set_xticks([])
    ax.set_yticks(np.arange(-.8, .2, .2))
    ax.set_yticklabels([])
    ax.set_ylim(-.8, .1)
    ax.set_xlim(0, np.max(edc_bkg) * 1.1)
    ax.set_xlabel('Intensity (a.u.)', fontdict=font)

    # add text
    ax.text(.0025, -.45, 'Background EDC')
    ax.text(.0008, .06, lbls[2])

    # colorbar
    pos = ax.get_position()
    cax = plt.axes([pos.x0+pos.width+0.01, pos.y0, 0.01, pos.height])
    cbar = plt.colorbar(c0, cax=cax, ticks=None)
    cbar.set_ticks([])
    cbar.set_clim(np.min(D.int_norm), np.max(D.int_norm))
    plt.show()

    # Save figure
    if print_fig:
        plt.savefig(save_dir + figname + '.pdf', dpi=100,
                    bbox_inches="tight", rasterized=True)


def fig8(print_fig=True):
    """figure 8

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Extraction LDA Fermi velocity
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    """

    figname = 'CSROfig8'

    os.chdir(data_dir)
    xz_lda = np.loadtxt('LDA_CSRO_xz.dat')
    os.chdir(home_dir)

    # full k-path
    # [0, 56, 110, 187, 241, 266, 325, 350]  = [G, X, S, G, Y, T, G, Z]

    # prepare data
    m, n = 8000, 351  # dimensions energy, full k-path
    size = 187 - 110
    bot, top = 3840, 4055  # restrict energy window
    data = np.array([xz_lda])  # combine data
    spec = np.reshape(data[0, :, 2], (n, m))  # reshape into n,m
    spec = spec[110:187, bot:top]  # restrict data to bot, top
    # spec = np.flipud(spec)
    spec_en = np.linspace(-8, 8, m)  # define energy data
    spec_en = spec_en[bot:top]  # restrict energy data
    spec_en = np.broadcast_to(spec_en, (spec.shape))
    spec_k = np.linspace(-np.sqrt(2) * np.pi / 5.5, 0, size)
    spec_k = np.transpose(
                np.broadcast_to(spec_k, (spec.shape[1], spec.shape[0])))
    max_pts = np.ones((size))

    # extract eigenenergies in the DFT plot
    for i in range(size):
        max_pts[i] = spec[i, :].argmax()
    ini = 40  # range where to search for
    fin = 48
    max_pts = max_pts[ini:fin]
    max_k = spec_k[ini:fin, 0]
    max_en = spec_en[0, max_pts.astype(int)]

    # initial guess
    p_max_i = np.array([0, 0])

    # fit
    p_max, c_max = curve_fit(utils.poly_1, max_k, max_en, p0=p_max_i)

    # extract Fermi velocity
    v_LDA = p_max[1]
    ev_LDA = np.sqrt(np.diag(c_max))[1]
    # k_F = -p_max[0] / p_max[1]  # Fermi momentum

    # for plotting
    xx = np.arange(-.43, -.27, .01)

    # create figure
    fig = plt.figure(figname, figsize=(8, 8), clear=True)
    ax = fig.add_subplot(131)
    ax.set_position([.2, .24, .5, .3])
    ax.tick_params(**kwargs_ticks)

    # plot DFT spectra
    c0 = ax.contourf(spec_k, spec_en, spec, 300, **kwargs_th, zorder=.1)
    ax.set_rasterization_zorder(.2)
    ax.plot(xx, utils.poly_1(xx, *p_max), 'C9--', linewidth=1)
    ax.plot(max_k, max_en, 'ro', ms=2)
    ax.plot([np.min(spec_k), 0], [0, 0], **kwargs_ef)

    # deocrate axes
    ax.set_yticks(np.arange(-1, 1, .1))
    ax.set_ylim(-.3, .1)
    ax.set_ylabel(r'$\omega$ (eV)', fontdict=font)
    ax.set_xlabel(r'$k_{\Gamma - \mathrm{S}}\, (\mathrm{\AA}^{-1})$',
                  fontdict=font)

    # add text
    ax.text(-.3, -.15, r'$v_\mathrm{LDA}=$' + str(np.round(v_LDA, 2)) +
            r' eV$\,\mathrm{\AA}$')

    # colorbar
    pos = ax.get_position()
    cax = plt.axes([pos.x0+pos.width+0.01, pos.y0, 0.01, pos.height])
    cbar = plt.colorbar(c0, cax=cax, ticks=None)
    cbar.set_ticks([])
    cbar.set_clim(np.min(spec), np.max(spec))
    plt.show()

    # Save figure
    if print_fig:
        plt.savefig(save_dir + figname + '.pdf', dpi=100,
                    bbox_inches="tight", rasterized=True)

    # save data
    os.chdir(data_dir)
    np.savetxt('Data_CSROfig8_v_LDA.dat', [v_LDA, ev_LDA])
    os.chdir(home_dir)

    return v_LDA, ev_LDA


def fig9(print_fig=True, load=True):
    """figure 9

    %%%%%%%%%%%%%%%%%%
    ReSigma vs ImSigma
    %%%%%%%%%%%%%%%%%%
    """

    figname = 'CSROfig9'

    if load:
        os.chdir(data_dir)
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
        os.chdir(home_dir)
    else:
        Z, eZ, Re, Loc_en, Width, eWidth, dims = fig6()
        v_LDA, ev_LDA = fig8()

    # Reshape data
    Re = np.reshape(np.ravel(Re), (dims[0], dims[1]))
    Loc_en = np.reshape(np.ravel(Loc_en), (dims[0], dims[1]))
    Width = np.reshape(np.ravel(Width), (dims[0], dims[1]))
    eWidth = np.reshape(np.ravel(eWidth), (dims[0], dims[1]))

    # create figure
    fig = plt.figure(figname, figsize=(8, 8), clear=True)

    # placeholders
    re = np.zeros(Re.shape)
    ere = np.zeros(Re.shape)
    im = np.zeros(Re.shape)
    eim = np.zeros(Re.shape)

    # labels
    lbls = [r'(a)  $T=1.3\,$K', r'(b)  $T=10\,$K', r'(c)  $T=20\,$K',
            r'(d)  $T=30\,$K']

    # colors
    Im_cols = np.array([[0, 1, 1], [0, .7, .7], [0, .4, .4], [0, 0, 0]])
    Re_cols = ['khaki', 'darkkhaki', 'goldenrod', 'darkgoldenrod']
    n_spec = 4  # number of spectra shown

    # positions of subplots
    position = ([.1, .55, .4, .4],
                [.1 + .41, .55, .4, .4],
                [.1, .55 - .41, .4, .4],
                [.1 + .41, .55 - .41, .4, .4])

    # constant part of HWHM fit
    offset = [.055, .052, .049, .05]

    n = 0  # counter
    for j in range(n_spec):

        # prepare data
        en = -Loc_en[j]
        width = Width[j]
        ewidth = eWidth[j]
        im[j] = width * v_LDA - offset[j]  # imaginary part of self-energy
        eim[j] = ewidth * v_LDA  # error
        re[j] = Re[j] / (1 - Z[j])   # Rescaling for cut-off condition
        ere[j] = ewidth * v_LDA / (1 - Z[j])  # error

        # panels
        ax = fig.add_subplot(2, 2, j+1)
        ax.set_position(position[j])
        ax.tick_params(**kwargs_ticks)

        # plot data
        ax.errorbar(en, im[j], eim[j],
                    color=Im_cols[j], lw=.5, capsize=2, fmt='d', ms=2)
        ax.errorbar(en, re[j], ere[j],
                    color=Re_cols[j], lw=.5, capsize=2, fmt='o', ms=2)

        # decorate axes
        if any(x == j for x in [0, 2]):
            ax.set_ylabel('Self energy (meV)', fontdict=font)
            ax.set_yticks(np.arange(0, .25, .05))
            ax.set_yticklabels(['0', '50', '100', '150', '200'])
            ax.set_xticks(np.arange(0, .1, .02))
            ax.set_xticklabels([])
        else:
            ax.set_yticks(np.arange(0, .25, .05))
            ax.set_yticklabels([])
        if any(x == j for x in [2, 3]):
            ax.set_xticks(np.arange(0, .1, .02))
            ax.set_xticklabels(['0', '-20', '-40', '-60', '-80', '-100'])
            ax.set_xlabel(r'$\omega\,(\mathrm{meV})$', fontdict=font)
        else:
            ax.set_xticks(np.arange(0, .1, .02))
            ax.set_xticklabels([])
        ax.set_xlim(0, .1)
        ax.set_ylim(-.01, .25)
        ax.grid(True, alpha=.2)

        # add text
        ax.text(.002, .235, lbls[j], fontdict=font)
        if j == 0:
            ax.text(.005, .15, r'$\mathfrak{Re}\Sigma \, (1-Z)^{-1}$',
                    fontsize=15, color=Re_cols[3])
            ax.text(.06, .014, r'$\mathfrak{Im}\Sigma$',
                    fontsize=15, color=Im_cols[2])
        n += 1

    plt.show()

    # Save figure
    if print_fig:
        plt.savefig(save_dir + figname + '.pdf', dpi=100,
                    bbox_inches="tight", rasterized=True)

    # save data
    os.chdir(data_dir)
    np.savetxt('Data_CSROfig9_re.dat', np.ravel(re))
    np.savetxt('Data_CSROfig9_ere.dat', np.ravel(ere))
    np.savetxt('Data_CSROfig9_im.dat', np.ravel(im))
    np.savetxt('Data_CSROfig9_eim.dat', np.ravel(eim))
    print('\n ~ Data saved (re, ere, im, eim)',
          '\n', '==========================================')
    os.chdir(home_dir)


def fig10(print_fig=True):
    """figure 10

    %%%%%%%%%%%%%%%%
    Quasiparticle Z
    %%%%%%%%%%%%%%%%
    """

    figname = 'CSROfig10'

    os.chdir(data_dir)
    Z_e = np.loadtxt('Data_CSROfig5_Z_e.dat')
    Z_b = np.loadtxt('Data_CSROfig6_Z_b.dat')
    eZ_b = np.loadtxt('Data_CSROfig6_eZ_b.dat')
    v_LDA_data = np.loadtxt('Data_CSROfig8_v_LDA.dat')
    v_LDA = v_LDA_data[0]
    C_B = np.genfromtxt('Data_C_Braden.csv', delimiter=',')
    C_M = np.genfromtxt('Data_C_Maeno.csv', delimiter=',')
    R_1 = np.genfromtxt('Data_R_1.csv', delimiter=',')
    R_2 = np.genfromtxt('Data_R_2.csv', delimiter=',')
    os.chdir(home_dir)
    print('\n ~ Data loaded (Zs, specific heat, transport data)',
          '\n', '==========================================')

    # useful parameter
    T = np.array([1.3, 10, 20, 30])  # temperatures
    hbar = 1.0545717e-34  # Planck constant
    NA = 6.022141e23  # Avogadro constant
    kB = 1.38065e-23  # Boltzmann constant
    a = 5.33e-10  # lattice parameter
    m_e = 9.109383e-31
    m_LDA = 1.6032

    # Sommerfeld constant for 2-dimensional systems
    gamma = ((np.pi * NA * kB ** 2 * a ** 2 / (3 * hbar ** 2)) *
             m_LDA * m_e)

    # heat capacity in units of Z
    Z_B = gamma / C_B[:, 1]
    Z_M = gamma / C_M[:, 1] * 1e3

    # fit for resistivity curve
    xx = np.array([1e-3, 1e4])
    yy = 2.3 * xx ** 2

    # create figure
    fig = plt.figure(figname, figsize=(10, 10), clear=True)
    ax = fig.add_subplot(121)
    ax.set_position([.2, .3, .3, .3])
    ax.tick_params(direction='in', length=1.5, width=.5, colors='k')

    # plot data beta band
    ax.errorbar(T, Z_b, eZ_b * v_LDA,
                color='m', lw=.5, capsize=2, fmt='o', ms=2)
    ax.fill_between([0, 50], .215, .3, alpha=.1, color='m')
    ax.plot(39, .229, 'm*')

    # plot data epsilon band
    ax.errorbar(T, Z_e, Z_e / v_LDA,
                color='r', lw=.5, capsize=2, fmt='d', ms=2)
    ax.fill_between([0, 50], 0.02, .08, alpha=.1, color='r')
    ax.plot(39, .052, 'r*')

    # plot Matsubara data
    # ax.plot(39, .326, 'C1+')  # Matsubara point
    # ax.plot(39, .175, 'r+')  # Matsubara point

    # plot heat capacity data
    ax.plot(C_B[:, 0], Z_B, 'o', ms=1, color='cadetblue')
    ax.plot(C_M[:, 0], Z_M, 'o', ms=1, color='slateblue')

    # decorate axes
    ax.arrow(28, .16, 8.5, .06, head_width=0.0, head_length=0,
             fc='k', ec='k')
    ax.arrow(28, .125, 8.5, -.06, head_width=0.0, head_length=0,
             fc='k', ec='k')
    ax.set_xscale("log", nonposx='clip')
    ax.set_yticks(np.arange(0, .5, .1))
    ax.set_xlim(1, 44)
    ax.set_ylim(0, .35)
    ax.set_xlabel(r'$T$ (K)')
    ax.set_ylabel(r'$Z$')

    # add text
    ax.text(1.2, .33, r'S. Nakatsuji $\mathit{et\, \,al.}$',
            color='slateblue')
    ax.text(1.2, .31, r'J. Baier $\mathit{et\, \,al.}$',
            color='cadetblue')
    ax.text(2.5e0, .25, r'$\bar{\beta}$-band', color='m')
    ax.text(2.5e0, .045, r'$\bar{\delta}$-band', color='r')
    ax.text(20, .135, 'DMFT')

    # Inset
    axi = fig.add_subplot(122)
    axi.set_position([.28, .39, .13, .08])
    axi.tick_params(**kwargs_ticks)

    # Plot resistivity data
    axi.loglog(np.sqrt(R_1[:, 0]), R_1[:, 1], 'o', ms=1,
               color='slateblue')
    axi.loglog(np.sqrt(R_2[:, 0]), R_2[:, 1], 'o', ms=1,
               color='slateblue')
    axi.loglog(xx, yy, 'k--', lw=1)

    # decorate axes
    axi.set_ylabel(r'$\rho\,(\mu \Omega \mathrm{cm})$')
    axi.set_xlim(1e-1, 1e1)
    axi.set_ylim(1e-2, 1e4)

    # add text
    axi.text(2e-1, 1e1, r'$\propto T^2$')

    plt.show()

    # Save figure
    if print_fig:
        plt.savefig(save_dir + figname + '.pdf', dpi=100,
                    bbox_inches="tight", rasterized=True)


def fig11(print_fig=True):
    """figure 11

    %%%%%%%%%%%%%%%%%%%%%%%%
    Tight binding model CSRO
    %%%%%%%%%%%%%%%%%%%%%%%%
    """

    figname = 'CSROfig11'

    # Initialize tight binding model
    kbnd = 2  # boundaries
    tb = utils.TB(a=np.pi, kbnd=kbnd, kpoints=300)
    param = utils.paramCSRO20_opt()
    tb.CSRO(param=param, e0=0, vert=True, proj=True)

    # load data
    bndstr = tb.bndstr
    coord = tb.coord
    X = coord['X']
    Y = coord['Y']
    Axz = bndstr['Axz']
    Ayz = bndstr['Ayz']
    Axy = bndstr['Axy']
    Bxz = bndstr['Bxz']
    Byz = bndstr['Byz']
    Bxy = bndstr['Bxy']

    # collect bands
    bands = (Axz, Ayz, Axy, Bxz, Byz, Bxy)

    # create figure
    fig = plt.figure(figname, figsize=(8, 8), clear=True)
    v_bnd = np.max(tb.FS)
    ax = fig.add_subplot(122)
    ax.set_position([.1, .3, .4, .4])
    ax.tick_params(**kwargs_ticks)

    # plot data
    for band in bands:
        ax.contour(X, Y, band, colors='black', ls='-',
                   levels=0, alpha=.2, zorder=.1)
    ax.contourf(tb.kx, tb.ky, tb.FS, 300, cmap='PuOr',
                vmin=-v_bnd, vmax=v_bnd, alpha=.05, zorder=.1)
    c0 = ax.contourf(tb.kx[250:750], tb.ky[250:750], tb.FS[250:750, 250:750],
                     300, cmap='PuOr', vmin=-v_bnd, vmax=v_bnd, zorder=.1)
    ax.set_rasterization_zorder(.2)
#    ax.plot([-1, 1], [1, 1], 'k-', lw=2)
#    ax.plot([-1, 1], [-1, -1], 'k-', lw=2)
#    ax.plot([1, 1], [-1, 1], 'k-', lw=2)
#    ax.plot([-1, -1], [-1, 1], 'k-', lw=2)

    # deocrate axes
    ax.set_xticks(np.arange(-kbnd - 1, kbnd + 1, 1))
    ax.set_yticks(np.arange(-kbnd - 1, kbnd + 1, 1))
#    ax.set_xticklabels([])
#    ax.set_yticklabels([])
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_xlabel(r'$k_x \, (\pi/a)$', fontdict=font)
    ax.set_ylabel(r'$k_y \, (\pi/b)$', fontdict=font)

    # add text
    ax.text(-.05, -.05, r'$\Gamma$', fontsize=18, color='r')
    ax.text(.95, .95, 'S', fontsize=18, color='r')
    ax.text(-.06, .95, 'X', fontsize=18, color='r')

    pos = ax.get_position()
    cax = plt.axes([pos.x0+pos.width + 0.01, pos.y0, 0.01, pos.height])
    cbar = plt.colorbar(c0, cax=cax, ticks=None)
    cbar.set_ticks([])
    cbar.set_clim(np.min(tb.FS), np.max(tb.FS))

    plt.show()

    # Save figure
    if print_fig:
        plt.savefig(save_dir + figname + '.pdf', dpi=100,
                    bbox_inches="tight", rasterized=True)


def fig12(print_fig=True):
    """figure 12

    %%%%%%%%%%%%%%%%%%%%%%%
    Tight binding model SRO
    %%%%%%%%%%%%%%%%%%%%%%%
    """

    figname = 'CSROfig12'

    # Initialize tight binding model
    kbnd = 2  # boundaries
    tb = utils.TB(a=np.pi, kbnd=kbnd, kpoints=300)
    param = utils.paramSRO()
    tb.SRO(param=param, e0=0, vert=True, proj=True)

    # load data
    bndstr = tb.bndstr
    coord = tb.coord
    X = coord['X']
    Y = coord['Y']
    xz = bndstr['xz']
    yz = bndstr['yz']
    xy = bndstr['xy']

    # collect bands
    bands = (xz, yz, xy)

    # create figure
    fig = plt.figure(figname, figsize=(8, 8), clear=True)
    v_bnd = np.max(tb.FS)
    ax = fig.add_subplot(122)
    ax.set_position([.1, .3, .4, .4])
    ax.tick_params(**kwargs_ticks)

    # plot data
    for band in bands:
        ax.contour(X, Y, band, colors='black', ls='-',
                   levels=0, alpha=.2)
    ax.contourf(tb.kx, tb.ky, tb.FS, 300, cmap='PuOr', zorder=.1,
                vmin=-v_bnd, vmax=v_bnd, alpha=.05)
    c0 = ax.contourf(tb.kx[250:750], tb.ky[250:750], tb.FS[250:750, 250:750],
                     300, cmap='PuOr', vmin=-v_bnd, vmax=v_bnd, zorder=.1)
    ax.set_rasterization_zorder(.2)
    ax.plot([-1, 1], [1, 1], 'k-', lw=2)
    ax.plot([-1, 1], [-1, -1], 'k-', lw=2)
    ax.plot([1, 1], [-1, 1], 'k-', lw=2)
    ax.plot([-1, -1], [-1, 1], 'k-', lw=2)
    ax.plot([-1, 0], [0, 1], 'k--', lw=1)
    ax.plot([-1, 0], [0, -1], 'k--', lw=1)
    ax.plot([0, 1], [1, 0], 'k--', lw=1)
    ax.plot([0, 1], [-1, 0], 'k--', lw=1)

    # deocrate axes
    ax.set_xticks(np.arange(-kbnd - 1, kbnd + 1, 1))
    ax.set_yticks(np.arange(-kbnd - 1, kbnd + 1, 1))
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_xlabel(r'$k_x \, (\pi/a)$', fontdict=font)
    ax.set_ylabel(r'$k_y \, (\pi/b)$', fontdict=font)

    pos = ax.get_position()
    cax = plt.axes([pos.x0+pos.width + 0.01, pos.y0, 0.01, pos.height])
    cbar = plt.colorbar(c0, cax=cax, ticks=None)
    cbar.set_ticks([])
    cbar.set_clim(np.min(tb.FS), np.max(tb.FS))

    plt.show()

    # Save figure
    if print_fig:
        plt.savefig(save_dir + figname + '.pdf', dpi=100,
                    bbox_inches="tight", rasterized=True)


def fig13(print_fig=True):
    """figure 13

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    TB along high symmetry directions, orbitally resolved
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    """

    figname = 'CSROfig13'

    # kpoints
    k_pts = 300
    x_GS = np.linspace(0, 1, k_pts)
    y_GS = np.linspace(0, 1, k_pts)
    x_SX = np.linspace(1, 0, k_pts)
    y_SX = np.ones(k_pts)
    x_XG = np.zeros(k_pts)
    y_XG = np.linspace(1, 0, k_pts)

    # full kpath
    x = (x_GS, x_SX, x_XG)
    y = (y_GS, y_SX, y_XG)

    # create figure
    fig = plt.figure(figname, figsize=(6, 6), clear=True)
    for i in range(len(x)):

        # calculate bandstructure
        en, spec, bndstr = utils.CSRO_eval_proj(x[i], y[i])
        k = np.sqrt(x[i] ** 2 + y[i] ** 2)  # norm
        v_bnd = .2  # set point of coloscale
        if i != 0:
            ax = fig.add_subplot(2, 3, i+1)
            ax.set_position([.1+i*.15+.15*(np.sqrt(2)-1),
                             .2, .15, .4])
            k = -k
        else:
            ax = fig.add_subplot(2, 3, i+1)
            ax.set_position([.1+i*.15, .2, .15*np.sqrt(2), .4])
        ax.tick_params(**kwargs_ticks)

        # plot data
        c0 = ax.contourf(k, en, spec, 300, zorder=.1,
                         cmap='PuOr', vmin=-v_bnd, vmax=v_bnd)
        ax.set_rasterization_zorder(.2)
        ax.plot([k[0], k[-1]], [0, 0], **kwargs_ef)
        for j in range(len(bndstr)):
            ax.plot(k, bndstr[j], 'k-', alpha=.2)

        # decorate axes
        if i == 0:
            ax.set_xticks([0, np.sqrt(2)])
            ax.set_xticklabels([r'$\Gamma$', 'S'])
            ax.set_yticks(np.arange(-1, 1, .2))
        elif i == 1:
            ax.set_xticks([-np.sqrt(2), -1])
            ax.set_xticklabels(['', 'X'])
            ax.set_yticks(np.arange(-1, 1, .2))
            ax.set_yticklabels([])
        elif i == 2:
            ax.set_xticks([-1, 0])
            ax.set_xticklabels(['', r'$\Gamma$'])
            ax.set_yticks(np.arange(-1, 1, .2))
            ax.set_yticklabels([])
        ax.set_ylim(np.min(en), np.max(en))

    # colorbar
    pos = ax.get_position()
    cax = plt.axes([pos.x0+pos.width+0.01, pos.y0, 0.01, pos.height])
    cbar = plt.colorbar(c0, cax=cax, ticks=None)
    cbar.set_ticks([])
    cbar.set_clim(np.min(spec), np.max(spec))

    plt.show()

    # Save figure
    if print_fig:
        plt.savefig(save_dir + figname + '.pdf', dpi=100,
                    bbox_inches="tight", rasterized=True)


def fig14(print_fig=True):
    """figure 14

    %%%%%%%%%%%%%%%%%%%%%%%%
    TB and density of states
    %%%%%%%%%%%%%%%%%%%%%%%%
    """

    figname = 'CSROfig14'

    # load data
    os.chdir(data_dir)
    Axz_dos = np.loadtxt('Data_CSRO20_Axz_kpts_5000.dat')
    Ayz_dos = np.loadtxt('Data_CSRO20_Ayz_kpts_5000.dat')
    Axy_dos = np.loadtxt('Data_CSRO20_Axy_kpts_5000.dat')
    Bxz_dos = np.loadtxt('Data_CSRO20_Bxz_kpts_5000.dat')
    Byz_dos = np.loadtxt('Data_CSRO20_Byz_kpts_5000.dat')
    Bxy_dos = np.loadtxt('Data_CSRO20_Bxy_kpts_5000.dat')
    os.chdir(home_dir)

    # collect data
    bands = (Ayz_dos, Axz_dos, Axy_dos, Byz_dos, Bxz_dos, Bxy_dos)

    # Placeholders
    En = ()
    _EF = ()
    DOS = ()
    N_bands = ()
    N_full = ()

    # create figure
    fig = plt.figure('DOS', figsize=(8, 8), clear=True)
    n = 0  # counter

    # calculate DOS
    for band in bands:
        n += 1
        ax = fig.add_subplot(2, 3, n)
        ax.tick_params(**kwargs_ticks)

        # histogram data, and normalize (density=true)
        dos, bins, patches = ax.hist(np.ravel(band), bins=150,
                                     density=True, alpha=.2, color='C8')
        en = np.zeros((len(dos)))  # placeholder energy
        for i in range(len(dos)):
            en[i] = (bins[i] + bins[i + 1]) / 2
        ef, _ef = utils.find(en, -0.002)  # fermi energy

        # integrate filling
        n_full = np.trapz(dos, x=en)  # consistency check
        n_band = np.trapz(dos[:_ef], x=en[:_ef])  # integrate up to EF

        # plot data
        ax.plot(en, dos, color='k', lw=.5)
        ax.fill_between(en[:_ef], dos[:_ef], 0, color='C1', alpha=.5)

        # decorate axes
        if n < 4:
            ax.set_position([.1+(n-1)*.29, .5, .28, .23])
            ax.set_xticks(np.arange(-.6, .3, .1))
            ax.set_xticklabels([])
        else:
            ax.set_position([.1+(n-4)*.29, .26, .28, .23])
            ax.set_xticks(np.arange(-.6, .3, .1))
        if n == 5:
            ax.set_xlabel(r'$\omega$ (eV)', fontdict=font)
        if any(x == n for x in [1, 4]):
            ax.set_ylabel('Intensity (a.u)', fontdict=font)
        ax.set_yticks(np.arange(0, 40, 10))
        ax.set_yticklabels([])
        ax.set_xlim(.21, -.37)

        # collect data
        N_full = N_full + (n_full,)
        N_bands = N_bands + (n_band,)
        En = En + (en,)
        _EF = _EF + (_ef,)
        DOS = DOS + (dos,)
    N = np.sum(N_bands)

    # prepare plot band structure
    k_pts = 200
    x_GS = np.linspace(0, 1, k_pts)
    y_GS = np.linspace(0, 1, k_pts)
    x_SX = np.linspace(1, 0, k_pts)
    y_SX = np.ones(k_pts)
    x_XG = np.zeros(k_pts)
    y_XG = np.linspace(1, 0, k_pts)

    # full k-path
    x = (x_GS, x_SX, x_XG)
    y = (y_GS, y_SX, y_XG)

    # colors
    cols = ['C1', 'C0', 'm', 'C8', 'C9', 'C3']
    fig = plt.figure('TB_eval', figsize=(8, 8), clear=True)
    for i in range(len(x)):
        en, spec, bndstr = utils.CSRO_eval_proj(x[i], y[i])
        k = np.sqrt(x[i] ** 2 + y[i] ** 2)
        if i != 0:
            ax = fig.add_subplot(3, 3, i+1)
            ax.set_position([.1 + i*.15+.15*(np.sqrt(2)-1), .2, .15, .4])
            k = -k
        else:
            ax = plt.subplot(3, 3, i+1)
            ax.set_position([.1+i*.15, .2, .15*np.sqrt(2), .4])
        ax.tick_params(**kwargs_ticks)

        # plot data
        for j in range(len(bndstr)):
            ax.plot(k, bndstr[j], color=cols[j])
        ax.plot([k[0], k[-1]], [0, 0], **kwargs_ef)
        if i == 0:
            ax.set_xticks([0, np.sqrt(2)])
            ax.set_xticklabels([r'$\Gamma$', 'S'])
            ax.set_yticks(np.arange(-1, 1, .2))
            ax.set_ylabel(r'$\omega$ (eV)', fontdict=font)

            # add text
            ax.text(.05, .25, '(a)', fontdict=font)
        elif i == 1:
            ax.set_xticks([-np.sqrt(2), -1])
            ax.set_xticklabels(['', 'X'])
            ax.set_yticks(np.arange(-1, 1, .2))
            ax.set_yticklabels([])
        elif i == 2:
            ax.set_xticks([-1, 0])
            ax.set_xticklabels(['', r'$\Gamma$'])
            ax.set_yticks(np.arange(-1, 1, .2))
            ax.set_yticklabels([])
        ax.set_xlim(xmin=k[0], xmax=k[-1])
        ax.set_ylim(ymax=np.max(en), ymin=np.min(en))

    # plot DOS
    n = 0   # counter
    for band in bands:
        dos = DOS[n]
        dos[0] = 0
        dos[-1] = 0
        ax = fig.add_subplot(339)
        ax.set_position([.1+3*.15+.01+.15*(np.sqrt(2)-1), .2,
                         .15 * np.sqrt(2), .4])
        ax.tick_params(**kwargs_ticks)
        ef, _ef = utils.find(En[n], 0.00)  # find Fermi level

        # plot DOS
        ax.plot(DOS[n], En[n], color='k', lw=.5)
        ax.fill_betweenx(En[n][:_ef], 0, DOS[n][:_ef],
                         color=cols[n], alpha=.7)

        # decorate axes
        ax.set_yticks(np.arange(-1, 1, .2))
        ax.set_yticklabels([])
        ax.set_xticks([])
        ax.set_ylim(np.min(en), np.max(en))
        ax.set_xlim(0, 35)
        n += 1

    # add text
    ax.text(1, .25, '(b)', fontdict=font)
    ax.set_xlabel(r'DOS $\partial_\omega\Omega_2(\omega)$', fontdict=font)
    print(N)  # consistency check

    plt.show()

    # Save figure
    if print_fig:
        plt.savefig(save_dir + figname + '.pdf', dpi=100,
                    bbox_inches="tight", rasterized=True)

    # save data
    os.chdir(data_dir)
    np.savetxt('Data_TB_DOS_Ayz.dat', DOS[0])
    np.savetxt('Data_TB_DOS_Axz.dat', DOS[1])
    np.savetxt('Data_TB_DOS_Axy.dat', DOS[2])
    np.savetxt('Data_TB_DOS_Byz.dat', DOS[3])
    np.savetxt('Data_TB_DOS_Bxz.dat', DOS[4])
    np.savetxt('Data_TB_DOS_Bxy.dat', DOS[5])
    np.savetxt('Data_TB_DOS_en_Ayz.dat', En[0])
    np.savetxt('Data_TB_DOS_en_Axz.dat', En[1])
    np.savetxt('Data_TB_DOS_en_Axy.dat', En[2])
    np.savetxt('Data_TB_DOS_en_Byz.dat', En[3])
    np.savetxt('Data_TB_DOS_en_Bxz.dat', En[4])
    np.savetxt('Data_TB_DOS_en_Bxy.dat', En[5])
    print('\n ~ Data saved Density of states',
          '\n', '==========================================')
    os.chdir(home_dir)


def fig15(print_fig=True):
    """figure 15

    %%%%%%%%%%%%%%%%%%
    DMFT Fermi surface
    %%%%%%%%%%%%%%%%%%
    """

    figname = 'CSROfig15'

    # load data
    os.chdir(data_dir)
    FS_DMFT_xy_data = np.loadtxt('FS_DMFT_xy.dat')
    FS_DMFT_xz_data = np.loadtxt('FS_DMFT_xz.dat')
    FS_DMFT_yz_data = np.loadtxt('FS_DMFT_yz.dat')
    FS_LDA_xy_data = np.loadtxt('FS_LDA_xy.dat')
    FS_LDA_xz_data = np.loadtxt('FS_LDA_xz.dat')
    FS_LDA_yz_data = np.loadtxt('FS_LDA_yz.dat')
    os.chdir(home_dir)

    m = 51

    # Reshape
    FS_DMFT_xy = np.reshape(FS_DMFT_xy_data, (m, m, 3))
    FS_DMFT_xz = np.reshape(FS_DMFT_xz_data, (m, m, 3))
    FS_DMFT_yz = np.reshape(FS_DMFT_yz_data, (m, m, 3))
    FS_LDA_xy = np.reshape(FS_LDA_xy_data, (m, m, 3))
    FS_LDA_xz = np.reshape(FS_LDA_xz_data, (m, m, 3))
    FS_LDA_yz = np.reshape(FS_LDA_yz_data, (m, m, 3))

    # Normalize
    FS_DMFT_xy = FS_DMFT_xy / np.max(FS_DMFT_xy[:, :, 2])
    FS_DMFT_xz = FS_DMFT_xz / np.max(FS_DMFT_xz[:, :, 2])
    FS_DMFT_yz = FS_DMFT_yz / np.max(FS_DMFT_yz[:, :, 2])
    FS_LDA_xy = FS_LDA_xy / np.max(FS_LDA_xy[:, :, 2])
    FS_LDA_xz = FS_LDA_xz / np.max(FS_LDA_xz[:, :, 2])
    FS_LDA_yz = FS_LDA_yz / np.max(FS_LDA_yz[:, :, 2])

    # Weight distribution
    FS_DMFT = (FS_DMFT_xz[:, :, 2] +
               FS_DMFT_yz[:, :, 2]) / 2 - FS_DMFT_xy[:, :, 2]
    FS_LDA = (FS_LDA_xz[:, :, 2] +
              FS_LDA_yz[:, :, 2]) / 2 - FS_LDA_xy[:, :, 2]

    # Flip data
    d1 = FS_DMFT
    d2 = np.fliplr(d1)
    d3 = np.flipud(d2)
    d4 = np.flipud(d1)
    l1 = FS_LDA
    l2 = np.fliplr(l1)
    l3 = np.flipud(l2)
    l4 = np.flipud(l1)

    # Extended Fermi Surface
    # DMFT = np.zeros((4 * m, 4 * m))
    # LDA = np.zeros((4 * m, 4 * m))
    # kx = np.linspace(-2, 2, 4 * m)
    # ky = np.linspace(-2, 2, 4 * m)
    # #Build data###
    # for i in range(m):
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

    # Placeholders
    DMFT = np.zeros((2 * m, 2 * m))
    LDA = np.zeros((2 * m, 2 * m))
    kx = np.linspace(-1, 1, 2 * m)
    ky = np.linspace(-1, 1, 2 * m)

    # Build data
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

    # Plot data
    v_d = np.max(DMFT)
    fig = plt.figure(figname, figsize=(8, 8), clear=True)
    ax = fig.add_subplot(121)
    ax.set_position([.1, .3, .4, .4])
    ax.tick_params(**kwargs_ticks)
    c0 = ax.contourf(kx, ky, DMFT, 300, cmap='PuOr', vmin=-v_d, vmax=v_d,
                     zorder=.1)
    ax.set_rasterization_zorder(.2)

    # decorate axes
    ax.set_xticks(np.arange(-2, 2, 1))
#    ax.set_xticklabels([])
    ax.set_yticks(np.arange(-2, 2, 1))
#    ax.set_yticklabels([])
    ax.set_ylim(-1, 1)
    ax.set_xlim(-1, 1)
    ax.set_xlabel(r'$k_x \, (\pi / a)$', fontdict=font)
    ax.set_ylabel(r'$k_y \, (\pi / b)$', fontdict=font)

    # add text
    ax.text(-.05, -.05, r'$\Gamma$', fontsize=18, color='r')
    ax.text(.95, .95, 'S', fontsize=18, color='r')
    ax.text(-.06, .95, 'X', fontsize=18, color='r')

    # colorbars
    pos = ax.get_position()
    cax = plt.axes([pos.x0+pos.width + 0.01, pos.y0, 0.01, pos.height])
    cbar = plt.colorbar(c0, cax=cax, ticks=None)
    cbar.set_ticks([])
    cbar.set_clim(np.min(DMFT), np.max(DMFT))

    plt.show()

    # Save figure
    if print_fig:
        plt.savefig(save_dir + figname + '.pdf', dpi=100,
                    bbox_inches="tight", rasterized=True)


def fig16(print_fig=True):
    """figure 16

    %%%%%%%%%%%%%%%%%%
    DMFT bandstructure calculation
    %%%%%%%%%%%%%%%%%%
    """

    figname = 'CSROfig16'

    # load data
    os.chdir(data_dir)
    xz_data = np.loadtxt('DMFT_CSRO_xz.dat')
    yz_data = np.loadtxt('DMFT_CSRO_yz.dat')
    xy_data = np.loadtxt('DMFT_CSRO_xy.dat')
#    xz_data = np.loadtxt('DMFT_CSRO_MAX_xz.dat')  # max entropy
#    yz_data = np.loadtxt('DMFT_CSRO_MAX_yz.dat')  # max entropy
#    xy_data = np.loadtxt('DMFT_CSRO_MAX_xy.dat')  # max entropy
    os.chdir(home_dir)

    # full k-path
    # [0, 56, 110, 187, 241, 266, 325, 350]  = [G, X, S, G, Y, T, G, Z]

    # prepare data
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
    v_bnd = .6 * np.max(DMFT)
    k = (np.flipud(k_SG), np.flipud(k_XS), np.flipud(k_GX))

    # collect data
    bndstr = (SG, XS, GX)

    # create figure
    fig = plt.figure(figname, figsize=(8, 8), clear=True)
    for i in range(len(bndstr)):
        if i != 0:
            ax = fig.add_subplot(3, 3, i+1)
            ax.set_position([.1+i*.15+.15*(np.sqrt(2)-1), .2, .15, .4])
            ax.tick_params(**kwargs_ticks)
        else:
            ax = plt.subplot(3, 3, i+1)
            ax.set_position([.1+i*.15, .2, .15*np.sqrt(2), .4])
            ax.tick_params(**kwargs_ticks)

        # plot data
        c0 = ax.contourf(k[i], en, bndstr[i], 200, cmap='PuOr',
                         vmax=v_bnd, vmin=-v_bnd, zorder=.1)
        ax.set_rasterization_zorder(.2)
        ax.plot([k[i][0], k[i][-1]], [0, 0], 'k:')

        # decorate axes
        if i == 0:
            ax.set_xticks([110, 186])
            ax.set_xticklabels([r'$\Gamma$', 'S'])
            ax.set_ylabel(r'$\omega$ (eV)', fontdict=font)
        elif i == 1:
            ax.set_xticks([56, 109])
            ax.set_xticklabels(['', 'X'])
            ax.set_yticklabels([])
        elif i == 2:
            ax.set_xticks([0, 55])
            ax.set_xticklabels(['', r'$\Gamma$'])
            ax.set_yticklabels([])
        ax.set_yticks(np.arange(-5, 5, .1))
        ax.set_ylim(-.85, .3)

    # colorbars
    pos = ax.get_position()
    cax = plt.axes([pos.x0+pos.width+0.01, pos.y0, 0.01, pos.height])
    cbar = plt.colorbar(c0, cax=cax, ticks=None)
    cbar.set_ticks([])
    cbar.set_clim(-v_bnd, v_bnd)

    plt.show()

    # Save figure
    if print_fig:
        plt.savefig(save_dir + figname + '.pdf', dpi=100,
                    bbox_inches="tight", rasterized=True)


def fig17(print_fig=True):
    """figure 17

    %%%%%%%%%%%%%%%%%%
    LDA bandstructure calculation
    %%%%%%%%%%%%%%%%%%
    """

    figname = 'CSROfig17'

    # load data
    os.chdir(data_dir)
    xz_data = np.loadtxt('LDA_CSRO_xz.dat')
    yz_data = np.loadtxt('LDA_CSRO_yz.dat')
    xy_data = np.loadtxt('LDA_CSRO_xy.dat')
    os.chdir(home_dir)

    # k-path
    # [0, 56, 110, 187, 241, 266, 325, 350]  = [G, X, S, G, Y, T, G, Z]

    # prepare data
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
    v_bnd = .7 * np.max(LDA)
    k = (np.flipud(k_SG), np.flipud(k_XS), np.flipud(k_GX))

    # collect data
    bndstr = (SG, XS, GX)

    # create figure
    fig = plt.figure(figname, figsize=(8, 8), clear=True)
    for i in range(len(bndstr)):
        if i != 0:
            ax = fig.add_subplot(3, 3, i+1)
            ax.set_position([.1+i*.15+.15*(np.sqrt(2)-1), .2, .15, .4])
            ax.tick_params(**kwargs_ticks)
        else:
            ax = fig.add_subplot(3, 3, i+1)
            ax.set_position([.1+i*.15, .2, .15*np.sqrt(2), .4])
            ax.tick_params(**kwargs_ticks)

        # plot data
        c0 = ax.contourf(k[i], en, bndstr[i], 200, cmap='PuOr',
                         vmax=v_bnd, vmin=-v_bnd, zorder=.1)
        ax.set_rasterization_zorder(.2)
        ax.plot([k[i][0], k[i][-1]], [0, 0], 'k:')

        if i == 0:
            ax.set_xticks([110, 186])
            ax.set_xticklabels([r'$\Gamma$', 'S'])
            ax.set_yticks(np.arange(-5, 2, .5))
            ax.set_ylabel(r'$\omega$ (eV)', fontdict=font)
        elif i == 1:
            ax.set_xticks([56, 109])
            ax.set_xticklabels(['', 'X'])
            ax.set_yticks(np.arange(-5, 2, .5))
            ax.set_yticklabels([])
        elif i == 2:
            ax.set_xticks([0, 55])
            ax.set_xticklabels(['', r'$\Gamma$'])
            ax.set_yticks(np.arange(-5, 2, .5))
            ax.set_yticklabels([])
        ax.set_ylim(-2.5, 1)

    # colorbars
    pos = ax.get_position()
    cax = plt.axes([pos.x0+pos.width+0.01, pos.y0, 0.01, pos.height])
    cbar = plt.colorbar(c0, cax=cax, ticks=None)
    cbar.set_ticks([])
    cbar.set_clim(-v_bnd, v_bnd)

    plt.show()

    # Save figure
    if print_fig:
        plt.savefig(save_dir + figname + '.pdf', dpi=100,
                    bbox_inches="tight", rasterized=True)


def fig18(print_fig=True):
    """figure 18

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    CSRO30 Experimental band structure
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    """

    figname = 'CSROfig18'

    file = '62421'
    gold = '62440'
    mat = 'CSRO30'
    year = '2017'
    sample = 'S13'

    # load FS
    D = ARPES.DLS(file, mat, year, sample)
    D.norm(gold=gold)
    D.restrict(bot=0, top=1, left=.12, right=.9)
    D.ang2kFS(D.ang, Ekin=22-4.5, lat_unit=True, a=5.33, b=5.55, c=11,
              V0=0, thdg=-9, tidg=9, phidg=-90)
    D.FS(e=0.005, ew=.02)
    D.FS_flatten()

    # load G-S cut
    file = '62444'
    gold = '62455'
    A1 = ARPES.DLS(file, mat, year, sample)
    A1.norm(gold)
    A1.ang2k(A1.ang, Ekin=22-4.5, lat_unit=True, a=5.33, b=5.55, c=11,
             V0=0, thdg=-8.3, tidg=0, phidg=-90)

    # load X-S cut
    file = '62449'
    gold = '62455'
    A2 = ARPES.DLS(file, mat, year, sample)
    A2.norm(gold=gold)
    A2.ang2k(A1.ang, Ekin=22-4.5, lat_unit=True, a=5.33, b=5.55, c=11,
             V0=0, thdg=-9, tidg=16, phidg=-90)

    def fig18a():
        ax = fig.add_subplot(4, 4, 1)
        ax.set_position([.08, .3, .2, .35])
        ax.tick_params(**kwargs_ticks)
        ax.contourf(A1.en_norm, A1.kys, A1.int_norm, 300, **kwargs_ex,
                    vmin=0, vmax=.8 * np.max(A1.int_norm), zorder=.1)
        ax.set_rasterization_zorder(.2)
        ax.plot([0, 0], [np.min(A1.kys), np.max(A1.kys)], **kwargs_ef)
        ax.plot([-.005, -.005], [np.min(A1.kys), np.max(A1.kys)],
                **kwargs_cut)

        # decorate axes
        ax.set_xticks(np.arange(-.08, .0, .02))
        ax.set_xticklabels(['-80', '-60', '-40', '-20', '0'])
        ax.set_yticks(np.arange(-5, 5, .5))
        ax.set_xlim(-.08, .005)
        ax.set_ylim(np.min(D.ky), np.max(D.ky))
        ax.set_xlabel(r'$\omega\,(\mathrm{meV})$', fontdict=font)
        ax.set_ylabel(r'$k_x \,(\pi/a)$', fontdict=font)

        # add text
        ax.text(-.075, .48, r'(a)', fontsize=12, color='c')
        ax.text(-.002, -.03, r'$\Gamma$', fontsize=12, color='r')
        ax.text(-.002, -1.03, r'Y', fontsize=12, color='r')

    def fig18c():
        ax = fig.add_subplot(4, 4, 15)
        panel_c_scale = ((np.max(D.kx) - np.min(D.kx)) /
                         (np.max(D.ky) - np.min(D.ky)))
        ax.set_position([.3+.35*panel_c_scale, .3, .2, .35])
        ax.tick_params(**kwargs_ticks)

        # plot data
        c0 = ax.contourf(-np.transpose(np.fliplr(A2.en_norm))-.002,
                         np.transpose(A2.kys),
                         np.transpose(np.fliplr(A2.int_norm)), 300,
                         **kwargs_ex, zorder=.1,
                         vmin=0, vmax=.8 * np.max(A2.int_norm))
        ax.set_rasterization_zorder(.2)
        ax.plot([0, 0], [np.min(A2.kys), np.max(A2.kys)], **kwargs_ef)

        # decorate axes
        ax.set_xticks(np.arange(0, .1, .02))
        ax.set_xticklabels(['0', '-20', '-40', '-60', '-80'])
        ax.set_yticks(np.arange(-5, 5, .5))
        ax.set_yticklabels([])
        ax.set_xlim(-.005, .08)
        ax.set_ylim(np.min(D.ky), np.max(D.ky))
        ax.set_xlabel(r'$\omega\,(\mathrm{meV})$', fontdict=font)

        # add text
        ax.text(0.0, .48, '(c)', fontdict=font)
        ax.text(-.002, -.03, 'X', fontsize=12, color='r')
        ax.text(-.002, -1.03, 'S', fontsize=12, color='r')

        # colorbar
        pos = ax.get_position()
        cax = plt.axes([pos.x0+pos.width+0.01, pos.y0, 0.01, pos.height])
        cbar = plt.colorbar(c0, cax=cax, ticks=None)
        cbar.set_ticks([])
        cbar.set_clim(np.min(A2.int_norm), np.max(A2.int_norm))

    def fig18b():
        ax = fig.add_subplot(4, 4, 2)
        panel_c_scale = ((np.max(D.kx) - np.min(D.kx)) /
                         (np.max(D.ky) - np.min(D.ky)))
        ax.set_position([.29, .3, .35 * panel_c_scale, .35])
        ax.tick_params(**kwargs_ticks)

        # plot data
        ax.contourf(D.kx, D.ky, np.flipud(D.map), 300, **kwargs_ex,
                    vmax=.9 * np.max(D.map),
                    vmin=.1 * np.max(D.map), zorder=.1)
        ax.set_rasterization_zorder(.2)
        ax.plot(A1.k[0], A1.k[1], **kwargs_cut)
        ax.plot(A2.k[0], A2.k[1], **kwargs_cut)

        # decorate axes
        ax.set_xlabel(r'$k_y \,(\pi/a)$', fontdict=font)
        ax.set_xticks(np.arange(-5, 5, .5))
        ax.set_yticks(np.arange(-5, 5, .5))
        ax.set_yticklabels([])
        ax.set_xlim(np.min(D.kx), np.max(D.kx))
        ax.set_ylim(np.min(D.ky), np.max(D.ky))

        # add text
        ax.text(-1.8, .48, '(b)', fontdict=font)
        ax.text(-.05, -.03, r'$\Gamma$', fontsize=12, color='r')
        ax.text(-.05, -1.03, 'Y', fontsize=12, color='r')
        ax.text(.95, -.03, 'X', fontsize=12, color='r')
        ax.text(.95, -1.03, 'S', fontsize=12, color='r')

    # create figure
    fig = plt.figure(figname, figsize=(8, 8), clear=True)
    fig18a()
    fig18b()
    fig18c()

    plt.show()

    # Save figure
    if print_fig:
        plt.savefig(save_dir + figname + '.pdf', dpi=100,
                    bbox_inches="tight", rasterized=True)


def fig19(print_fig=True):
    """figure 19

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    CSRO30 Gamma - S cut epsilon pocket
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    """

    figname = 'CSROfig19'

    mat = 'CSRO30'
    year = '2017'
    sample = 'S13'
    file = '62488'
    gold = '62492'

    # load data
    D = ARPES.DLS(file, mat, year, sample)
    D.norm(gold=gold)
    D.bkg()
    D.ang2k(D.ang, Ekin=72-4.5, lat_unit=True, a=5.33, b=5.55, c=11,
            V0=0, thdg=-10, tidg=0, phidg=45)

    # MDC
    mdc_val = .00
    mdcw_val = .005
    mdc = np.zeros(D.ang.shape)
    for i in range(len(D.ang)):
        val, mdc_idx = utils.find(D.en_norm[i, :], mdc_val)
        val, mdcw_idx = utils.find(D.en_norm[i, :], mdc_val - mdcw_val)
        mdc[i] = np.sum(D.int_norm[i, mdcw_idx:mdc_idx])
    mdc = mdc / np.max(mdc)  # normalize

    plt.figure('MDC', figsize=(7, 5), clear=True)

    # start fitting MDC
    d = 1e5

    # initial parameters
    p_mdc_i = np.array(
                [-.4, .4, .65, .9, 1.25, 1.4, 1.7,
                 .05, .05, .05, .1, .1, .05, .05,
                 .4, .8, .3, .4, .5, .5, 1,
                 .06, -0.02, .02])

    # boundaries
    bounds_bot = np.concatenate((p_mdc_i[0:-3] - np.inf, p_mdc_i[-3:] - d))
    bounds_top = np.concatenate((p_mdc_i[0:-3] + np.inf, p_mdc_i[-3:] + d))
    p_mdc_bounds = (bounds_bot, bounds_top)

    # MDC fit
    p_mdc, cov_mdc = curve_fit(
            utils.lor_7, D.k[1], mdc, p_mdc_i, bounds=p_mdc_bounds)
    b_mdc = utils.poly_2(D.k[1], *p_mdc[-3:])  # background
    f_mdc = utils.lor_7(D.k[1], *p_mdc) - b_mdc  # fit
    f_mdc[0] = -.05
    f_mdc[-1] = -.05
    plt.plot(D.k[1], mdc, 'bo')
    plt.plot(D.k[1], f_mdc)
    plt.plot(D.k[1], b_mdc, 'k--')

    # create figure
    fig = plt.figure(figname, figsize=(6, 6), clear=True)
    ax = fig.add_subplot(121)
    ax.set_position([.2, .3, .5, .5])
    ax.tick_params(**kwargs_ticks)

    # plot data
    c0 = ax.contourf(D.kxs, D.en_norm+.002, D.int_norm, 300, **kwargs_ex,
                     vmin=0, vmax=.7 * np.max(D.int_norm), zorder=.1)
    ax.set_rasterization_zorder(.2)
    ax.plot([np.min(D.kxs), np.max(D.kxs)], [0, 0], **kwargs_ef)

    # plot MDC
    ax.plot(D.k[1]*.95, (mdc - b_mdc) / 25 + .005, 'o',
            markersize=1.5, color='C9')
    ax.fill(D.k[1]*.95, (f_mdc) / 25 + .005, alpha=.2, color='C9')
    ax.plot(D.k[1]*.95, (f_mdc) / 25 + .005, color='b', linewidth=.5)

    # label colors and positions
    cols = ['m', 'm', 'k', 'r', 'r', 'k', 'm', 'C1']
    alphas = [0.5, 0.5, .5, 1, 1, .5, 0.5, 0.5, 0.5]
    lws = [0.5, 0.5, .5, 1, 1, .5, 0.5, 0.5, 0.5]
    p_mdc[6 + 16] *= 1.5

    # plot Lorentzians
    for i in range(7):
        ax.plot(D.k[1]*.95, (utils.lor(D.k[1], p_mdc[i], p_mdc[i + 7],
                p_mdc[i + 14],
                p_mdc[-3]*0, p_mdc[-2]*0, p_mdc[-1]*0))/25+.002,
                lw=lws[i], color=cols[i], alpha=alphas[i])

    # decorate axes
    ax.set_yticks(np.arange(-.15, .1, .05))
    ax.set_yticklabels(['-150', '-100', '-50', '0', '50'])
    ax.set_xticks(np.arange(0, 3, 1))
    ax.set_xticklabels([r'$\Gamma$', 'S', r'$\Gamma$'])
    ax.set_ylim(-.15, .05)
    ax.set_xlim(np.min(D.kxs), np.max(D.kxs))
    ax.set_ylabel(r'$\omega\,(\mathrm{meV})$', fontdict=font)
    ax.set_xlabel(r'$k_x \,(\pi/a)$', fontdict=font)

    # add text
    ax.text(.87*.95, .029, r'$\gamma$', fontsize=12, color='r')
    ax.text(1.2*.95, .034, r'$\gamma$', fontsize=12, color='r')

    # colorbars
    pos = ax.get_position()
    cax = plt.axes([pos.x0+pos.width+0.01, pos.y0, 0.01, pos.height])
    cbar = plt.colorbar(c0, cax=cax, ticks=None)
    cbar.set_ticks([])
    cbar.set_clim(np.min(D.int_norm), np.max(D.int_norm))

    plt.show()

    # Save figure
    if print_fig:
        plt.savefig(save_dir + figname + '.pdf', dpi=100,
                    bbox_inches="tight", rasterized=True)


def fig20(print_fig=True, load=True):
    """figure 20

    %%%%%%%%%%%%%%%
    Fit dispersions
    %%%%%%%%%%%%%%%
    """

    figname = 'CSROfig20'

    # load data
    os.chdir(data_dir)
    GX_alpha_1 = np.loadtxt('dispersion_GX_alpha_1.dat')
    GX_alpha_2 = np.loadtxt('dispersion_GX_alpha_2.dat')
    GX_beta_1 = np.loadtxt('dispersion_GX_beta_1.dat')
    GX_beta_2 = np.loadtxt('dispersion_GX_beta_2.dat')
    GX_gamma_1 = np.loadtxt('dispersion_GX_gamma_1.dat')
    GX_gamma_2 = np.loadtxt('dispersion_GX_gamma_2.dat')
    GX_delta = np.loadtxt('dispersion_GX_delta.dat')

    GS_alpha_1 = np.loadtxt('dispersion_GS_alpha_1.dat')
    GS_alpha_2 = np.loadtxt('dispersion_GS_alpha_2.dat')
    GS_beta_1 = np.loadtxt('dispersion_GS_beta_1.dat')
    GS_beta_2 = np.loadtxt('dispersion_GS_beta_2.dat')
    GS_gamma_1 = np.loadtxt('dispersion_GS_gamma_1.dat')
    GS_gamma_2 = np.loadtxt('dispersion_GS_gamma_2.dat')
    GS_delta = np.loadtxt('dispersion_GS_delta.dat')

    XS_branch_1 = np.loadtxt('dispersion_XS_branch_1.dat')
    XS_branch_2 = np.loadtxt('dispersion_XS_branch_2.dat')

    int_alpha_1 = np.loadtxt('dispersion_int_alpha_1.dat')
    int_alpha_2 = np.loadtxt('dispersion_int_alpha_2.dat')
    int_beta_1 = np.loadtxt('dispersion_int_beta_1.dat')
    int_beta_2 = np.loadtxt('dispersion_int_beta_2.dat')
    int_gamma_1 = np.loadtxt('dispersion_int_gamma_1.dat')
    int_gamma_2 = np.loadtxt('dispersion_int_gamma_2.dat')

    alpha_1 = np.loadtxt('coords_CSRO20_alpha_1.dat')
    alpha_2 = np.loadtxt('coords_CSRO20_alpha_2.dat')
    beta_1 = np.loadtxt('coords_CSRO20_beta_1.dat')
    beta_2 = np.loadtxt('coords_CSRO20_beta_2.dat')
    gamma_1 = np.loadtxt('coords_CSRO20_gamma_alt_1.dat')
    gamma_2 = np.loadtxt('coords_CSRO20_gamma_alt_2.dat')
    delta = np.loadtxt('coords_CSRO20_delta.dat')
    os.chdir(home_dir)

    coords = (GX_alpha_1, GX_alpha_2, GX_beta_1, GX_beta_2,
              GX_gamma_1, GX_gamma_2, GX_delta,  # 0 - 6
              GS_alpha_1, GS_alpha_2, GS_beta_1, GS_beta_2,
              GS_gamma_1, GS_gamma_2, GS_delta,  # 7 - 13
              XS_branch_1, XS_branch_2,  # 14, 15
              int_alpha_1, int_alpha_2, int_beta_1, int_beta_2,
              int_gamma_1, int_gamma_2,  # 16 - 21
              alpha_1, alpha_2, beta_1, beta_2, gamma_1, gamma_2,
              delta)  # 22 - 28

    # placeholders
    Kx = ()
    Ky = ()
    En = ()

    # transform into k-space
    n = 0  # counter (datasets)
    m = 1  # counter (datapoints)
    for coord in coords:
        x = np.zeros(len(coord))
        y = np.zeros(len(coord))
        en = np.zeros(len(coord))

        if n < 22:
            for i in range(len(coord)):
                x[i] = coord[i][0]
                y[i] = 0
                en[i] = coord[i][1]
                m += 1  # regularization
        elif n >= 22:
            for i in range(len(coord)):
                x[i] = coord[i][0]
                y[i] = coord[i][1]
                en[i] = 0
                m += 1

        kx = np.ones(x.size)
        ky = np.ones(y.size)

        for i in range(y.size):
            if any(x == n for x in np.arange(0, 7, 1)):
                k, k_V0 = utils.ang2k(x[i], Ekin=22-4.5, lat_unit=True,
                                      a=5.2, b=5.7, c=11, V0=0, thdg=9.3,
                                      tidg=y[i], phidg=90)
            elif any(x == n for x in np.arange(7, 14, 1)):
                k, k_V0 = utils.ang2k(x[i], Ekin=48, lat_unit=True,
                                      a=5.5, b=5.5, c=11, V0=0, thdg=2.5,
                                      tidg=0, phidg=45)
            elif any(x == n for x in np.arange(14, 16, 1)):
                k, k_V0 = utils.ang2k(x[i], Ekin=22-4.5, lat_unit=True,
                                      a=5.2, b=5.7, c=11, V0=0, thdg=6.3,
                                      tidg=y[i]-15.7, phidg=90)
            elif any(x == n for x in np.arange(16, 22, 1)):
                k, k_V0 = utils.ang2k(x[i], Ekin=22-4.5, lat_unit=True,
                                      a=5.2, b=5.7, c=11, V0=0, thdg=9.2,
                                      tidg=-4.2, phidg=90)
            elif any(x == n for x in np.arange(22, 29, 1)):
                k, k_V0 = utils.ang2k(x[i], Ekin=22-4.5, lat_unit=True,
                                      a=5.2, b=5.7, c=11, V0=0, thdg=8.7,
                                      tidg=y[i]-4, phidg=88)

            kx[i] = k[0]
            ky[i] = k[1]

        Kx = Kx + (kx,)
        Ky = Ky + (ky,)
        En = En + (en,)
        n += 1

    # maximum iterations
    it_max = 1000000

    # initial parameters
    p = utils.paramSRO()

    t1 = p['t1']
    t2 = p['t2']
    t3 = p['t3']
    t4 = p['t4']
    t5 = p['t5']
    t6 = p['t6']
    mu = p['mu']
    so = p['so']

    P = np.array([t1, t2, t3, t4, t5, t6, mu, so])

    # load data
    if load:
        os.chdir(data_dir)
        it = np.loadtxt('Data_CSROfig20_it.dat')
        J = np.loadtxt('Data_CSROfig20_J.dat')
        P = np.loadtxt('Data_CSROfig20_P.dat')
        os.chdir(home_dir)
    else:
        # optimize parameters
        it, J, P = utils.optimize_TB(Kx, Ky, En, it_max, P)

    J_n = J / m * 1e3

    # save data
    os.chdir(data_dir)
    np.savetxt('Data_CSROfig20_it.dat', np.ravel(it))
    np.savetxt('Data_CSROfig20_J.dat', np.ravel(J_n))
    np.savetxt('Data_CSROfig20_P.dat', np.ravel(P))
    print('\n ~ Data saved (iterations, cost)',
          '\n', '==========================================')
    os.chdir(home_dir)

    # create figure
    fig = plt.figure(figname, clear=True, figsize=(7, 7))
    ax = fig.add_axes([.25, .25, .5, .5])
    print(J_n[-1])

    # plot cost
    ax.plot(it, J_n, 'C8o', ms=2)
    ax.plot([0, np.max(it)], [np.min(J_n), np.min(J_n)], **kwargs_ef)
    ax.tick_params(**kwargs_ticks)

    # decorate axes
    ax.set_xlim([np.min(it), np.max(it)])
    ax.set_xlabel('iterations (t)', fontdict=font)
    ax.set_ylabel(r'fitness $\xi (t)$ (meV)', fontdict=font)
    ax.arrow(2080, 9.2, 0, -1.2, head_width=160,
             head_length=.3, fc='k', ec='k')
    ax.arrow(5950, 7.5, 0, -1.2, head_width=160,
             head_length=.3, fc='k', ec='k')

    plt.show()

    # Save figure
    if print_fig:
        plt.savefig(save_dir + figname + '.pdf',
                    dpi=100, bbox_inches="tight", rasterized=True)

    return it_max, J, P


def fig21(print_fig=True):
    """figure 21

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Fermi surface extraction points
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    """

    figname = 'CSROfig21'

    # load data
    os.chdir(data_dir)
    alpha_1 = np.loadtxt('coords_CSRO20_alpha_1.dat')
    alpha_2 = np.loadtxt('coords_CSRO20_alpha_2.dat')
    beta_1 = np.loadtxt('coords_CSRO20_beta_1.dat')
    beta_2 = np.loadtxt('coords_CSRO20_beta_2.dat')
    beta_3 = np.loadtxt('coords_CSRO20_beta_3.dat')
    gamma_1 = np.loadtxt('coords_CSRO20_gamma_1.dat')
    gamma_2 = np.loadtxt('coords_CSRO20_gamma_2.dat')
    gamma_3 = np.loadtxt('coords_CSRO20_gamma_3.dat')
    delta = np.loadtxt('coords_CSRO20_delta.dat')
    os.chdir(home_dir)

    coords = (alpha_1, alpha_2, beta_1, beta_2, beta_3, gamma_1, gamma_2,
              gamma_3, delta)

    mat = 'CSRO20'
    year = '2017'
    sample = 'S6'

    # load data for FS map
    file = '62087'
    gold = '62081'
    D = ARPES.DLS(file, mat, year, sample)
    D.norm(gold=gold)
    D.restrict(bot=0, top=1, left=.12, right=.9)
    D.FS(e=0.02, ew=.03)
    D.FS_flatten()
    D.ang2kFS(D.ang, Ekin=22-4.5, lat_unit=True, a=5.33, b=5.55, c=11,
              V0=0, thdg=8.7, tidg=4, phidg=88)

    # create figure
    fig = plt.figure(figname, figsize=(8, 8), clear=True)
    ax = fig.add_subplot(132)
    # ax.set_position([.37, .3, .28, .35])
    ratio = (np.max(D.ky) - np.min(D.ky))/(np.max(D.kx) - np.min(D.kx))
    ax.set_position([.277+.01+.22, .3, .28/ratio, .28])
    ax.tick_params(**kwargs_ticks)

    # plot data
    ax.contourf(D.kx, D.ky, np.flipud(D.map), 300, **kwargs_ex,
                vmax=.9 * np.max(D.map), vmin=.3 * np.max(D.map), zorder=.1)
    ax.set_rasterization_zorder(.2)

#    # Tight Binding Model
#    param = utils.paramCSRO20_opt()  # Load parameters
#    tb = utils.TB(a=np.pi, kbnd=2, kpoints=200)  # Initialize
#    tb.CSRO(param)  # Calculate bandstructure
#
#    plt.figure(figname)
#    bndstr = tb.bndstr  # Load bandstructure
#    coord = tb.coord  # Load coordinates
#
#    # read dictionaries
#    X = coord['X']
#    Y = coord['Y']
#    Axy = bndstr['Axy']
#    Bxz = bndstr['Bxz']
#    Byz = bndstr['Byz']
#    bands = (Axy, Bxz, Byz)
#
#    # loop over bands
#    n = 0  # counter
#    for band in bands:
#        n += 1
#        ax.contour(X, Y, band, colors='w', linestyles=':', levels=0,
#                   linewidths=1)

    # decorate axes
    ax.set_xlabel(r'$k_y \,(\pi/b)$', fontdict=font)
    ax.set_ylabel(r'$k_x \,(\pi/a)$', fontdict=font)
    ax.set_xticks([-.5, 0, .5, 1])
    ax.set_yticks([-1.5, -1, -.5, 0, .5])
    ax.set_xlim(np.min(D.kx), np.max(D.kx))
    ax.set_ylim(np.min(D.ky), np.max(D.ky))

    # add text
    ax.text(-.05, -.03, r'$\Gamma$', fontsize=12, color='r')
    ax.text(-.05, -1.03, r'Y', fontsize=12, color='r')
    ax.text(.95, -.03, r'X', fontsize=12, color='r')
    ax.text(.95, -1.03, r'S', fontsize=12, color='r')

    # transform extraction points into k-space
    for coord in coords:
        x = np.zeros(len(coord))
        y = np.zeros(len(coord))

        for i in range(len(coord)):
            x[i] = coord[i][0]
            y[i] = coord[i][1]

        kx = np.ones(x.size)
        ky = np.ones(y.size)

        for i in range(y.size):
            k, k_V0 = utils.ang2k(x[i], Ekin=22-4.5, lat_unit=True,
                                  a=5.33, b=5.55, c=11, V0=0, thdg=8.7,
                                  tidg=y[i]-4, phidg=88)
            kx[i] = k[0]
            ky[i] = k[1]

            plt.plot(kx, ky, 'ro', ms=.2, alpha=.3)

    plt.show()

    # Save figure
    if print_fig:
        plt.savefig(save_dir + figname + '.pdf',
                    dpi=100, bbox_inches="tight", rasterized=True)


def fig22(print_fig=True):
    """figure 22

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Tight binding model folded SRO
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    """

    figname = 'CSROfig22'

    # Initialize tight binding model
    kbnd = 1  # boundaries
    tb = utils.TB(a=np.pi, kbnd=kbnd, kpoints=300)
    param = utils.paramSRO()
    tb.SRO_folded(param=param, e0=0, vert=False, proj=False)

    # load data
    bndstr = tb.bndstr
    coord = tb.coord
    X = coord['X']
    Y = coord['Y']
    xz = bndstr['xz']
    yz = bndstr['yz']
    xy = bndstr['xy']
    xz_q = bndstr['xz_q']
    yz_q = bndstr['yz_q']
    xy_q = bndstr['xy_q']
    # collect bands
    bands = (xz, yz, xy, xz_q, yz_q, xy_q)

    # create figure
    fig = plt.figure(figname, figsize=(8, 8), clear=True)
    ax = fig.add_subplot(121)
    ax.set_position([.1, .3, .4, .4])
    ax.tick_params(**kwargs_ticks)

    # plot data
    n = 0  # counter
    for band in bands:
        n += 1
        if n < 4:
            ax.contour(X, Y, band, colors='C0', ls='-', zorder=.1,
                       linewidths=2, levels=0, alpha=1)
        else:
            ax.contour(X, Y, band, colors='C8', linewidths=.5, zorder=.1,
                       levels=0)
    ax.set_rasterization_zorder(.2)
    ax.plot([-1, 0], [0, 1], 'k-', lw=2)
    ax.plot([-1, 0], [0, -1], 'k-', lw=2)
    ax.plot([0, 1], [1, 0], 'k-', lw=2)
    ax.plot([0, 1], [-1, 0], 'k-', lw=2)
    ax.plot([-.5, -.5], [-.5, .5], **kwargs_ef)
    ax.plot([.5, .5], [-.5, .5], **kwargs_ef)
    ax.plot([-.5, .5], [.5, .5], **kwargs_ef)
    ax.plot([-.5, .5], [-.5, -.5], **kwargs_ef)

    # deocrate axes
    ax.set_xticks(np.arange(-kbnd - .5, kbnd + 1, .5))
    ax.set_yticks(np.arange(-kbnd - .5, kbnd + 1, .5))
    ax.set_xticklabels(['', '-2', '-1', '0', '1', '2'])
    ax.set_yticklabels(['', '-2', '-1', '0', '1', '2'])
    ax.set_xlim(-.5, .5)
    ax.set_ylim(-.5, .5)
    ax.set_xlabel(r'$k_x \, (\pi/a_o)$', fontdict=font)
    ax.set_ylabel(r'$k_y \, (\pi/a_o)$', fontdict=font)

    # add text
#    ax.text(-.975, .9, '(a)', fontdict=font)
    ax.text(-.47, .43, '(a)', fontdict=font)

    # Initialize tight binding model
    kbnd = 2  # boundaries
    tb = utils.TB(a=np.pi, kbnd=kbnd, kpoints=300)
    param = utils.paramSRO()
    tb.CSRO(param=param, e0=0, vert=False, proj=False)

    # load data
    bndstr = tb.bndstr
    coord = tb.coord
    X = coord['X']
    Y = coord['Y']
    Axz = bndstr['Axz']
    Ayz = bndstr['Ayz']
    Axy = bndstr['Axy']
    Bxz = bndstr['Bxz']
    Byz = bndstr['Byz']
    Bxy = bndstr['Bxy']

    # collect bands
    bands = (Axz, Ayz, Axy, Bxz, Byz, Bxy)

    # create figure
    fig = plt.figure(figname, figsize=(8, 8))
    ax2 = fig.add_subplot(122)
    ax2.set_position([.52, .3, .4, .4])
    ax2.tick_params(**kwargs_ticks)

    # label colors
    lblc = ['r', 'r', 'C0', 'C4', 'C9', 'C8']

    # plot data
    n = 0  # counter
    for band in bands:
        ax2.contour(X, Y, band, ls='-', colors=lblc[n],
                    linewidths=2, levels=0, alpha=1, zorder=.1)
        n += 1
    ax2.set_rasterization_zorder(.2)
    ax2.plot([-2, 0], [0, 2], 'k-', lw=2)
    ax2.plot([-2, 0], [0, -2], 'k-', lw=2)
    ax2.plot([0, 2], [2, 0], 'k-', lw=2)
    ax2.plot([0, 2], [-2, 0], 'k-', lw=2)
    ax2.plot([-1, -1], [-1, 1], **kwargs_ef)
    ax2.plot([1, 1], [-1, 1], **kwargs_ef)
    ax2.plot([-1, 1], [1, 1], **kwargs_ef)
    ax2.plot([-1, 1], [-1, -1], **kwargs_ef)

    # deocrate axes
    ax2.set_xticks(np.arange(-kbnd - 1, kbnd + 1, 1))
    ax2.set_yticks(np.arange(-kbnd - 1, kbnd + 1, 1))
    ax2.set_yticklabels([])
    ax2.set_xlim(-1, 1)
    ax2.set_ylim(-1, 1)
    ax2.set_xlabel(r'$k_x \, (\pi/a_o)$', fontdict=font)

    # add text
#    ax2.text(-1.95, 1.8, '(b)', fontdict=font)
    ax2.text(-.94, .86, '(b)', fontdict=font)

    plt.show()

    # Save figure
    if print_fig:
        plt.savefig(save_dir + figname + '.pdf', dpi=100,
                    bbox_inches="tight", rasterized=True)


def fig23(print_fig=True, load=True):
    """figure 23

    %%%%%%%%%%%%%%%%%%%%
    Fit dispersions + FS
    %%%%%%%%%%%%%%%%%%%%
    """

    figname = 'CSROfig23'

    # load data
    os.chdir(data_dir)
    GX_alpha_1 = np.loadtxt('dispersion_GX_alpha_1.dat')
    GX_alpha_2 = np.loadtxt('dispersion_GX_alpha_2.dat')
    GX_beta_1 = np.loadtxt('dispersion_GX_beta_1.dat')
    GX_beta_2 = np.loadtxt('dispersion_GX_beta_2.dat')
    GX_gamma_1 = np.loadtxt('dispersion_GX_gamma_1.dat')
    GX_gamma_2 = np.loadtxt('dispersion_GX_gamma_2.dat')
    GX_delta = np.loadtxt('dispersion_GX_delta.dat')

    GS_alpha_1 = np.loadtxt('dispersion_GS_alpha_1.dat')
    GS_alpha_2 = np.loadtxt('dispersion_GS_alpha_2.dat')
    GS_beta_1 = np.loadtxt('dispersion_GS_beta_1.dat')
    GS_beta_2 = np.loadtxt('dispersion_GS_beta_2.dat')
    GS_gamma_1 = np.loadtxt('dispersion_GS_gamma_1.dat')
    GS_gamma_2 = np.loadtxt('dispersion_GS_gamma_2.dat')
    GS_delta = np.loadtxt('dispersion_GS_delta.dat')

    XS_branch_1 = np.loadtxt('dispersion_XS_branch_1.dat')
    XS_branch_2 = np.loadtxt('dispersion_XS_branch_2.dat')

    alpha_1 = np.loadtxt('coords_CSRO20_alpha_1.dat')
    alpha_2 = np.loadtxt('coords_CSRO20_alpha_2.dat')
    beta_1 = np.loadtxt('coords_CSRO20_beta_1.dat')
    beta_2 = np.loadtxt('coords_CSRO20_beta_2.dat')
    beta_3 = np.loadtxt('coords_CSRO20_beta_3.dat')
    gamma_1 = np.loadtxt('coords_CSRO20_gamma_1.dat')
    gamma_2 = np.loadtxt('coords_CSRO20_gamma_2.dat')
    gamma_3 = np.loadtxt('coords_CSRO20_gamma_3.dat')
    delta = np.loadtxt('coords_CSRO20_delta.dat')
    os.chdir(home_dir)

    coords = (GX_alpha_1, GX_alpha_2, GX_beta_1, GX_beta_2,
              GX_gamma_1, GX_gamma_2, GX_delta,  # 0 - 6
              GS_alpha_1, GS_alpha_2, GS_beta_1, GS_beta_2,
              GS_gamma_1, GS_gamma_2, GS_delta,  # 7 - 13
              XS_branch_1, XS_branch_2,  # 14, 15
              alpha_1, alpha_2, beta_1, beta_2, beta_3, gamma_1, gamma_2,
              gamma_3, delta)  # 16 - 24

    # placeholders
    Kx = ()
    Ky = ()
    En = ()

    # transform into k-space
    n = 0  # counter (datasets)
    m = 1  # counter (datapoints)
    for coord in coords:
        x = np.zeros(len(coord))
        y = np.zeros(len(coord))
        en = np.zeros(len(coord))

        if n < 16:
            for i in range(len(coord)):
                x[i] = coord[i][0]
                y[i] = 0
                en[i] = coord[i][1]
                m +=  2 * 2.4   # regularization
        elif n >= 16:
            for i in range(len(coord)):
                x[i] = coord[i][0]
                y[i] = coord[i][1]
                en[i] = 0
                m += 1

        kx = np.ones(x.size)
        ky = np.ones(y.size)

        for i in range(y.size):
            if any(x == n for x in np.arange(0, 7, 1)):
                k, k_V0 = utils.ang2k(x[i], Ekin=22-4.5, lat_unit=True,
                                      a=5.33, b=5.55, c=11, V0=0, thdg=9.3,
                                      tidg=y[i], phidg=90)
            elif any(x == n for x in np.arange(7, 14, 1)):
                k, k_V0 = utils.ang2k(x[i], Ekin=40 - 4.5, lat_unit=True,
                                      a=5.5, b=5.5, c=11, V0=0, thdg=2.5,
                                      tidg=0, phidg=42)
            elif any(x == n for x in np.arange(14, 16, 1)):
                k, k_V0 = utils.ang2k(x[i], Ekin=22-4.5, lat_unit=True,
                                      a=5.33, b=5.55, c=11, V0=0, thdg=6.3,
                                      tidg=y[i]-16, phidg=90)
            elif any(x == n for x in np.arange(16, 25, 1)):
                k, k_V0 = utils.ang2k(x[i], Ekin=22-4.5, lat_unit=True,
                                      a=5.33, b=5.55, c=11, V0=0, thdg=8.7,
                                      tidg=y[i]-4, phidg=88)

            kx[i] = k[0]
            ky[i] = k[1]

        Kx = Kx + (kx,)
        Ky = Ky + (ky,)
        En = En + (en,)
        n += 1

    # maximum iterations
    it_max = 10000

    # initial parameters
    p = utils.paramSRO()

    t1 = p['t1']
    t2 = p['t2']
    t3 = p['t3']
    t4 = p['t4']
    t5 = p['t5']
    t6 = p['t6']
    mu = p['mu']
    so = p['so']

    P = np.array([t1, t2, t3, t4, t5, t6, mu, so])

    # load data
    if load:
        os.chdir(data_dir)
        it = np.loadtxt('Data_CSROfig23_it.dat')
        J = np.loadtxt('Data_CSROfig23_J.dat')
        P = np.loadtxt('Data_CSROfig23_P.dat')
        os.chdir(home_dir)
    else:
        # optimize parameters
        it, J, P = utils.optimize_TB(Kx, Ky, En, it_max, P)

    J_n = J / m * 1e3
    fig = plt.figure(figname, clear=True, figsize=(7, 7))
    ax = fig.add_axes([.25, .25, .5, .5])
    print('J_final='+str(J_n[-1]))
    # plot cost
    ax.plot(it, J_n, 'C8o', ms=2)
    ax.plot([0, np.max(it)], [np.min(J_n), np.min(J_n)], **kwargs_ef)
    ax.tick_params(**kwargs_ticks)

    # decorate axes
    ax.set_xlim([np.min(it), np.max(it)])
    ax.set_xlabel('iterations $t$', fontdict=font)
    ax.set_ylabel(r'fitness $\xi (\theta_t)$ (meV)', fontdict=font)
#    ax.arrow(2080, 9.2, 0, -1.2, head_width=160,
#             head_length=.3, fc='k', ec='k')
#    ax.arrow(5950, 7.5, 0, -1.2, head_width=160,
#             head_length=.3, fc='k', ec='k')

    plt.show()

    # Save figure
    if print_fig:
        plt.savefig(save_dir + figname + '.pdf',
                    dpi=100, bbox_inches="tight", rasterized=True)

    # save data
    os.chdir(data_dir)
    np.savetxt('Data_CSROfig23_it.dat', np.ravel(it))
    np.savetxt('Data_CSROfig23_J.dat', np.ravel(J))
    np.savetxt('Data_CSROfig23_P.dat', np.ravel(P))
    print('\n ~ Data saved (iterations, cost)',
          '\n', '==========================================')
    os.chdir(home_dir)

    return it_max, J, P


def fig25(print_fig=True):
    """figure 25

    %%%%%%%%%%%%%%%%%%%%%
    self energy + Z + DOS
    %%%%%%%%%%%%%%%%%%%%%
    """

    figname = 'CSROfig25'

    os.chdir(data_dir)
    Z_e = np.loadtxt('Data_CSROfig5_Z_e.dat')
    Z_b = np.loadtxt('Data_CSROfig6_Z_b.dat')
    eZ_b = np.loadtxt('Data_CSROfig6_eZ_b.dat')
    dims = np.loadtxt('Data_CSROfig6_dims.dat', dtype=np.int32)
    Loc_en = np.loadtxt('Data_CSROfig6_Loc_en.dat')
    Loc_en = np.reshape(np.ravel(Loc_en), (dims[0], dims[1]))
    v_LDA_data = np.loadtxt('Data_CSROfig8_v_LDA.dat')
    v_LDA = v_LDA_data[0]
    re = np.loadtxt('Data_CSROfig9_re.dat')
    ere = np.loadtxt('Data_CSROfig9_ere.dat')
    im = np.loadtxt('Data_CSROfig9_im.dat')
    eim = np.loadtxt('Data_CSROfig9_eim.dat')
#    C_B = np.genfromtxt('Data_C_Braden.csv', delimiter=',')
#    C_M = np.genfromtxt('Data_C_Maeno.csv', delimiter=',')
#    R_1 = np.genfromtxt('Data_R_1.csv', delimiter=',')
#    R_2 = np.genfromtxt('Data_R_2.csv', delimiter=',')
    os.chdir(home_dir)
    print('\n ~ Data loaded (Zs, specific heat, transport data)',
          '\n', '==========================================')

    print('Z_e='+str(np.mean(Z_e)))
    print('Z_b='+str(np.mean(Z_b)))

    # Reshape data
    re = np.reshape(np.ravel(re), (dims[0], dims[1]))
    ere = np.reshape(np.ravel(ere), (dims[0], dims[1]))
    im = np.reshape(np.ravel(im), (dims[0], dims[1]))
    eim = np.reshape(np.ravel(eim), (dims[0], dims[1]))

    # useful parameter
    T = np.array([1.3, 10, 20, 30])  # temperatures
    hbar = 1.0545717e-34  # Planck constant
    NA = 6.022141e23  # Avogadro constant
    kB = 1.38065e-23  # Boltzmann constant
    a = 5.33e-10  # lattice parameter
    m_e = 9.109383e-31
    m_LDA = 1.6032

    # Sommerfeld constant for 2-dimensional systems
    gamma = ((np.pi * NA * kB ** 2 * a ** 2 / (3 * hbar ** 2)) *
             m_LDA * m_e)

    # heat capacity in units of Z
#    Z_B = gamma / C_B[:, 1]
#    Z_M = gamma / C_M[:, 1] * 1e3

    print('gamma='+str(gamma / np.mean(Z_e)))

#    # fit for resistivity curve
#    xx = np.array([1e-3, 1e4])
#    yy = 2.3 * xx ** 2

    # create figure
    fig = plt.figure(figname, figsize=(10, 10), clear=True)

#    ax1 = fig.add_subplot(131)
#    ax1.set_position([.08, .3, .25, .25])
#    ax1.tick_params(direction='in', length=1.5, width=.5, colors='k')
#
#    # plot data
#    spec = 2
#    en = -Loc_en[spec]
#    ax1.errorbar(en, im[spec], eim[spec],
#                 color=[0, .4, .4], lw=.5, capsize=2, fmt='d', ms=2)
#    ax1.errorbar(en, re[spec], ere[spec],
#                 color='goldenrod', lw=.5, capsize=2, fmt='o', ms=2)
#
#    # decorate axes
#    ax1.set_ylabel('Self energy (meV)', fontdict=font)
#    ax1.set_yticks(np.arange(0, .25, .05))
#    ax1.set_yticklabels(['0', '50', '100', '150', '200'])
#    ax1.set_xticks(np.arange(0, .1, .02))
#    ax1.set_xticklabels(['0', '-20', '-40', '-60', '-80', '-100'])
#    ax1.set_xlabel(r'$\omega\,(\mathrm{meV})$', fontdict=font)
#    ax1.set_xlim(0, .1)
#    ax1.set_ylim(-.01, .25)
#    ax1.grid(True, alpha=.2)
#
#    # add text
#    ax1.text(.005, .2, r'$\mathfrak{Re}\Sigma(\omega) \, (1-Z)^{-1}$',
#             fontsize=12, color='goldenrod')
#    ax1.text(.06, .014, r'$\mathfrak{Im}\Sigma(\omega)$',
#             fontsize=12, color=[0, .4, .4])
#    ax1.text(.002, .23, '(a)', fontdict=font)

    ax2 = fig.add_subplot(132)
    ax2.set_position([.4, .3, .25, .25])
    ax2.tick_params(direction='in', length=1.5, width=.5, colors='k')

    # plot data beta band
    ax2.errorbar(T, Z_b, eZ_b * v_LDA,
                 color='m', lw=.5, capsize=2, fmt='o', ms=2)
    ax2.fill_between([0, 50], .215, .3, alpha=.1, color='m')
    ax2.plot(39, .229, 'm*')

    # plot data epsilon band
    ax2.errorbar(T, Z_e, Z_e / v_LDA,
                 color='r', lw=.5, capsize=2, fmt='d', ms=2)
    ax2.fill_between([0, 50], 0.02, .08, alpha=.1, color='r')
    ax2.plot(39, .052, 'r*')

    # plot Matsubara data
    # ax.plot(39, .326, 'C1+')  # Matsubara point
    # ax.plot(39, .175, 'r+')  # Matsubara point

    # plot heat capacity data
#    ax2.plot(C_B[:, 0], Z_B, 'o', ms=1, color='cadetblue')
#    ax2.plot(C_M[:, 0], Z_M, 'o', ms=1, color='slateblue')

    # decorate axes
    ax2.arrow(28, .16, 8.5, .06, head_width=0.0, head_length=0,
              fc='k', ec='k')
    ax2.arrow(28, .125, 8.5, -.06, head_width=0.0, head_length=0,
              fc='k', ec='k')
    ax2.set_xscale("log", nonposx='clip')
    ax2.set_yticks(np.arange(0, .5, .1))
    ax2.set_xlim(1, 44)
    ax2.set_ylim(0, .35)
    ax2.set_xlabel(r'$T$ (K)', fontdict=font)
    ax2.set_ylabel(r'$Z$', fontdict=font)

    # add text
#    ax2.text(1.1, .11, r'S. Nakatsuji $\mathit{et\, \,al.}$',
#             color='slateblue')
#    ax2.text(1.1, .09, r'J. Baier $\mathit{et\, \,al.}$',
#             color='cadetblue')
    ax2.text(1.1, .32, '(a)', fontdict=font)
    ax2.text(2.5e0, .25, r'$\alpha$-band', color='m')
    ax2.text(2.5e0, .045, r'$\gamma$-band', color='r')
    ax2.text(20, .135, 'DMFT')

#    # Inset
#    axi = fig.add_subplot(133)
#    axi.set_position([.63, .39, .13, .08])
#    axi.tick_params(**kwargs_ticks)
#
#    # Plot resistivity data
#    axi.loglog(np.sqrt(R_1[:, 0]), R_1[:, 1], 'o', ms=1,
#               color='slateblue')
#    axi.loglog(np.sqrt(R_2[:, 0]), R_2[:, 1], 'o', ms=1,
#               color='slateblue')
#    axi.loglog(xx, yy, 'k--', lw=1)
#
#    # decorate axes
#    axi.set_ylabel(r'$\rho\,(\mu \Omega \mathrm{cm})$')
#    axi.set_xlim(1e-1, 1e1)
#    axi.set_ylim(1e-2, 1e4)
#
#    # add text
#    axi.text(2e-1, 1e1, r'$\propto T^2$')

    ax3 = fig.add_axes([.72, .3, .25, .25])
    ax3.tick_params(**kwargs_ticks)

    os.chdir(data_dir)
    DMFT_DOS_xy_dn = np.loadtxt('DMFT_DOS_xy_dn.dat')
    DMFT_DOS_yz_dn = np.loadtxt('DMFT_DOS_yz_dn.dat')
    DMFT_DOS_xz_dn = np.loadtxt('DMFT_DOS_xz_dn.dat')
    DMFT_DOS_xy_up = np.loadtxt('DMFT_DOS_xy_up.dat')
    DMFT_DOS_yz_up = np.loadtxt('DMFT_DOS_yz_up.dat')
    DMFT_DOS_xz_up = np.loadtxt('DMFT_DOS_xz_up.dat')
    os.chdir(home_dir)

    top = 5000
    bot = 2000

    DMFT_xy = DMFT_DOS_xy_dn[bot:top, 1] + DMFT_DOS_xy_up[bot:top, 1]
    DMFT_yz = DMFT_DOS_yz_dn[bot:top, 1] + DMFT_DOS_yz_up[bot:top, 1]
    DMFT_xz = DMFT_DOS_xz_dn[bot:top, 1] + DMFT_DOS_xz_up[bot:top, 1]

    En_DMFT = DMFT_DOS_xy_dn[bot:top, 0]

    DOS = (DMFT_xy, DMFT_yz, DMFT_xz)
    En = (En_DMFT, En_DMFT, En_DMFT)

    cols = ['darkred', 'b', 'k']

    for i in range(1):
        ax3.plot(En[i], DOS[i], color=cols[i], lw=1)
        ax3.fill_between(En[i], DOS[i], np.zeros(len(DOS[i])),
                         color=cols[i], alpha=.1)
#    ax3.legend([r'$d_{xy}$', r'$d_{xz}$', r'$d_{yz}$'],
#               fontsize=12, frameon=False, loc='center left')
    ax3.text(-3.8, 1.65, '(b) DMFT', fontdict=font)
    ax3.text(-1.7, .12, r'$xy$', color=cols[0], fontsize=12)
    ax3.set_yticks(np.arange(0, 2, .5))
    ax3.set_xticklabels(np.arange(-4, 4, 1), fontdict=font)
    ax3.plot([0, 0], [0, 2], **kwargs_ef)
    ax3.set_xlim(-4, 2)
    ax3.set_ylim(0, 1.8)
    ax3.text(0.2, 1.3, 'vHs', fontdict=font)
    ax3.set_xlabel(r'$\omega$ (eV)', fontdict=font)
    ax3.set_ylabel(r'PDOS (eV$^{-1}$)', fontdict=font)

#    # Inset
    axi = fig.add_axes([.75, .4, .11, .11])
    for i in [1, 2]:
        axi.plot(En[i], DOS[i], color=cols[i], lw=.5)
    axi.plot([0, 0], [0, 10], **kwargs_ef)
    axi.set_xticks(np.arange(-4, 4, 2))
    axi.set_xlim(-4, 2)
    axi.set_ylim(0, 1.8)
    axi.text(-1.8, 1.2, '$yz$', color=cols[1])
    axi.text(-1.8, .9, '$xz$', color=cols[2])

    plt.show()

    # Save figure
    if print_fig:
        plt.savefig(save_dir + figname + '.png', dpi=300,
                    bbox_inches="tight")


def fig26(print_fig=True):
    """figure 26

    %%%%%%%%%%%%%%%%%%%%%%%%%%%
    Fermi surface counting CSRO
    %%%%%%%%%%%%%%%%%%%%%%%%%%%
    """

    figname = 'CSROfig26'

    # calculate band structure
    kbnd = 1  # boundaries
    tb = utils.TB(a=np.pi, kbnd=kbnd, kpoints=200)
    param = utils.paramCSRO_fit()
    tb.CSRO(param=param, e0=0.00, vert=True, proj=True)

    # get vertices
    VX = tb.VX
    VY = tb.VY

    # vertices of pockets
    alpha = np.array([VX[3][3], VY[3][3]])
    beta = np.array([VX[3][2], VY[3][2]])
    gamma_1 = np.array([VX[4][0], VY[4][0]])
    gamma_2 = np.array([VX[4][1], VY[4][1]])
    gamma_3 = np.array([VX[4][2], VY[4][2]])
    gamma_4 = np.array([VX[4][3], VY[4][3]])
    delta_1 = np.array([VX[3][0], VY[3][0]])
    delta_2 = np.array([VX[3][1], VY[3][1]])
    delta_3 = np.array([VX[3][4], VY[3][4]])
    delta_4 = np.array([VX[3][5], VY[3][5]])

    bands = (alpha, beta, gamma_1, gamma_2, gamma_3, gamma_4,
             delta_1, delta_2, delta_3, delta_4)  # collect bands

    fig = plt.figure(figname, figsize=(7, 7), clear=True)

    cols = ['C1', 'm', 'b', 'r']
    lbls = [r'(a)  $electron$', r'(b)  $hole$',
            r'(c)  $electron$', r'(d)  $hole$']
    loc_x = [.08, .3, .52, .74]
    FS_areas = np.zeros(4)  # placeholders
    BZ = 4  # area of BZ

    n = 0  # counter
    m = 0
    for band in bands:
        if any(x == m for x in [0, 1, 2, 6]):
            FS_areas[n] = utils.area(band[0, :], band[1, :])
            ax = fig.add_subplot(1, 4, n+1)
            ax.set_position([loc_x[n], .4, .2, .2])
            ax.tick_params(**kwargs_ticks)
            ax.plot([-1, -1], [-1, 1], **kwargs_ef)
            ax.plot([1, 1], [-1, 1], **kwargs_ef)
            ax.plot([-1, 1], [1, 1], **kwargs_ef)
            ax.plot([-1, 1], [-1, -1], **kwargs_ef)
            n += 1

        ax.plot(band[0, :], band[1, :], color=cols[n-1], lw=1)
        ax.fill(band[0, :], band[1, :], color=cols[n-1], alpha=.1)

        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_xlim(-1.4, 1.4)
        ax.set_ylim(-1.4, 1.4)

        if any(x == m for x in [0, 1]):
            ax.text(-1.3, 1.1, lbls[n-1])
        elif any(x == m for x in [2, 6]):
            ax.text(-1.3, 1.1, lbls[n-1])
#        if m == 0:
#            ax.text(-.85, -.9, '1. BZ', fontdict=font)

        m += 1
    FS_areas = np.abs(FS_areas) / BZ
    FS_areas[2] *= 4
    FS_areas[3] *= 4

    al = FS_areas[0]  # electron
    be = FS_areas[1]  # hole
    ga = FS_areas[2]  # electron
    de = FS_areas[3]  # hole

    n = (2 +  # full bands from folding
         al +
         ga +
         (1 - de) +
         (1 - be))

    print(n)

    plt.show()

    # Save figure
    if print_fig:
        plt.savefig(save_dir + figname + '.pdf', dpi=100,
                    bbox_inches="tight", rasterized=True)


def fig27(print_fig=True):
    """figure 27

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Fermi surface counting CSRO unfolded
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    """

    figname = 'CSROfig27'

    # calculate band structure
    kbnd = 2  # boundaries
    tb = utils.TB(a=np.pi, kbnd=kbnd, kpoints=200)
    param = utils.paramCSRO_fit()
    tb.SRO(param=param, e0=0.00, vert=True, proj=True)

    # get vertices
    VX = tb.VX
    VY = tb.VY

    # vertices of pockets
    alpha_1 = np.array([VX[0][0], VY[0][0]])
    alpha_2 = np.array([VX[0][1], VY[0][1]])
    alpha_3 = np.array([VX[0][4], VY[0][4]])
    alpha_4 = np.array([VX[0][5], VY[0][5]])
    beta = np.array([VX[2][8], VY[2][8]])
    gamma_1 = np.array([VX[1][0], VY[1][0]])
    gamma_2 = np.array([VX[1][1], VY[1][1]])
    gamma_3 = np.array([VX[1][2], VY[1][2]])
    gamma_4 = np.array([VX[1][3], VY[1][3]])
    delta_1 = np.array([VX[0][2], VY[0][2]])
    delta_2 = np.array([VX[0][3], VY[0][3]])
    delta_3 = np.array([VX[0][6], VY[0][6]])
    delta_4 = np.array([VX[0][7], VY[0][7]])

    bands = (alpha_1, alpha_2, alpha_3, alpha_4, beta,
             gamma_1, gamma_2, gamma_3, gamma_4,
             delta_1, delta_2, delta_3, delta_4)  # collect bands

    fig = plt.figure(figname, figsize=(7, 7), clear=True)

    cols = ['c', 'b', 'm', 'C1']
    lbls = [r'(a)  $hole$', r'(b)  $electron$',
            r'(c)  $hole$', r'(d)  $electron$']
    loc_x = [.08, .3, .52, .74]
    FS_areas = np.zeros(4)  # placeholders
    BZ = 4  # area of BZ

    n = 0  # counter
    m = 0
    for band in bands:
        if any(x == m for x in [0, 4, 5, 9]):
            FS_areas[n] = utils.area(band[0, :], band[1, :])
            ax = fig.add_subplot(1, 4, n+1)
            ax.set_position([loc_x[n], .4, .2, .2])
            ax.tick_params(**kwargs_ticks)
            ax.plot([-1, -1], [-1, 1], **kwargs_ef)
            ax.plot([1, 1], [-1, 1], **kwargs_ef)
            ax.plot([-1, 1], [1, 1], **kwargs_ef)
            ax.plot([-1, 1], [-1, -1], **kwargs_ef)
#            ax.plot([-1, 0], [0, 1], 'k--', lw=0.5, alpha=.5)
#            ax.plot([-1, 0], [0, -1], 'k--', lw=0.5, alpha=.5)
#            ax.plot([0, 1], [1, 0], 'k--', lw=0.5, alpha=.5)
#            ax.plot([0, 1], [-1, 0], 'k--', lw=0.5, alpha=.5)
            n += 1

        ax.plot(band[0, :], band[1, :], color=cols[n-1], lw=1)
        ax.fill(band[0, :], band[1, :], color=cols[n-1], alpha=.1)

        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_xlim(-1.4, 1.4)
        ax.set_ylim(-1.4, 1.4)

        props = dict(fc='w', ec='w', alpha=0.8,
                     pad=.0)

        if any(x == m for x in [0, 4]):
            ax.text(-1.3, 1.1, lbls[n-1], bbox=props)
        elif any(x == m for x in [5, 9]):
            ax.text(-1.3, 1.1, lbls[n-1], bbox=props)
#        if m == 0:
#            ax.text(-.6, -.9, '1. BZ', fontdict=font)

        m += 1
    FS_areas = np.abs(FS_areas) / BZ

    al = FS_areas[0]  # hole
    be = FS_areas[1]  # electron
    ga = FS_areas[2]  # hole
    de = FS_areas[3]  # electron

    n = (2 * (1 - al) +
         2 * be +
         2 * (1 - ga) +
         2 * de)

    print((1-al), be, (1-ga), de)

    plt.show()

    # Save figure
    if print_fig:
        plt.savefig(save_dir + figname + '.pdf', dpi=100,
                    bbox_inches="tight", rasterized=True)


def fig28(print_fig=True):
    """figure 28

    %%%%%%%%%%%%%%%%%%%%%%%%%%
    Fermi surface counting SRO
    %%%%%%%%%%%%%%%%%%%%%%%%%%
    """

    figname = 'CSROfig28'

    # calculate band structure
    kbnd = 2  # boundaries
    tb = utils.TB(a=np.pi, kbnd=kbnd, kpoints=200)
    param = utils.paramSRO()
    tb.SRO(param=param, e0=0.00, vert=True, proj=True)

    # get vertices
    VX = tb.VX
    VY = tb.VY

    # vertices of pockets
    alpha_1 = np.array([VX[0][0], VY[0][0]])
    alpha_2 = np.array([VX[0][1], VY[0][1]])
    alpha_3 = np.array([VX[0][2], VY[0][2]])
    alpha_4 = np.array([VX[0][3], VY[0][3]])
    beta = np.array([VX[2][8], VY[2][8]])
    gamma = np.array([VX[1][8], VY[1][8]])

    bands = (alpha_1, alpha_2, alpha_3, alpha_4, beta, gamma)

    fig = plt.figure(figname, figsize=(7, 7), clear=True)

    cols = ['c', 'b', 'm', 'C1']
    lbls = [r'(a)  $hole$', r'(b)  $electron$',
            r'(c)  $electron$']
    loc_x = [.08, .3, .52, .74]
    FS_areas = np.zeros(3)  # placeholders
    BZ = 4  # area of BZ

    n = 0  # counter
    m = 0
    for band in bands:
        if any(x == m for x in [0, 4, 5]):
            FS_areas[n] = utils.area(band[0, :], band[1, :])
            ax = fig.add_subplot(1, 4, n+1)
            ax.set_position([loc_x[n], .4, .2, .2])
            ax.tick_params(**kwargs_ticks)
            ax.plot([-1, -1], [-1, 1], **kwargs_ef)
            ax.plot([1, 1], [-1, 1], **kwargs_ef)
            ax.plot([-1, 1], [1, 1], **kwargs_ef)
            ax.plot([-1, 1], [-1, -1], **kwargs_ef)
            n += 1

        ax.plot(band[0, :], band[1, :], color=cols[n-1], lw=1)
        ax.fill(band[0, :], band[1, :], color=cols[n-1], alpha=.1)

        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_xlim(-1.4, 1.4)
        ax.set_ylim(-1.4, 1.4)

        props = dict(fc='w', ec='w', alpha=0.8,
                     pad=.0)

        if any(x == m for x in [0, 4]):
            ax.text(-1.3, 1.1, lbls[n-1], bbox=props)
        elif any(x == m for x in [5]):
            ax.text(-1.3, 1.1, lbls[n-1], bbox=props)
#        if m == 0:
#            ax.text(-.6, -.9, '1. BZ', fontdict=font)

        m += 1
    FS_areas = np.abs(FS_areas) / BZ

    al = FS_areas[0]  # hole
    be = FS_areas[1]  # electron
    ga = FS_areas[2]  # hole

    n = (2 * (1 - al) +
         2 * be +
         2 * ga)

    print(1-al, be, ga)

    plt.show()

    # Save figure
    if print_fig:
        plt.savefig(save_dir + figname + '.pdf', dpi=100,
                    bbox_inches="tight", rasterized=True)


def fig29(print_fig=True):
    """figure 29

    %%%%%%%%%%%%%%%%%%%%%%%%%
    Fermi surface folded CSRO
    %%%%%%%%%%%%%%%%%%%%%%%%%
    """

    figname = 'CSROfig29'

    # calculate band structure
    kbnd = 1.5  # boundaries
    tb = utils.TB(a=np.pi, kbnd=kbnd, kpoints=100)
    param = utils.paramCSRO20_opt()
    tb.SRO_folded(param=param, e0=-0.00, vert=True, proj=True)

    # get vertices
    VX = tb.VX
    VY = tb.VY

    # vertices of pocket
    alpha_1 = np.array([VX[0][0], VY[0][0]])
    alpha_2 = np.array([VX[0][2], VY[0][2]])
    alpha_3 = np.array([VX[0][3], VY[0][3]])
    alpha_4 = np.array([VX[0][6], VY[0][6]])
    beta = np.array([VX[2][2], VY[2][2]])
    gamma_1 = np.array([VX[1][0], VY[1][0]])
    gamma_2 = np.array([VX[1][1], VY[1][1]])
    gamma_3 = np.array([VX[1][2], VY[1][2]])
    gamma_4 = np.array([VX[1][3], VY[1][3]])
    delta_1 = np.array([VX[0][1], VY[0][1]])
    delta_2 = np.array([VX[0][4], VY[0][4]])
    delta_3 = np.array([VX[0][5], VY[0][5]])
    delta_4 = np.array([VX[0][7], VY[0][7]])

    f_alpha = np.array([VX[3][4], VY[3][4]])
    f_beta_1 = np.array([VX[5][0], VY[5][0]])
    f_beta_2 = np.array([VX[5][1], VY[5][1]])
    f_beta_3 = np.array([VX[5][2], VY[5][2]])
    f_beta_4 = np.array([VX[5][3], VY[5][3]])
    f_gamma_1 = np.array([VX[4][2], VY[4][2]])
#    f_gamma_2 = np.array([VX[4][1], VY[4][1]])
#    f_gamma_3 = np.array([VX[4][2], VY[4][2]])
#    f_gamma_4 = np.array([VX[4][3], VY[4][3]])
#    f_gamma_5 = np.array([VX[4][4], VY[4][4]])
    f_delta = np.array([VX[3][5], VY[3][5]])

    bands = (alpha_1, alpha_2, alpha_3, alpha_4, beta,
             gamma_1, gamma_2, gamma_3, gamma_4,
             delta_1, delta_2, delta_3, delta_4)

    f_bands = (f_alpha, f_beta_1, f_beta_2, f_beta_3, f_beta_4,
               f_gamma_1, f_delta)

    fig = plt.figure(figname, figsize=(10, 10), clear=True)
    ax = fig.add_subplot(144)
    ax.set_position([.723, .3, .28*2/3, .28])
    ax.tick_params(**kwargs_ticks)

    cols = ['c', 'b', 'm', 'C1']

    n = 0  # counter
    m = 0
    for band in bands:
        if any(x == m for x in [0, 4, 5, 9]):
            n += 1
        ax.plot(band[0, :], band[1, :], color=cols[n-1], lw=1, zorder=0)
        ax.fill(band[0, :], band[1, :], color=cols[n-1], alpha=.1,
                zorder=0)
        m += 1

    n = 0
    m = 0
    for band in f_bands:
        if any(x == m for x in [0, 1, 5, 6]):
            n += 1
        ax.plot(band[0, :], band[1, :], color=cols[n-1], lw=1, ls='--',
                zorder=0)
#        ax.fill(band[0, :], band[1, :], color=cols[n], alpha=.1)
        m += 1

    ax.plot([-.5, -.5], [-.5, .5], 'k-', lw=1.5)
    ax.plot([.5, .5], [-.5, .5], 'k-', lw=1.5)
    ax.plot([-.5, .5], [.5, .5], 'k-', lw=1.5)
    ax.plot([-.5, .5], [-.5, -.5], 'k-', lw=1.5)
    ax.plot([-1, 0], [0, 1], **kwargs_ef, alpha=1)
    ax.plot([-1, 0], [0, -1], **kwargs_ef, alpha=1)
    ax.plot([0, 1], [1, 0], **kwargs_ef, alpha=1)
    ax.plot([0, 1], [-1, 0], **kwargs_ef, alpha=1)

#    ax.fill_between([-1, 0], [-5, -5], [0, -1], color='w', alpha=1,
#                    zorder=2)
#    ax.fill_between([0, 1], [-5, -5], [-1, 0], color='w', alpha=1,
#                    zorder=2)
#    ax.fill_between([-1, 0], [0, 1], [5, 5], color='w', alpha=1,
#                    zorder=2)
#    ax.fill_between([0, 1], [1, 0], [5, 5], color='w', alpha=1,
#                    zorder=2)
    ax.fill_between([-1, 1], [-5, -5], [-1, -1], color='w', alpha=1,
                    zorder=2)
    ax.fill_between([-1, 1], [1, 1], [5, 5], color='w', alpha=1,
                    zorder=2)

    c1x = -.7
    c1y = -1.5
    l1 = .1
    c2x = -.7
    c2y = -1.8
    l2 = .07
    ax.plot([c1x-l1, c1x], [c1y, c1y+l1], **kwargs_ef)
    ax.plot([c1x-l1, c1x], [c1y, c1y-l1], **kwargs_ef)
    ax.plot([c1x, c1x+l1], [c1y+l1, c1y], **kwargs_ef)
    ax.plot([c1x, c1x+l1], [c1y-l1, c1y], **kwargs_ef)
    ax.text(c1x+.15, c1y-.03, 'tetr. BZ')
    ax.plot([c2x-l2, c2x-l2], [c2y-l2, c2y+l2], 'k-', lw=1.5)
    ax.plot([c2x+l2, c2x+l2], [c2y-l2, c2y+l2], 'k-', lw=1.5)
    ax.plot([c2x-l2, c2x+l2], [c2y+l2, c2y+l2], 'k-', lw=1.5)
    ax.plot([c2x-l2, c2x+l2], [c2y-l2, c2y-l2], 'k-', lw=1.5)
    ax.text(c2x+.15, c2y-.03, 'orth. BZ')
    ax.plot([-1, 0], [0, 1], **kwargs_ef, alpha=.2)
    ax.plot([-1, 0], [0, -1], **kwargs_ef, alpha=.2)
    ax.plot([0, 1], [1, 0], **kwargs_ef, alpha=.2)
    ax.plot([0, 1], [-1, 0], **kwargs_ef, alpha=.2)

    ux = .2
    uy = -1.5
    fx = .2
    fy = -1.8
    l0 = .07
    ax.plot([ux-l0, ux-l0], [uy-l0, uy], color=cols[0], lw=1, ls='-')
    ax.plot([ux-l0, ux], [uy-l0, uy-l0], color=cols[0], lw=1, ls='-')
    ax.plot([ux+l0, ux+l0], [uy-l0, uy], color=cols[1], lw=1, ls='-')
    ax.plot([ux, ux+l0], [uy-l0, uy-l0], color=cols[1], lw=1, ls='-')
    ax.plot([ux+l0, ux+l0], [uy, uy+l0], color=cols[2], lw=1, ls='-')
    ax.plot([ux, ux+l0], [uy+l0, uy+l0], color=cols[2], lw=1, ls='-')
    ax.plot([ux-l0, ux-l0], [uy, uy+l0], color=cols[3], lw=1, ls='-')
    ax.plot([ux-l0, ux], [uy+l0, uy+l0], color=cols[3], lw=1, ls='-')
    ax.fill_between([ux-l0, ux], [uy-l0, uy-l0], [uy, uy],
                    color=cols[0], alpha=.1, zorder=3)
    ax.fill_between([ux, ux+l0], [uy-l0, uy-l0], [uy, uy],
                    color=cols[1], alpha=.1, zorder=3)
    ax.fill_between([ux, ux+l0], [uy, uy], [uy+l0, uy+l0],
                    color=cols[2], alpha=.1, zorder=3)
    ax.fill_between([ux-l0, ux], [uy, uy], [uy+l0, uy+l0],
                    color=cols[3], alpha=.1, zorder=3)
    ax.text(ux+.15, uy-.05, 'unfolded')
    ax.plot([fx-l0, fx-l0], [fy-l0, fy], color=cols[0], lw=1, ls='--')
    ax.plot([fx-l0, fx], [fy-l0, fy-l0], color=cols[0], lw=1, ls='--')
    ax.plot([fx+l0, fx+l0], [fy-l0, fy], color=cols[1], lw=1, ls='--')
    ax.plot([fx, fx+l0], [fy-l0, fy-l0], color=cols[1], lw=1, ls='--')
    ax.plot([fx+l0, fx+l0], [fy, fy+l0], color=cols[2], lw=1, ls='--')
    ax.plot([fx, fx+l0], [fy+l0, fy+l0], color=cols[2], lw=1, ls='--')
    ax.plot([fx-l0, fx-l0], [fy, fy+l0], color=cols[3], lw=1, ls='--')
    ax.plot([fx-l0, fx], [fy+l0, fy+l0], color=cols[3], lw=1, ls='--')
    ax.text(fx+.15, fy-.05, 'folded')

    # ax.text(-.93, .8, '(d)', fontsize=12)
    ax.text(-.75, -1.25, r'$\alpha$', color=cols[0], fontsize=12)
    ax.text(-.55, -1.25, r'$\beta$', color=cols[1], fontsize=12)
    ax.text(-.35, -1.25, r'$\gamma$', color=cols[2], fontsize=12)
    ax.text(-.15, -1.25, r'$\delta$', color=cols[3], fontsize=12)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_xlim(-1, 1)
    ax.set_ylim(-2, 1)

    plt.show()

    # Save figure
    if print_fig:
        plt.savefig(save_dir + figname + '.pdf', dpi=100,
                    bbox_inches="tight", rasterized=True)


def fig30(print_fig=True):
    """figure 30

    %%%%%%%%%%%%%%%%%
    xFig1 2nd version
    %%%%%%%%%%%%%%%%%
    """

    figname = 'CSROfig30'

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

    # Figure panels
    def fig30a():
        # calculate band structure
        kbnd = 1.5  # boundaries
        tb = utils.TB(a=np.pi, kbnd=kbnd, kpoints=200)
        tb.SRO_folded(param=param, e0=-0.00, vert=True, proj=True)

        # get vertices
        VX = tb.VX
        VY = tb.VY

        # vertices of pocket
        alpha_1 = np.array([VX[0][0], VY[0][0]])
        alpha_2 = np.array([VX[0][2], VY[0][2]])
        alpha_3 = np.array([VX[0][3], VY[0][3]])
        alpha_4 = np.array([VX[0][6], VY[0][6]])
        beta = np.array([VX[2][2], VY[2][2]])
        gamma_1 = np.array([VX[1][0], VY[1][0]])
        gamma_2 = np.array([VX[1][1], VY[1][1]])
        gamma_3 = np.array([VX[1][2], VY[1][2]])
        gamma_4 = np.array([VX[1][3], VY[1][3]])
        delta_1 = np.array([VX[0][1], VY[0][1]])
        delta_2 = np.array([VX[0][4], VY[0][4]])
        delta_3 = np.array([VX[0][5], VY[0][5]])
        delta_4 = np.array([VX[0][7], VY[0][7]])

        f_alpha = np.array([VX[3][4], VY[3][4]])
        f_beta_1 = np.array([VX[5][0], VY[5][0]])
        f_beta_2 = np.array([VX[5][1], VY[5][1]])
        f_beta_3 = np.array([VX[5][2], VY[5][2]])
        f_beta_4 = np.array([VX[5][3], VY[5][3]])
        f_gamma_1 = np.array([VX[4][2], VY[4][2]])
    #    f_gamma_2 = np.array([VX[4][1], VY[4][1]])
    #    f_gamma_3 = np.array([VX[4][2], VY[4][2]])
    #    f_gamma_4 = np.array([VX[4][3], VY[4][3]])
    #    f_gamma_5 = np.array([VX[4][4], VY[4][4]])
        f_delta = np.array([VX[3][5], VY[3][5]])

        bands = (alpha_1, alpha_2, alpha_3, alpha_4, beta,
                 gamma_1, gamma_2, gamma_3, gamma_4,
                 delta_1, delta_2, delta_3, delta_4)

        f_bands = (f_alpha, f_beta_1, f_beta_2, f_beta_3, f_beta_4,
                   f_gamma_1, f_delta)

        ax = fig.add_subplot(144)
        ax.set_position([.08, .3, .28*2/3, .28])
        # ax.set_position([.723, .3, .28*2/3, .28])
        ax.tick_params(**kwargs_ticks)

        cols = ['c', 'b', 'm', 'C1']

        n = 0  # counter
        m = 0
        for band in bands:
            if any(x == m for x in [0, 4, 5, 9]):
                n += 1
            ax.plot(band[0, :], band[1, :], color=cols[n-1], lw=1, zorder=0)
            ax.fill(band[0, :], band[1, :], color=cols[n-1], alpha=.1,
                    zorder=0)
            m += 1

        n = 0
        m = 0
        for band in f_bands:
            if any(x == m for x in [0, 1, 5, 6]):
                n += 1
            ax.plot(band[0, :], band[1, :], color=cols[n-1], lw=1, ls='--',
                    zorder=0)
    #        ax.fill(band[0, :], band[1, :], color=cols[n], alpha=.1)
            m += 1

        ax.plot([-.5, -.5], [-.5, .5], 'k-', lw=1.5)
        ax.plot([.5, .5], [-.5, .5], 'k-', lw=1.5)
        ax.plot([-.5, .5], [.5, .5], 'k-', lw=1.5)
        ax.plot([-.5, .5], [-.5, -.5], 'k-', lw=1.5)
        ax.plot([-1, 0], [0, 1], **kwargs_ef, alpha=1)
        ax.plot([-1, 0], [0, -1], **kwargs_ef, alpha=1)
        ax.plot([0, 1], [1, 0], **kwargs_ef, alpha=1)
        ax.plot([0, 1], [-1, 0], **kwargs_ef, alpha=1)

#        ax.fill_between([-1, 0], [-5, -5], [0, -1], color='w', alpha=1,
#                        zorder=2)
#        ax.fill_between([0, 1], [-5, -5], [-1, 0], color='w', alpha=1,
#                        zorder=2)
#        ax.fill_between([-1, 0], [0, 1], [5, 5], color='w', alpha=1,
#                        zorder=2)
#        ax.fill_between([0, 1], [1, 0], [5, 5], color='w', alpha=1,
#                        zorder=2)
        ax.fill_between([-1, 1], [-5, -5], [-1, -1], color='w', alpha=1,
                        zorder=2)
        ax.fill_between([-1, 1], [1, 1], [5, 5], color='w', alpha=1,
                        zorder=2)
        ax.plot([-.46, .46], [-1, -1], 'k--', lw=.5)

        c1x = -.7
        c1y = -1.5
        l1 = .1
        c2x = -.7
        c2y = -1.8
        l2 = .07
        ax.plot([c1x-l1, c1x], [c1y, c1y+l1], **kwargs_ef)
        ax.plot([c1x-l1, c1x], [c1y, c1y-l1], **kwargs_ef)
        ax.plot([c1x, c1x+l1], [c1y+l1, c1y], **kwargs_ef)
        ax.plot([c1x, c1x+l1], [c1y-l1, c1y], **kwargs_ef)
        ax.text(c1x+.15, c1y-.03, 'tetr. BZ')
        ax.plot([c2x-l2, c2x-l2], [c2y-l2, c2y+l2], 'k-', lw=1.5)
        ax.plot([c2x+l2, c2x+l2], [c2y-l2, c2y+l2], 'k-', lw=1.5)
        ax.plot([c2x-l2, c2x+l2], [c2y+l2, c2y+l2], 'k-', lw=1.5)
        ax.plot([c2x-l2, c2x+l2], [c2y-l2, c2y-l2], 'k-', lw=1.5)
        ax.text(c2x+.15, c2y-.03, 'orth. BZ')
        ax.plot([-1, 0], [0, 1], **kwargs_ef, alpha=.2)
        ax.plot([-1, 0], [0, -1], **kwargs_ef, alpha=.2)
        ax.plot([0, 1], [1, 0], **kwargs_ef, alpha=.2)
        ax.plot([0, 1], [-1, 0], **kwargs_ef, alpha=.2)

        ux = .2
        uy = -1.5
        fx = .2
        fy = -1.8
        l0 = .07
        ax.plot([ux-l0, ux-l0], [uy-l0, uy], color=cols[0], lw=1, ls='-')
        ax.plot([ux-l0, ux], [uy-l0, uy-l0], color=cols[0], lw=1, ls='-')
        ax.plot([ux+l0, ux+l0], [uy-l0, uy], color=cols[1], lw=1, ls='-')
        ax.plot([ux, ux+l0], [uy-l0, uy-l0], color=cols[1], lw=1, ls='-')
        ax.plot([ux+l0, ux+l0], [uy, uy+l0], color=cols[2], lw=1, ls='-')
        ax.plot([ux, ux+l0], [uy+l0, uy+l0], color=cols[2], lw=1, ls='-')
        ax.plot([ux-l0, ux-l0], [uy, uy+l0], color=cols[3], lw=1, ls='-')
        ax.plot([ux-l0, ux], [uy+l0, uy+l0], color=cols[3], lw=1, ls='-')
        ax.fill_between([ux-l0, ux], [uy-l0, uy-l0], [uy, uy],
                        color=cols[0], alpha=.1, zorder=3)
        ax.fill_between([ux, ux+l0], [uy-l0, uy-l0], [uy, uy],
                        color=cols[1], alpha=.1, zorder=3)
        ax.fill_between([ux, ux+l0], [uy, uy], [uy+l0, uy+l0],
                        color=cols[2], alpha=.1, zorder=3)
        ax.fill_between([ux-l0, ux], [uy, uy], [uy+l0, uy+l0],
                        color=cols[3], alpha=.1, zorder=3)
        ax.text(ux+.15, uy-.05, 'unfolded')
        ax.plot([fx-l0, fx-l0], [fy-l0, fy], color=cols[0], lw=1, ls='--')
        ax.plot([fx-l0, fx], [fy-l0, fy-l0], color=cols[0], lw=1, ls='--')
        ax.plot([fx+l0, fx+l0], [fy-l0, fy], color=cols[1], lw=1, ls='--')
        ax.plot([fx, fx+l0], [fy-l0, fy-l0], color=cols[1], lw=1, ls='--')
        ax.plot([fx+l0, fx+l0], [fy, fy+l0], color=cols[2], lw=1, ls='--')
        ax.plot([fx, fx+l0], [fy+l0, fy+l0], color=cols[2], lw=1, ls='--')
        ax.plot([fx-l0, fx-l0], [fy, fy+l0], color=cols[3], lw=1, ls='--')
        ax.plot([fx-l0, fx], [fy+l0, fy+l0], color=cols[3], lw=1, ls='--')
        ax.text(fx+.15, fy-.05, 'folded')

        ax.text(-.93, .8, '(a)', fontsize=12)
        ax.text(-.75, -1.25, r'$\alpha$', color=cols[0], fontsize=12)
        ax.text(-.55, -1.25, r'$\beta$', color=cols[1], fontsize=12)
        ax.text(-.35, -1.25, r'$\gamma$', color=cols[2], fontsize=12)
        ax.text(-.15, -1.25, r'$\delta$', color=cols[3], fontsize=12)
        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_xlim(-1, 1)
        ax.set_ylim(-2, 1)

    def fig30b():
        ax = fig.add_subplot(141)
        # ax.set_position([.08, .3, .22, .28])
        ax.set_position([.277, .3, .22, .28])
        ax.tick_params(**kwargs_ticks)

        # plot data
        ax.contourf(A1.en_norm, A1.kys, A1.int_norm, 300, **kwargs_ex,
                    vmin=.1*np.max(A1.int_norm), vmax=.7*np.max(A1.int_norm),
                    zorder=.1)
        ax.set_rasterization_zorder(.2)
        ax.plot([0, 0], [np.min(A1.kys), np.max(A1.kys)], **kwargs_ef)
        ax.plot([-.005, -.005], [np.min(A1.kys), np.max(A1.kys)], 'r-.', lw=.5)

        # decorate axes
        ax.set_xlim(-.06, .03)
        ax.set_ylim(np.min(D.ky), np.max(D.ky))
        ax.set_xticks(np.arange(-.06, .03, .02))
        ax.set_xticklabels(['-60', '-40', '-20', '0', '20'])
        ax.set_yticks([-1.5, -1, -.5, 0, .5])
        ax.set_yticklabels([])
        ax.set_xlabel(r'$\omega\,(\mathrm{meV})$', fontdict=font)
        # ax.set_ylabel(r'$k_x \,(\pi/a)$', fontdict=font)
        ax.plot((mdc - b_mdc) / 30 + .001, A1.k[1], 'o', ms=1.5, color='C9')
        ax.fill(f_mdc / 30 + .001, A1.k[1], alpha=.2, color='C9')

        # add text
        ax.text(-.058, .57, '(b)', fontsize=12)
        ax.text(.024, -.03, r'$\Gamma$', fontsize=12, color='k')
        ax.text(.024, -1.03, 'Y', fontsize=12, color='k')

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
            ax.plot((utils.lor(A1.k[1], p_mdc[i], p_mdc[i+8], p_mdc[i+16],
                    p_mdc[-3], p_mdc[-2], p_mdc[-1]) - b_mdc) / 30 + .001,
                    A1.k[1], lw=.5, color=cols[i])
            ax.text(p_mdc[i+16]/5+corr[i], p_mdc[i]-.03, lbls[i],
                    fontsize=10, color=cols[i])
        ax.plot(f_mdc / 30 + .001, A1.k[1], color='k', lw=.5)

    def fig30c():
        ax = fig.add_subplot(142)
        # ax.set_position([.31, .3, .28/ratio, .28])
        ax.set_position([.277+.01+.22, .3, .28/ratio, .28])
        ax.tick_params(**kwargs_ticks)

        # plot data
        ax.contourf(D.kx, D.ky, np.flipud(D.map), 300, **kwargs_ex, zorder=.1,
                    vmax=.9 * np.max(D.map), vmin=.3 * np.max(D.map))
        ax.set_rasterization_zorder(.2)
        ax.plot(A1.k[0], A1.k[1], 'r-.', lw=.5)
        ax.plot(A2.k[0], A2.k[1], 'r-.', lw=.5)

        # decorate axes
        # ax.set_xlabel(r'$k_y \,(\pi/b)$', fontdict=font)
        ax.set_xticklabels([])

        # add text
        ax.text(-.65, .56, r'(c)', fontsize=12, color='w')
        ax.text(-.05, -.03, r'$\Gamma$', fontsize=12, color='w')
        ax.text(-.05, -1.03, r'Y', fontsize=12, color='w')
        ax.text(.95, -.03, r'X', fontsize=12, color='w')
        ax.text(.95, -1.03, r'S', fontsize=12, color='w')

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

        # loop over bands
        n = 0  # counter
        for band in bands:
            n += 1
            ax.contour(X, Y, band, colors='w', linestyles=':', levels=0,
                       linewidths=1)

        ax.set_xticks([-.5, 0, .5, 1])
        ax.set_yticks([-1.5, -1, -.5, 0, .5])
        ax.set_yticklabels([])
        ax.set_xlim(np.min(D.kx), np.max(D.kx))
        ax.set_ylim(np.min(D.ky), np.max(D.ky))

    def fig30d():
        ax = fig.add_subplot(143)
        ax.set_position([.277+.01+.22+.28/ratio+.01, .3, .17, .28])
        ax.tick_params(**kwargs_ticks)

        # plot data
        c0 = ax.contourf(-np.transpose(np.fliplr(A2.en_norm)),
                         np.transpose(A2.kys),
                         np.transpose(np.fliplr(A2.int_norm)), 300,
                         **kwargs_ex, zorder=.1,
                         vmin=.1*np.max(A2.int_norm),
                         vmax=.7*np.max(A2.int_norm))
        ax.set_rasterization_zorder(.2)
        ax.plot([0, 0], [np.min(A2.kys), np.max(A2.kys)], **kwargs_ef)

        # decorate axes
        ax.set_xticks(np.arange(0, .08, .02))
        ax.set_xticklabels(['0', '-20', '-40', '-60'])
        ax.set_yticks([-1.5, -1, -.5, 0, .5])
        ax.set_yticklabels([])
        ax.set_xlabel(r'$\omega\,(\mathrm{meV})$', fontdict=font)
        ax.set_xlim(-.01, .06)
        ax.set_ylim(np.min(D.ky), np.max(D.ky))

        # add text
        ax.text(-.0085, .56, '(d)', fontsize=12)
        ax.text(-.008, -.03, 'X', fontsize=12, color='k')
        ax.text(-.008, -1.03, 'S', fontsize=12, color='k')

        pos = ax.get_position()
        cax = plt.axes([pos.x0+pos.width+0.01,
                        pos.y0, 0.01, pos.height])
        cbar = plt.colorbar(c0, cax=cax, ticks=None)
        cbar.set_ticks([])
        cbar.set_clim(np.min(A2.int_norm), np.max(A2.int_norm))

    fig = plt.figure(figname, figsize=(10, 10), clear=True)
    fig30a()
    fig30b()
    fig30c()
    fig30d()
    fig.show()

    # Save figure
    if print_fig:
        fig.savefig(save_dir + figname + '.pdf', dpi=100,
                    bbox_inches="tight", rasterized=True)


def fig31(print_fig=True):
    """figure 31

    %%%%%%%%%%%%%%%%%
    xFig3 self energy
    %%%%%%%%%%%%%%%%%
    """

    figname = 'CSROfig31'

    os.chdir(data_dir)
    Z_e = np.loadtxt('Data_CSROfig5_Z_e.dat')
    Z_b = np.loadtxt('Data_CSROfig6_Z_b.dat')
    dims = np.loadtxt('Data_CSROfig6_dims.dat', dtype=np.int32)
    Loc_en = np.loadtxt('Data_CSROfig6_Loc_en.dat')
    Loc_en = np.reshape(np.ravel(Loc_en), (dims[0], dims[1]))
    re = np.loadtxt('Data_CSROfig9_re.dat')
    ere = np.loadtxt('Data_CSROfig9_ere.dat')
    im = np.loadtxt('Data_CSROfig9_im.dat')
    eim = np.loadtxt('Data_CSROfig9_eim.dat')
    os.chdir(home_dir)
    print('\n ~ Data loaded (Zs, specific heat, transport data)',
          '\n', '==========================================')

    print('Z_e='+str(np.mean(Z_e)))
    print('Z_b='+str(np.mean(Z_b)))

    # Reshape data
    re = np.reshape(np.ravel(re), (dims[0], dims[1]))
    ere = np.reshape(np.ravel(ere), (dims[0], dims[1]))
    im = np.reshape(np.ravel(im), (dims[0], dims[1]))
    eim = np.reshape(np.ravel(eim), (dims[0], dims[1]))

#    # fit for resistivity curve
#    xx = np.array([1e-3, 1e4])
#    yy = 2.3 * xx ** 2

    # create figure
    fig = plt.figure(figname, figsize=(10, 10), clear=True)

    ax1 = fig.add_subplot(131)
    ax1.set_position([.08, .3, .25, .25])
    ax1.tick_params(direction='in', length=1.5, width=.5, colors='k')

    # plot data
    spec = 2
    en = -Loc_en[spec]
    ax1.errorbar(en, im[spec], eim[spec],
                 color=[0, .4, .4], lw=.5, capsize=2, fmt='d', ms=2)
    ax1.errorbar(en, re[spec], ere[spec],
                 color='goldenrod', lw=.5, capsize=2, fmt='o', ms=2)
    ax1.fill_between([.083, .095], -1, 1, color='k', alpha=.1)

    # decorate axes
    ax1.arrow(.089, .075, 0, .025,  head_width=0.003, head_length=0.01,
              fc='k', ec='k', lw=.5)
    ax1.set_ylabel('Self energy (meV)', fontdict=font)
    ax1.set_yticks(np.arange(0, .3, .05))
    ax1.set_yticklabels(['0', '50', '100', '150', '200', '250'])
    ax1.set_xticks(np.arange(0, .12, .02))
    ax1.set_xticklabels(['0', '-20', '-40', '-60', '-80', '-100'])
    ax1.set_xlabel(r'$\omega\,(\mathrm{meV})$', fontdict=font)
    ax1.set_xlim(0, .1)
    ax1.set_ylim(-.01, .25)
    ax1.grid(True, alpha=.2)

    # add text
    ax1.text(.005, .18, r'$\Re\Sigma (\omega) \, (1-Z)^{-1}$',
             fontsize=12, color='goldenrod')
    ax1.text(.05, .014, r'$\Im\Sigma (\omega)$',
             fontsize=12, color=[0, .4, .4])
    ax1.text(.084, .06, r'$\omega_\mathrm{FL}$', fontdict=font)

    plt.show()

    # Save figure
    if print_fig:
        plt.savefig(save_dir + figname + '.pdf', dpi=100,
                    bbox_inches="tight", rasterized=True)


def fig32(print_fig=True):
    """figure 32

    %%%%%%%%%%%%%%%%%%%%%%%%%%%
    Resolution effects on alpha
    %%%%%%%%%%%%%%%%%%%%%%%%%%%
    """

    figname = 'CSROfig32'
    fig = plt.figure(figname, figsize=(6, 6), clear=True)

    ax1 = fig.add_subplot(231)
    ax1.set_position([.1, .55, .2, .2])
    ax1.tick_params(**kwargs_ticks)

    ax2 = fig.add_subplot(232)
    ax2.set_position([.32, .55, .2, .2])
    ax2.tick_params(**kwargs_ticks)

    ax3 = fig.add_subplot(233)
    ax3.set_position([.54, .55, .2, .2])
    ax3.tick_params(**kwargs_ticks)

    ax4 = fig.add_subplot(234)
    ax4.set_position([.1, .3, .2, .2])
    ax4.tick_params(**kwargs_ticks)

    ax5 = fig.add_subplot(235)
    ax5.set_position([.32, .3, .2, .2])
    ax5.tick_params(**kwargs_ticks)

    ax6 = fig.add_subplot(236)
    ax6.set_position([.54, .3, .2, .2])
    ax6.tick_params(**kwargs_ticks)

#    Ax = [ax1, ax2, ax3, ax4]
    kB = 8.617e-5  # Boltzmann constant
    Ts = np.array([1.3, 10, 20, 30])
    v_F = 2.34
    k_F = -.28
    n = 5000
    k = np.linspace(-1, .0, n)
    lda = (k-k_F) * v_F
    en = np.linspace(lda[0], lda[-1], n)
    res = 200

    # EDC values
    edc_b_val = -.36  # EDC beta band
    edcw_b_val = .01

    # boundaries of fit
    top_b = .01
    bot_b = -.015
    left_b = -.45
    right_b = -.15

    val, _edc_b = utils.find(k, edc_b_val)
    val, _edcw_b = utils.find(k, edc_b_val - edcw_b_val)
    val, _top_b = utils.find(en, top_b)
    val, _bot_b = utils.find(en, bot_b)
    val, _left_b = utils.find(k, left_b)
    val, _right_b = utils.find(k, right_b)
    val, k_F_idx = utils.find(k, k_F)
    val, e_F_idx = utils.find(en, 0)

    G0 = .02  # Impurity scattering
    eta = 10.6
    G = G0 + eta * en**2

    int_b = np.zeros(len(Ts))

    ImS0 = G * v_F + (np.pi * kB * 0)**2
    ReS0 = np.imag(hilbert(ImS0)) / 20
    ReS0 -= ReS0[e_F_idx]

    FD0 = utils.FDsl(en, kB*0, 0, 1, 0, 0)
    FDres0 = utils.FDconvGauss(en, kB*0, 0, 1, .008, 0, 1, 0)
    spec0 = np.zeros((len(k), len(en)))

    for j in range(len(k)):
        spec0[j, :] = ImS0 / ((en-lda[j]-ReS0)**2 + ImS0**2)

    ax1.contourf(k, en, np.transpose(spec0 * FD0), res, **kwargs_ex,
                 vmin=0, vmax=20, zorder=.1)
    ax1.set_rasterization_zorder(.2)
    ax1.plot(k[:e_F_idx+7], lda[:e_F_idx+7], 'k--', lw=1)
    ax1.plot([-2, 2], [0, 0], **kwargs_ef)
    ax1.set_xticks([-1, 0])
    ax1.set_yticks(np.arange(-.08, .08, .04))
    ax1.set_yticklabels(['-80', '-40', '0', '40'])
    ax1.set_ylabel(r'$\omega$ (meV)')
    ax1.set_xticklabels(['S', r'$\Gamma$'])
    ax1.set_xlim(-1, .0)
    ax1.set_ylim(-.1, .03)
    ax1.text(-.95, .012, '(a)')
    ax1.text(-.8, .012, r'$f(\omega, 0),\,\Sigma(\omega)$')
    ax1.text(-.25, -.05, r'$\epsilon_\mathbf{k}^b$')

    ax2.contourf(k, en, np.transpose(spec0 * FDres0), res, **kwargs_ex,
                 vmin=0, vmax=20, zorder=.1)
    ax2.set_rasterization_zorder(.2)
    ax2.plot(k[:e_F_idx+7], lda[:e_F_idx+7], 'k--', lw=1)
    ax2.plot([-2, 2], [0, 0], **kwargs_ef)
    ax2.set_xticks([-1, 0])
    ax2.set_yticks(np.arange(-.08, .08, .04))
    ax2.set_xticklabels(['S', r'$\Gamma$'])
    ax2.set_yticklabels([])
    ax2.set_xlim(-1, .0)
    ax2.set_ylim(-.1, .03)
    ax2.text(-.95, .012, '(b)')
    ax2.text(-.8, .012, r'$\otimes \,\mathcal{R}(\Delta \omega, \Delta k)$')

    ImS = G * v_F + (np.pi * kB * 30)**2
    ReS = np.imag(hilbert(ImS)) / 20
    ReS -= ReS[e_F_idx]
    spec_res = np.zeros((len(k), len(en)))
    FDres = utils.FDconvGauss(en, kB*30, 0, 1, .008, 0, 1, 0)

    for j in range(len(k)):
        spec_res[j, :] = ImS / ((en-lda[j]-ReS)**2 + ImS**2)

    ax3.contourf(k, en, np.transpose(spec_res * FDres), res, **kwargs_ex,
                 vmin=0, vmax=20, zorder=.1)
    ax3.set_rasterization_zorder(.2)
    ax3.plot(k[:e_F_idx+7], lda[:e_F_idx+7], 'k--', lw=1)
    ax3.plot([-2, 2], [0, 0], **kwargs_ef)

    box = {'ls': '--', 'color': 'r', 'lw': .5}
    ax3.plot([k[_left_b], k[_left_b]],
             [en[_top_b], en[_bot_b]], **box)
    ax3.plot([k[_right_b], k[_right_b]],
             [en[_top_b], en[_bot_b]], **box)
    ax3.plot([k[_left_b], k[_right_b]],
             [en[_top_b], en[_top_b]], **box)
    ax3.plot([k[_left_b], k[_right_b]],
             [en[_bot_b], en[_bot_b]], **box)

    ax3.set_xticks([-1, 0])
    ax3.set_yticks(np.arange(-.08, .08, .04))
    ax3.set_xticklabels(['S', r'$\Gamma$'])
    ax3.set_yticklabels([])
    ax3.set_xlim(-1, .0)
    ax3.set_ylim(-.1, .03)
    ax3.text(-.95, .012, '(c)')
    ax3.text(-.8, .012, r'$T \,> 0\,$K')

    # self-energy plot
#    ax6.plot(en, ImS0-G0*v_F)
#    dReS0 = -(ReS0[e_F_idx-1] - ReS0[e_F_idx])/(en[1]-en[0])
#    Z = 1 / (1-dReS0)
#    ax6.plot(en, ReS0/(1-Z))
#    ax6.set_xlim(-.1, .0)
#    ax6.set_ylim(0, .25)
    cols = ['c', 'C0', 'b', 'k']
    for i in range(len(Ts)):
        T = Ts[i]
        ImS = G * v_F + (np.pi * kB * T)**2
        ReS = np.imag(hilbert(ImS)) / 20
        ReS -= ReS[e_F_idx]

        spec = np.zeros((len(k), len(en)))
        for j in range(len(k)):
            spec[j, :] = ImS / ((en-lda[j]-ReS)**2 + ImS**2)

        FD = utils.FDconvGauss(en, kB*T, 0, 1, .008, 0, 1, 0)
        spec *= FD

        # integrates around Ef
        int_b[i] = np.sum(spec[_left_b:_right_b, _bot_b:_top_b])

        # plot data if necessary
    #    ax = Ax[i]
    #    ax.contourf(k, en, np.transpose(spec), 10, **kwargs_ex,
    #                vmin=0, vmax=20)
    #    ax.plot(k, lda, 'k--')
    #
    #    ax.set_ylim(-.1, .03)
    #    ax.set_xlim(-1, 0)

    #    int_b[i] = int_b[i] / int_b[0]

        MDC = spec[:, e_F_idx]
        EDC = spec[k_F_idx]

        ax4.plot(k, MDC, color=cols[i], lw=1)
        ax5.plot(en, EDC, color=cols[i], lw=1)
        ax6.plot(T, int_b[i] / int_b[0], 'o', ms=3, color=cols[i])

    ax4.set_xlim(-1, .0)
    ax4.set_ylim(-.05*np.max(MDC), 1.25*np.max(MDC))
    ax4.set_yticks([])
    ax4.set_xticks([-1, 0])
    ax4.set_xticklabels(['S', r'$\Gamma$'])
    ax4.set_ylabel('Intensity (arb. u.)')
    ax4.text(-.95, 15, r'(d) MDC @ $E_\mathrm{F}$')

    ax5.set_xticks(np.arange(-.08, .08, .04))
    ax5.set_xticklabels(['-80', '-40', '0', '40'])
    ax5.set_xlabel(r'$\omega$ (meV)')
    ax5.set_xlim(-.1, .03)
    ax5.set_ylim(-.05*np.max(EDC), 1.3*np.max(EDC))
    ax5.set_yticks([])
    ax5.text(-.095, 22.5, r'(e) EDC @ $k_\mathrm{F}$')

    ax6.tick_params(labelleft='off', labelright='on')
    ax6.yaxis.set_label_position('right')
    ax6.set_xticks(np.arange(0, 40, 10))
    ax6.set_yticks(np.arange(0, 1.1, .1))
    ax6.set_xlabel(r'$T$ (K)', fontdict=font)
    ax6.set_xlim(0, 32)
    ax6.set_ylim(.65, 1.1)
    ax6.grid(True, alpha=.2)
    ax6.text(1, 1.04, '(f) Box integrated')
    plt.show()

    # Save figure
    if print_fig:
        plt.savefig(save_dir + figname + '.pdf', dpi=100,
                    bbox_inches="tight", rasterized=True)


def fig33(print_fig=True):
    """figure 33

    %%%%%%%%%%%%%%%%%%%%%%%%%%%
    Resolution effects extended
    %%%%%%%%%%%%%%%%%%%%%%%%%%%
    """

    figname = 'CSROfig33'
    fig = plt.figure(figname, figsize=(6, 6), clear=True)

    kB = 8.617e-5  # Boltzmann constant
    Ts = np.array([1.3, 10, 20, 30])
    v_F = 2.34
    k_F = -.28
    n = 3000
    k = np.linspace(-1, .0, n)
    lda = (k-k_F) * v_F
    en = np.linspace(lda[0], lda[-1], n)
    res = 200

    # EDC values
    edc_b_val = -.36  # EDC beta band
    edcw_b_val = .01

    # boundaries of fit
    top_b = .01
    bot_b = -.015
    left_b = -.45
    right_b = -.15

    val, _edc_b = utils.find(k, edc_b_val)
    val, _edcw_b = utils.find(k, edc_b_val - edcw_b_val)
    val, _top_b = utils.find(en, top_b)
    val, _bot_b = utils.find(en, bot_b)
    val, _left_b = utils.find(k, left_b)
    val, _right_b = utils.find(k, right_b)
    val, k_F_idx = utils.find(k, k_F)
    val, e_F_idx = utils.find(en, 0)

    G0 = .02  # Impurity scattering
    eta = 10.6
    G = G0 + eta * en**2

    int_b = np.zeros(len(Ts))

    lbls = ['(a)', '(b)', '(c)', '(d)']
    cols = ['c', 'C0', 'b', 'k']
    ax1box = fig.add_subplot(3, 5, 4+1)
    ax1box.set_position([.1+(4*.165)+.02, .72, .15, .15])
    ax1box.tick_params(**kwargs_ticks)
    for i in range(len(Ts)):
        ax1 = fig.add_subplot(3, 5, i+1)
        ax1.set_position([.1+(i*.165), .72, .15, .15])
        ax1.tick_params(**kwargs_ticks)

        T = Ts[i]
        ImS0 = (eta*en**2) * v_F + (np.pi*kB*T)**2
        ReS0 = np.imag(hilbert(ImS0)) / 20
        ReS0 -= ReS0[e_F_idx]
        FD = utils.FDsl(en, kB*T, 0, 1, 0, 0)
        spec0 = np.zeros((len(k), len(en)))

        for j in range(len(k)):
            spec0[j, :] = ImS0 / ((en-lda[j]-ReS0)**2 + ImS0**2)

        int_b[i] = np.sum(spec0[_left_b:_right_b, _bot_b:_top_b])
        ax1box.plot(T, 1, 'o', ms=3, color=cols[i])
        spec0[spec0 > 20] = 20
        ax1.contourf(k, en, np.transpose(spec0 * FD), res, **kwargs_ex,
                     vmin=0, vmax=20, zorder=.1)
        ax1.set_rasterization_zorder(.2)
        ax1.plot(k[:e_F_idx], lda[:e_F_idx], 'k--', lw=1)
        ax1.plot([-2, 2], [0, 0], **kwargs_ef)
        ax1.set_xticks([-1, 0])
        ax1.set_xticklabels([])
        ax1.set_yticks(np.arange(-.08, .08, .04))

        if i == 0:
            ax1.set_yticklabels(['-80', '-40', '0', '40'], fontsize=8)
            ax1.set_ylabel(r'$\omega$ (meV)', fontsize=8)
            ax1.text(-.25, -.05, r'$\epsilon_\mathbf{k}^b$', fontsize=8)
            ax1.set_title(r'$T_0=\,$'+str(T)+r'$\,$K', fontsize=8)
        else:
            ax1.set_yticklabels([])
            ax1.set_title(r'$T=\,$'+str(T)+r'$\,$K', fontsize=8)

        ax1.set_xlim(-1, .0)
        ax1.set_ylim(-.1, .03)
        ax1.text(-.95, .012, lbls[i], fontsize=8)

        box = {'ls': '--', 'color': 'r', 'lw': .5}
        ax1.plot([k[_left_b], k[_left_b]],
                 [en[_top_b], en[_bot_b]], **box)
        ax1.plot([k[_right_b], k[_right_b]],
                 [en[_top_b], en[_bot_b]], **box)
        ax1.plot([k[_left_b], k[_right_b]],
                 [en[_top_b], en[_top_b]], **box)
        ax1.plot([k[_left_b], k[_right_b]],
                 [en[_bot_b], en[_bot_b]], **box)

    ax1box.plot([0, 40], [1, 1], **kwargs_ef)
    ax1box.tick_params(labelleft='off', labelright='on')
    ax1box.yaxis.set_label_position('right')
    ax1box.set_title(r'$\Sigma\,$box$\,(T\,)$ / $\Sigma\,$box$\,(T_0)$',
                     fontsize=8)
    ax1box.set_xticks(np.arange(0, 40, 10))
    ax1box.set_xticklabels([])
    ax1box.set_yticks(np.arange(.7, 1.1, .1))
    ax1box.set_yticklabels([0.7, 0.8, 0.9, 1.0], fontsize=8)
    ax1box.set_xlim(0, 32)
    ax1box.set_ylim(.65, 1.1)
    ax1box.grid(True, alpha=.2)
    ax1box.text(1, 1.04, '(e) no elast. scat.', fontsize=8)

    lbls = ['(f)', '(g)', '(h)', '(i)']
    cols = ['c', 'C0', 'b', 'k']
    ax2box = fig.add_subplot(3, 5, 9+1)
    ax2box.set_position([.1+(4*.165)+.02, .56, .15, .15])
    ax2box.tick_params(**kwargs_ticks)
    for i in range(len(Ts)):
        ax2 = fig.add_subplot(3, 5, i+6)
        ax2.set_position([.1+(i*.165), .56, .15, .15])
        ax2.tick_params(**kwargs_ticks)

        T = Ts[i]
        ImS = G * v_F + (np.pi*kB*T)**2
        ReS = np.imag(hilbert(ImS)) / 20
        ReS -= ReS[e_F_idx]
        FD = utils.FDsl(en, kB*T, 0, 1, 0, 0)
        spec = np.zeros((len(k), len(en)))

        for j in range(len(k)):
            spec[j, :] = ImS / ((en-lda[j]-ReS)**2 + ImS**2)
        spec[spec > 20] = 20
        int_b[i] = np.sum(spec[_left_b:_right_b, _bot_b:_top_b])
        ax2box.plot(T, int_b[i] / int_b[0], 'o', ms=3, color=cols[i])

        ax2.contourf(k, en, np.transpose(spec * FD), res, **kwargs_ex,
                     vmin=0, vmax=20, zorder=.1)
        ax2.set_rasterization_zorder(.2)
        ax2.plot(k[:e_F_idx], lda[:e_F_idx], 'k--', lw=1)
        ax2.plot([-2, 2], [0, 0], **kwargs_ef)
        ax2.set_xticks([-1, 0])
        ax2.set_xticklabels([])
        ax2.set_yticks(np.arange(-.08, .08, .04))

        if i == 0:
            ax2.set_yticklabels(['-80', '-40', '0', '40'], fontsize=8)
            ax2.set_ylabel(r'$\omega$ (meV)', fontsize=8)
        else:
            ax2.set_yticklabels([])

        ax2.set_xlim(-1, .0)
        ax2.set_ylim(-.1, .03)
        ax2.text(-.95, .012, lbls[i], fontsize=8)

        box = {'ls': '--', 'color': 'r', 'lw': .5}
        ax2.plot([k[_left_b], k[_left_b]],
                 [en[_top_b], en[_bot_b]], **box)
        ax2.plot([k[_right_b], k[_right_b]],
                 [en[_top_b], en[_bot_b]], **box)
        ax2.plot([k[_left_b], k[_right_b]],
                 [en[_top_b], en[_top_b]], **box)
        ax2.plot([k[_left_b], k[_right_b]],
                 [en[_bot_b], en[_bot_b]], **box)

    ax2box.plot([0, 40], [1, 1], **kwargs_ef)
    ax2box.tick_params(labelleft='off', labelright='on')
    ax2box.yaxis.set_label_position('right')
    ax2box.set_xticks(np.arange(0, 40, 10))
    ax2box.set_xticklabels([])
    ax2box.set_yticks(np.arange(.7, 1.1, .1))
    ax2box.set_yticklabels([0.7, 0.8, 0.9, 1.0], fontsize=8)
    ax2box.set_xlim(0, 32)
    ax2box.set_ylim(.65, 1.1)
    ax2box.grid(True, alpha=.2)
    ax2box.text(1, 1.04, '(j) $+$ elast. scat.', fontsize=8)

    lbls = ['(k)', '(l)', '(m)', '(n)']
    cols = ['c', 'C0', 'b', 'k']
    ax3box = fig.add_subplot(3, 5, 15)
    ax3box.set_position([.1+(4*.165)+.02, .4, .15, .15])
    ax3box.tick_params(**kwargs_ticks)
    for i in range(len(Ts)):
        ax3 = fig.add_subplot(3, 5, i+11)
        ax3.set_position([.1+(i*.165), .4, .15, .15])
        ax3.tick_params(**kwargs_ticks)

        T = Ts[i]
        ImS = G * v_F + (np.pi*kB*T)**2
        ReS = np.imag(hilbert(ImS)) / 20
        ReS -= ReS[e_F_idx]
        FD = utils.FDconvGauss(en, kB*T, 0, 1, .01, 0, 1, 0)
        spec = np.zeros((len(k), len(en)))

        for j in range(len(k)):
            spec[j, :] = ImS / ((en-lda[j]-ReS)**2 + ImS**2)
        spec[spec > 20] = 20
        int_b[i] = np.sum(spec[_left_b:_right_b, _bot_b:_top_b])
        ax3box.plot(T, int_b[i] / int_b[0], 'o', ms=3, color=cols[i])

        ax3.contourf(k, en, np.transpose(spec * FD), res, **kwargs_ex,
                     vmin=0, vmax=20, zorder=.1)
        ax3.set_rasterization_zorder(.2)
        ax3.plot(k[:e_F_idx], lda[:e_F_idx], 'k--', lw=1)
        ax3.plot([-2, 2], [0, 0], **kwargs_ef)
        ax3.set_xticks([-1, 0])
        ax3.set_xticklabels(['S', r'$\Gamma$'], fontsize=8)
        ax3.set_yticks(np.arange(-.08, .08, .04))

        if i == 0:
            ax3.set_yticklabels(['-80', '-40', '0', '40'], fontsize=8)
            ax3.set_ylabel(r'$\omega$ (meV)', fontsize=8)
        else:
            ax3.set_yticklabels([])

        ax3.set_xlim(-1, .0)
        ax3.set_ylim(-.1, .03)
        ax3.text(-.95, .012, lbls[i], fontsize=8)

        box = {'ls': '--', 'color': 'r', 'lw': .5}
        ax3.plot([k[_left_b], k[_left_b]],
                 [en[_top_b], en[_bot_b]], **box)
        ax3.plot([k[_right_b], k[_right_b]],
                 [en[_top_b], en[_bot_b]], **box)
        ax3.plot([k[_left_b], k[_right_b]],
                 [en[_top_b], en[_top_b]], **box)
        ax3.plot([k[_left_b], k[_right_b]],
                 [en[_bot_b], en[_bot_b]], **box)

    ax3box.plot([0, 40], [1, 1], **kwargs_ef)
    ax3box.tick_params(labelleft='off', labelright='on')
    ax3box.yaxis.set_label_position('right')
    ax3box.set_xticks(np.arange(0, 40, 10))
    ax3box.set_yticks(np.arange(.7, 1.1, .1))
    ax3box.set_xticklabels([0, 10, 20, 30], fontsize=8)
    ax3box.set_yticklabels([0.7, 0.8, 0.9, 1.0], fontsize=8)
    ax3box.set_xlabel(r'$T$ (K)', fontsize=8)
    ax3box.set_xlim(0, 32)
    ax3box.set_ylim(.65, 1.1)
    ax3box.grid(True, alpha=.2)
    ax3box.text(1, 1.04, '(o) $+$ instr. res.', fontsize=8)
    plt.show()

    # Save figure
    if print_fig:
        plt.savefig(save_dir + figname + '.pdf', dpi=180,
                    bbox_inches="tight", rasterized=True)


def fig34(print_fig=True):
    """figure 34

    %%%%%%%%%%%%%%%%%%%%%%%
    Resolution effects EDCs
    %%%%%%%%%%%%%%%%%%%%%%%
    """

    figname = 'CSROfig34'
    fig = plt.figure(figname, figsize=(6, 6), clear=True)

    ax1 = fig.add_axes([.1, .4, .3, .3])
    ax1.tick_params(**kwargs_ticks)

    ax2 = fig.add_axes([.43, .4, .3, .3])
    ax2.tick_params(**kwargs_ticks)

    kB = 8.617e-5  # Boltzmann constant
    Ts = np.array([1.3, 10, 20, 30])
    v_F = 2.34
    k_F = -.28
    n = 3000
    k = np.linspace(-.36, -.25, n)
    lda = (k-k_F) * v_F
    en = np.linspace(lda[0], lda[-1], n)

    # EDC values
    edc_b_val = -.36  # EDC beta band
    edcw_b_val = .01

    val, k_F_idx = utils.find(k, k_F)
    val, e_F_idx = utils.find(en, 0)

    G0 = .02  # Impurity scattering
    eta = 10.6
    G = G0 + eta * en**2

    cols = ['c', 'C0', 'b', 'k']

    for i in range(len(Ts)):
        T = Ts[i]
        ImS = G * v_F + (np.pi*kB*T)**2
        ReS = np.imag(hilbert(ImS)) / 20
        ReS -= ReS[e_F_idx]
        FD = utils.FDconvGauss(en, kB*T, 0, 1, .01, 0, 1, 0)
        EDC = np.zeros(len(en))
        EDC = ImS / ((en-lda[k_F_idx]-ReS)**2 + ImS**2)
        EDC *= FD
        ax1.plot(en, EDC, '-', color=cols[i])

    ax1.plot([-1, 1], [0, 0], **kwargs_ef)
    ax1.set_xticks(np.arange(-.1, .04, .02))
    ax1.set_xticklabels(['-100', '-80', '-60', '-40', '-20', '0', '20'])
    ax1.set_xlabel(r'$\omega$ (meV)', fontdict=font)
    ax1.set_xlim(-.1, .03)
    ax1.set_yticks([])
    ax1.set_ylabel('Intensity (arb. u.)', fontdict=font)
    ax1.text(-.095, 22.5, r'(a) EDC @ $k_\mathrm{F}$', fontsize=10)
    ax1.text(.006, 18, r'$1.3\,$K', color=cols[0])
    ax1.text(.006, 15, r'$10\,$K', color=cols[1])
    ax1.text(.006, 12, r'$20\,$K', color=cols[2])
    ax1.text(.006, 9, r'$30\,$K', color=cols[3])

    # data loading for figure
    files = [25, 26, 27, 28]
    gold = 14
    mat = 'CSRO20'
    year = 2017
    sample = 'S1'

    edc_b_val = -.36  # EDC beta band
    edcw_b_val = .01

    for i in [0, 3]:
        D = ARPES.Bessy(files[i], mat, year, sample)
        if i == 0:
            D.int_amp(1.52)  # renoramlize intensity for this spectrum
        D.norm(gold=gold)
        D.bkg()
        D.restrict(bot=.7, top=.9, left=.33, right=.5)

        # Transform data
        if i == 0:
            D.ang2k(D.ang, Ekin=48, lat_unit=True, a=5.5, b=5.5, c=11,
                    V0=0, thdg=2.4, tidg=0, phidg=45)
        else:
            D.ang2k(D.ang, Ekin=48, lat_unit=True, a=5.5, b=5.5, c=11,
                    V0=0, thdg=2.8, tidg=0, phidg=45)
        int_norm = D.int_norm
        en_norm = D.en_norm - .008
        val, _edc_b = utils.find(D.kxs[:, 0], edc_b_val)
        val, _edcw_b = utils.find(D.kxs[:, 0], edc_b_val - edcw_b_val)
        edc_b = (np.sum(int_norm[_edcw_b:_edc_b, :], axis=0) /
                 (_edc_b - _edcw_b + 1))

        bkg_b = utils.Shirley(edc_b)
        edc_b -= bkg_b
        bkg_b = utils.Shirley(edc_b)
        edc_b = edc_b / np.sum(edc_b)
        ax2.plot(en_norm[_edc_b], edc_b, 'o', ms=2, color=cols[i])
        ax2.plot([-1, 1], [0, 0], **kwargs_ef)

    ax2.set_xlim([-.1, .03])
    ax2.set_ylim([-.0004, .0088])
    ax2.text(-.095, .0078, r'(b) $\alpha$-EDC @ $k_\mathrm{F}$', fontsize=10)
    ax2.set_xticks(np.arange(-.1, .04, .02))
    ax2.set_xticklabels(['-100', '-80', '-60', '-40', '-20', '0', '20'])
    ax2.set_xlabel(r'$\omega$ (meV)', fontdict=font)
    ax2.set_xlim(-.1, .03)
    ax2.set_yticks([])
    ax2.text(.006, .006, r'$1.3\,$K', color=cols[0])
    ax2.text(.006, .0032, r'$30\,$K', color=cols[3])
    plt.show()

    # Save figure
    if print_fig:
        plt.savefig(save_dir + figname + '.pdf', dpi=200,
                    bbox_inches="tight", rasterized=True)


def fig35(print_fig=True):
    """figure 35

    %%%%%%%%%%%%%%%%%%%%%
    Heat capacity effects
    %%%%%%%%%%%%%%%%%%%%%
    """

    figname = 'CSROfig35'

    # load data
    os.chdir(data_dir)
    DOS_xy = np.loadtxt('Data_TB_DOS_Axy.dat')
    DOS_yz = np.loadtxt('Data_TB_DOS_Byz.dat')
    DOS_xz = np.loadtxt('Data_TB_DOS_Bxz.dat')

    DOS_xy_2 = np.loadtxt('Data_TB_DOS_Bxy.dat')
    DOS_yz_2 = np.loadtxt('Data_TB_DOS_Ayz.dat')
    DOS_xz_2 = np.loadtxt('Data_TB_DOS_Axz.dat')

    En_xy = np.loadtxt('Data_TB_DOS_en_Axy.dat')
    En_yz = np.loadtxt('Data_TB_DOS_en_Byz.dat')
    En_xz = np.loadtxt('Data_TB_DOS_en_Bxz.dat')

    En_xy_2 = np.loadtxt('Data_TB_DOS_en_Bxy.dat')
    En_yz_2 = np.loadtxt('Data_TB_DOS_en_Ayz.dat')
    En_xz_2 = np.loadtxt('Data_TB_DOS_en_Axz.dat')

    C_B = np.genfromtxt('Data_C_Braden.csv', delimiter=',')
    C_M = np.genfromtxt('Data_C_Maeno.csv', delimiter=',')

#    DMFT_DOS_xy_dn = np.loadtxt('DMFT_DOS_xy_dn.dat')
#    DMFT_DOS_yz_dn = np.loadtxt('DMFT_DOS_yz_dn.dat')
#    DMFT_DOS_xz_dn = np.loadtxt('DMFT_DOS_xz_dn.dat')
#    DMFT_DOS_xy_up = np.loadtxt('DMFT_DOS_xy_up.dat')
#    DMFT_DOS_yz_up = np.loadtxt('DMFT_DOS_yz_up.dat')
#    DMFT_DOS_xz_up = np.loadtxt('DMFT_DOS_xz_up.dat')
    os.chdir(home_dir)

#    top = 4500
#    bot = 3500
#    DMFT_xy = DMFT_DOS_xy_dn[bot:top, 1] +\
#              DMFT_DOS_xy_up[bot:top, 1]
#    DMFT_yz = DMFT_DOS_yz_dn[bot:top, 1] +\
#              DMFT_DOS_yz_up[bot:top, 1]
#    DMFT_xz = DMFT_DOS_xz_dn[bot:top, 1] +\
#              DMFT_DOS_xz_up[bot:top, 1]
#    En_DMFT = DMFT_DOS_xy_dn[bot:top, 0]

    DOS = (DOS_xy, DOS_yz, DOS_xz, DOS_xy_2, DOS_yz_2, DOS_xz_2)
    En = (En_xy, En_yz, En_xz, En_xy_2, En_yz_2, En_xz_2)
    T = np.arange(.4, 40, .05)
    kB = 8.6173303e-5
    C = np.ones(len(T))
    Cp = np.ones(len(T))
    U = np.ones(len(T))
    stp = 20
    J = 1.60218e-19
    mols = 6.022140857e23
    Gamma1 = 0
    Gamma2 = 0
    fig_sub = plt.figure('specific heat', clear=True)

    for i in range(len(En)):
        en = En[i]
        dos = DOS[i]

        nbins = len(en)

        expnt = np.ones((nbins, len(T)))
        FD = np.ones((nbins, len(T)))
        dFD = np.ones((nbins, len(T)))
        expnt_ext = np.ones((stp * nbins - (stp - 1), len(T)))
        FD_ext = np.ones((stp * nbins - (stp - 1), len(T)))
        dFD_ext = np.ones((stp * nbins - (stp - 1), len(T)))

#        dE = en[1] - en[0]
        En_ext = np.linspace(en[0], en[-1], FD_ext[:, 0].size)
        DOS_ext = interp1d(en, dos, kind='cubic')

        EF = 0.0026

        for t in range(len(T)):
            expnt[:, t] = (en - EF) / (kB * T[t])
            expnt_ext[:, t] = (En_ext - EF) / (kB * T[t])
            FD[:, t] = 1 / (np.exp(expnt[:, t]) + 1)
            FD_ext[:, t] = 1 / (np.exp(expnt_ext[:, t]) + 1)

        for e in range(nbins):
            dFD[e, :-1] = np.diff(FD[e, :]) / (T[2] - T[1])
            dFD[e, -1] = dFD[e, -2]

        for e in range(dFD_ext[:, 0].size):
            dFD_ext[e, :-1] = np.diff(FD_ext[e, :]) / (T[2] - T[1])
            dFD_ext[e, -1] = dFD_ext[e, -2]

        v_max = .6 * np.max(dFD_ext)
        ax3 = fig_sub.add_subplot(223)
        ax3.pcolormesh(T, En_ext, dFD_ext, cmap='PuOr',
                       vmax=v_max, vmin=-v_max)
        ax3.plot(T, 4 * kB * T, 'r--', lw=1)
        ax3.plot(T, -4 * kB * T, 'r--', lw=1)
        ax3.set_ylim(-.01, .01)

        edc_val, edc_idx = utils.find(T, 5)
        ax4 = fig_sub.add_subplot(224)
        ax4.plot(dFD_ext[:, edc_idx], En_ext, 'C8o', ms=1)
        ax4.set_ylim(-.01, .01)

        Cpext = Cp
        Uext = U
        Cext = C

        for t in range(len(T)):
            Cp[t] = np.trapz((en - EF) * dos * dFD[:, t], x=en)
            U[t] = np.trapz((en - EF) * dos * FD[:, t], x=en)
            Cpext[t] = np.trapz((En_ext - EF)*DOS_ext(En_ext)*dFD_ext[:, t],
                                x=En_ext)
            Uext[t] = np.trapz((En_ext - EF)*DOS_ext(En_ext)*FD_ext[:, t],
                               x=En_ext)

        C[:-1] = np.diff(U[:]) / (T[2] - T[1])
        C[-1] = C[-2]
        Cext[:-1] = np.diff(Uext[:]) / (T[2] - T[1])
        Cext[-1] = Cext[-2]
        pre = J*mols * 1000

        gamma2 = pre * Cpext / T
        gamma1 = pre * C / T

        Gamma1 = Gamma1 + gamma1
        Gamma2 = Gamma2 + gamma2

        ax1 = fig_sub.add_subplot(221)
        ax1.plot(En_ext, DOS_ext(En_ext), 'rs', ms=1)
        ax1.plot(en, dos)
        ax1.plot([EF+2*kB*T[-1], EF+2*kB*T[-1]], [0, np.max(dos)], 'r--')
        ax1.plot([EF-2*kB*T[-1], EF-2*kB*T[-1]], [0, np.max(dos)], 'r--')
        ax1.set_xlim(-.1, .1)

    fig = plt.figure(figname, figsize=(6, 6), clear=True)
    ax = fig.add_axes([.3, .3, .4, .4])
#    ax2.plot(T, gamma1 * factor)
    ax.plot(C_B[:, 0], C_B[:, 1] * 1e3, 'o', ms=1, color='k')
    ax.plot(C_M[:, 0], C_M[:, 1], 'o', ms=1, color='b')
    ax.text(4e-1, 130, r'J. Baier $\mathit{et\,\,al.}$', color='k')
    ax.text(4e-1, 120, r'S. Nakatsuji $\mathit{et\,\,al.}$', color='b')
    ax.text(4e-1, 110, r'TBA model ($\varepsilon_\mathbf{k} + 2.6\,$meV)',
            color='r')
    ax.set_xscale("log", nonposx='clip')
    ax.set_xlim(3e-1, 20)
    ax.set_ylim(80, 210)
    ax.set_xlabel(r'$T$ (K)')
    ax.set_ylabel(r'$c_p$ (mJ$\,$mol$^{-1}\,$K$^{-2}$)')

    ax.plot(T, Gamma1 * 2.85, 'ro', ms=3)
#    ax.plot(T, Gamma2 * 2.9, 'ro', ms=3)
    plt.show()

    # Save figure
    if print_fig:
        plt.savefig(save_dir + figname + '.pdf', dpi=100,
                    bbox_inches="tight", rasterized=True)
