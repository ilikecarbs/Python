#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 08:59:09 2018

@author: ilikecarbs

%%%%%%%%%%%%%%%%%%%%%%%%%%
   PhD_chapter_Concepts
%%%%%%%%%%%%%%%%%%%%%%%%%%

**Conceptual figures for thesis**

.. note::
        To-Do:
            -
"""

import utils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
from scipy.special import sph_harm
from scipy.stats import exponnorm
from scipy.optimize import curve_fit

import ARPES


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


def fig1(print_fig=True):
    """figure 1

    %%%%%%%%%%%%%%%%%%%%%%%%%%%
    Photoemission principle DOS
    %%%%%%%%%%%%%%%%%%%%%%%%%%%
    """

    figname = 'CONfig1'

    fig = plt.figure(figname, figsize=(8, 8), clear=True)

    ax = fig.add_axes([.2, .2, .6, .6])

    xx = np.linspace(0, 1.5, 100)
    xx_fill = np.linspace(0, np.sqrt(2), 100)
    yy = xx ** 2
    yy_fill = xx_fill ** 2

    x_off = 3.5
    y_off = 4.5
    yy_2 = yy + y_off
    yy_2_fill = yy_fill + y_off
    xx_2 = xx * utils.FDsl(yy_2, *[.05, 2+y_off, 1, 0, 0]) + x_off
    xx_2_fill = xx_fill * utils.FDsl(yy_2, *[.05, 2.2+y_off, 1, 0, 0]) + x_off

    core_1 = utils.lor(yy_2, *[.7+y_off, .03, .12, 0, 0, 0]) + x_off
    core_2 = utils.lor(yy_2, *[1.2+y_off, .03, .1, 0, 0, 0]) + x_off

    # plot DOS solid
    ax.plot(xx, yy + 2, 'k')
    ax.plot(xx_fill, 4*np.ones(len(xx_fill)), 'k-', lw=.5)
    ax.plot([0, 1.2], [.7, .7], 'k-', alpha=.5, lw=2)
    ax.plot([0, 1], [1.2, 1.2], 'k-', alpha=.5, lw=2)
    ax.fill_between(xx_fill, yy_fill + 2, 4, alpha=.1, color='k')

    # plot spectrum
    ax.plot(xx_2, yy_2 + 2, 'k')
    ax.plot(xx_2_fill, (4+y_off)*np.ones(len(xx_2_fill)), 'k-', lw=.5)
    ax.plot(core_1, yy_2, 'k')
    ax.plot(core_2, yy_2, 'k')
    ax.fill_between(xx_2_fill, yy_2_fill + 2, 4 + y_off, alpha=.5, color='C8')
    ax.fill_between(core_1, yy_2, 4 + y_off, alpha=.5, color='C8')
    ax.fill_between(core_2, yy_2, 4 + y_off, alpha=.5, color='C8')

    # plot arrow and excitation
    x_arr_1 = np.linspace(.88, 1.5, 50)
    y_arr_1 = np.sin(x_arr_1*50)/10 + x_arr_1*.7 + .9
    ax.plot(x_arr_1, y_arr_1, 'c-')
    ax.arrow(x_arr_1[0], y_arr_1[0], -.1, -.09,
             head_width=0.1, head_length=0.1,
             ec='c', fc='c', lw=2)
    ax.text(1.55, 2, r'$h\nu$')
    ax.plot([0, x_off], [y_off, y_off], **kwargs_ef)
    ax.plot(.5, 1.2, 'o', mec='k', mfc='w')
    ax.plot(.5, 1.2+y_off, 'ko')
    ax.plot([.5, x_off], [1.2+y_off, 1.2+y_off], **kwargs_ef)
    ax.arrow(.5, 1.2, 0, y_off-.2, head_width=0.1, head_length=0.1,
             fc='k', ec='k', lw=.5)
    x_arr_2 = np.linspace(.88, 1.5, 50) + .5
    y_arr_2 = np.sin(x_arr_2*50)/10 + x_arr_2*.7 + 3.3
    ax.plot(x_arr_2, y_arr_2, 'c-')
    ax.arrow(x_arr_2[0], y_arr_2[0], -.1, -.09,
             head_width=0.1, head_length=0.1,
             ec='c', fc='c', lw=2)
    ax.plot(1, 3.8, 'o', mec='k', mfc='w')
    ax.plot(1, 3.8+y_off, 'ko')
    ax.plot([1, x_off], [3.8+y_off, 3.8+y_off], **kwargs_ef)
    ax.arrow(1, 3.8, 0, y_off-.2, head_width=0.1, head_length=0.1,
             fc='k', ec='k', lw=.5)
    ax.text(2.05, 4.75, r'$h\nu$')

    # plot helper lines
    ax.plot([4.5, 9.], [1.2+y_off, 1.2+y_off], **kwargs_ef)
    ax.plot([7.7, 8.4], [y_off, y_off], **kwargs_ef)
    ax.plot([1, 9.], [1.2, 1.2], **kwargs_ef)
    ax.plot([np.sqrt(2), 8.4], [4, 4], **kwargs_ef)
    ax.arrow(8, 5, 0, .58, head_width=0.1, head_length=0.1,
             fc='k', ec='k', lw=.5)
    ax.arrow(8, 5, 0, -.4, head_width=0.1, head_length=0.1,
             fc='k', ec='k', lw=.5)
    ax.text(8.1, 5, r'$E_\mathrm{kin}$')
    ax.arrow(8, 4.25, 0, .15, head_width=0.1, head_length=0.1,
             fc='k', ec='k', lw=.5)
    ax.arrow(8, 4.25, 0, -.15, head_width=0.1, head_length=0.1,
             fc='k', ec='k', lw=.5)
    ax.text(8.1, 4.13, r'$\Phi$')
    ax.arrow(8, 3, 0, .88, head_width=0.1, head_length=0.1,
             fc='k', ec='k', lw=.5)
    ax.arrow(8, 3, 0, -1.7, head_width=0.1, head_length=0.1,
             fc='k', ec='k', lw=.5)
    ax.text(8.1, 2.5, r'$E_\mathrm{B}$')
    ax.arrow(8.8, 3, 0, 2.58, head_width=0.1, head_length=0.1,
             fc='k', ec='k', lw=.5)
    ax.arrow(8.8, 3, 0, -1.7, head_width=0.1, head_length=0.1,
             fc='k', ec='k', lw=.5)
    ax.text(8.9, 3.5, r'$h\nu$')

    # plot axis
    ax.arrow(0, 0, 0, 5, head_width=0.1, head_length=0.1,
             fc='k', ec='k', lw=1.5)
    ax.arrow(0, 0, 2.5, 0, head_width=0.1, head_length=0.1,
             fc='k', ec='k', lw=1.5)
    ax.arrow(0+x_off, 0+y_off, 0, 5, head_width=0.1, head_length=0.1,
             fc='k', ec='k', lw=1.5)
    ax.arrow(0+x_off, 0+y_off, 2.5, 0, head_width=0.1, head_length=0.1,
             fc='k', ec='k', lw=1.5)
    ax.set_xlim(-1, 10)
    ax.set_ylim(-1, 10)
    plt.axis('off')

    # add text
    ax.text(4.5, 5.8, 'core levels')
    ax.text(4.7, 7.5, 'valence band')
    ax.text(-.7, 3.9, r'$E_\mathrm{F}$')
    ax.text(-.7, 4.4, r'$E_\mathrm{vac}$')
    ax.text(.4, 5.9, r'$e^-$')
    ax.text(.9, 8.5, r'$e^-$')
    ax.text(-.8, 6.5, r'SAMPLE', fontdict=font)
    ax.text(4, 9, r'SPECTRUM', fontdict=font)
    ax.text(2.7, -.1, r'DOS$(\omega)$', fontdict=font)
    ax.text(2.7+x_off, -.1+y_off, r'DOS$(\omega)$', fontdict=font)
    ax.text(-.12, 5.2, r'$\omega$', fontdict=font)
    ax.text(-.12+x_off, 5.2+y_off, r'$\omega$', fontdict=font)
    plt.show()

    # Save figure
    if print_fig:
        plt.savefig(save_dir + figname + '.png', dpi=600, bbox_inches="tight")


def fig2(print_fig=True):
    """figure 2

    %%%%%%%%%%%%%%%%%%%%%
    Electron transmission
    %%%%%%%%%%%%%%%%%%%%%
    """

    figname = 'CONfig2'

    fig = plt.figure(figname, figsize=(8, 8), clear=True)

    ax = fig.add_axes([.2, .2, .6, .6])

    # Create a sphere
    k_i = 2.5
    k_f = 2

    k_th = np.pi/3
    off = 4
    th = np.linspace(0, np.pi, 100)
    x_i = k_i * np.cos(th)
    y_i = k_i * np.sin(th)
    x_f = k_f * np.cos(th)
    y_f = k_f * np.sin(th)
    kx_i = k_i * np.cos(k_th)
    ky_i = k_i * np.sin(k_th)
    ky_f = np.sqrt(k_f ** 2 - kx_i ** 2)

    kxs_i = np.linspace(-k_f, k_f, 100)
    kys_i = np.sqrt(k_i ** 2 - kxs_i ** 2)

    kx_p = np.linspace(0, k_f, 50)
    kx_n = np.linspace(0, -k_f, 50)
    ky_p = kx_p * (kys_i[-1] / k_f)
    ky_n = kx_n * -(kys_i[0] / k_f)

    ax.arrow(0, 0, kx_i-.1, ky_i-.15, head_width=0.1, head_length=0.1,
             fc='r', ec='r', lw=1.5)
    ax.arrow(0, 0, 0, ky_i-.1, head_width=0.1, head_length=0.1,
             fc='k', ec='k', lw=.5)
    ax.arrow(0, 0, kx_i-.06, 0, head_width=0.1, head_length=0.1,
             fc='k', ec='k', lw=.5)

    ax.arrow(0, off, kx_i-.1, ky_f-.15, head_width=0.1, head_length=0.1,
             fc='r', ec='r', lw=1.5)
    ax.arrow(0, off, 0, ky_f-.1, head_width=0.1, head_length=0.1,
             fc='k', ec='k', lw=.5)
    ax.arrow(0, off, kx_i-.06, 0, head_width=0.1, head_length=0.1,
             fc='k', ec='k', lw=.5)

    ax.plot(kxs_i, kys_i, 'C8-', lw=3)
    ax.plot([kxs_i[-1], kxs_i[-1]], [kys_i[-1], off], 'k--', lw=.5)
    ax.plot([kxs_i[0], kxs_i[0]], [kys_i[0], off], 'k--', lw=.5)
    ax.plot(kx_p, ky_p, 'k--', lw=.5)
    ax.plot(kx_n, ky_n, 'k--', lw=.5)
    ax.fill_between(kx_p, ky_p, kys_i[50:], facecolor='C8', alpha=.1,
                    edgecolor='w')
    ax.fill_between(kx_n, ky_n, kys_i[50:], facecolor='C8', alpha=.1,
                    edgecolor='w')
    ax.plot([kx_i, kx_i], [0, off+ky_f], **kwargs_ef)
    ax.plot(x_i, y_i, 'k-')
    ax.plot(x_f, y_f+off, 'C8-')
    ax.fill_between([-3, 3], [3.6, 3.6], [4, 4], color='C8', alpha=.2)
    ax.fill_between([-3, 3], [-.4, -.4], [0, 0], color='k', alpha=.2)
    ax.set_xlim(-4, 4)
    ax.set_ylim(-1, 7)

    ax.arrow(-k_f, 1, 0, .35, head_width=0.1, head_length=0.1,
             fc='k', ec='k', lw=.5)
    ax.arrow(-k_f, 1, 0, -.9, head_width=0.1, head_length=0.1,
             fc='k', ec='k', lw=.5)

    ax.plot(kx_i, ky_i, 'o', mec='k', mfc='w')
    ax.plot(kx_i, ky_f+off, 'o', mec='k', mfc='k')
    plt.axis('off')

    # add text
    ax.text(-2, .35, r'$\sqrt{\frac{2m V_0}{\hbar^2}}$', fontdict=font)
    ax.text(1.3, 5.7, r'$e^-$', fontdict=font)
    ax.text(.03, .7, r'$\theta_\mathrm{int}$')
    ax.text(.1, 4.4, r'$\theta$')
    ax.text(.65, 1.8, r'$\mathbf{K}_f$', color='r')
    ax.text(.7, 5.3, r'$\mathbf{k}_f$', color='r')
    ax.text(-.4, 1.8, r'$\mathbf{K}_f^\perp$')
    ax.text(-.4, 5.2, r'$\mathbf{k}_f^\perp$')
    ax.text(.8, -.3, r'$\mathbf{K}_f^\parallel$')
    ax.text(.8, 3.7, r'$\mathbf{k}_f^\parallel$')
    ax.text(-1.8, 3.73, 'Surface')
    ax.text(-1.8, 4.27, 'Vacuum')
    ax.text(-1.8, -.27, 'Bulk')
    ax.text(-.1, -.27, r'$\Gamma$')
    x1, y1 = 0, 1  # for angle text
    x2, y2 = 0.45, 0.8  # for angle text

    # angle text
    ax.annotate("",
                xy=(x1, y1), xycoords='data',
                xytext=(x2, y2), textcoords='data',
                arrowprops=dict(arrowstyle="-",
                                color="k",
                                shrinkA=0, shrinkB=0,
                                patchA=None,
                                patchB=None,
                                connectionstyle="arc3,rad=.25"
                                ))
    x1, y1 = 0, .7+off
    x2, y2 = 0.4, 0.5+off
    ax.annotate("",
                xy=(x1, y1), xycoords='data',
                xytext=(x2, y2), textcoords='data',
                arrowprops=dict(arrowstyle="-",
                                color="k",
                                shrinkA=0, shrinkB=0,
                                patchA=None,
                                patchB=None,
                                connectionstyle="arc3,rad=.25"
                                ))
    plt.show()

    # Save figure
    if print_fig:
        plt.savefig(save_dir + figname + '.png', dpi=600, bbox_inches="tight")


def fig3(print_fig=True):
    """figure 3

    %%%%%%%
    EDC/MDC
    %%%%%%%
    """

    figname = 'CONfig3'

    fig = plt.figure(figname, figsize=(8, 8), clear=True)

    ax1 = fig.add_subplot(2, 2, 1)
    ax1.set_position([.1, .5, .3, .4])
    ax1.tick_params(**kwargs_ticks)
    T = 48

    a = 3.885
    t1 = -0.23
    t2 = 0.087
    mu = -0.272
    W = 0.2
    gamma = 0.02
    A = 4
    kB = 8.617e-5

    w = np.linspace(0.05, -0.5, int(1e3))
    k = np.linspace(-1, 1, int(1e3))

    def sig(gamma, A, W, w):
        return (- complex(0, 1) * (gamma + A * w ** 2) / (1 + (w / W) ** 4)
                + 1 / np.sqrt(2) * w / W * (gamma * (1 + (w / W) ** 2)
                - A * W ** 2 * (1 - (w / W) ** 2)) / (1 + (w / W) ** 4))

    def Ximod(t1, t2, a, mu, k):
        return (2 * t1 * (np.cos(1 / np.sqrt(2) * k * a) +
                          np.cos(1 / np.sqrt(2) * k * a)) +
                4 * t2 * np.cos(1 / np.sqrt(2) * k * a) *
                np.cos(1 / np.sqrt(2) * k * a) - mu)

    def G(k, w, t1, t2, a, mu, gamma, A, W):
        return 1 / (w - Ximod(t1, t2, a, mu, k) - sig(gamma, A, W, w))

    [K, E] = np.meshgrid(k, w)

    model = (-1 / np.pi * np.imag(G(K, E, t1, t2, a, mu, gamma, A, W)) *
             utils.FDsl(E, *[T*kB, 0, 1, 0, 0]))

    # build MDC / EDC
    mdc_val, mdc_idx = utils.find(w, -.1)
    mdc = model[mdc_idx]
    edc_val, edc_idx = utils.find(k, .4)
    edc = model[:, edc_idx]

    # coherent / incoheren weight EDC
    p_edc_coh = np.array([-.05, 2e-2, 6.6e-1, 0, 0, 0])
    p_edc_inc = np.array([-.15, 1.1e-1, 5.5e-1, 0, 0, 0])
    f_coh = utils.lor(w, *p_edc_coh) * utils.FDsl(w, *[T*kB*2, 0, 1, 0, 0])
    f_coh[0] = 0
    f_coh[-1] = 0
    f_inc = utils.lor(w, *p_edc_inc) * utils.FDsl(w, *[T*kB*2, 0, 1, 0, 0])
    f_inc[0] = 0
    f_inc[-1] = 0

    # plot data
    c0 = ax1.pcolormesh(k, w, model, cmap=cm.bone_r)
    ax1.plot([k[0], k[-1]], [0, 0], **kwargs_ef)
    ax1.plot([k[0], k[-1]], [mdc_val, mdc_val], ls='-.', color='C8', lw=.5)
    ax1.plot([edc_val, edc_val], [w[0], w[-1]], ls='-.', color='C8', lw=.5)

    ax2 = fig.add_subplot(2, 2, 2)
    ax2.set_position([.1, .29, .3, .2])
    ax2.tick_params(**kwargs_ticks)
    ax2.plot(k, mdc, 'ko-', lw=1, ms=1)

    ax3 = fig.add_subplot(2, 2, 3)
    ax3.set_position([.41, .5, .2, .4])
    ax3.tick_params(**kwargs_ticks)
    ax3.plot([0, 100], [0, 0], **kwargs_ef)
    ax3.plot(edc, w, 'ko-', lw=1, ms=1)
    ax3.fill(f_coh, w, alpha=.5, color='b')
    ax3.fill(f_inc, w, alpha=.5, color='C0')

    # decorate axes
    ax1.set_xticklabels([])
    ax1.set_xlim(-1, 1)
    ax1.set_ylim(-.5, .05)
    ax1.set_ylabel(r'$\omega$', fontdict=font)
    ax2.set_yticks([])
    ax2.set_xlim(-1, 1)
    ax2.set_ylim(0, 1.1 * np.max(mdc))
    ax2.set_xlabel(r'$k$ $(\pi/a)$', fontdict=font)
    ax2.set_ylabel('MDC intensity', fontdict=font)
    ax3.set_yticklabels([])
    ax3.set_xticks([])
    ax3.set_xlim(0, 1.1 * np.max(edc))
    ax3.set_ylim(-.5, .05)
    ax3.set_xlabel('EDC intensity', fontdict=font)

    # add text
    ax1.text(-.95, .02, r'(a)', fontdict=font)
    ax2.text(-.95, 5.5, r'(b)', fontdict=font)
    ax3.text(.5, .02, r'(c)', fontdict=font)

    ax3.text(4, -.11, r'$\mathcal{A}_\mathrm{coh}\,(k, \omega)$',
             fontsize=12, color='b')
    ax3.text(2, -.2, r'$\mathcal{A}_\mathrm{inc}\,(k, \omega)$',
             fontsize=12, color='C0')

    # colorbar
    pos = ax3.get_position()
    cax = plt.axes([pos.x0+pos.width + 0.01,
                    pos.y0, 0.01, pos.height])
    cbar = plt.colorbar(c0, cax=cax, ticks=None)
    cbar.set_ticks([])
#    cbar.set_clim(np.min(D.map), np.max(D.map))
    plt.show()

    # Save figure
    if print_fig:
        plt.savefig(save_dir + figname + '.png', dpi=600, bbox_inches="tight")


def fig4(print_fig=True):
    """figure 4

    %%%%%%%%%%%%%%%%%%
    Experimental Setup
    %%%%%%%%%%%%%%%%%%
    """

    figname = 'CONfig4'

    fig = plt.figure(figname, figsize=(8, 8), clear=True)

    ax = fig.add_axes([.1, .1, .8, .8], projection='3d')

    ax.tick_params(**kwargs_ticks)

    # Create a sphere
    k_i = 3

    k_phi = np.pi/3
    k_th = np.pi/3

#    phi, theta = np.mgrid[0.0:0.5*np.pi:180j, 0.0:2.0*np.pi:720j]
#    x_i = k_i * np.sin(phi) * np.cos(theta)
#    y_i = k_i * np.sin(phi) * np.sin(theta)
#    z_i = k_i * np.cos(phi)

    x_phi = 1.1 * np.cos(np.linspace(0, k_phi, 100))
    y_phi = 1.1 * np.sin(np.linspace(0, k_phi, 100))
    z_phi = np.zeros(100)

    x_th = 1 * np.sin(np.linspace(0, k_th, 100)) * np.cos(k_phi)
    y_th = 1 * np.sin(np.linspace(0, k_th, 100)) * np.sin(k_phi)
    z_th = 1 * np.cos(np.linspace(0, k_th, 100))

    y_hv = np.linspace(-2, -.25, 100)
    x_hv = .2*np.sin(y_hv*50)
    z_hv = -y_hv

    kx_i = k_i * np.sin(k_th) * np.cos(k_phi)
    ky_i = k_i * np.sin(k_th) * np.sin(k_phi)
    kz_i = k_i * np.cos(k_th)

    ax.quiver(0, 0, 0, 0, 0, 2.5, arrow_length_ratio=.08,
              color='k')
    ax.quiver(0, 0, 0, 0, 2.5, 0, arrow_length_ratio=.06,
              color='k')
    ax.quiver(0, 0, 0, 2.5, 0, 0, arrow_length_ratio=.08,
              color='k')
    ax.quiver(0, 0, 0, kx_i-.1, ky_i-.1, kz_i-.1, arrow_length_ratio=.08, lw=2,
              color='r')
    ax.quiver(x_hv[-1], y_hv[-1], z_hv[-1], .1, .3, -.2,
              arrow_length_ratio=.6, color='c')

    ax.plot([0, 0], [0, 0], [0, 0], 'o', mec='k', mfc='w', ms=5)
    ax.plot([kx_i, kx_i], [ky_i, ky_i], [kz_i, kz_i],
            'o', mec='k', mfc='k', ms=5)
    ax.plot([kx_i, kx_i], [ky_i, ky_i], [0, kz_i], **kwargs_ef)
    ax.plot([0, kx_i], [0, ky_i], [kz_i, kz_i], **kwargs_ef)
    ax.plot([0, kx_i], [0, ky_i], [0, 0], **kwargs_ef)
    ax.plot([kx_i, kx_i], [0, ky_i], [0, 0], **kwargs_ef)
    ax.plot([0, kx_i], [ky_i, ky_i], [0, 0], **kwargs_ef)
    ax.plot([0, kx_i], [ky_i, ky_i], [0, 0], **kwargs_ef)
    ax.plot(x_phi, y_phi, z_phi, 'C0')
    ax.plot(x_th, y_th, z_th, 'C0')
    ax.plot(x_hv, y_hv, z_hv, 'c')
    ax.set_xlim([-3, 3])
    ax.set_ylim([-3, 3])
    ax.set_zlim([-2, 4])
    ax.set_aspect("equal")
    # ax.plot_surface(x_i, y_i, z_i, color="r", alpha=.05)

    # add text
    ax.text(0, -2.2, 2, r'$hv$', fontdict=font)
    ax.text(0, 1.65, 1., r'$e^-$', fontdict=font)
    ax.text(0, .1, .35, r'$\theta$', fontdict=font)
    ax.text(.75, .3, 0, r'$\phi$', fontdict=font)
    ax.text(2.9, 0, 0, r'$x$', fontdict=font)
    ax.text(0, 2.6, 0, r'$y$', fontdict=font)
    ax.text(0, -.1, 2.6, r'$z$', fontdict=font)
    ax.text(3.5, 1.5, -0.25, 'SAMPLE', fontdict=font)

    kwargs_cyl = {'alpha': .05, 'color': 'k'}  # keywords cylinder

    # Cylinder
    r = 3
    x_cyl = np.linspace(-r, r, 100)
    z_cyl = np.linspace(-1, 0, 100)
    X_cyl, Z_cyl = np.meshgrid(x_cyl, z_cyl)
    Y_cyl = np.sqrt(r**2 - X_cyl**2)

    x_cir = r * np.cos(np.linspace(0, 2*np.pi, 360))
    y_cir = r * np.sin(np.linspace(0, 2*np.pi, 360))

    R, Phi = np.meshgrid(np.linspace(0, r, 100), np.linspace(0, 2*np.pi, 100))

    X_cir = R * np.cos(Phi)
    Y_cir = R * np.sin(Phi)
    Z_ceil = np.zeros((100, 100))
    Z_floor = -np.ones((100, 100))

    # draw cylinder
    ax.plot_surface(X_cyl, Y_cyl, Z_cyl, **kwargs_cyl)
    ax.plot_surface(X_cyl, -Y_cyl, Z_cyl, **kwargs_cyl)
    ax.plot_surface(X_cir, Y_cir, Z_floor, **kwargs_cyl)
    ax.plot_surface(X_cir, Y_cir, Z_ceil, **kwargs_cyl)
    ax.plot(x_cir, y_cir, 'k-', alpha=.1)
    ax.plot(x_cir, y_cir, -1, 'k--', alpha=.1, lw=.5)
    plt.axis('off')
    ax.view_init(elev=20, azim=30)

    plt.show()

    # Save figure
    if print_fig:
        plt.savefig(save_dir + figname + '.png', dpi=600, bbox_inches="tight")


def fig5(print_fig=True):
    """figure 5

    %%%%%%%%%%%%%%%%
    eg, t2g orbitals
    %%%%%%%%%%%%%%%%
    """

    figname = 'CONfig5'

    # create figure
    fig = plt.figure(figname, figsize=(8, 8), clear=True)
    ax = fig.add_axes([.1, .1, .8, .8], projection='3d')

    theta_1d = np.linspace(0, np.pi, 300)
    phi_1d = np.linspace(0, 2*np.pi, 300)

    theta_2d, phi_2d = np.meshgrid(theta_1d, phi_1d)
    xyz_2d = np.array([np.sin(theta_2d) * np.sin(phi_2d),
                      np.sin(theta_2d) * np.cos(phi_2d),
                      np.cos(theta_2d)])

    colormap = cm.ScalarMappable(cmap=plt.get_cmap("PRGn"))
    colormap.set_clim(-.45, .45)

    l_ = 2  # angular momentum

    # build orbitals
    dz2 = sph_harm(0, l_, phi_2d, theta_2d)

    dxz = ((sph_harm(-1, l_, phi_2d, theta_2d)
           - sph_harm(1, l_, phi_2d, theta_2d))
           / np.sqrt(2))

    dyz = (1j * (sph_harm(-1, l_, phi_2d, theta_2d)
           + sph_harm(1, l_, phi_2d, theta_2d))
           / np.sqrt(2))

    dxy = (1j * (sph_harm(-2, l_, phi_2d, theta_2d)
           - sph_harm(2, l_, phi_2d, theta_2d))
           / np.sqrt(2))

    dx2y2 = ((sph_harm(-2, l_, phi_2d, theta_2d)
             + sph_harm(2, l_, phi_2d, theta_2d))
             / np.sqrt(2))

    dz2_r = np.abs(dz2.real)*xyz_2d
    dxz_r = np.abs(dxz.real)*xyz_2d
    dyz_r = np.abs(dyz.real)*xyz_2d
    dxy_r = np.abs(dxy.real)*xyz_2d
    dx2y2_r = np.abs(dx2y2.real)*xyz_2d

    orbitals_r = (dxy_r, dxz_r, dyz_r, dz2_r, dx2y2_r)
    orbitals = (dxy, dxz, dyz, dz2, dx2y2)

    # locations
    x = [0, 0, 0, 0, 0]
    y = [1.1, 2.6, 4.1, 1.85, 3.35]
    z = [1, 1, 1, 2.65, 2.65]

    # plot orbitals
    for i in range(5):
        ax.plot_surface(orbitals_r[i][0]+x[i], orbitals_r[i][1]+y[i],
                        orbitals_r[i][2]+z[i],
                        facecolors=colormap.to_rgba(orbitals[i].real),
                        rstride=2, cstride=2)

    # surfaces
    X_t2g = np.zeros((2, 2))
    z_t2g = [.2, 1.9]
    y_t2g = [.2, 4.9]
    Y_t2g, Z_t2g = np.meshgrid(y_t2g, z_t2g)

    X_eg = np.zeros((2, 2))
    z_eg = [1.9, 3.6]
    y_eg = [.2, 4.9]
    Y_eg, Z_eg = np.meshgrid(y_eg, z_eg)

    # plot surfaces
    ax.plot_surface(X_t2g, Y_t2g, Z_t2g, alpha=.2, color='b')
    ax.plot_surface(X_eg, Y_eg, Z_eg, alpha=.2, color='C0')
    ax.quiver(0, 0, 0, 0, 0, 1, arrow_length_ratio=.08,
              color='k')
    ax.quiver(0, 0, 0, 0, 1, 0, arrow_length_ratio=.06,
              color='k')
    ax.quiver(0, 0, 0, 1, 0, 0, arrow_length_ratio=.08,
              color='k')
    ax.set_xlim(1, 4)
    ax.set_ylim(2.5, 5.5)
    ax.set_zlim(1, 4)
    plt.axis('off')
    ax.view_init(elev=20, azim=30)

    # add text
    ax.text(1.25, 0, 0, '$x$', fontdict=font)
    ax.text(0, 1.1, -.05, '$y$', fontdict=font)
    ax.text(0, -.05, 1.1, '$z$', fontdict=font)

    ax.text(0, 1.05, 1.45, r'$d_{xy}$', fontdict=font)
    ax.text(0, 2.5, 1.5, r'$d_{yz}$', fontdict=font)
    ax.text(0, 3.85, 1.5, r'$d_{xz}$', fontdict=font)
    ax.text(0, 1.75, 3.35, r'$d_{z^2}$', fontdict=font)
    ax.text(0, 3.1, 3.05, r'$d_{x^2-y^2}$', fontdict=font)

    ax.text(0, .3, 3.3, r'$e_{g}$', fontsize=15, color='k')
    ax.text(0, .3, 1.6, r'$t_{2g}$', fontsize=15, color='k')

    plt.show()

    # Save figure
    if print_fig:
        plt.savefig(save_dir + figname + '.png', dpi=600, bbox_inches="tight")


def fig6(print_fig=True):
    """figure 6

    %%%%%%%%%%%%%%%%%%
    Manipulator angles
    %%%%%%%%%%%%%%%%%%
    """

    figname = 'CONfig6'

    fig = plt.figure(figname, figsize=(8, 8), clear=True)

    ax = fig.add_axes([.1, .1, .8, .8], projection='3d')

    k_i = 3
    phi, theta = np.mgrid[0.0:0.5*np.pi:180j, 0.0:2.0*np.pi:720j]
    x_i = k_i * np.sin(phi) * np.cos(theta)
    y_i = k_i * np.sin(phi) * np.sin(theta)
    z_i = k_i * np.cos(phi)

    # draw hemisphere
    ax.plot_surface(x_i, y_i, z_i, color='C8', alpha=.1)

    # detector angles
    angdg = np.linspace(-20, 20, 100)
    thdg = 10
    tidg = 20
    phidg = 0

    # angle x-axis
    phi = np.pi/8

    # angle indicators
    k = utils.det_angle(k_i, angdg, thdg, tidg, phidg)
    k_ti = utils.det_angle(2.2, np.linspace(0, thdg, 30), thdg, tidg, phidg)
    k_0 = utils.det_angle(k_i, 0, 0, tidg, phidg)
#    k_m = det_angle(.6, angdg, thdg, tidg, phidg)
    k_full = utils.det_angle(k_i, np.linspace(-90, 90, 200), 0, tidg, phidg)

    # angle indicator
    x_th_m = np.zeros(50)
    y_th_m = 1.5 * np.sin(np.linspace(0, tidg*np.pi/180, 50))
    z_th_m = 1.5 * np.cos(np.linspace(0, tidg*np.pi/180, 50))

    x_phi_m = 1.7 * np.sin(np.linspace(0, phi, 50))
    y_phi_m = 1.7 * np.cos(np.linspace(0, phi, 50))
    z_phi_m = np.zeros(50)

    # lines
    ax.quiver(0, 0, 0, 0, 0, k_i, color='k', arrow_length_ratio=.06, lw=2)
    ax.quiver(0, 0, 0, k_i*np.sin(phi), k_i*np.cos(phi), 0,
              color='k', arrow_length_ratio=.06, lw=2)
    ax.quiver(0, 0, 0, k_i*np.sin(phi-np.pi/2), k_i*np.cos(phi-np.pi/2), 0,
              color='k', arrow_length_ratio=.06, lw=2)
    ax.plot([0, 0], [0, k_i], [0, 0], 'k--', lw=1)
    ax.plot([0, k[0, 50]], [0, k[1, 50]], [0, k[2, 50]], 'k--', lw=1)
    ax.plot([0, k_0[0]], [0, k_0[1]], [0, k_0[2]], 'k--', lw=1)

    ax.plot(k_full[0], k_full[1], k_full[2], **kwargs_ef)
    ax.plot([0, k[0, -1]], [0, k[1, -1]], [0, k[2, -1]], **kwargs_ef)
    ax.plot([0, k[0, 0]], [0, k[1, 0]], [0, k[2, 0]], **kwargs_ef)
    ax.plot(k[0], k[1], k[2], 'r-', lw=3)
    ax.plot(k_ti[0], k_ti[1], k_ti[2], 'C0-')
    ax.plot(x_th_m, y_th_m, z_th_m, 'C0-')
    ax.plot(x_phi_m, y_phi_m, z_phi_m, 'C0-')

    # Cylinder
    kwargs_cyl = {'alpha': .05, 'color': 'k'}  # keywords cylinder
    r = 3
    x_cyl = np.linspace(-r, r, 100)
    z_cyl = np.linspace(-1, 0, 100)
    X_cyl, Z_cyl = np.meshgrid(x_cyl, z_cyl)
    Y_cyl = np.sqrt(r**2 - X_cyl**2)

    x_cir = r * np.cos(np.linspace(0, 2*np.pi, 360))
    y_cir = r * np.sin(np.linspace(0, 2*np.pi, 360))

    R, Phi = np.meshgrid(np.linspace(0, r, 100), np.linspace(0, 2*np.pi, 100))

    X_cir = R * np.cos(Phi)
    Y_cir = R * np.sin(Phi)
    Z_ceil = np.zeros((100, 100))
    Z_floor = -np.ones((100, 100))

    # draw cylinder
    ax.plot_surface(X_cyl, Y_cyl, Z_cyl, **kwargs_cyl)
    ax.plot_surface(X_cyl, -Y_cyl, Z_cyl, **kwargs_cyl)
    ax.plot_surface(X_cir, Y_cir, Z_floor, **kwargs_cyl)
    ax.plot_surface(X_cir, Y_cir, Z_ceil, **kwargs_cyl)
    ax.plot(x_cir, y_cir, 'k-', alpha=.1)
    ax.plot(x_cir, y_cir, -1, 'k--', alpha=.1, lw=.5)

    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 2])
    ax.set_zlim([-1, 3])

    plt.axis('off')
    ax.view_init(elev=20, azim=50)

    # add text
    ax.text(-.2, .2, 2.8, r'Detector angles  $\alpha$', fontsize=15, color='r')
    ax.text(1.5, 3.2, 0, '$x$', fontsize=15, color='k')
    ax.text(-2.9, 1.2, 0, '$y$', fontsize=15, color='k')
    ax.text(0, -.1, 3.1, '$z$', fontsize=15, color='k')
    ax.text(-.2, -.2, 1, r'$\Theta$', fontsize=15, color='k')
    ax.text(-.2, .53, 1.8, r'$\chi$', fontsize=15, color='k')
    ax.text(.48, 1.32, 0, r'$\Phi$', fontsize=15, color='k')
    ax.text(3., 3, 0, 'SAMPLE', fontsize=15, color='k')
    plt.show()

    # Save figure
    if print_fig:
        plt.savefig(save_dir + figname + '.png', dpi=600, bbox_inches="tight")


def fig7(print_fig=True):
    """figure 7

    %%%%%%%%%%%%
    Mirror plane
    %%%%%%%%%%%%
    """

    figname = 'CONfig7'

    fig = plt.figure(figname, figsize=(8, 8), clear=True)
    ax = fig.add_axes([.1, .1, .8, .8], projection='3d')
    ax.tick_params(**kwargs_ticks)

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
    dxy_r = np.abs(dxy.real)*xyz_2d

    ax.plot_surface(dxy_r[0]*3, dxy_r[1]*3,
                    dxy_r[2]*.0, alpha=.1,
                    facecolors=colormap.to_rgba(dxy.real),
                    rstride=2, cstride=2)

    X = np.zeros((2, 2))
    z = [0, 3.5]
    y = [-3, 3]
    Y, Z = np.meshgrid(y, z)
    ax.plot_surface(X, Y, Z, alpha=.2, color='C8')

    angdg = np.linspace(-15, 15, 100)
    tidg = 0
    k_1 = utils.det_angle(4, angdg, -40, tidg, 90)
    # k_2 = utils.det_angle(4, angdg, 0, 40, 0)
    y_hv = np.linspace(-2, -.25, 100)
    x_hv = .2*np.sin(y_hv*50)
    z_hv = -y_hv

    kx_i = k_i * np.sin(k_th) * np.cos(k_phi)
    ky_i = k_i * np.sin(k_th) * np.sin(k_phi)
    kz_i = k_i * np.cos(k_th)

    ax.quiver(0, 0, 0, 0, 0, 2.5, arrow_length_ratio=.08,
              color='k')
    ax.quiver(0, 0, 0, 0, 2.5, 0, arrow_length_ratio=.06,
              color='k')
    ax.quiver(0, 0, 0, 2.5, 0, 0, arrow_length_ratio=.08,
              color='k')
    ax.quiver(0, 0, 0, kx_i-.1, ky_i-.1, kz_i-.1, arrow_length_ratio=.08, lw=2,
              color='r')
    ax.quiver(x_hv[-1], y_hv[-1], z_hv[-1], .1, .3, -.2,
              arrow_length_ratio=.6, color='c')

    ax.quiver(0, -1.5, 1.5, 0, .7, .7, arrow_length_ratio=.2,
              color='b', lw=2)
    ax.quiver(0, -1.5, 1.5, .8, 0, 0, arrow_length_ratio=.2,
              color='b', lw=2)
    ax.plot([0, 0], [0, 0], [0, 0], 'o', mec='k', mfc='w', ms=5)
    ax.plot([kx_i, kx_i], [ky_i, ky_i], [kz_i, kz_i],
            'o', mec='k', mfc='k', ms=5)
    ax.plot([0, k_1[0, 0]], [0, k_1[1, 0]], [0, k_1[2, 0]], **kwargs_ef)
    ax.plot([0, k_1[0, -1]], [0, k_1[1, -1]], [0, k_1[2, -1]], **kwargs_ef)
    ax.plot(k_1[0], k_1[1], k_1[2], 'r-', lw=3)
    # ax.plot(k_2[0], k_2[1], k_2[2], 'r-', lw=3, alpha=.1)
    ax.plot(x_hv, y_hv, z_hv, 'c', lw=2)
    ax.set_xlim([-3, 3])
    ax.set_ylim([-3, 3])
    ax.set_zlim([-2, 4])
    ax.set_aspect("equal")
    # ax.plot_surface(x_i, y_i, z_i, color="r", alpha=.05)

    # add text
    ax.text(0, -2.2, 2, r'$hv$', fontdict=font)
    ax.text(0, 2.3, 2., r'$e^-$', fontdict=font)
    ax.text(2.9, 0, 0, r'$x$', fontdict=font)
    ax.text(0, 2.6, 0, r'$y$', fontdict=font)
    ax.text(0, -.1, 2.6, r'$z$', fontdict=font)
    ax.text(2.8, 2.5, -0.25, 'SAMPLE', fontdict=font)
    ax.text(2., 1.3, 0, r'$d_{xy}$', fontdict=font)
    ax.text(0, -2.6, 2.6, 'Mirror plane', fontdict=font)
    ax.text(1.9, 0, 1.85, r'$\bar{\sigma}$', fontdict=font)
    ax.text(.8, 0, 2.15, r'$\bar{\pi}$', fontdict=font)
    ax.text(0, 2.2, 3.5, r'Detector', fontdict=font)
    kwargs_cyl = {'alpha': .05, 'color': 'k'}  # keywords cylinder

    # Cylinder
    r = 3
    x_cyl = np.linspace(-r, r, 100)
    z_cyl = np.linspace(-1, 0, 100)
    X_cyl, Z_cyl = np.meshgrid(x_cyl, z_cyl)
    Y_cyl = np.sqrt(r**2 - X_cyl**2)

    x_cir = r * np.cos(np.linspace(0, 2*np.pi, 360))
    y_cir = r * np.sin(np.linspace(0, 2*np.pi, 360))

    R, Phi = np.meshgrid(np.linspace(0, r, 100), np.linspace(0, 2*np.pi, 100))

    X_cir = R * np.cos(Phi)
    Y_cir = R * np.sin(Phi)
    Z_ceil = np.zeros((100, 100))
    Z_floor = -np.ones((100, 100))

    # draw cylinder
    ax.plot_surface(X_cyl, Y_cyl, Z_cyl, **kwargs_cyl)
    ax.plot_surface(X_cyl, -Y_cyl, Z_cyl, **kwargs_cyl)
    ax.plot_surface(X_cir, Y_cir, Z_floor, **kwargs_cyl)
    ax.plot_surface(X_cir, Y_cir, Z_ceil, **kwargs_cyl)
    ax.plot(x_cir, y_cir, 'k-', alpha=.1)
    ax.plot(x_cir, y_cir, -1, 'k--', alpha=.1, lw=.5)
    plt.axis('off')
    ax.view_init(elev=20, azim=50)

    plt.show()

    # Save figure
    if print_fig:
        plt.savefig(save_dir + figname + '.png', dpi=600, bbox_inches="tight")


def fig8(print_fig=True):
    """figure 8

    %%%%%%%%%%%%%%%%%%
    Data normalization
    %%%%%%%%%%%%%%%%%%
    """

    figname = 'CONfig8'

    mat = 'CSRO20'
    year = '2017'
    sample = 'S6'
    gold = '62091'

    D = ARPES.DLS(gold, mat, year, sample)
    D.norm(gold=gold)

    fig = plt.figure(figname, figsize=(8, 8), clear=True)
    ax1 = fig.add_subplot(321)
    ax1.set_position([.1, .5, .35, .35])
    ax1.tick_params(**kwargs_ticks)

    # plot data
    c0 = ax1.contourf(D.ang, D.en, np.transpose(D.int), 200, **kwargs_ex)

    # decorate axes
    ax1.set_xticklabels([])
    ax1.set_ylabel(r'$E_\mathrm{kin}$ (eV)', fontdict=font)

    ax2 = fig.add_subplot(322)
    ax2.set_position([.1, .14, .35, .35])
    ax2.tick_params(**kwargs_ticks)

    # plot data
    ax2.contourf(D.angs, D.en_norm, D.int_norm, 200, **kwargs_ex)
    ax2.plot([D.ang[0], D.ang[-1]], [0, 0], **kwargs_ef)

    # decorate axes
    ax2.set_ylabel(r'$\omega$ (eV)', fontdict=font)
    ax2.set_xlabel(r'Detector angles', fontdict=font)

    # add text
    ax1.text(-15, 17.652, '(a)', fontdict=font)
    ax2.text(-15, 0.02, '(b)', fontdict=font)

    # colorbar
    pos = ax1.get_position()
    cax = plt.axes([pos.x0, pos.y0+pos.height+.01,
                    pos.width, .01])
    cbar = plt.colorbar(c0, cax=cax, ticks=None, orientation='horizontal')
    cbar.set_ticks([])

    # some constant
    Ef_ini = 17.645
    T_ini = 6
    bnd = 1
    ch = 300

    kB = 8.6173303e-5  # Boltzmann constant

    # create figure
    ax3 = fig.add_subplot(323)
    ax3.set_position([.55, .63, .35, .22])
    enval, inden = utils.find(D.en, Ef_ini-0.12)  # energy window

    # plot data
    ax3.plot(D.en[inden:], D.int[ch, inden:], 'bo', ms=2)

    # initial guess
    p_ini_FDsl = [T_ini * kB, Ef_ini, np.max(D.int[ch, :]), 20, 0]

    # Placeholders
    T_fit = np.zeros(len(D.ang))
    Res = np.zeros(len(D.ang))
    Ef = np.zeros(len(D.ang))
    norm = np.zeros(len(D.ang))

    # Fit loop
    for i in range(len(D.ang)):
        try:
            p_FDsl, c_FDsl = curve_fit(utils.FDsl, D.en[inden:],
                                       D.int[i, inden:], p_ini_FDsl)
        except RuntimeError:
            print("Error - convergence not reached")

        # Plots data at this particular channel
        if i == ch:
            ax3.plot(D.en[inden:], utils.FDsl(D.en[inden:],
                     *p_FDsl), 'r-')

        T_fit[i] = p_FDsl[0] / kB
        Res[i] = np.sqrt(T_fit[i] ** 2 - T_ini ** 2) * 4 * kB
        Ef[i] = p_FDsl[1]  # Fit parameter

    # Fit Fermi level fits with a polynomial
    p_ini_poly2 = [Ef[ch], 0, 0, 0]
    p_poly2, c_poly2 = curve_fit(utils.poly_2, D.ang[bnd:-bnd],
                                 Ef[bnd:-bnd], p_ini_poly2)
    Ef_fit = utils.poly_2(D.ang, *p_poly2)

    # boundaries if strong curvature in Fermi level
    mx = np.max(D.en) - np.max(Ef_fit)
    mn = np.min(Ef_fit) - np.min(D.en)
    for i in range(len(D.ang)):
        mx_val, mx_idx = utils.find(D.en, Ef_fit[i] + mx)
        mn_val, mn_idx = utils.find(D.en, Ef_fit[i] - mn)
        norm[i] = np.sum(D.int[i, mn_idx:mx_idx])  # normalization

    # Plot data
    ax4 = fig.add_subplot(324)
    ax4.set_position([.55, .14, .35, .22])
    ax4.plot(D.ang, Res * 1e3, 'bo', ms=3)
    print("Resolution ~" + str(np.mean(Res)) + "eV")
    ax5 = fig.add_subplot(325)
    ax5.set_position([.55, .37, .35, .22])
    ax5.plot(D.ang, Ef, 'bo', ms=3)
    ax5.plot(D.ang[bnd], Ef[bnd], 'ro')
    ax5.plot(D.ang[-bnd], Ef[-bnd], 'ro')
    ax5.plot(D.ang, Ef_fit, 'r-')

    # decorate axes
    ax3.tick_params(**kwargs_ticks)
    ax3.set_ylim(0, 1400)
    ax4.tick_params(**kwargs_ticks)
    ax5.set_xticklabels([])
    ax3.xaxis.set_label_position('top')
    ax3.tick_params(labelbottom='off', labeltop='on')
    ax5.tick_params(**kwargs_ticks)
    ax3.set_xlabel(r'$\omega$', fontdict=font)
    ax3.set_ylabel('Intensity (a.u.)', fontdict=font)
    ax4.set_ylabel('Resolution (meV)', fontdict=font)
    ax4.set_xlabel('Detector angles', fontdict=font)
    ax5.set_ylabel(r'$\omega$ (eV)', fontdict=font)
    ax5.set_ylim(D.en[0], D.en[-1])

    # add text
    ax3.text(17.558, 1250, '(c)', fontdict=font)
    ax5.text(-17, 17.648, '(d)', fontdict=font)
    ax4.text(-17, 12.1, '(e)', fontdict=font)

    plt.show()

    # Save figure
    if print_fig:
        plt.savefig(save_dir + figname + '.png', dpi=600, bbox_inches="tight")
