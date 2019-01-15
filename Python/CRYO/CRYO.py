#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 11:14:47 2018

@author: denyssutter
"""

import numpy as np
import matplotlib.pyplot as plt
import CRYO_utils as utils
import pandas as pd
import os
from scipy.optimize import curve_fit


# Directory paths
save_dir = '/Users/denyssutter/Documents/CondenZero/Figures/'
data_dir = '/Users/denyssutter/Documents/CondenZero/Data/'
home_dir = '/Users/denyssutter/Documents/CondenZero/'

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

kwargs_ticks = {'bottom': True,
                'top': True,
                'left': True,
                'right': True,
                'direction': 'in',
                'length': 1.5,
                'width': .5,
                'colors': 'black'}


def FirstCool_v1(print_fig=True):

    figname = 'xFig_FirstCool_v1'

    fig = plt.figure(figname, figsize=(6, 6), clear=True)

    ax = fig.add_axes([.1, .1, .8, .8])
    ax.tick_params(**kwargs_ticks)
    axi1 = fig.add_axes([.5, .5, .35, .35])
    axi1.tick_params(**kwargs_ticks)
    axi2 = fig.add_axes([.68, .15, .2, .2])
    axi2.tick_params(**kwargs_ticks)

    os.chdir(data_dir)
    data = pd.read_csv('Data_23052018_1p4K.csv').values

    t0 = 164
    t = data[:, 0] / 60 - t0
    T = data[:, 2]

    t0_val, t0_idx = utils.find(t, 0)

    # plot figure
    ax.plot(t, T, 'k-')
    ax.plot([-2, -2], [0, 100], 'k--', lw=1)
    ax.plot([19, 19], [0, 100], 'k--', lw=1)
    ax.plot([-2, 19], [100, 100], 'k--', lw=1)
    ax.plot([-2, -21], [100, 156], 'k--', lw=1)
    ax.plot([19, 92], [100, 156], 'k--', lw=1)

    ax.plot([90, 90], [0, 6], 'k--', lw=1)
    ax.plot([108, 108], [0, 6], 'k--', lw=1)
    ax.plot([90, 108], [6, 6], 'k--', lw=1)
    ax.plot([90, 36], [6, 20], 'k--', lw=1)
    ax.plot([108, 102], [6, 20], 'k--', lw=1)

    ax.fill_between(t[:t0_idx], 0, T[:t0_idx], color='C0', alpha=.3)
    ax.set_xlim(-150, 108)
    ax.set_ylim(0, 1.05 * T[0])
    ax.set_ylabel(r'$T$ (K)', fontdict=font)
    ax.set_xlabel('$t-t_0$ (min)', fontdict=font)
    ax.grid(True, alpha=.2)

    ax.text(-110, 80, r'LN$_2$ cooldown', fontdict=font)

    axi1.plot(t, T, 'k-')
    axi1.plot([t[0], t[-1]], [4.2, 4.2], ls='--', color='C1')
    axi1.fill_between(t[:t0_idx], 0, T[:t0_idx], color='C0', alpha=.3)
    axi1.text(10, 10, r'$T=4.2\,$K', color='C1', fontsize=12)
    axi1.text(4, 60, 'LHe$_4$ cooldown', fontdict=font)
    axi1.set_xlim(-2, 19)
    axi1.set_ylim(0, 100)

    axi2.plot(t, T, 'k-')

    axi2.plot([t[0], t[-1]], [1.5, 1.5], 'r--')
    axi2.set_xlim(90, 108)
    axi2.set_ylim(0, 6)
    axi2.text(93, .5, r'$T_\mathrm{base}\approx 1.5\,$K',
              color='r', fontsize=12)
    axi2.text(91, 4.7, '1K-pot pumpdown')
    axi2.set_xticklabels([])

    plt.show()

    # Save figure
    if print_fig:
        plt.savefig(save_dir + figname + '.pdf', dpi=100,
                    bbox_inches="tight", rasterized=True)


def FirstCool_v2(print_fig=True):

    figname = 'xFig_FirstCool_v2'

    fig = plt.figure(figname, figsize=(8, 8), clear=True)

    ax1 = fig.add_axes([.3, .3, .4, .4])
    ax1.tick_params(**kwargs_ticks)
    ax2 = fig.add_axes([.45, .45, .2, .2])
    ax2.tick_params(**kwargs_ticks)

    os.chdir(data_dir)
    data = pd.read_csv('Data_23052018_1p4K.csv').values

    t0 = 164
    t = data[:, 0] / 60 - t0
    T = data[:, 2]

    t0_val, t0_idx = utils.find(t, 0)
    t1_val, t1_idx = utils.find(t, 12)

    ax1.plot(t, T, 'k-')
    ax1.plot([t[0], 70], [4.2, 4.2], ls='--', color='b')
    ax1.plot([12, 12], [0, 15], ls='--', color='b')
    ax1.plot([90, 90], [0, 6], 'k--', lw=1)
    ax1.plot([108, 108], [0, 6], 'k--', lw=1)
    ax1.plot([90, 108], [6, 6], 'k--', lw=1)
    ax1.plot([90, 34], [6, 38], 'k--', lw=1)
    ax1.plot([108, 93], [6, 38], 'k--', lw=1)
    ax1.fill_between(t[:t0_idx-1], 0, T[:t0_idx-1],
                     color='steelblue', alpha=.2)
    ax1.fill_between(t[t0_idx-1:t1_idx], 0, T[t0_idx-1:t1_idx],
                     color='b', alpha=.5)
    ax1.text(-8, 65, 'pre-cooling with LN$_2$', rotation=90, color='steelblue')
    ax1.text(9.5, 60, r'cool-down time $\simeq 12$ min',
             rotation=90, color='b')
    ax1.text(22, 8, r'$T_\mathrm{L^4He}=4.2\,$K', color='b', fontsize=12)
    ax1.set_ylabel(r'Temperature $T$ (K)', fontdict=font)
    ax1.set_xlabel('Time $t$ (min)', fontdict=font)
    ax1.set_xlim(-10, 108)
    ax1.set_ylim(0, 100)

    ax2.plot(t, T, 'k-')
    ax2.plot([t[0], t[-1]], [1.5, 1.5], 'r--')
    ax2.set_xlim(90, 108)
    ax2.set_ylim(0, 6)
    ax2.text(93, .5, r'$T_\mathrm{base}\approx 1.5\,$K',
             color='r', fontsize=12)
    ax2.text(91, 4.9, '1K-pot pumpdown')
    ax2.set_xticklabels([])

    plt.show()

    # Save figure
    if print_fig:
        plt.savefig(save_dir + figname + '.pdf', dpi=100,
                    bbox_inches="tight", rasterized=True)


def Stabilization(print_fig=True):

    figname = 'xFig_Stabilization'

    fig = plt.figure(figname, figsize=(6, 6), clear=True)

    ax1 = fig.add_subplot(221)
    ax1.tick_params(**kwargs_ticks)

    ax2 = fig.add_subplot(222)
    ax2.tick_params(**kwargs_ticks)

    ax3 = fig.add_subplot(223)
    ax3.tick_params(**kwargs_ticks)

    os.chdir(data_dir)
    data = pd.read_csv('Data_23052018_1p4K.csv').values

    t0 = 164
    t = data[:, 0] / 60 - t0

    T = data[:, 2]

    ti_val, ti_idx = utils.find(t, 39)
    tf_val, tf_idx = utils.find(t, 82)

    t_low = t[ti_idx:tf_idx]
    T_low = T[ti_idx:tf_idx]

    ax1.plot(t, T, 'k-')
    ax1.plot([ti_val, ti_val], [0, 50], 'r-')
    ax1.plot([tf_val, tf_val], [0, 50], 'r-')

    ax1.set_xlim(-150, 108)
    ax1.set_ylim(0, 1.05 * T[0])
    ax1.set_ylabel(r'$T$ (K)', fontdict=font)
    ax1.set_xlabel('$t-t_0$ (min)', fontdict=font)
    ax1.grid(True, alpha=.2)

    ax2.plot(t_low, T_low, 'C0.')

    n, bins, patches = ax3.hist(T_low, density=True)

    T_bin = bins[:-1] + np.diff(bins)/2
    mean = np.mean(T_low)
    std_dev = np.std(T_low)
    temp = np.linspace(bins[0], bins[-1], 200)
    p_i = np.array([mean, std_dev, 17, 0, 0, 0])

    # fit boundaries
    eps = 1e-9
    bounds_bot = np.concatenate((p_i[0:-3] - np.inf, p_i[-3:] - eps))
    bounds_top = np.concatenate((p_i[0:-3] + np.inf, p_i[-3:] + eps))
    p_bounds = (bounds_bot, bounds_top)

    popt, pcov = curve_fit(utils.gauss, T_bin, n, p0=p_i, bounds=p_bounds)
    fit = utils.gauss(temp, *popt)

    ax3.plot(T_bin, n, 'r.')
    ax3.plot(temp, fit, 'r-')
    ax3.arrow(popt[0], popt[2]/2, -popt[1], 0, head_width=.5,
              head_length=.002, fc='k', ec='k', zorder=2)
    ax3.arrow(popt[0], popt[2]/2, popt[1], 0, head_width=.5,
              head_length=.002, fc='k', ec='k', zorder=2)
    plt.show()

    print(popt[1])

    # Save figure
    if print_fig:
        plt.savefig(save_dir + figname + '.pdf', dpi=100,
                    bbox_inches="tight", rasterized=True)


def Logo(print_fig=True):

    figname = 'Logo'
    fig = plt.figure(figname, figsize=(6, 6), clear=True)
    ax = fig.add_axes([.2, .4, .4, .2])

    kB = 8.6173303e-5
    T = np.array([0, 4, 8, 12, 18])
    cols = ['navy', 'steelblue', 'lightblue', 'powderblue', 'lightcyan']
    x = np.linspace(-.006, .006, 300)

    y = np.zeros((len(T), len(x)))

    for i in range(len(T)):
        y[i] = utils.FDsl(x, kB*T[i], 0, 1, 0, 0)
        ax.plot(x, y[i], color=cols[i], lw=5)
    ax.plot(x, y[0], color='navy', lw=5)
    # ax.text(-.005, .45, r'Conden$\,\,$Zero', fontsize=25)
    plt.axis('off')
    plt.show()

    # Save figure
    if print_fig:
        plt.savefig(save_dir + figname + '.pdf', dpi=100,
                    bbox_inches="tight", rasterized=True)


def Logo2(print_fig=True):

    figname = 'Logo2'
    fig = plt.figure(figname, figsize=(3, 3), clear=True)
    ax = fig.add_axes([0, 0, 1, 1])

    CZ_orange = np.array([205, 131, 58])/256
#    CZ_gray = np.array([120, 120, 118])/156
#    CZ_black = np.array([71, 71, 69])/256
    CZ = np.array([35, 54, 58])/256
    kB = 8.6173303e-5
    T = np.array([0, 8])
#    cols = ['lightblue', 'steelblue', 'navy']
    a = .003
    x = np.linspace(-a, a, 300)

    y = np.zeros((len(T), len(x)))

    circle = plt.Circle((0, a), .0054, color=CZ)
#    x_c = .0054*np.cos(np.linspace(0, 2*np.pi, 600))
#    y_c = .0054*np.sin(np.linspace(0, 2*np.pi, 600))+a

    for i in range(len(T)):
        y[i] = utils.FDsl(x, kB*T[i], 0, 2*a, 0, 0)
#    ax.plot(x[4:-4], y[1][4:-4], '-', color=CZ_orange, lw=20)
#    ax.plot(x, y[0], color='w', lw=20)
    ax.plot(x, y[0], color='w', lw=20)
    ax.plot(x[4:-4], y[1][4:-4], '-', color=CZ_orange, lw=20)
    ax.plot(x[:100], y[0][:100], color='w', lw=20)
    ax.plot(x[-100:], y[0][-100:], color='w', lw=20)
#    ax.text(-.005, .45, r'Conden$\,\,$Zero', fontsize=25)
#    ax.plot(x[110:190], y[1][110:190], '-', color=CZ_orange, lw=20)
    ax.add_artist(circle)
    ax.set_xlim(-.0055, .0055)
    ax.set_ylim(-.0025, .0085)
#    ax.set_rasterization_zorder(2)
    plt.axis('off')
    plt.show()

    # Save figure
    if print_fig:
        plt.savefig(save_dir + figname + '.pdf', dpi=100,
                    bbox_inches="tight", rasterized=True)
        plt.savefig(save_dir + figname + '.png', dpi=1000,
                    bbox_inches="tight")
