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


# Directory paths
save_dir = '/Users/denyssutter/Documents/Startup/Figures/'
data_dir = '/Users/denyssutter/Documents/Startup/Data/'
home_dir = '/Users/denyssutter/Documents/Startup/'

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


def FirstCool(print_fig=True):

    figname = 'xFig_FirstCool'

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
