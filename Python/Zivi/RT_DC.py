#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 15:35:59 2018

@author: denyssutter
"""

import numpy as np
import matplotlib.pyplot as plt
import utils_RTDC as utils


# Directory paths
save_dir = '/Users/denyssutter/Documents/library/Python/Zivi/'

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


def Cell_Def(Q=1.2e-11, r_0=7e-6, print_fig=True):
    """Plots model of cell deformation

    **model deformation of elastic sphere and
    elastic shell with surface tension**

    Args
    ----
    :Q:             liquid flow
    :r_0:           cell radius
    :print_fig:     produce figure


    Return
    ------
    Plot
    """

    figname = 'Cell_Def'

    """
    %%%%%%%%%%%%%%%%%%%%%%%
       Other parameters
    %%%%%%%%%%%%%%%%%%%%%%%
    """

    l_0 = 2e-5  # channel side length
    R_0 = 1.094 * l_0 / 2  # equivalent radius
    K_2 = 2.096  # constant

    u = K_2 * Q / l_0**2  # Poiseuille flow at inifinity (liquid velocity)

    eh = 3.4e-3  # unit of stiffness of shell
    e_0 = 270  # unit of stiffness of sphere

    Eh = eh * 20  # multiples of the unit
    E_0 = e_0 * 20  # muplitples of the unit

    nu = .5  # compressibility (0.5 ~incompressible)
    gamma = 0.1 * Eh  # surface tension
    eta = .015  # viscosity
    lambd = r_0 / R_0  # cell radius relative to channel radius

    N = 20  # number of equations taken into account, N=20 high enough

    """
    %%%%%%%%%%%%%%%
       Modelling
    %%%%%%%%%%%%%%%
    """

    fn, v_equil = utils.f_n(N=N, lambd=lambd)  # expansion coefficients
    gn, v_equil = utils.g_n(N=N, lambd=lambd)  # expansion coefficients

    v_0 = u / v_equil  # velocity of the cell

    sig_c = eta * v_0 / r_0  # characteristic stress
    th = np.linspace(0, 2*np.pi, 600)  # polar angle

    # keyword arguments for calculations
    kwargs_sh = {'th': th, 'gamma': gamma, 'Eh': Eh, 'sig_c': sig_c,
                 'nu': nu, 'r_0': r_0, 'fn': fn, 'gn': gn}

    kwargs_sp = {'th': th, 'E_0': E_0, 'sig_c': sig_c,
                 'nu': nu, 'r_0': r_0, 'fn': fn, 'gn': gn}

    # extract deformation
    A_sh, d_sh, x_sh, z_sh = utils.deformation_sh(**kwargs_sh)
    u_r_sh, u_th_sh = utils.displacement_sh(**kwargs_sh)
    A_sp, d_sp, x_sp, z_sp = utils.deformation_sp(**kwargs_sp)
    u_r_sp, u_th_sp = utils.displacement_sp(**kwargs_sp)

    """
    %%%%%%%%%%%%%%
       Plotting
    %%%%%%%%%%%%%%
    """
    x0 = r_0*np.cos(th)
    z0 = r_0*np.sin(th)

    fig = plt.figure(figname, figsize=(6, 6), clear=True)
    ax_1 = fig.add_subplot(221)
    ax_1.tick_params(**kwargs_ticks)
    ax_1.set_position([.13, .51, .4, .4])
    ax_1.plot(th/np.pi, u_r_sh/r_0, 'k-')
    ax_1.plot(th/np.pi, u_th_sh/r_0, 'k--')
    ax_1.set_xlim(0, 1)
    ax_1.set_xticklabels([])
    ax_1.set_ylabel(r'$u_i / r_0$', fontdict=font)
    ax_1.legend(('elastic shell, $u_r$', r'elastic shell, $u_\theta$'), loc=1)

    ax_2 = fig.add_subplot(222)
    ax_2.tick_params(**kwargs_ticks)
    ax_2.set_position([.54, .51, .4, .4])
    ax_2.plot(x_sh, z_sh, 'ko', ms=1)
    ax_2.plot(x0, z0, 'k--')
    ax_2.fill_between(x_sh, 0, z_sh, color='C8', alpha=1)
    ax_2.plot([0, 0], [-1.3*r_0, 1.3*r_0], 'k-.', lw=.5)
    ax_2.set_xticks([])
    ax_2.set_yticks([])
    ax_2.set_xlim(-1.7*r_0, 1.7*r_0)
    ax_2.set_ylim(-1.7*r_0, 1.7*r_0)
    ax_2.text(-.4*r_0, 0*r_0, r'$d_\mathrm{sh} =$'+str(np.round(d_sh, 3)),
              color='k', fontsize=12)
    ax_2.text(-.1*r_0, 1.38*r_0,
              r'$Eh =$'+str(1e3*np.round(Eh, 4))+r'$\,$nN/$\mu$m',
              color='k', fontsize=12)

    ax_3 = fig.add_subplot(223)
    ax_3.tick_params(**kwargs_ticks)
    ax_3.set_position([.13, .08, .4, .4])
    ax_3.plot(th/np.pi, u_r_sp/r_0, 'r-')
    ax_3.plot(th/np.pi, u_th_sp/r_0, 'r--')
    ax_3.set_xlim(0, 1)
    ax_3.set_xlabel(r'$\theta/\pi$', fontdict=font)
    ax_3.set_ylabel(r'$u_i / r_0$', fontdict=font)
    ax_3.legend(('elastic sphere, $u_r$', r'elastic sphere, $u_\theta$'),
                loc=1)

    ax_4 = fig.add_subplot(224)
    ax_4.tick_params(**kwargs_ticks)
    ax_4.set_position([.54, .08, .4, .4])
    ax_4.plot(x_sp, z_sp, 'ro', ms=1)
    ax_4.plot(x0, z0, 'k--')
    ax_4.fill_between(x_sp, 0, z_sp, color='C8', alpha=1)
    ax_4.plot([0, 0], [-1.3*r_0, 1.3*r_0], 'k-.', lw=.5)
    ax_4.set_xticks([])
    ax_4.set_yticks([])
    ax_4.set_xlim(-1.7*r_0, 1.7*r_0)
    ax_4.set_ylim(-1.7*r_0, 1.7*r_0)
    ax_4.text(-.4*r_0, 0*r_0, r'$d_\mathrm{sp} =$'+str(np.round(d_sp, 3)),
              color='r', fontsize=12)
    ax_4.text(.2*r_0, 1.38*r_0, r'$E =$'+str(1e-3*np.round(E_0, 3))+r'$\,$kPa',
              color='k', fontsize=12)

    plt.show()

    # Save figure
    if print_fig:
        plt.savefig(save_dir + figname + '.pdf', dpi=100,
                    bbox_inches="tight", rasterized=True)
