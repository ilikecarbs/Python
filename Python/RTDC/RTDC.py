#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 15:35:59 2018

@author: denyssutter

%%%%%%%%%%
   RTDC
%%%%%%%%%%

**Scripts for figures and called by RTDC_main**

.. note::
        To-Do:
            -
"""

import numpy as np
import matplotlib.pyplot as plt
import RTDC_utils as utils
import os


# Directory paths
save_dir = '/Users/denyssutter/Documents/Denys/Zivi/Figs/'
data_dir = '/Users/denyssutter/Documents/Denys/Zivi/data/'

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


def Cell_Def(Q=1.2e-11, r_0=7e-6, save_data=False, print_fig=True):
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

    Eh = eh * 6.5  # multiples of the unit
    E_0 = e_0 * 9.5  # muplitples of the unit

    nu = .5  # compressibility (0.5 ~incompressible)
    gamma = 0.1 * Eh  # surface tension
    eta = .015  # viscosity
    lambd = r_0 / R_0  # cell radius relative to channel radius

    N = 40  # number of equations taken into account, N=20 high enough

    """
    %%%%%%%%%%%%%%%
       Modelling
    %%%%%%%%%%%%%%%
    """

    fn, v_equil = utils.f_n(N=N, lambd=lambd)  # expansion coefficients
    gn, v_equil = utils.g_n(N=N, lambd=lambd)  # expansion coefficients

    v_0 = u / v_equil  # velocity of the cell

    sig_c = eta * v_0 / r_0  # characteristic stress
    th = np.linspace(-np.pi, np.pi, 600)  # polar angle

    # keyword arguments for calculations
    kwargs_sh = {'th': th, 'gamma': gamma, 'Eh': Eh, 'sig_c': sig_c,
                 'nu': nu, 'r_0': r_0, 'fn': fn, 'gn': gn}

    kwargs_sp = {'th': th, 'E_0': E_0, 'sig_c': sig_c,
                 'nu': nu, 'r_0': r_0, 'fn': fn, 'gn': gn}

    # extract deformation
    A_sh, d_sh, x_sh, z_sh = utils.def_sh(**kwargs_sh)
    u_r_sh, u_th_sh = utils.disp_sh(**kwargs_sh)
    A_sp, d_sp, x_sp, z_sp = utils.def_sp(**kwargs_sp)
    u_r_sp, u_th_sp = utils.disp_sp(**kwargs_sp)

    if save_data:
        os.chdir(data_dir)
        np.savetxt('coord_sh.dat', np.array([x_sh[range(0, len(th), 10)],
                                            z_sh[range(0, len(th), 10)]]))
        np.savetxt('coord_sp.dat', np.array([x_sp[range(0, len(th), 10)],
                                            z_sp[range(0, len(th), 10)]]))

    """
    %%%%%%%%%%%%%%
       Plotting
    %%%%%%%%%%%%%%
    """

    # circle
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
    ax_2.plot(x_sh, z_sh, 'k-')
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
    ax_4.plot(x_sp, z_sp, 'r-')
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


def Stream_Func(Q=1.2e-11, r_0=9e-6, print_fig=True):
    """Plots stream function and velocity dependency

    **Stream function according to Mietke et al., parameters obtained
    by strategy laid out by Mietke et al.**

    Args
    ----
    :Q:             liquid flow
    :r_0:           cell radius
    :print_fig:     produce figure


    Return
    ------
    Plot
    """

    figname = 'Stream_Func'

    """
    %%%%%%%%%%%%%%%%%%%%%%%
       Other parameters
    %%%%%%%%%%%%%%%%%%%%%%%
    """

    N = 20
    l_0 = 2e-5  # channel side length
    R_0 = 1.094 * l_0 / 2  # equivalent radius
    K_2 = 2.096  # constant

    u = K_2 * Q / l_0**2  # Poiseuille flow at inifinity (liquid velocity)

    lambd = r_0 / R_0

    # parameters
    z_bnd = 1.5  # boundary
    R_0 = 1/lambd
    grid = 100
    num_lev = 25

    """
    %%%%%%%%%%%%%%%
       Modelling
    %%%%%%%%%%%%%%%
    """

    A_n, B_n, C_n, D_n, v_equil = utils.Cpts(N, lambd)

    x_grid = np.linspace(-R_0, R_0, grid)
    z_grid = np.linspace(-z_bnd, z_bnd, grid)

    Z, X = np.meshgrid(z_grid, x_grid)
    r = np.sqrt(X**2 + Z**2)

    cos_th = Z/r

    n_i = 0
    Psi = 0
    for n in np.arange(2, N+2, 2):
        Psi += utils.GB(n, cos_th) * (A_n[n_i]*r**n + B_n[n_i]*r**(-n+1) +
                                      C_n[n_i]*r**(n+2) + D_n[n_i]*r**(-n+3))
        n_i += 1

    Psi[r < 1] = 0

    levels = np.linspace(-1, 1, num_lev)**3/(2*lambd**2)

    th = np.linspace(0, 2*np.pi, 600)  # polar angle
    x0 = 1*np.cos(th)
    z0 = 1*np.sin(th)

    # load data
    try:
        os.chdir(data_dir)
        lambds_v_equil = np.loadtxt('lambds_v_equil.dat')
    except FileNotFoundError:
        print('Set valid save directory (data_dir) in RTDC.py')

    lambds = lambds_v_equil[0, :]
    v_equils = lambds_v_equil[1, :]
    v_val, v_idx = utils.find(v_equils, v_equil)

    """
    %%%%%%%%%%%%%%
       Plotting
    %%%%%%%%%%%%%%
    """

    fig = plt.figure(figname, figsize=(6, 6), clear=True)
    ax1 = fig.add_axes([.05, .3, .4, .4])
    ax1.tick_params(**kwargs_ticks)
    ax1.contour(Z, X, Psi, levels, colors='C0', linewidths=1)
    ax1.plot(x0, z0, 'k-', lw=3)
    ax1.arrow(-.3, -.1, .6, 0, head_width=.1,
              head_length=.1, fc='k', ec='k')
    ax1.text(-.5, .1, r'$v_\mathrm{cell}$ = ' + str(round(100*u/v_equil, 1)) +
             r'$\,$cm/s')
    ax1.plot([-z_bnd, z_bnd], [-R_0, -R_0], 'k-', lw=3)
    ax1.plot([-z_bnd, z_bnd], [R_0, R_0], 'k-', lw=3)
    ax1.set_ylim(-1.5, 1.5)
    ax1.set_xlim(-1.5, 1.5)
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])
    ax1.set_title('$Q = ' + str(round(Q*1e10, 3)) + r'\,\mu$L/s, $\quadr_0 =$'
                  + str(round(1e6*r_0, 2)) + r'$\,\mu$m')

    ax2 = fig.add_axes([.57, .3, .4, .4])
    ax2.plot(lambds, 1/v_equils, 'k-')
    ax2.plot(lambds[v_idx], 1/v_equils[v_idx], 'ro')
    ax2.set_ylabel(r'$v_\mathrm{cell}/u$', fontdict=font)
    ax2.set_xlabel(r'$\lambda$', fontdict=font)
    ax2.set_ylim(.5, 1)
    ax2.set_xlim(0, 1)
    plt.show()

    # Save figure
    if print_fig:
        plt.savefig(save_dir + figname + '.pdf', dpi=100,
                    bbox_inches="tight", rasterized=True)


def Coefficients(res=100, N=40):
    """Calculates and produces data files of expansion coefficients

    **Paramters fn, gn and v_equil for given resolution res
    saved as .dat files for faster computation of area-vs-deformation plots**

    Args
    ----
    :res:   Number of points in the cell radius array
    :N:     degree of coefficient fn or gn


    Return
    ------
    :Fn.dat:            fn's of size (N-1, res)
    :Gn.dat:            gn's of size (N-1, res)
    :V_equil.dat:       Equilibrium velocities of size (res)
    :lambs_v_equil.dat  data for lambda vs v_equil plot (2, 200)
    """

    # coefficients for area-vs-def plots

    l_0 = 2e-5  # channel side length
    R_0 = 1.094 * l_0 / 2  # equivalent radius

    r_var = np.linspace(5e-7, 9e-6, res)
    lambds = r_var / R_0

    Fn = np.zeros((N-1, len(r_var)))
    Gn = np.zeros((N-1, len(r_var)))
    V_equil = np.zeros(len(r_var))

    n = 0
    for lambd in lambds:
        Fn[:, n], V_equil[n] = utils.f_n(N=N, lambd=lambd)
        Gn[:, n], V_equil[n] = utils.g_n(N=N, lambd=lambd)
        n += 1

    try:
        os.chdir(data_dir)
        np.savetxt('Fn.dat', Fn)
        np.savetxt('Gn.dat', Gn)
        np.savetxt('V_equil.dat', V_equil)
        np.savetxt('r_var.dat', r_var)
    except FileNotFoundError:
        print('Set valid save directory (data_dir) in RTDC.py')

    # coefficients for lamda vs velocity plot
    lambds = np.arange(0, 1, .005)
    lambds_v_equil = np.zeros((2, len(lambds)))
    for i in range(len(lambds)):
        A_n, B_n, C_n, D_n, v_equil = utils.Cpts(N, lambds[i])
        lambds_v_equil[0, i] = lambds[i]
        lambds_v_equil[1, i] = v_equil

    try:
        os.chdir(data_dir)
        np.savetxt('lambds_v_equil.dat', lambds_v_equil)
    except FileNotFoundError:
        print('Set valid save directory (data_dir) in RTDC.py')


def Fit_Shell(x_0, z_0, Eh_ini=.1, Q=1.2e-11, gamma_pre=.1, it_max=500,
              alpha=5e-3, print_fig=True):
    """Plots Shell fit

    **Fitting data with shell model**

    Args
    ----
    :Eh_ini:    initial stiffness guess
    :Q:         flow rate
    :gamma_pre: pre-factor of surface tension
    :it_max:    maximum iterations
    :alpha:     learning rate
    :print_fig: print figure (True / False)


    Return
    ------
    Plot
    """

    figname = 'Fit_Shell'

    x_s_ini = np.sum(x_0) / len(x_0)
    z_s_ini = np.sum(z_0) / len(z_0)

    A_0 = utils.area(x_0, z_0)  # area of data
    r_0 = np.sqrt(A_0 / np.pi)  # radius of circle with area A_0

    """
    %%%%%%%%%%%%%%%%
       Parameters
    %%%%%%%%%%%%%%%%
    """

    l_0 = 2e-5  # channel side length
    R_0 = 1.094 * l_0 / 2  # equivalent radius
    K_2 = 2.096  # constant
    u = K_2 * Q / l_0**2  # Poiseuille flow at inifinity (liquid velocity)

    N = 20  # number of equations taken into account, N=20 high enough

    lambd = r_0 / R_0

    nu = .5  # compressibility (0.5 ~incompressible)
    gamma = gamma_pre * Eh_ini  # surface tension
    eta = .015  # viscosity

    fn, v_equil = utils.f_n(N=N, lambd=lambd)  # expansion coefficients
    gn, v_equil = utils.g_n(N=N, lambd=lambd)  # expansion coefficients
    v_0 = u / v_equil  # velocity of the cell
    sig_c = eta * v_0 / r_0  # characteristic stress
    P_ini = np.array([Eh_ini, x_s_ini, z_s_ini])

    """
    %%%%%%%%%%%%%%%%
       Optimizing
    %%%%%%%%%%%%%%%%
    """

    kwargs_opt_sh = {'x_0': x_0, 'z_0': z_0, 'gamma_pre': gamma_pre,
                     'sig_c': sig_c, 'nu': nu, 'r_0': r_0, 'fn': fn, 'gn': gn,
                     'it_max': it_max, 'alpha': alpha, 'P': P_ini}

    it, J, P = utils.optimize_sh(**kwargs_opt_sh)
    print(P)
    Eh = P[0]
    x_s = P[1]
    z_s = P[2]

    """
    %%%%%%%%%%%%%%
       Results
    %%%%%%%%%%%%%%
    """

    th = np.linspace(0, 2*np.pi, 600)
    gamma = gamma_pre * Eh

    kwargs_sh = {'th': th, 'gamma': gamma, 'Eh': Eh, 'sig_c': sig_c,
                 'nu': nu, 'r_0': r_0, 'fn': fn, 'gn': gn}

    # extract deformation
    A_sh, d_sh, x_sh, z_sh = utils.def_sh(**kwargs_sh)

    # plot
    fig = plt.figure(figname, figsize=(8, 8), clear=True)
    ax1 = fig.add_axes([.07, .3, .4, .4])
    ax1.plot(x_0*1e6, z_0*1e6, 'ko', ms=5)
    ax1.plot((x_sh + P[1])*1e6, (z_sh + P[2])*1e6, 'r-')
    ax1.set_xlabel(r'$x$ ($\mu$m)', fontdict=font)
    ax1.set_ylabel(r'$z$ ($\mu$m)', fontdict=font)
    ax1.text(x_s*1e6-3.5, z_s*1e6+1,
             r'$Eh_\mathrm{fit}=$'+str(np.round(Eh*1e3, 1))+r'$\,$nN/$\mu$m',
             fontdict=font)
    ax1.text(x_s*1e6-3., z_s*1e6-1,
             r'$A_\mathrm{fit}=$'+str(np.round(A_sh*1e12, 1)) +
             r'$\,\mu \mathrm{m}^2$',
             fontdict=font)
    ax1.text(x_s*1e6-2.5, z_s*1e6-3,
             r'$d_\mathrm{fit}=$'+str(np.round(d_sh, 3)),
             fontdict=font)
    ax2 = fig.add_axes([.58, .3, .4, .4])
    ax2.plot(it, J, 'k-')
    ax2.set_xlim(0, it_max)
    ax2.set_ylim(0, np.max(J))
    ax2.set_xlabel('iterations', fontdict=font)
    ax2.set_ylabel(r'$\xi\, (\mu$m)', fontdict=font)

    plt.show()

    # Save figure
    if print_fig:
        plt.savefig(save_dir + figname + '.pdf', dpi=100,
                    bbox_inches="tight", rasterized=True)


def Fit_Sphere(x_0, z_0, E_0_ini=1e6, Q=1.2e-11, it_max=500, alpha=5e-3,
               print_fig=True):
    """Plots Sphere fit

    **Fitting data with shell model**

    Args
    ----
    :E_0_ini:   initial stiffness guess
    :Q:         flow rate
    :it_max:    maximum iterations
    :alpha:     learning rate
    :print_fig: print figure (True / False)


    Return
    ------
    Plot
    """

    figname = 'Fit_Sphere'

    x_s_ini = np.sum(x_0) / len(x_0)
    z_s_ini = np.sum(z_0) / len(z_0)

    A_0 = utils.area(x_0, z_0)  # area of data
    r_0 = np.sqrt(A_0 / np.pi)  # radius of circle with area A_0

    """
    %%%%%%%%%%%%%%%%
       Parameters
    %%%%%%%%%%%%%%%%
    """

    l_0 = 2e-5  # channel side length
    R_0 = 1.094 * l_0 / 2  # equivalent radius
    K_2 = 2.096  # constant
    u = K_2 * Q / l_0**2  # Poiseuille flow at inifinity (liquid velocity)

    N = 20  # number of equations taken into account, N=20 high enough

    lambd = r_0 / R_0

    nu = .5  # compressibility (0.5 ~incompressible)
    eta = .015  # viscosity

    fn, v_equil = utils.f_n(N=N, lambd=lambd)  # expansion coefficients
    gn, v_equil = utils.g_n(N=N, lambd=lambd)  # expansion coefficients
    v_0 = u / v_equil  # velocity of the cell
    sig_c = eta * v_0 / r_0  # characteristic stress
    P_ini = np.array([E_0_ini, x_s_ini, z_s_ini])

    """
    %%%%%%%%%%%%%%%%
       Optimizing
    %%%%%%%%%%%%%%%%
    """

    kwargs_opt_sp = {'x_0': x_0, 'z_0': z_0,
                     'sig_c': sig_c, 'nu': nu, 'r_0': r_0, 'fn': fn, 'gn': gn,
                     'it_max': it_max, 'alpha': alpha, 'P': P_ini}

    it, J, P = utils.optimize_sp(**kwargs_opt_sp)
    print(P)
    E_0 = P[0]
    x_s = P[1]
    z_s = P[2]

    """
    %%%%%%%%%%%%%%
       Results
    %%%%%%%%%%%%%%
    """

    th = np.linspace(0, 2*np.pi, 600)

    kwargs_sp = {'th': th, 'E_0': E_0, 'sig_c': sig_c,
                 'nu': nu, 'r_0': r_0, 'fn': fn, 'gn': gn}

    # extract deformation
    A_sp, d_sp, x_sp, z_sp = utils.def_sp(**kwargs_sp)

    # plot
    fig = plt.figure(figname, figsize=(8, 8), clear=True)
    ax1 = fig.add_axes([.07, .3, .4, .4])
    ax1.plot(x_0*1e6, z_0*1e6, 'ko', ms=5)
    ax1.plot((x_sp + P[1])*1e6, (z_sp + P[2])*1e6, 'r-')
    ax1.set_xlabel(r'$x$ ($\mu$m)', fontdict=font)
    ax1.set_ylabel(r'$z$ ($\mu$m)', fontdict=font)
    ax1.text(x_s*1e6-3.5, z_s*1e6+1,
             r'$E_{0, \mathrm{fit}}=$'+str(np.round(E_0*1e-3, 3))+r'$\,$kPa',
             fontdict=font)
    ax1.text(x_s*1e6-3., z_s*1e6-1,
             r'$A_\mathrm{fit}=$'+str(np.round(A_sp*1e12, 1)) +
             r'$\,\mu \mathrm{m}^2$',
             fontdict=font)
    ax1.text(x_s*1e6-2.5, z_s*1e6-3,
             r'$d_\mathrm{fit}=$'+str(np.round(d_sp, 3)),
             fontdict=font)
    ax2 = fig.add_axes([.58, .3, .4, .4])
    ax2.plot(it, J, 'k-')
    ax2.set_xlim(0, it_max)
    ax2.set_ylim(0, np.max(J))
    ax2.set_xlabel('iterations', fontdict=font)
    ax2.set_ylabel(r'$\xi\, (\mu$m)', fontdict=font)

    plt.show()

    # Save figure
    if print_fig:
        plt.savefig(save_dir + figname + '.pdf', dpi=100,
                    bbox_inches="tight", rasterized=True)


def Area_vs_Def(Q=1.2e-11, eh=3.4e-3, e_0=270, print_fig=True):
    """returns plot

    **Area vs Deformation plot for elastic shell and sphere**

    Args
    ----
    :Q:         flow rate
    :eh:        unit of stiffness of shell
    :e_0:       unit of stiffness of sphere
    :print_fig: print figure (True / False)


    Return
    ------
    Plot
    """

    figname = 'Area_vs_Def_Q0p0'+str(int(Q*1e12))

    # Qs = 1e-11*np.linspace(4, 32, 100)  # range of flow

    l_0 = 2e-5  # channel side length
    K_2 = 2.096  # constant

    u = K_2 * Q / l_0**2  # Poiseuille flow at inifinity (liquid velocity)

    eh = 3.4e-3  # stiffness unit of shell
    e_0 = 270  # stiffness unit of sphere

    nu = .5  # compressibility (0.5 ~incompressible)
    gamma = 0  # surface tension
    eta = .015  # viscosity
    th = np.linspace(0, 2*np.pi, 600)  # polar angle

    # load data
    try:
        os.chdir(data_dir)
        V_equil = np.loadtxt('V_equil.dat')
        Fn = np.loadtxt('Fn.dat')
        Gn = np.loadtxt('Gn.dat')
        r_var = np.loadtxt('r_var.dat')
    except FileNotFoundError:
        print('Set valid save directory (data_dir) in RTDC.py')

    """
    %%%%%%%%%%%%%%%%%%%
       Generate data
    %%%%%%%%%%%%%%%%%%%
    """

    # multiplications of stiffness unit
    K_sp = np.array([1, 2, 3, 4, 5, 6, 8, 10])
    K_sh = np.array([1, 2, 3, 4, 5, 7, 9, 12])

    # placeholders
    A_sh = np.zeros((len(K_sh), len(r_var)))
    Def_sh = np.zeros((len(K_sh), len(r_var)))
    A_sp = np.zeros((len(K_sp), len(r_var)))
    Def_sp = np.zeros((len(K_sp), len(r_var)))

    for k in range(len(K_sp)):
        for i in range(len(r_var)):
            v_equil = V_equil[i]
            fn = Fn[:, i]
            gn = Gn[:, i]
            v_0 = u / v_equil
            sig_c = eta * v_0 / r_var[i]

            k_sp = {'th': th, 'E_0': K_sp[k]*e_0, 'sig_c': sig_c,
                    'nu': nu, 'r_0': r_var[i], 'fn': fn, 'gn': gn}

            k_sh = {'th': th, 'gamma': gamma, 'Eh': K_sh[k]*eh,
                    'sig_c': sig_c, 'nu': nu, 'r_0': r_var[i],
                    'fn': fn, 'gn': gn}

            # fill in data
            A_sp[k, i], Def_sp[k, i], x_d, z_d = utils.def_sp(**k_sp)
            A_sh[k, i], Def_sh[k, i], x_d, z_d = utils.def_sh(**k_sh)

    """
    %%%%%%%%%%%%%%
       Plotting
    %%%%%%%%%%%%%%
    """

    # colorscales
    cols1 = plt.cm.magma_r(np.linspace(.1, 1, len(K_sp)))
    cols2 = plt.cm.viridis_r(np.linspace(.1, 1, len(K_sh)))

    fig = plt.figure(figname, figsize=(6, 6), clear=True)
    ax1 = fig.add_subplot(121)
    ax1.set_position([.12, .55, .6, .4])
    ax1.tick_params(**kwargs_ticks)
    ax1.plot(0, 1, 'wo', label=r'$Q=$'+str(Q*1e9)+r'$\,\mu$l/s')
    ax1.plot(0, 1, 'wo', label=r'$E_0=$'+str(e_0)+r'$\,$Pa')
    for k in range(len(K_sp)):
        ax1.plot(A_sp[k]*1e12, Def_sp[k], '-', color=cols1[k], lw=2,
                 label=str(K_sp[k])+r'$E_0$')

    ax1.legend(bbox_to_anchor=(1.05, 1.0, 0, .0), loc=2, borderaxespad=0.,
               fontsize=8)
    ax1.set_yticks(np.arange(0, .05, .01))
    ax1.set_xticks(np.arange(0, 250, 50))
    ax1.set_xticklabels([])
    ax1.set_xlim(0, 200)
    ax1.set_ylim(0, .03)
    ax1.set_ylabel(r'Deformation (sphere)', fontdict=font)
    ax1.grid(True, alpha=.2)

    ax2 = fig.add_subplot(122)
    ax2.set_position([.12, .1, .6, .4])
    ax2.tick_params(**kwargs_ticks)
    ax2.plot(0, 1, 'wo', label=r'$Q=$'+str(Q*1e9)+r'$\,\mu$l/s')
    ax2.plot(0, 1, 'wo', label=r'$(Eh)_0=$'+str(eh*1e3)+r'$\,$nN/$\mu$m')

    for k in range(len(K_sh)):
        ax2.plot(A_sh[k]*1e12, Def_sh[k], '-', color=cols2[k], lw=2,
                 label=str(K_sh[k])+r'$(Eh)_0$')
    ax2.legend(bbox_to_anchor=(1.05, 1.0, 0, .0), loc=2, borderaxespad=0.,
               fontsize=8)
    ax2.set_yticks(np.arange(0, .05, .01))
    ax2.set_xticks(np.arange(0, 250, 50))
    ax2.set_xlim(0, 200)
    ax2.set_ylim(0, .04)
    ax2.set_xlabel(r'Area ($\mu \mathrm{m}^2$)', fontdict=font)
    ax2.set_ylabel(r'Deformation (shell)', fontdict=font)
    ax2.grid(True, alpha=.2)

    plt.show()

    # Save figure
    if print_fig:
        plt.savefig(save_dir + figname + '.pdf', dpi=100,
                    bbox_inches="tight", rasterized=True)
