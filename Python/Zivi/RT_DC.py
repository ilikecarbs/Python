#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 15:35:59 2018

@author: denyssutter
"""

import numpy as np
import matplotlib.pyplot as plt
import utils_SI
import utils_def


# Dictionaries
font = {'family': 'serif',
        'style': 'normal',
        'color': 'black',
        'weight': 'ultralight',
        'size': 12,
        }

th = np.linspace(0, 2*np.pi, 600)

# external parameters
Q = 1.2e-11  # flow
l_0 = 2e-5  # length
R_0 = 1.094 * l_0 / 2
K_2 = 2.096

u = K_2 * Q / l_0**2

eh = 3.4e-3  # unit of stiffness of shell
e_0 = 270  # unit of stiffness of sphere

Eh = eh * 10  # multiples of the unit
E_0 = e_0 * 10  # muplitples of the unit

nu = .5  # compressibility (0.5 ~incompressible)
gamma = 0.1 * Eh  # surface tension
eta = .015  # viscosity
lambd = .9
r_0 = lambd*R_0

N = 10

lambd = r_0 / R_0

fn, v_equil = utils_SI.f_n(N=N, lambd=lambd)
gn, v_equil = utils_SI.g_n(N=N, lambd=lambd)

v_0 = u / v_equil

print('u = '+str(u))
print('v_equil = '+str(v_equil))
print('v = '+str(v_0))

# characteristic stress
sig_c = eta * v_0 / r_0

kwargs_sh = {'th': th, 'gamma': gamma, 'Eh': Eh, 'sig_c': sig_c,
             'nu': nu, 'r_0': r_0, 'fn': fn, 'gn': gn}

kwargs_sp = {'th': th, 'E_0': E_0, 'sig_c': sig_c,
             'nu': nu, 'r_0': r_0, 'fn': fn, 'gn': gn}

"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Modeling relative displacements
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""

A_sh, d_sh, x_sh, z_sh = utils_def.deformation_sh(**kwargs_sh)
u_r_sh, u_th_sh = utils_def.displacement_sh(**kwargs_sh)
A_sp, d_sp, x_sp, z_sp = utils_def.deformation_sp(**kwargs_sp)
u_r_sp, u_th_sp = utils_def.displacement_sp(**kwargs_sp)

x0 = r_0*np.cos(th)
z0 = r_0*np.sin(th)


fig_dis = plt.figure('displacements', figsize=(8, 8), clear=True)
ax_dis_1 = fig_dis.add_subplot(221)
ax_dis_1.set_position([.15, .55, .4, .4])
ax_dis_1.plot(th/np.pi, u_r_sh/r_0, 'k-')
ax_dis_1.plot(th/np.pi, u_th_sh/r_0, 'k--')
ax_dis_1.plot(th/np.pi, u_r_sp/r_0, 'r-')
ax_dis_1.plot(th/np.pi, u_th_sp/r_0, 'r--')
ax_dis_1.set_xlim(0, 1)
ax_dis_1.set_xlabel(r'$\theta/\pi$', fontdict=font)
ax_dis_1.set_ylabel(r'$u_i / r_0$', fontdict=font)
ax_dis_1.legend(('elastic shell, $u_r$', r'elastic shell, $u_\theta$',
                 'elastic sphere, $u_r$', r'elastic sphere, $u_\theta$'))

ax_dis_2 = fig_dis.add_subplot(222)
ax_dis_2.set_position([.56, .55, .4, .4])
ax_dis_2.plot(x_sh, z_sh, 'ko', ms=1)
ax_dis_2.plot(x_sp, z_sp, 'ro', ms=1)
ax_dis_2.plot(x0, z0, 'k--')
ax_dis_2.fill_between(x_sp, 0, z_sp, color='C8', alpha=1)
ax_dis_2.plot([0, 0], [-1.3*r_0, 1.3*r_0], 'k-.', lw=.5)
ax_dis_2.set_xticks([])
ax_dis_2.set_yticks([])
ax_dis_2.set_xlim(-1.5*r_0, 1.5*r_0)
ax_dis_2.set_ylim(-1.5*r_0, 1.5*r_0)
ax_dis_2.text(-.2*r_0, .2*r_0,
              r'$d_\mathrm{sp} =$'+str(np.round(d_sp, 3)),
              color='r', fontsize=12)
ax_dis_2.text(-.2*r_0, -.2*r_0,
              r'$d_\mathrm{sh} =$'+str(np.round(d_sh, 3)),
              color='k', fontsize=12)

plt.show()
# %%
N = 10
r_var = np.linspace(1e-6, 1.05e-5, 10)
Area_sh = np.zeros(len(r_var))
Def_sh = np.zeros(len(r_var))
Area_sp = np.zeros(len(r_var))
Def_sp = np.zeros(len(r_var))

ax_dis_3 = fig_dis.add_subplot(223)
ax_dis_3.set_position([.15, .08, .4, .4])

for k in np.array([1, 2, 3, 4, 5, 6, 8, 10]):
    for i in range(len(r_var)):
        lambd = r_var[i] / R_0
#        dummy, dummy, v_equil = utils_SI.Coeff(N=N, lambd=lambd)

        fn, v_equil = utils_SI.f_n(N=N, lambd=lambd)
        gn, v_equil = utils_SI.g_n(N=N, lambd=lambd)
        v_0 = u / v_equil
        sig_c = eta * v_0 / r_var[i]

        kwargs_sh = {'th': th, 'gamma': gamma, 'Eh': k*eh, 'sig_c': sig_c,
                     'nu': nu, 'r_0': r_var[i], 'fn': fn, 'gn': gn}

        kwargs_sp = {'th': th, 'E_0': k*e_0, 'sig_c': sig_c,
                     'nu': nu, 'r_0': r_var[i], 'fn': fn, 'gn': gn}

        Area_sp[i], Def_sp[i], x_d, z_d = utils_def.deformation_sp(**kwargs_sp)
#       Area_sh[i], Def_sh[i], x_d, z_d = utils_def.deformation_sh(**kwargs_sh)

#    ax_dis_3.plot(Area_sh*1e12, Def_sh*1e2, 'k-')
    ax_dis_3.plot(Area_sp*1e12, Def_sp, 'r-')

ax_dis_3.set_xlim(0, 200)
ax_dis_3.set_ylim(0, 3)
ax_dis_3.set_xlabel(r'Area ($\mu \mathrm{m}^2$)', fontdict=font)
ax_dis_3.set_ylabel(r'Deformation', fontdict=font)
plt.show()

# %%
plt.savefig('rel_deformation' + '.pdf', dpi=100,
            bbox_inches="tight", rasterized=True)
