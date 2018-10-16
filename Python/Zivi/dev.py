#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 17:20:12 2018

@author: denyssutter
"""
import utils_RTDC as utils
import time
import numpy as np
import matplotlib.pyplot as plt

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
#%%

start = time.time()


end = time.time()
print(end-start)

# %%
Q = 1.2e-11

#Qs = 1e-11*np.linspace(4, 32, 100)
l_0 = 2e-5  # channel side length
R_0 = 1.094 * l_0 / 2  # equivalent radius
K_2 = 2.096  # constant

u = K_2 * Q / l_0**2  # Poiseuille flow at inifinity (liquid velocity)

eh = 3.4e-3  # unit of stiffness of shell
e_0 = 270  # unit of stiffness of sphere

nu = .5  # compressibility (0.5 ~incompressible)
gamma = 0  # surface tension
eta = .015  # viscosity
th = np.linspace(0, 2*np.pi, 600)  # polar angle

N = 40  # number of equations taken into account, N=20 high enough

V_equil = np.loadtxt('V_equil.dat')
Fn = np.loadtxt('Fn.dat')
Gn = np.loadtxt('Gn.dat')
r_var = np.loadtxt('r_var.dat')

K_sp = np.array([1, 2, 3, 4, 5, 6, 8, 10])
K_sh = np.array([1, 2, 3, 4, 5, 7, 9, 12])

#A_sh = np.zeros((len(K_sh), len(r_var), len(Qs)))
#Def_sh = np.zeros((len(K_sh), len(r_var), len(Qs)))
#A_sp = np.zeros((len(K_sp), len(r_var), len(Qs)))
#Def_sp = np.zeros((len(K_sp), len(r_var), len(Qs)))

A_sh = np.zeros((len(K_sh), len(r_var)))
Def_sh = np.zeros((len(K_sh), len(r_var)))
A_sp = np.zeros((len(K_sp), len(r_var)))
Def_sp = np.zeros((len(K_sp), len(r_var)))

q_i = 0
#for q in Qs:
for k in range(len(K_sp)):
    for i in range(len(r_var)):

        v_equil = V_equil[i]
        fn = Fn[:, i]
        gn = Gn[:, i]
        u = K_2 * Q / l_0**2  # Poiseuille flow at inifinity
        v_0 = u / v_equil
        sig_c = eta * v_0 / r_var[i]

        k_sp = {'th': th, 'E_0': K_sp[k]*e_0, 'sig_c': sig_c,
                'nu': nu, 'r_0': r_var[i], 'fn': fn, 'gn': gn}

        k_sh = {'th': th, 'gamma': gamma, 'Eh': K_sh[k]*eh,
                'sig_c': sig_c, 'nu': nu, 'r_0': r_var[i],
                'fn': fn, 'gn': gn}

        A_sp[k, i], Def_sp[k, i], x_d, z_d = utils.def_sp(**k_sp)
        A_sh[k, i], Def_sh[k, i], x_d, z_d = utils.def_sh(**k_sh)
#    q_i += 1

#%%
lblx = np.array([26, 66, 102, 132, 153, 170, 180, 175])
lbly = np.array([.02, .018, .016, .014, .012, .010, .0065, .003])
fig = plt.figure('velocity', figsize=(8, 6), clear=True)
ax1 = fig.add_subplot(121)
ax1.set_position([.1, .3, .4, .4])
ax1.tick_params(**kwargs_ticks)
for i in range(len(K_sp)):
    ax1.plot(A_sp[i]*1e12, Def_sp[i], 'k-')
    ax1.text(lblx[i], lbly[i], str(K_sp[i])+r'$E_0$')
ax1.set_yticks(np.arange(0, .05, .01))
ax1.set_xlim(0, 200)
ax1.set_ylim(0, .03)
ax1.set_xlabel(r'Area ($\mu \mathrm{m}^2$)', fontdict=font)
ax1.set_ylabel(r'Deformation', fontdict=font)

ax2 = fig.add_subplot(122)
ax2.set_position([.55, .3, .4, .4])
ax2.tick_params(**kwargs_ticks)
for i in range(len(K_sh)):
    ax2.plot(A_sh[i]*1e12, Def_sh[i], 'b-')
    ax2.text(lblx[i], lbly[i], str(K_sh[i])+r'$(Eh)_0$')
ax2.set_yticks(np.arange(0, .05, .01))
ax2.set_xlim(0, 200)
ax2.set_ylim(0, .04)
#ax2.set_yticklabels([])
ax2.set_xlabel(r'Area ($\mu \mathrm{m}^2$)', fontdict=font)
#ax2.set_ylabel(r'Deformation', fontdict=font)

plt.show()
#%%

fig = plt.figure('A-def-v', figsize=(8, 8), clear=True)
ax1 = fig.add_subplot(121)
ax1.set_position([.1, .3, .4, .4])
ax1.tick_params(**kwargs_ticks)

#for q in range(len(Qs)):
for i in range(len(K_sp)):
    ax1.plot(A_sp[i, :, 0]*1e12, Def_sp[i, :, 0], 'k-')
#        ax1.plot(A_sp[-1, :, q]*1e12, Def_sp[-1, :, q], 'r-')
#ax1.set_yticks(np.arange(0, .05, .01))
ax1.set_xlim(0, 200)
ax1.set_ylim(0, .1)
#ax1.set_xlabel(r'Area ($\mu \mathrm{m}^2$)', fontdict=font)
#ax1.set_ylabel(r'Deformation', fontdict=font)

plt.show()

# %%
