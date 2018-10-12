#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 17:20:12 2018

@author: denyssutter
"""
import utils_RTDC as utils
import time
import numpy as np
import os
#%%


N = 6
lambd = .7
z_bnd = 1.5
grid = 100
#A = utils.A_alpha(n=2, k=2, lambd=lambd)


start = time.time()

#a_n, b_n, v_equil = utils.Coeff(N, lambd)
A_n, B_n, C_n, D_n, v_equil = utils.Cpts(N, lambd)

end = time.time()
print(end - start)



# %%

N = 10
r_var = np.linspace(1e-6, 9e-6, 20)
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

        fn, v_equil = utils.f_n(N=N, lambd=lambd)
        gn, v_equil = utils.g_n(N=N, lambd=lambd)
        v_0 = u / v_equil
        sig_c = eta * v_0 / r_var[i]

        kwargs_sh = {'th': th, 'gamma': gamma, 'Eh': k*eh, 'sig_c': sig_c,
                     'nu': nu, 'r_0': r_var[i], 'fn': fn, 'gn': gn}

        kwargs_sp = {'th': th, 'E_0': k*e_0, 'sig_c': sig_c,
                     'nu': nu, 'r_0': r_var[i], 'fn': fn, 'gn': gn}

        Area_sp[i], Def_sp[i], x_d, z_d = utils.deformation_sp(**kwargs_sp)
#       Area_sh[i], Def_sh[i], x_d, z_d = utils_def.deformation_sh(**kwargs_sh)

#    ax_dis_3.plot(Area_sh*1e12, Def_sh*1e2, 'k-')
    ax_dis_3.plot(Area_sp*1e12, Def_sp, 'r-')

ax_dis_3.set_xlim(0, 200)
ax_dis_3.set_ylim(0, .03)
ax_dis_3.set_xlabel(r'Area ($\mu \mathrm{m}^2$)', fontdict=font)
ax_dis_3.set_ylabel(r'Deformation', fontdict=font)
plt.show()

#%%
fig_vel = plt.figure('Velocities', figsize=(8, 8), clear=True)
ax_1 = fig_vel.add_subplot(221)

N = 10

lambd = np.linspace(0, 1, 10)
V_equil = np.zeros(len(lambd))

for i in range(len(lambd)):
    dummy, dummy, v_equil = utils.Coeff(N, lambd[i])
    V_equil[i] = 1/v_equil

ax_1.plot(lambd, V_equil, 'ko')

plt.show()
# %%
plt.savefig('rel_deformation' + '.pdf', dpi=100,
            bbox_inches="tight", rasterized=True)

#%%

def A_alpha(n, k, lambd):
    A = -utils.I_2(2*k+n+2) / factorial(2*k) * lambd**(n+2*k+3)

    if k > 0:
        A += utils.I_1(2*k+n+1) / (factorial(2*k-2)*(4*k+1)) * lambd**(n+2*k+3)
    if 2*k == n:
        A += (4*k+5)*factorial(2*k+2)/4 * np.pi

    return A

def A_beta(n, k, lambd):
    A = -utils.I_1(2*k+n+3)/(factorial(2*k)*(4*k+5)) * lambd**(n+2*k+5)
    if 2*k == n:
        A += (4*k+3)*factorial(2*k+2)/4 * np.pi

    return A


def B_alpha(n, k, lambd):
    B = (utils.I_3(2*k+n+1)/factorial(2*k) +
         utils.I_2(2*k+n)*(2*k+2)*(2*k+1)/(factorial(2*k)*(4*k+1)))*lambd**(n+2*k+1)
    if 2*k == n:
        B += np.pi * (4*k+3)*factorial(2*k+2)/(4*(4*k+1))
    if 2*k == n-2:
        B += np.pi * (2*k+2)*(2*k+1)*factorial(2*k+2)/4

    return B


def B_beta(n, k, lambd):
    B = -utils.I_2(2*k+n+2)/(factorial(2*k)*(4*k+5)) * lambd**(n+2*k+3)
    if 2*k == n:
        B += np.pi * factorial(2*k+2)/4
    if 2*k == n-2:
        B += np.pi * (2*k+2)*(4*k+3)*(2*k+1)*factorial(2*k+2)/(4*(4*k+5))

    return B


def Coeff(N, lambd):
    K = N
    M = np.zeros((K, N))
    k = 0
    for k_i in np.arange(0, K, 2):
        n_i = 0
        for n in np.arange(0, N, 2):
            M[k_i, n_i] = A_alpha(n=n, k=k, lambd=lambd)
            M[k_i+1, n_i] = A_beta(n=n, k=k, lambd=lambd)
            M[k_i, n_i+int(N/2)] = B_alpha(n=n, k=k, lambd=lambd)
            M[k_i+1, n_i+int(N/2)] = B_beta(n=n, k=k, lambd=lambd)
            n_i += 1
        k += 1

    vec = np.zeros(N)
    vec[0] = -1
    vec[1] = -2/5*lambd**2
    vec[2] = 2/5*lambd**2

    M[:, int(N/2)] = vec
    RHS = np.zeros(N)
    RHS[0] = -1

    coeff = np.linalg.solve(M, RHS)

    v_equil = coeff[int(N/2)]
    coeff[int(N/2)] = 0

    a_n = coeff[:int(N/2)]
    b_n = coeff[int(N/2):]

    return M, a_n, b_n, v_equil

M_ref, a_ref, b_ref, v_equil_ref = Coeff(N, lambd)