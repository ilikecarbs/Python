#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 14:26:47 2018

@author: denyssutter
"""
from numpy.polynomial import legendre as L
import numpy as np
from scipy.special import iv, kv
from math import factorial
from scipy import integrate


def I_1(n):
    return integrate.quad(lambda x: x**(n-1) /
                          (iv(1, x)**2 - iv(0, x) * iv(2, x)), 0, 100)[0]


def I_2(n):
    return integrate.quad(lambda x: x**n *
                          (iv(1, x) * kv(1, x) + iv(2, x) * kv(0, x)) /
                          (iv(1, x)**2 - iv(0, x) * iv(2, x)), 0, 100)[0]


def I_3(n):
    return integrate.quad(lambda x: -x**(n-1) /
                          (iv(1, x)**2 - iv(0, x) * iv(2, x)), 0, 100)[0]


def A_alpha(n, k, lambd, I1, I2):
    A = -I2[2*k+n+2] / factorial(2*k) * lambd**(n+2*k+3)

    if k > 0:
        A += I1[2*k+n+1] / (factorial(2*k-2)*(4*k+1)) * lambd**(n+2*k+3)
    A[int(2*k/2)] += (4*k+5)*factorial(2*k+2)/4 * np.pi

    return A


def A_beta(n, k, lambd, I1):
    A = -I1[2*k+n+3]/(factorial(2*k)*(4*k+5)) * lambd**(n+2*k+5)
    A[int(2*k/2)] += (4*k+3)*factorial(2*k+2)/4 * np.pi

    return A


def B_alpha(n, k, lambd, I2, I3):
    B = (I3[2*k+n+1]/factorial(2*k) +
         I2[2*k+n]*(2*k+2)*(2*k+1)/(factorial(2*k)*(4*k+1)))*lambd**(n+2*k+1)
    B[int(2*k/2)] += np.pi * (4*k+3)*factorial(2*k+2)/(4*(4*k+1))
    try:
        B[int(2*k/2+1)] += np.pi * (2*k+2)*(2*k+1)*factorial(2*k+2)/4
    except IndexError:
        return B
    return B


def B_beta(n, k, lambd, I2):
    B = -I2[2*k+n+2]/(factorial(2*k)*(4*k+5)) * lambd**(n+2*k+3)
    B[int(2*k/2)] += np.pi * factorial(2*k+2)/4
    try:
        B[int(2*k/2+1)] += np.pi*((2*k+2)*(4*k+3)*(2*k+1) *
                                  factorial(2*k+2)/(4*(4*k+5)))
    except IndexError:
        return B
    return B


def Coeff(N, lambd):
    I1 = np.zeros(2*N)
    I2 = np.zeros(2*N)
    I3 = np.zeros(2*N)
    for i in range(2*N):
        I1[i] = I_1(i)
        I2[i] = I_2(i)
        I3[i] = I_3(i)

    K = N
    M = np.zeros((K, N))
    k = 0
    n = np.arange(0, N, 2)
    for k_i in np.arange(0, K, 2):
        M[k_i, :int(N/2)] = A_alpha(n=n, k=k, lambd=lambd, I1=I1, I2=I2)
        M[k_i, int(N/2):] = B_alpha(n=n, k=k, lambd=lambd, I2=I2, I3=I3)
        M[k_i+1, :int(N/2)] = A_beta(n=n, k=k, lambd=lambd, I1=I1)
        M[k_i+1, int(N/2):] = B_beta(n=n, k=k, lambd=lambd, I2=I2)
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

    return a_n, b_n, v_equil


def Cpts(N, lambd):
    a_n, b_n, v_equil = Coeff(N=N, lambd=lambd)

    A_n = np.zeros(len(a_n))
    B_n = np.zeros(len(a_n))
    C_n = np.zeros(len(a_n))
    D_n = np.zeros(len(a_n))

    n_i = 0
    for n in np.arange(2, N, 2):
        B_n[n_i] = ((-1)**(n/2+1) * np.pi *
                    (factorial(n)/2*a_n[n_i] +
                     n*(n-1)*factorial(n)/(2*(2*n+1)) * b_n[n_i+1]))
        n_i += 1

    # add last part separately
    B_n[-1] = (-1)**(N/2+1) * np.pi * factorial(N)/2*a_n[-1]

    n_i = 0
    for n in np.arange(2, N+2, 2):
        D_n[n_i] = ((-1)**(n/2+1) * np.pi *
                    factorial(n)/(2*(2*n-3)) * b_n[n_i])

        A_n[n_i] = -(2*n+1)/2 * B_n[n_i] - (2*n-1)/2 * D_n[n_i]
        C_n[n_i] = (2*n-1)/2 * B_n[n_i] + (2*n-3)/2 * D_n[n_i]

        n_i += 1

#    A_n[0] += 1 - v_equil
#    A_n[1] += -2/5 * lambd**2 * v_equil
#    C_n[0] += 2/5 * lambd**2 * v_equil
    return A_n, B_n, C_n, D_n, v_equil


def f_n(N, lambd):
    A_n, B_n, C_n, D_n, v_equil = Cpts(N=N, lambd=lambd)

    fn = np.zeros(N-1)

    n_i = 0
    for n in np.arange(1, len(fn), 2):
        fn[n] = 2*((2*n+3)/n * C_n[n_i] + (2*n-1)/(n+1) * D_n[n_i])
        n_i += 1
    return fn, v_equil


def g_n(N, lambd):
    A_n, B_n, C_n, D_n, v_equil = Cpts(N=N, lambd=lambd)

    gn = np.zeros(N-1)

    n_i = 0
    for n in np.arange(1, len(gn), 2):
        gn[n] = -((n-1)/n * A_n[n_i] + (n+2)/(n+1) * B_n[n_i] +
                  (n+3)/n * C_n[n_i] + (n-2)/(n+1) * D_n[n_i])
        n_i += 1
    return gn, v_equil


def P_n(x, n):
    c = np.zeros(n+1)
    c[n] = 1
    P_n = L.legval(x, c, tensor=True)

    return P_n


def dP_n(x, n):
    c = np.zeros(n+1)
    c[n] = 1
    dc = L.legder(c)

    dP_n = L.legval(x, dc, tensor=True)

    return dP_n


def area(x, y):
    a = 0
    x0, y0 = x[0], y[0]
    for i in range(len(x)-1):
        x1 = x[i+1]
        y1 = y[i+1]
        dx = x1 - x0
        dy = y1 - y0
        a += 0.5 * (y0 * dx - x0 * dy)
        x0 = x1
        y0 = y1
    return a


def alpha_sh(n, gamma, Eh, sig_c, nu, r_0, fn, gn):
    if np.mod(n, 2) == 1:
        nom_alpha = (n*(n+1)+nu-1)*fn[n] + n*(n+1)*(1+nu)*gn[n]
        denom_alpha = (n*(n+1)-2) * (Eh + gamma*(n*(n+1)+nu-1))

        alpha = sig_c * r_0 ** 2 * nom_alpha / denom_alpha
    else:
        alpha = 0
    return alpha


def beta_sh(n, gamma, Eh, sig_c, nu, r_0, fn, gn):
    if np.mod(n, 2) == 1:
        nom_beta = (1+nu)*(fn[n]+2*gn[n]) + gamma/(Eh)*(1-nu**2)*(n*(n+1)-2)
        denom_beta = (n*(n+1)-2)*(Eh+gamma*(n*(n+1)+nu-1))

        beta = sig_c * r_0 ** 2 * nom_beta / denom_beta
    else:
        beta = 0
    return beta


def alpha_sp(n, E_0, sig_c, nu, r_0, fn, gn):
    G = E_0 / (2*(1+nu))

    if np.mod(n, 2) == 1:
        nom_alpha = (fn[n]*(2*nu-2*(nu-1)*n**2+nu*n-1) +
                     gn[n]*n*(n+1)*((2*nu-1)*n-nu+2))
        denom_alpha = (n-1)*(nu+n**2+2*nu*n+n+1)

        alpha = sig_c * r_0 / (2 * G) * nom_alpha / denom_alpha
    else:
        alpha = 0
    return alpha


def beta_sp(n, E_0, sig_c, nu, r_0, fn, gn):
    G = E_0 / (2*(1+nu))  # shear stress

    if np.mod(n, 2) == 1:
        nom_beta = (fn[n]*(-nu+(2*nu-1)*n+2) +
                    gn[n]*(nu-2*(nu-1)*n**2+(3*nu-1)*n+1))
        denom_beta = (n-1)*(nu+n**2+2*nu*n+n+1)

        beta = sig_c * r_0 / (2 * G) * nom_beta / denom_beta
    else:
        beta = 0
    return beta


def displacement_sh(th, gamma, Eh, sig_c, nu, r_0, fn, gn):
    u_r = 0
    u_th = 0

    for i in range(len(fn)):
        if i >= 2:
            al = alpha_sh(i, gamma=gamma, Eh=Eh, sig_c=sig_c, nu=nu, r_0=r_0,
                          fn=fn, gn=gn)
            u_r += al * P_n(np.cos(th), i)
        if i >= 2:
            be = beta_sh(i, gamma=gamma, Eh=Eh, sig_c=sig_c, nu=nu, r_0=r_0,
                         fn=fn, gn=gn)
            u_th += be * dP_n(np.cos(th), i) * -np.sin(th)

    return u_r, u_th


def displacement_sp(th, E_0, sig_c, nu, r_0, fn, gn):
    u_r = 0
    u_th = 0

    for i in range(len(fn)):
        if i >= 2:
            al = alpha_sp(i, E_0=E_0, sig_c=sig_c, nu=nu, r_0=r_0,
                          fn=fn, gn=gn)
            u_r += al * P_n(np.cos(th), i)
        if i >= 2:
            be = beta_sp(i, E_0=E_0, sig_c=sig_c, nu=nu, r_0=r_0,
                         fn=fn, gn=gn)
            u_th += be * dP_n(np.cos(th), i) * -np.sin(th)

    return u_r, u_th


def deformation_sh(th, gamma, Eh, sig_c, nu, r_0, fn, gn):
    u_r, u_th = displacement_sh(th=th, gamma=gamma, Eh=Eh, sig_c=sig_c, nu=nu,
                                r_0=r_0, fn=fn, gn=gn)

    x_d = np.zeros(len(u_r))
    z_d = np.zeros(len(u_r))

    for i in range(len(u_r)):
        x_d[i] = (r_0 + u_r[i]) * np.sin(th[i]) + u_th[i] * np.cos(th[i])
        z_d[i] = (r_0 + u_r[i]) * np.cos(th[i]) - u_th[i] * np.sin(th[i])

    A = np.abs(area(x_d, z_d))
    P = 0
    for i in range(len(x_d)-1):
        P += np.sqrt((x_d[i+1]-x_d[i])**2 + (z_d[i+1]-z_d[i])**2)
    c = 2 * np.sqrt(np.pi * A) / P
    d = 1-c

    return A, d, x_d, z_d


def deformation_sp(th, E_0, sig_c, nu, r_0, fn, gn):
    u_r, u_th = displacement_sp(th=th, E_0=E_0, sig_c=sig_c, nu=nu,
                                r_0=r_0, fn=fn, gn=gn)

    x_d = np.zeros(len(u_r))
    z_d = np.zeros(len(u_r))

    for i in range(len(u_r)):
        x_d[i] = (r_0 + u_r[i]) * np.sin(th[i]) + u_th[i] * np.cos(th[i])
        z_d[i] = (r_0 + u_r[i]) * np.cos(th[i]) - u_th[i] * np.sin(th[i])

    A = np.abs(area(x_d, z_d))
    P = 0
    for i in range(len(x_d)-1):
        P += np.sqrt((x_d[i+1]-x_d[i])**2 + (z_d[i+1]-z_d[i])**2)
    c = 2 * np.sqrt(np.pi * A) / P
    d = 1-c

    return A, d, x_d, z_d
