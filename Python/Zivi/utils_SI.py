#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 15:33:26 2018

@author: denyssutter
"""

import numpy as np
from scipy.special import iv, kv
from math import factorial


def I1(n):
    x = np.arange(.01, 40, .01)
    integrand = -x**(n-1) / (iv(1, x)**2 - iv(0, x) * iv(2, x))
    return np.trapz(x, integrand)


def I3(n):
    x = np.arange(.01, 40, .01)
    integrand = x**(n-1) / (iv(1, x)**2 - iv(0, x) * iv(2, x))
    return np.trapz(x, integrand)


def I2(n):
    x = np.arange(.01, 40, .01)
    integrand = -x**n * ((iv(1, x) * kv(1, x) + iv(2, x) * kv(0, x)) /
                         (iv(1, x)**2 - iv(0, x) * iv(2, x)))
    return np.trapz(x, integrand)


def A_alpha(n, k, lambd):
    A = -I2(2*k+n+2) / factorial(2*k) * lambd**(n+2*k+3)

    if k > 0:
        A += I1(2*k+n+1) / (factorial(2*k-2)*(4*k+1)) * lambd**(n+2*k+3)
    if 2*k == n:
        A += (4*k+5)*factorial(2*k+2)/4 * np.pi

    return A


def A_beta(n, k, lambd):
    A = -I1(2*k+n+3)/(factorial(2*k)*(4*k+5)) * lambd**(n+2*k+5)
    if 2*k == n:
        A += (4*k+3)*factorial(2*k+2)/4 * np.pi

    return A


def B_alpha(n, k, lambd):
    B = (I3(2*k+n+1)/factorial(2*k) +
         I2(2*k+n)*(2*k+2)*(2*k+1)/(factorial(2*k)*(4*k+1)))*lambd**(n+2*k+1)
    if 2*k == n:
        B += np.pi * (4*k+3)*factorial(2*k+2)/(4*(4*k+1))
    if 2*k == n-2:
        B += np.pi * (2*k+2)*(2*k+1)*factorial(2*k+2)/4

    return B


def B_beta(n, k, lambd):
    B = -I2(2*k+n+2)/(factorial(2*k)*(4*k+5)) * lambd**(n+2*k+3)
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

    return a_n, b_n, v_equil


def ABCD_n(N, lambd):
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

    A_n[0] += 1 - v_equil
    A_n[1] += -2/5 * lambd**2 * v_equil
    C_n[0] += 2/5 * lambd**2 * v_equil
    return A_n, B_n, C_n, D_n, v_equil


def f_n(N, lambd):
    A_n, B_n, C_n, D_n, v_equil = ABCD_n(N=N, lambd=lambd)

    fn = np.zeros(N-1)

    n_i = 0
    for n in np.arange(1, len(fn), 2):
        fn[n] = 2*((2*n+3)/n * C_n[n_i] + (2*n-1)/(n+1) * D_n[n_i])
        n_i += 1
    return fn, v_equil


def g_n(N, lambd):
    A_n, B_n, C_n, D_n, v_equil = ABCD_n(N=N, lambd=lambd)

    gn = np.zeros(N-1)

    n_i = 0
    for n in np.arange(1, len(gn), 2):
        gn[n] = -((n-1)/n * A_n[n_i] + (n+2)/(n+1) * B_n[n_i] +
                  (n+3)/n * C_n[n_i] + (n-2)/(n+1) * D_n[n_i])
        n_i += 1
    return gn, v_equil
