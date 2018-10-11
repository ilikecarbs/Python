#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 14:26:47 2018

@author: denyssutter
"""
from numpy.polynomial import legendre as L
import numpy as np


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
