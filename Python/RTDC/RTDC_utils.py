#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 14:26:47 2018

@author: denyssutter

%%%%%%%%%%%%%%%%
   RTDC_utils
%%%%%%%%%%%%%%%%

**Useful functions for calculations and fit procedure**

.. note::
        To-Do:
            -
"""

from numpy.polynomial import legendre as L
import numpy as np
from scipy.special import iv, kv
from math import factorial
from scipy import integrate
import time
from joblib import Parallel, delayed
import multiprocessing


def find(array, val):
    """returns array[_val], _val

    **Searches entry in array closest to val.**

    Args
    ----
    :array:     entry value in array
    :_val:      index of entry

    Return
    ------
    :array[_val]:   entry value in array
    :_val:          index of entry
    """

    array = np.asarray(array)
    _val = (np.abs(array - val)).argmin()

    return array[_val], _val


def I_1(n):
    """returns integral I_1

    **Integral S_2 in Haberman paper**

    Args
    ----
    :n:         degree

    Return
    ------
    :I_1:       S_2 integral
    """

    I_1 = integrate.quad(lambda x: x**(n-1) /
                         (iv(1, x)**2 - iv(0, x) * iv(2, x)), 0, 100)[0]

    return I_1


def I_2(n):
    """returns integral I_2

    **Integral S_4 in Haberman paper**

    Args
    ----
    :n:         degree

    Return
    ------
    :I_2:       S_4 integral
    """

    I_2 = integrate.quad(lambda x: x**n *
                         (iv(1, x) * kv(1, x) + iv(2, x) * kv(0, x)) /
                         (iv(1, x)**2 - iv(0, x) * iv(2, x)), 0, 100)[0]

    return I_2


def I_3(n):
    """returns integral I_3

    **Integral S_3 in Haberman paper**

    Args
    ----
    :n:         degree

    Return
    ------
    :I_3:       S_3 integral
    """

    I_3 = integrate.quad(lambda x: -x**(n-1) /
                         (iv(1, x)**2 - iv(0, x) * iv(2, x)), 0, 100)[0]

    return I_3


def A_alpha(n, k, lambd, I1, I2):
    """returns A

    **From SM equation (S1) definitions, Exponent of I1 is corrected**

    Args
    ----
    :n, k:      degrees
    :lambd:     relative radius of cell to the channel
    :I1, I2:    Integrals from Haberman paper


    Return
    ------
    :A:         A_alpha(n, k)
    """

    A = -I2[2*k+n+2] / factorial(2*k) * lambd**(n+2*k+3)

    if k > 0:
        A += I1[2*k+n+1] / (factorial(2*k-2)*(4*k+1)) * lambd**(n+2*k+3)
    A[int(2*k/2)] += (4*k+5)*factorial(2*k+2)/4 * np.pi

    return A


def A_beta(n, k, lambd, I1):
    """returns A

    **From SM equation (S1) definitions, Exponent of I1 is corrected**

    Args
    ----
    :n, k:      degrees
    :lambd:     relative radius of cell to the channel
    :I1:        Integral from Haberman paper


    Return
    ------
    :A:         A_beta(n, k)
    """

    A = -I1[2*k+n+3]/(factorial(2*k)*(4*k+5)) * lambd**(n+2*k+5)
    A[int(2*k/2)] += (4*k+3)*factorial(2*k+2)/4 * np.pi

    return A


def B_alpha(n, k, lambd, I2, I3):
    """returns B

    **From SM equation (S1) definitions.**

    Args
    ----
    :n, k:      degrees
    :lambd:     relative radius of cell to the channel
    :I2, I3:    Integrals from Haberman paper


    Return
    ------
    :B:         B_alpha(n, k)
    """

    B = (I3[2*k+n+1]/factorial(2*k) +
         I2[2*k+n]*(2*k+2)*(2*k+1)/(factorial(2*k)*(4*k+1)))*lambd**(n+2*k+1)
    B[int(2*k/2)] += np.pi * (4*k+3)*factorial(2*k+2)/(4*(4*k+1))
    try:
        B[int(2*k/2+1)] += np.pi * (2*k+2)*(2*k+1)*factorial(2*k+2)/4
    except IndexError:
        return B
    return B


def B_beta(n, k, lambd, I2):
    """returns B

    **From SM equation (S1) definitions, Exponent of I2 is corrected**

    Args
    ----
    :n, k:      degrees
    :lambd:     relative radius of cell to the channel
    :I2:        Integral from Haberman paper


    Return
    ------
    :B:         B_beta(n, k)
    """

    B = -I2[2*k+n+2]/(factorial(2*k)*(4*k+5)) * lambd**(n+2*k+3)
    B[int(2*k/2)] += np.pi * factorial(2*k+2)/4
    try:
        B[int(2*k/2+1)] += np.pi*((2*k+2)*(4*k+3)*(2*k+1) *
                                  factorial(2*k+2)/(4*(4*k+5)))
    except IndexError:
        return B
    return B


def Coeff(N, lambd):
    """returns a_n, b_n, v_equil

    **Solve system of linear equations equation (S1)**

    .. note::
        Corrections equation S1:
            - correct u -> -u
            - correct 2/5*lambda**2 -> -2/5*lambda**2*u
            - correct -2/5*lambda**2 -> 2/5*lambda**2*u

    Args
    ----
    :N:         max. number of equations taken into account
    :lambd:     relative radius of cell to the channel


    Return
    ------
    :a_n:       a_n coefficient of n-th degree
    :b_n:       b_n coefficient of n-th degree
    :v_equil:   equilibrium cell velocity
    """

    # Build arrays with integrals
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

    # Build Matrix of linear system eq. 1 of SM
    for k_i in np.arange(0, K, 2):
        M[k_i, :int(N/2)] = A_alpha(n=n, k=k, lambd=lambd, I1=I1, I2=I2)
        M[k_i, int(N/2):] = B_alpha(n=n, k=k, lambd=lambd, I2=I2, I3=I3)
        M[k_i+1, :int(N/2)] = A_beta(n=n, k=k, lambd=lambd, I1=I1)
        M[k_i+1, int(N/2):] = B_beta(n=n, k=k, lambd=lambd, I2=I2)
        k += 1

    # b_0 is set to zero. Use this coloumn of matrix M to solve for vec
    vec = np.zeros(N)
    vec[0] = -1  # units of u
    vec[1] = -2/5*lambd**2  # units of u
    vec[2] = 2/5*lambd**2  # units of u

    # replace b_0 coloumn
    M[:, int(N/2)] = vec
    RHS = np.zeros(N)
    RHS[0] = -1  # entry of first equation in the system right hand side

    # solve linear system
    coeff = np.linalg.solve(M, RHS)

    v_equil = coeff[int(N/2)]  # extract u

    # set b_0 equal to zero.
    coeff[int(N/2)] = 0

    a_n = coeff[:int(N/2)]
    b_n = coeff[int(N/2):]

    return a_n, b_n, v_equil


def Cpts(N, lambd):
    """returns A_n, B_n, C_n, D_n, v_equil

    **Build Components from Coefficients from SM equations (S2a-d)**

    Args
    ----
    :N:         max. number of equations taken into account
    :lambd:     relative radius of cell to the channel


    Return
    ------
    :A_n:       A_n coefficient of n-th degree
    :B_n:       B_n coefficient of n-th degree
    :C_n:       C_n coefficient of n-th degree
    :D_n:       D_n coefficient of n-th degree
    :v_equil:   equilibrium cell velocity
    """

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

    return A_n, B_n, C_n, D_n, v_equil


def f_n(N, lambd):
    """returns fn, v_equil

    **f_n coefficients from SM equation (S3a)**

    Args
    ----
    :N:         max. number of equations taken into account
    :lambd:     relative radius of cell to the channel


    Return
    ------
    :fn:        gn coefficients
    """

    A_n, B_n, C_n, D_n, v_equil = Cpts(N=N, lambd=lambd)

    fn = np.zeros(N-1)

    n_i = 0
    for n in np.arange(1, len(fn), 2):
        fn[n] = 2*((2*n+3)/n * C_n[n_i] + (2*n-1)/(n+1) * D_n[n_i])
        n_i += 1
    return fn, v_equil


def g_n(N, lambd):
    """returns gn, v_equil

    **g_n coefficients from SM equation (S3b)**

    Args
    ----
    :N:         max. number of equations taken into account
    :lambd:     relative radius of cell to the channel


    Return
    ------
    :gn:        gn coefficients
    """

    A_n, B_n, C_n, D_n, v_equil = Cpts(N=N, lambd=lambd)

    gn = np.zeros(N-1)

    n_i = 0
    for n in np.arange(1, len(gn), 2):
        gn[n] = -((n-1)/n * A_n[n_i] + (n+2)/(n+1) * B_n[n_i] +
                  (n+3)/n * C_n[n_i] + (n-2)/(n+1) * D_n[n_i])
        n_i += 1
    return gn, v_equil


def P_n(x, n):
    """returns P_n

    **Legendre polynomial of n-th degree**

    Args
    ----
    :n:     degree
    :x:     cos(theta)


    Return
    ------
    :P_n:   Legendre polynomial of n-th degree
    """

    c = np.zeros(n+1)
    c[n] = 1
    P_n = L.legval(x, c, tensor=True)

    return P_n


def dP_n(x, n):
    """returns dP_n

    **derivative of Legendre polynomial of n-the degree**

    Args
    ----
    :n:     degree
    :x:     cos(theta)


    Return
    ------
    :dP_n:  derivative of Legendre polynomial
    """

    c = np.zeros(n+1)
    c[n] = 1
    dc = L.legder(c)

    dP_n = L.legval(x, dc, tensor=True)

    return dP_n


def GB(n, x):
    """returns GB

    **Gegenbauer polynomial for equation 18**

    Args
    ----
    :n:     n-th degree
    :x:     cos(theta)


    Return
    ------
    :GB:    Gegenbauer polynomial
    """

    GB = (P_n(x, n-2) - P_n(x, n)) / (2*n-1)
    return GB


def area(x, y):
    """returns a

    **Measure area a of shape with coordinates (x, y) from Green's theorem**

    Args
    ----
    :x:         x-coordinates
    :y:         y-coordinates


    Return
    ------
    :a:         area
    """
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
    """returns alpha

    **Expansion coefficients for shell model: equation 16a**

    Args
    ----
    :n:         n-th term
    :gamma:     surface tension
    :Eh:        stiffness of shell
    :sig_c:     characteristic stress
    :nu:        Poisson ratio
    :r_0:       estimated radius undeformed cell
    :fn:        fn-coefficients
    :gn:        gn-coefficients


    Return
    ------
    :alpha:     n-th coefficient of expansion
    """

    nom_alpha = (n*(n+1)+nu-1)*fn[n] + n*(n+1)*(1+nu)*gn[n]
    denom_alpha = (n*(n+1)-2) * (Eh + gamma*(n*(n+1)+nu-1))

    alpha = sig_c * r_0 ** 2 * nom_alpha / denom_alpha

    return alpha


def beta_sh(n, gamma, Eh, sig_c, nu, r_0, fn, gn):
    """returns beta

    **Expansion coefficients for shell model: equation 16b**

    Args
    ----
    :n:         n-th term
    :gamma:     surface tension
    :Eh:        stiffness of shell
    :sig_c:     characteristic stress
    :nu:        Poisson ratio
    :r_0:       estimated radius undeformed cell
    :fn:        fn-coefficients
    :gn:        gn-coefficients


    Return
    ------
    :beta:      n-th coefficient of expansion
    """

    nom_beta = (1+nu)*(fn[n]+2*gn[n]) + gamma/(Eh)*(1-nu**2)*(n*(n+1)-2)
    denom_beta = (n*(n+1)-2)*(Eh+gamma*(n*(n+1)+nu-1))

    beta = sig_c * r_0 ** 2 * nom_beta / denom_beta

    return beta


def alpha_sp(n, E_0, sig_c, nu, r_0, fn, gn):
    """returns alpha

    **Expansion coefficients for sphere model: equation 25a**

    Args
    ----
    :n:         n-th term
    :E_0:       stiffness of sphere
    :sig_c:     characteristic stress
    :nu:        Poisson ratio
    :r_0:       estimated radius undeformed cell
    :fn:        fn-coefficients
    :gn:        gn-coefficients


    Return
    ------
    :alpha:     n-th coefficient of expansion
    """

    G = E_0 / (2*(1+nu))  # shear stress

    nom_alpha = (fn[n]*(2*nu-2*(nu-1)*n**2+nu*n-1) +
                 gn[n]*n*(n+1)*((2*nu-1)*n-nu+2))
    denom_alpha = (n-1)*(nu+n**2+2*nu*n+n+1)

    alpha = sig_c * r_0 / (2 * G) * nom_alpha / denom_alpha

    return alpha


def beta_sp(n, E_0, sig_c, nu, r_0, fn, gn):
    """returns beta

    **Expansion coefficients for sphere model: equation 25b**

    Args
    ----
    :n:         n-th term
    :E_0:       stiffness of sphere
    :sig_c:     characteristic stress
    :nu:        Poisson ratio
    :r_0:       estimated radius undeformed cell
    :fn:        fn-coefficients
    :gn:        gn-coefficients


    Return
    ------
    :beta:      n-th coefficient of expansion
    """
    G = E_0 / (2*(1+nu))  # shear stress

    nom_beta = (fn[n]*(-nu+(2*nu-1)*n+2) +
                gn[n]*(nu-2*(nu-1)*n**2+(3*nu-1)*n+1))
    denom_beta = (n-1)*(nu+n**2+2*nu*n+n+1)

    beta = sig_c * r_0 / (2 * G) * nom_beta / denom_beta

    return beta


def disp_sh(th, gamma, Eh, sig_c, nu, r_0, fn, gn):
    """returns u_r, u_th

    **Calculates displacement for shell model: equation 12**

    Args
    ----
    :th:        polar angle
    :gamma:     surface tension
    :E_0:       stiffness of shell
    :sig_c:     characteristic stress
    :nu:        Poisson ratio
    :r_0:       estimated radius undeformed cell
    :fn:        fn-coefficients
    :gn:        gn-coefficients


    Return
    ------
    :u_r:       radial displacement
    :u_th:      polar displacement
    """

    u_r = 0  # radial displacement
    u_th = 0  # polar displacement

    n = np.arange(2, len(fn), 1)
    al = alpha_sh(n, gamma=gamma, Eh=Eh, sig_c=sig_c, nu=nu, r_0=r_0,
                  fn=fn, gn=gn)
    be = beta_sh(n, gamma=gamma, Eh=Eh, sig_c=sig_c, nu=nu, r_0=r_0,
                 fn=fn, gn=gn)
    be[np.arange(2, len(fn), 2)-2] = 0

    for i in np.arange(2, len(fn), 1):
        u_r += al[i-2] * P_n(np.cos(th), i)
        u_th += be[i-2] * dP_n(np.cos(th), i) * -np.sin(th)

    return u_r, u_th


def disp_sp(th, E_0, sig_c, nu, r_0, fn, gn):
    """returns u_r, u_th

    **Calculates displacement for sphere model: equation 23, 24**

    Args
    ----
    :th:        polar angle
    :E_0:       stiffness of sphere
    :sig_c:     characteristic stress
    :nu:        Poisson ratio
    :r_0:       estimated radius undeformed cell
    :fn:        fn-coefficients
    :gn:        gn-coefficients


    Return
    ------
    :u_r:       radial displacement
    :u_th:      polar displacement
    """

    u_r = 0
    u_th = 0

    n = np.arange(2, len(fn), 1)
    al = alpha_sp(n, E_0=E_0, sig_c=sig_c, nu=nu, r_0=r_0,
                  fn=fn, gn=gn)
    be = beta_sp(n, E_0=E_0, sig_c=sig_c, nu=nu, r_0=r_0,
                 fn=fn, gn=gn)
    for i in np.arange(2, len(fn), 1):
        u_r += al[i-2] * P_n(np.cos(th), i)
        u_th += be[i-2] * dP_n(np.cos(th), i) * -np.sin(th)

    return u_r, u_th


def def_sh(th, gamma, Eh, sig_c, nu, r_0, fn, gn):
    """returns A, d, x_d, z_d

    **Calculates area, deformation and coordinates of deformed shell**

    Args
    ----
    :th:        polar angle
    :gamma:     surface tension
    :Eh:        stiffness of shell
    :sig_c:     characteristic stress
    :nu:        Poisson ratio
    :r_0:       estimated radius undeformed cell
    :fn:        fn-coefficients
    :gn:        gn-coefficients


    Return
    ------
    :A:         area
    :d:         deformation
    :x_d:       x-coordinate
    :z_d:       z-coordinate
    """

    u_r, u_th = disp_sh(th=th, gamma=gamma, Eh=Eh, sig_c=sig_c, nu=nu,
                        r_0=r_0, fn=fn, gn=gn)

    # transform into polar coordinates. Defined on page 2027
    x_d = (r_0 + u_r) * np.sin(th) + u_th * np.cos(th)
    z_d = (r_0 + u_r) * np.cos(th) - u_th * np.sin(th)

    A = np.abs(area(x_d, z_d))
    P = 0
    for i in range(len(x_d)-1):
        P += np.sqrt((x_d[i+1]-x_d[i])**2 + (z_d[i+1]-z_d[i])**2)
    c = 2 * np.sqrt(np.pi * A) / P
    d = 1-c

    return A, d, x_d, z_d


def def_sp(th, E_0, sig_c, nu, r_0, fn, gn):
    """returns A, d, x_d, z_d

    **Calculates area, deformation and coordinates of deformed sphere**

    Args
    ----
    :th:        polar angle
    :E_0:       stiffness of sphere
    :sig_c:     characteristic stress
    :nu:        Poisson ratio
    :r_0:       estimated radius undeformed cell
    :fn:        fn-coefficients
    :gn:        gn-coefficients


    Return
    ------
    :A:         area
    :d:         deformation
    :x_d:       x-coordinate
    :z_d:       z-coordinate
    """

    u_r, u_th = disp_sp(th=th, E_0=E_0, sig_c=sig_c, nu=nu,
                        r_0=r_0, fn=fn, gn=gn)

    # transform into polar coordinates. Defined on page 2027
    x_d = (r_0 + u_r) * np.sin(th) + u_th * np.cos(th)
    z_d = (r_0 + u_r) * np.cos(th) - u_th * np.sin(th)

    A = np.abs(area(x_d, z_d))
    P = 0
    for i in range(len(x_d)-1):
        P += np.sqrt((x_d[i+1]-x_d[i])**2 + (z_d[i+1]-z_d[i])**2)
    c = 2 * np.sqrt(np.pi * A) / P
    d = 1-c

    return A, d, x_d, z_d


def R2(x, z, th=np.pi/2):
    """returns x_rot, z_rot

    **rotates x, z coordinates by angle th**

    Args
    ----
    :x:       x-coordinates from data
    :z:       z-coordinates from data
    :th:      rotation angle


    Return
    ------
    :x_rot:   rotated x-coordinates
    :z_rot:   rotated z-coordinates
    """

    # vectorize input
    vec = np.transpose(np.array([x, z]))
    vec_rot = np.copy(vec)

    # 2-dim rotation matrix
    R = np.array([[np.cos(th), -np.sin(th)],
                  [np.sin(th), np.cos(th)]])

    # rotate dataset
    for i in range(vec.shape[0]):
        vec_rot[i] = np.matmul(R, vec[i])
    x_rot = vec_rot[:, 0]
    z_rot = vec_rot[:, 1]

    return x_rot, z_rot


def cost_sh(x_0, z_0, gamma, sig_c, nu, r_0, fn, gn, Eh, x_s, z_s):
    """returns J

    **Calculates the cost of the shell model**

    Args
    ----
    :x_0:       x-coordinates from data
    :z_0:       z-coordinates from data
    :gamma:     tension of shell
    :sig_c:     characteristic stress
    :nu:        Poisson ratio
    :r_0:       estimated radius undeformed cell
    :fn:        fn-coefficients
    :gn:        gn-coefficients
    :Eh:        stiffness of shell
    :x_s:       center of mass x-coordinate
    :z_s:       center of mass z-coordinate

    Return
    ------
    :J:         cost
    """

    r_exp = np.sqrt((x_0-x_s)**2 + (z_0-z_s)**2)
    th_exp = np.zeros(len(r_exp))

    half_idx = int(len(r_exp)/2)
    th_exp[:half_idx] = np.arccos((z_0[:half_idx]-z_s) / r_exp[:half_idx])
    th_exp[half_idx:] = 2*np.pi - np.arccos((z_0[half_idx:]-z_s) /
                                            r_exp[half_idx:])

    A_sh, d_sh, x_sh, z_sh = def_sh(th_exp, gamma, Eh, sig_c, nu, r_0, fn, gn)

    r_sh = np.sqrt(x_sh**2 + z_sh**2)

    J = np.sum(np.abs(r_exp - r_sh)) / len(x_0) * 1e6

    return J


def cost_test(x_0, z_0, gamma, sig_c, nu, r_0, fn, gn, Eh, x_s, z_s):
    """returns J

    **Test: Calculates the cost of the shell model with different costs**

    Args
    ----
    :x_0:       x-coordinates from data
    :z_0:       z-coordinates from data
    :gamma:     tension of shell
    :sig_c:     characteristic stress
    :nu:        Poisson ratio
    :r_0:       estimated radius undeformed cell
    :fn:        fn-coefficients
    :gn:        gn-coefficients
    :Eh:        stiffness of shell
    :x_s:       center of mass x-coordinate
    :z_s:       center of mass z-coordinate

    Return
    ------
    :J:         cost
    """

    r_exp = np.sqrt((x_0-x_s)**2 + (z_0-z_s)**2)
    th_exp = np.zeros(len(r_exp))

    half_idx = int(len(r_exp)/2)
    th_exp[:half_idx] = np.arccos((z_0[:half_idx]-z_s) / r_exp[:half_idx])
    th_exp[half_idx:] = 2*np.pi - np.arccos((z_0[half_idx:]-z_s) /
                                            r_exp[half_idx:])

    A_0 = area(x_0, z_0)
    P = 0
    for i in range(len(x_0)-1):
        P += np.sqrt((x_0[i+1]-x_0[i])**2 + (z_0[i+1]-z_0[i])**2)
    c_0 = 2 * np.sqrt(np.pi * A_0) / P
    d_0 = 1-c_0

    A_sh, d_sh, x_sh, z_sh = def_sh(th_exp, gamma, Eh, sig_c, nu, r_0, fn, gn)

    r_sh = np.sqrt(x_sh**2 + z_sh**2)

    J_r = np.sum(np.abs(r_exp - r_sh)) / len(x_0) * 1e6
    J_A = np.sqrt(np.sum(np.abs(A_0 - A_sh)) * 1e12)
    J_d = np.sum(np.abs(d_0 - d_sh)) * 1e6

    J = J_r + J_A + J_d

    return J


def d_cost_sh(x_0, z_0, gamma, sig_c, nu, r_0, fn, gn, P, d):
    """returns dJ

    **Calculates the derivative of the cost of the shell model
    of parameter P[d]**

    Args
    ----
    :x_0:       x-coordinates from data
    :z_0:       z-coordinates from data
    :gamma:     tension of shell
    :sig_c:     characteristic stress
    :nu:        Poisson ratio
    :r_0:       estimated radius undeformed cell
    :fn:        fn-coefficients
    :gn:        gn-coefficients
    :P:         P[0]: Eh, P[1]: x_s, P[2]: z_s
    :d:         derivative with respect to parameter d


    Return
    ------
    :dJ:    derivative of cost
    """

    eps = 1e-9

    P_p = np.copy(P)  # placeholder
    P_n = np.copy(P)  # placeholder
    P_p[d] += eps
    P_n[d] -= eps

    # derivative of parameter P[d]
    dJ = (cost_sh(x_0, z_0, gamma, sig_c, nu, r_0, fn, gn, *P_p) -
          cost_sh(x_0, z_0, gamma, sig_c, nu, r_0, fn, gn, *P_n)) / (2 * eps)

    return dJ


def cost_deriv_sh(x_0, z_0, gamma, sig_c, nu, r_0, fn, gn, P):
    """returns dJ

    **Calculates the cost gradients of the shell model**

    Args
    ----
    :x_0:       x-coordinates from data
    :z_0:       z-coordinates from data
    :gamma:     tension of shell
    :sig_c:     characteristic stress
    :nu:        Poisson ratio
    :r_0:       estimated radius undeformed cell
    :fn:        fn-coefficients
    :gn:        gn-coefficients
    :P:         P[0]: Eh, P[1]: x_s, P[2]: z_s

    Return
    ------
    :DJ:        gradient of cost w.r.t. parameters
    """

    # parallelize
    num_cores = multiprocessing.cpu_count()
    inputs = range(3)

    DJ = np.array(Parallel(n_jobs=num_cores)(delayed(d_cost_sh)
                                             (x_0, z_0, gamma, sig_c,
                                              nu, r_0, fn, gn, P, i)
                  for i in inputs))

    return DJ


def optimize_sh(x_0, z_0, gamma_pre, sig_c, nu, r_0, fn, gn,
                it_max, alpha, P):
    """returns it, J, P

    **Optimizes the shell model and returns the cost and parameters**

    Args
    ----
    :x_0:       x-coordinates from data
    :z_0:       z-coordinates from data
    :gamma:     tension of shell
    :sig_c:     characteristic stress
    :nu:        Poisson ratio
    :r_0:       estimated radius undeformed cell
    :fn:        fn-coefficients
    :gn:        gn-coefficients
    :it_max:    maximum of iterations
    :alpha:     learning rate
    :P:         initial parameters: P[0]: stiffness, P[1]: x_s, P[2]: z_s

    Return
    ------
    :it:        iteratons
    :J:         cost
    :P:         optimized parameters
    """

    J = np.array([])  # cost
    it = np.array([])  # iterations
    m = np.zeros(P.size)  # start values
    v = np.zeros(P.size)  # start values
    beta1 = .9  # parameter Adam optimizer
    beta2 = .999  # parameter Adam optimizer
    epsilon = 1e-8  # preventing from dividing by zero
    Alpha = np.array([alpha, alpha*1e-7, alpha*1e-7])  # learning rates

    # start optimizing
    start_time = time.time()

    for i in range(it_max):
        gamma = gamma_pre * P[0]
        J = np.append(J, cost_sh(x_0, z_0, gamma, sig_c, nu, r_0, fn, gn, *P))
        it = np.append(it, i)
        DJ = cost_deriv_sh(x_0, z_0, gamma, sig_c, nu, r_0, fn, gn, P)
        lr = Alpha * np.sqrt((1 - beta2) / (1 - beta1))  # rate

        # update parameters
        m = beta1 * m + (1 - beta1) * DJ
        v = beta2 * v + ((1 - beta2) * DJ) * DJ
        P = P - lr * m / (np.sqrt(v) + epsilon)

        # display every 10 iterations
        if np.mod(i, 50) == 0:
            print('iteration nr. ' + str(i))
            print("--- %s seconds ---" % (time.time() - start_time))
            print("J = " + str(J[-1]))
        if np.mod(i, 50) == 0:
            print('iteration nr. ' + str(i))
            print("--- %s seconds ---" % (time.time() - start_time))
            print("Eh = " + str(P[0]))

    return it, J, P


def cost_sp(x_0, z_0, sig_c, nu, r_0, fn, gn, E_0, x_s, z_s):
    """returns J

    **Calculates the cost of the sphere model**

    Args
    ----
    :x_0:       x-coordinates from data
    :z_0:       z-coordinates from data
    :gamma:     tension of shell
    :sig_c:     characteristic stress
    :nu:        Poisson ratio
    :r_0:       estimated radius undeformed cell
    :fn:        fn-coefficients
    :gn:        gn-coefficients
    :E_0:       stiffness of sphere
    :x_s:       center of mass x-coordinate
    :z_s:       center of mass z-coordinate

    Return
    ------
    :J:         cost
    """

    r_exp = np.sqrt((x_0-x_s)**2 + (z_0-z_s)**2)
    th_exp = np.zeros(len(r_exp))

    half_idx = int(len(r_exp)/2)
    th_exp[:half_idx] = np.arccos((z_0[:half_idx]-z_s) / r_exp[:half_idx])
    th_exp[half_idx:] = 2*np.pi - np.arccos((z_0[half_idx:]-z_s) /
                                            r_exp[half_idx:])

    A_sp, d_sp, x_sp, z_sp = def_sp(th_exp, E_0, sig_c, nu, r_0, fn, gn)

    r_sp = np.sqrt(x_sp**2 + z_sp**2)

    J = np.sum(np.abs(r_exp - r_sp)) / len(x_0) * 1e6

    return J


def d_cost_sp(x_0, z_0, sig_c, nu, r_0, fn, gn, P, d):
    """returns dJ

    **Calculates the derivative of the cost of the sphere model
    of parameter P[d]**

    Args
    ----
    :x_0:       x-coordinates from data
    :z_0:       z-coordinates from data
    :gamma:     tension of shell
    :sig_c:     characteristic stress
    :nu:        Poisson ratio
    :r_0:       estimated radius undeformed cell
    :fn:        fn-coefficients
    :gn:        gn-coefficients
    :P:         P[0]: E_0, P[1]: x_s, P[2]: z_s
    :d:         derivative with respect to parameter d


    Return
    ------
    :dJ:    derivative of cost
    """

    eps = 1e-9

    P_p = np.copy(P)  # placeholder
    P_n = np.copy(P)  # placeholder
    P_p[d] += eps
    P_n[d] -= eps

    # derivative of parameter P[d]
    dJ = (cost_sp(x_0, z_0, sig_c, nu, r_0, fn, gn, *P_p) -
          cost_sp(x_0, z_0, sig_c, nu, r_0, fn, gn, *P_n)) / (2 * eps)

    return dJ


def cost_deriv_sp(x_0, z_0, sig_c, nu, r_0, fn, gn, P):
    """returns dJ

    **Calculates the cost gradients of the sphere model**

    Args
    ----
    :x_0:       x-coordinates from data
    :z_0:       z-coordinates from data
    :sig_c:     characteristic stress
    :nu:        Poisson ratio
    :r_0:       estimated radius undeformed cell
    :fn:        fn-coefficients
    :gn:        gn-coefficients
    :P:         P[0]: E_0, P[1]: x_s, P[2]: z_s

    Return
    ------
    :DJ:        gradient of cost w.r.t. parameters
    """

    # parallelize
    num_cores = multiprocessing.cpu_count()
    inputs = range(3)

    DJ = np.array(Parallel(n_jobs=num_cores)(delayed(d_cost_sp)
                                             (x_0, z_0, sig_c,
                                              nu, r_0, fn, gn, P, i)
                  for i in inputs))

    return DJ


def optimize_sp(x_0, z_0, sig_c, nu, r_0, fn, gn,
                it_max, alpha, P):
    """returns it, J, P

    **Optimizes the sphere model and returns the cost and parameters**

    Args
    ----
    :x_0:       x-coordinates from data
    :z_0:       z-coordinates from data
    :sig_c:     characteristic stress
    :nu:        Poisson ratio
    :r_0:       estimated radius undeformed cell
    :fn:        fn-coefficients
    :gn:        gn-coefficients
    :it_max:    maximum of iterations
    :alpha:     learning rate
    :P:         initial parameters: P[0]: stiffness, P[1]: x_s, P[2]: z_s

    Return
    ------
    :it:        iteratons
    :J:         cost
    :P:         optimized parameters
    """

    J = np.array([])  # cost
    it = np.array([])  # iterations
    m = np.zeros(P.size)  # start values
    v = np.zeros(P.size)  # start values
    beta1 = .9  # parameter Adam optimizer
    beta2 = .999  # parameter Adam optimizer
    epsilon = 1e-8  # preventing from dividing by zero
    Alpha = np.array([alpha, alpha*1e-9, alpha*1e-9])  # learning rates

    # start optimizing
    start_time = time.time()

    for i in range(it_max):
        J = np.append(J, cost_sp(x_0, z_0, sig_c, nu, r_0, fn, gn, *P))
        it = np.append(it, i)
        DJ = cost_deriv_sp(x_0, z_0, sig_c, nu, r_0, fn, gn, P)
        lr = Alpha * np.sqrt((1 - beta2) / (1 - beta1))  # rate

        # update parameters
        m = beta1 * m + (1 - beta1) * DJ
        v = beta2 * v + ((1 - beta2) * DJ) * DJ
        P = P - lr * m / (np.sqrt(v) + epsilon)

        # display every 10 iterations
        if np.mod(i, 50) == 0:
            print('iteration nr. ' + str(i))
            print("--- %s seconds ---" % (time.time() - start_time))
            print("J = " + str(J[-1]))
        if np.mod(i, 50) == 0:
            print('iteration nr. ' + str(i))
            print("--- %s seconds ---" % (time.time() - start_time))
            print("E0 = " + str(P[0]))

    return it, J, P
