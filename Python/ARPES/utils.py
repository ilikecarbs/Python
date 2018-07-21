#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 09:57:01 2018

@author: ilikecarbs

%%%%%%%%%%%%%%%%%%%%%
        utils
%%%%%%%%%%%%%%%%%%%%%

**Useful helper functions, mainly used for ARPES.py**

.. note::
        To-Do:
            -
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as la
import utils
from scipy.stats import exponnorm
from scipy import special
from scipy.ndimage.filters import gaussian_filter


def find(array, val):
    """returns array[_val], _val

    **Searches entry in array closest to val.**

    Args
    ----------
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


def Shirley(EDC):
    """returns b

    **Generates a Shirley background from given edc**

    Args
    ----------
    :array:     entry value in array
    :_val:      index of entry

    Return
    ------
    :array[_val]:   entry value in array
    :_val:          index of entry
    """

    n = len(EDC)
    A = 1e-5
    shirley = np.ones((n))
    shirley[-1] = EDC[-1]
    shirley[-2] = EDC[-1]
    it = 10  # iterations

    # start algorithm
    for k in range(it):
        for i in range(n - 2):
            SUM = 0.
            for j in np.arange(i + 1, n):
                SUM += (EDC[j] - shirley[j])
            shirley[i] = shirley[-1] + A * SUM
        A = A * (1. + (EDC[0] - shirley[0]) / EDC[0])

    return shirley


def paramSRO():
    """returns param

    **Parameter set of TB model Sr2RuO4 arXiv:1212.3994v1**

    Args
    ----------

    Return
    ------
    :param:   parameter dictionary

    - t1:   Nearest neighbour for out-of-plane orbitals large
    - t2:   Nearest neighbour for out-of-plane orbitals small
    - t3:   Nearest neighbour for dxy orbitals
    - t4:   Next nearest neighbour for dxy orbitals
    - t5:   Next next nearest neighbour for dxy orbitals
    - t6:   Off diagonal matrix element
    - mu:   Chemical potential
    - so:   spin orbit coupling
    """

    param = dict([('t1', .145), ('t2', .016), ('t3', .081), ('t4', .039),
                  ('t5', .005), ('t6', 0), ('mu', .122), ('so', .032)])
    return param


def paramCSRO20():
    """returns param

    **parameter set of TB model D. Sutter et al. :-)**

    Args
    ----------

    Return
    ------
    :param:   parameter dictionary

    - t1:   Nearest neighbour for out-of-plane orbitals large
    - t2:   Nearest neighbour for out-of-plane orbitals small
    - t3:   Nearest neighbour for dxy orbitals
    - t4:   Next nearest neighbour for dxy orbitals
    - t5:   Next next nearest neighbour for dxy orbitals
    - t6:   Off diagonal matrix element
    - mu:   Chemical potential
    - so:   spin orbit coupling
    """

    param = dict([('t1', .115), ('t2', .002), ('t3', .071), ('t4', .039),
                  ('t5', .012), ('t6', 0), ('mu', .084), ('so', .037)])
    return param


def paramCSRO30():
    """returns param

    **Parameter test set CSRO30**

    Args
    ----------

    Return
    ------
    :param:   parameter dictionary

    - t1:   Nearest neighbour for out-of-plane orbitals large
    - t2:   Nearest neighbour for out-of-plane orbitals small
    - t3:   Nearest neighbour for dxy orbitals
    - t4:   Next nearest neighbour for dxy orbitals
    - t5:   Next next nearest neighbour for dxy orbitals
    - t6:   Off diagonal matrix element
    - mu:   Chemical potential
    - so:   spin orbit coupling
    """

    param = dict([('t1', .1), ('t2', .005), ('t3', .081), ('t4', .04),
                  ('t5', .01), ('t6', 0), ('mu', .08), ('so', .04)])
    return param


class TB:
    """
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
             Tight binding class
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    **Tight binding models for Sr2RuO4 and Ca1.8Sr0.2RuO4**
    """
    def __init__(self, a=np.pi, kbnd=1, kpoints=100):
        """returns self.a, self.coord

        **Initializing tight binding class**

        Args
        ----------
        :a:         TB lattice constant in units pi/a
        :kbnd:      boundary of model in units pi/a
        :kpoints:   k-mesh granularity

        Return
        ------
        :self.a:        lattice constant
        :self.coord:    k-mesh
        """

        x = np.linspace(-kbnd, kbnd, kpoints)
        y = np.linspace(-kbnd, kbnd, kpoints)
        [X, Y] = np.meshgrid(x, y)
        self.a = a
        self.coord = dict([('x', x), ('y', y), ('X', X), ('Y', Y)])

    def SRO(self, param=paramSRO(), e0=0, vert=False, proj=False):
        """returns self.bndstr, self.kx, self.ky, self.FS

        **Calculates band structure from 3 band tight binding model**

        Args
        ----------
        :param:     TB parameters
        :e0:        chemical potential shift
        :vert:      'True': plots useful numeration of vertices for figures
        :proj:      'True': projects onto orbitals

        Return
        ------
        :self.bndstr:   band structure dictionary: yz, xz and xy
        :self.kx:       kx coordinates (vert=True and proj=True)
        :self.ky:       ky coordinates (vert=True and proj=True)
        :self.FS:       FS coordinates (vert=True and proj=True)
        """

        # Load TB parameters
        t1 = param['t1']  # Nearest neighbour for out-of-plane orbitals large
        t2 = param['t2']  # Nearest neighbour for out-of-plane orbitals small
        t3 = param['t3']  # Nearest neighbour for dxy orbitals
        t4 = param['t4']  # Next nearest neighbour for dxy orbitals
        t5 = param['t5']  # Next next nearest neighbour for dxy orbitals
        t6 = param['t6']  # Off diagonal matrix element
        mu = param['mu']  # Chemical potential
        so = param['so']  # spin orbit coupling

        coord = self.coord
        a = self.a
        x = coord['x']
        y = coord['y']
        X = coord['X']
        Y = coord['Y']

        # Hopping terms
        fyz = - 2 * t2 * np.cos(X * a) - 2 * t1 * np.cos(Y * a)
        fxz = - 2 * t1 * np.cos(X * a) - 2 * t2 * np.cos(Y * a)
        fxy = - 2 * t3 * (np.cos(X * a) + np.cos(Y * a)) - \
            4 * t4 * (np.cos(X * a) * np.cos(Y * a)) - \
            2 * t5 * (np.cos(2 * X * a) + np.cos(2 * Y * a))
        off = - 4 * t6 * (np.sin(X * a) * np.sin(Y * a))

        # Placeholders energy eigenvalues
        yz = np.ones((len(x), len(y)))
        xz = np.ones((len(x), len(y)))
        xy = np.ones((len(x), len(y)))

        # Tight binding Hamiltonian
        def H(i, j):
            H = np.array([[fyz[i, j] - mu, off[i, j] + complex(0, so), -so],
                          [off[i, j] - complex(0, so), fxz[i, j] - mu,
                           complex(0, so)],
                          [-so, -complex(0, so), fxy[i, j] - mu]])
            return H

        # Diagonalization of symmetric Hermitian matrix on k-mesh
        for i in range(len(x)):
            for j in range(len(y)):
                val = la.eigvalsh(H(i, j))
                val = np.real(val)
                yz[i, j] = val[0]
                xz[i, j] = val[1]
                xy[i, j] = val[2]

        # Band structure
        bndstr = (yz, xz, xy)

        # Placeholder for contours
        C = ()

        # Projectors
        Pyz = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]])
        Pxz = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
        Pxy = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 1]])

        # Generate contours C
        n = 0
        for bnd in bndstr:
            n += 1
            plt.subplot(1, 3, n)
            c = plt.contour(X, Y, bnd, colors='black',
                            linestyles='-', levels=e0)
            C = C + (c,)
            plt.axis('equal')

        # Get vertices and label them, useful for plotting separate
        # parts of the Fermi surface pockets with different colors
        if vert:
            if proj:  # Also project eigenbasis onto orbitals from here

                # Placeholders for projected FS
                kx = np.linspace(np.min(x), np.max(x), 1000)
                ky = np.linspace(np.min(y), np.max(y), 1000)
                FS = np.zeros((len(kx), len(ky)))

            for n_bnd in range(len(bndstr)):
                p = C[n_bnd].collections[0].get_paths()
                p = np.asarray(p)
                plt.figure('SRO_vertices')

                # Placeholders vertices
                V_x = ()
                V_y = ()

                # Get vertices and plot them
                for i in range(len(p)):
                    v = p[i].vertices
                    v_x = v[:, 0]
                    v_y = v[:, 1]
                    V_x = V_x + (v_x,)
                    V_y = V_y + (v_y,)
                    plt.plot(v_x, v_y)
                    plt.axis('equal')
                    plt.text(v_x[0], v_y[0], str(i))
                    plt.show()

                if proj:  # Do projection for all vertices
                    for j in range(len(V_x)):  # Loop over all vertices
                        for i in range(len(V_x[j])):  # Loop over k-points

                            #  Find values and diagonalize
                            val_x, ind_x = utils.find(x, V_x[j][i])
                            val_y, ind_y = utils.find(y, V_y[j][i])
                            val_proj, vec_proj = la.eigh(H(ind_x, ind_y))
                            val_proj = np.real(val_proj)

                            # orbital weights
                            wyz = np.real(
                                    np.sum(
                                            np.conj(vec_proj[:, n_bnd]) *
                                            (Pyz * vec_proj[:, n_bnd])))
                            wxz = np.real(
                                    np.sum(
                                            np.conj(vec_proj[:, n_bnd]) *
                                            (Pxz * vec_proj[:, n_bnd])))
                            wxy = np.real(
                                    np.sum(
                                            np.conj(vec_proj[:, n_bnd]) *
                                            (Pxy * vec_proj[:, n_bnd])))

                            # Total out-of-plane weight
                            wz = wyz + wxz

                            # Weight for divergence colorscale
                            w = wz - wxy

                            # Build Fermi surface
                            xval, _xval = utils.find(kx, V_x[j][i])
                            yval, _yval = utils.find(ky, V_y[j][i])
                            FS[_xval, _yval] = w

            FS = gaussian_filter(FS, sigma=10, mode='constant')
            self.kx = kx
            self.ky = ky
            self.FS = FS

        self.bndstr = dict([('yz', yz), ('xz', xz), ('xy', xy)])

    def CSRO(self, param=paramCSRO20(), e0=0, vert=False, proj=True):
        """returns self.bndstr, self.kx, self.ky, self.FS

        **Calculates band structure from 6 band tight binding model**

        Args
        ----------
        :param:     TB parameters
        :e0:        chemical potential shift
        :vert:      'True': plots useful numeration of vertices for figures
        :proj:      'True': projects onto orbitals

        Return
        ------
        :self.bndstr:   band structure dictionary: yz, xz and xy
        :self.kx:       kx coordinates (vert=True and proj=True)
        :self.ky:       ky coordinates (vert=True and proj=True)
        :self.FS:       FS coordinates (vert=True and proj=True)
        """

        # Load TB parameters
        t1 = param['t1']  # Nearest neighbour for out-of-plane orbitals large
        t2 = param['t2']  # Nearest neighbour for out-of-plane orbitals small
        t3 = param['t3']  # Nearest neighbour for dxy orbitals
        t4 = param['t4']  # Next nearest neighbour for dxy orbitals
        t5 = param['t5']  # Next next nearest neighbour for dxy orbitals
        t6 = param['t6']  # Off diagonal matrix element
        mu = param['mu']  # Chemical potential
        so = param['so']  # spin orbit coupling

        coord = self.coord
        a = self.a
        x = coord['x']
        y = coord['y']
        X = coord['X']
        Y = coord['Y']

        # Hopping terms
        fx = -2 * np.cos((X + Y) / 2 * a)
        fy = -2 * np.cos((X - Y) / 2 * a)
        f4 = -2 * t4 * (np.cos(X * a) + np.cos(Y * a))
        f5 = -2 * t5 * (np.cos((X + Y) * a) + np.cos((X - Y) * a))
        f6 = -2 * t6 * (np.cos(X * a) - np.cos(Y * a))

        # Placeholders energy eigenvalues
        Ayz = np.ones((len(x), len(y)))
        Axz = np.ones((len(x), len(y)))
        Axy = np.ones((len(x), len(y)))
        Byz = np.ones((len(x), len(y)))
        Bxz = np.ones((len(x), len(y)))
        Bxy = np.ones((len(x), len(y)))

        # TB submatrix
        def A(i, j):
            A = np.array([[-mu, complex(0, so) + f6[i, j], -so],
                          [-complex(0, so) + f6[i, j], -mu, complex(0, so)],
                          [-so, -complex(0, so), -mu + f4[i, j] + f5[i, j]]])
            return A

        # TB submatrix
        def B(i, j):
            B = np.array([[t2 * fx[i, j] + t1 * fy[i, j], 0, 0],
                          [0, t1 * fx[i, j] + t2 * fy[i, j], 0],
                          [0, 0, t3 * (fx[i, j] + fy[i, j])]])
            return B

        # Tight binding Hamiltonian
        def H(i, j):
            C1 = np.concatenate((A(i, j), B(i, j)), 1)
            C2 = np.concatenate((B(i, j), A(i, j)), 1)
            H = np.concatenate((C1, C2), 0)
            return H

        # Diagonalization of symmetric Hermitian matrix on k-mesh
        for i in range(len(x)):
            for j in range(len(y)):
                val = la.eigvalsh(H(i, j))
                val = np.real(val)
                Ayz[i, j] = val[0]
                Axz[i, j] = val[1]
                Axy[i, j] = val[2]
                Byz[i, j] = val[3]
                Bxz[i, j] = val[4]
                Bxy[i, j] = val[5]

        # Band structure
        bndstr = (Ayz, Axz, Axy, Byz, Bxz, Bxy)

        # Placeholder for contours
        C = ()

        # Projectors
        PAyz = np.array([[1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]])
        PAxz = np.array([[0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]])
        PAxy = np.array([[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0],
                         [0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]])
        PByz = np.array([[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0],
                         [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]])
        PBxz = np.array([[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0]])
        PBxy = np.array([[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1]])

        # Generate contours C
        n = 0
        for bnd in bndstr:
            n += 1
            plt.subplot(2, 3, n)
            c = plt.contour(X, Y, bnd, colors='black',
                            linestyles='-', levels=e0)
            C = C + (c,)
            plt.axis('equal')

        # Get vertices and label them, useful for plotting separate
        # parts of the Fermi surface pockets with different colors
        if vert:
            if proj:  # Also project eigenbasis onto orbitals from here

                # Placeholders for projected FS
                kx = np.linspace(np.min(x), np.max(x), 1000)
                ky = np.linspace(np.min(y), np.max(y), 1000)
                FS = np.zeros((len(kx), len(ky)))

            for n_bnd in range(len(bndstr)):
                p = C[n_bnd].collections[0].get_paths()
                p = np.asarray(p)
                plt.figure('CSRO_vertices')

                # Placeholders vertices
                V_x = ()
                V_y = ()

                # Get vertices and plot them
                for i in range(len(p)):
                    v = p[i].vertices
                    v_x = v[:, 0]
                    v_y = v[:, 1]
                    V_x = V_x + (v_x,)
                    V_y = V_y + (v_y,)
                    plt.plot(v_x, v_y)
                    plt.axis('equal')
                    plt.text(v_x[0], v_y[0], str(i))
                    plt.show()

                if proj:  # Do projection for all vertices
                    for j in range(len(V_x)):  # Loop over all vertices
                        for i in range(len(V_x[j])):  # Loop over k-points

                            # Find values and diagonalize
                            val_x, ind_x = utils.find(x, V_x[j][i])
                            val_y, ind_y = utils.find(y, V_y[j][i])
                            eval_proj, evec_proj = la.eigh(H(ind_x, ind_y))
                            eval_proj = np.real(eval_proj)

                            # orbital weights
                            wAyz = np.real(
                                    np.sum(
                                            np.conj(evec_proj[:, n_bnd]) *
                                            (PAyz * evec_proj[:, n_bnd])))
                            wAxz = np.real(
                                    np.sum(
                                            np.conj(evec_proj[:, n_bnd]) *
                                            (PAxz * evec_proj[:, n_bnd])))
                            wAxy = np.real(
                                    np.sum(
                                            np.conj(evec_proj[:, n_bnd]) *
                                            (PAxy * evec_proj[:, n_bnd])))
                            wByz = np.real(
                                    np.sum(
                                            np.conj(evec_proj[:, n_bnd]) *
                                            (PByz * evec_proj[:, n_bnd])))
                            wBxz = np.real(
                                    np.sum(
                                            np.conj(evec_proj[:, n_bnd]) *
                                            (PBxz * evec_proj[:, n_bnd])))
                            wBxy = np.real(
                                    np.sum(
                                            np.conj(evec_proj[:, n_bnd]) *
                                            (PBxy * evec_proj[:, n_bnd])))

                            # Total out-of-plane weight
                            wz = wAyz + wAxz + wByz + wBxz

                            # Total in-plane weight
                            wxy = wAxy + wBxy

                            # Weight for divergence colorscale
                            w = wz - wxy

                            # Build Fermi surface
                            xval, _xval = utils.find(kx, V_x[j][i])
                            yval, _yval = utils.find(ky, V_y[j][i])
                            FS[_xval, _yval] = w + 0

            FS = gaussian_filter(FS, sigma=10, mode='constant')
            self.kx = kx
            self.ky = ky
            self.FS = FS

        self.bndstr = dict([('Ayz', Ayz), ('Axz', Axz), ('Axy', Axy),
                            ('Byz', Byz), ('Bxz', Bxz), ('Bxy', Bxy)])

    def single(self, param):
        """returns self.bndstr, self.kx, self.ky, self.FS

        **Calculates single band tight binding band structure**

        Args
        ----------
        :param:     TB parameters

        Return
        ------
        :self.bndstr:   band structure dictionary: yz, xz and xy
        """

        # Load TB parameters
        t1 = param['t1']  # Nearest neighbour hopping
        t2 = param['t2']  # Next nearest neighbour hopping
        t3 = param['t3']  # NNN neighbour hopping
        t4 = param['t4']  # NNNN hopping
        t5 = param['t5']  # NNNNN hopping
        mu = param['mu']  # Chemical potential

        coord = self.coord
        a = self.a

        X = coord['X']
        Y = coord['Y']

        bndstr = (- mu -
                  2 * t1 * (np.cos(X * a) + np.cos(Y * a)) -
                  4 * t2 * (np.cos(X * a) * np.cos(Y * a)) -
                  2 * t3 * (np.cos(2 * X * a) + np.cos(2 * Y * a)) -
                  4 * t4 * (np.cos(2 * X * a) * np.cos(Y * a) +
                            np.cos(X * a) * np.cos(2 * Y * a)) -
                  4 * t5 * (np.cos(2 * X * a) * np.cos(2 * Y * a)))

        self.bndstr = dict([('bndstr', bndstr)])


def CSRO_eval(x, y):
    a = np.pi
    #Load TB parameters
    param = paramCSRO20()  
    t1 = param['t1']; t2 = param['t2']; t3 = param['t3']
    t4 = param['t4']; t5 = param['t5']; t6 = param['t6']
    mu = param['mu']; l = param['l']
    en = np.linspace(-.65, .3, 500)
    spec = np.zeros((len(en), len(x)))
    #Hopping terms
    fx = -2 * np.cos((x + y) / 2 * a)
    fy = -2 * np.cos((x - y) / 2 * a)
    f4 = -2 * t4 * (np.cos(x * a) + np.cos(y * a))
    f5 = -2 * t5 * (np.cos((x + y) * a) + np.cos((x - y) * a))
    f6 = -2 * t6 * (np.cos(x * a) - np.cos(y * a))
    #Placeholders energy eigenvalues
    Ayz = np.ones(len(x)); Axz = np.ones(len(x))
    Axy = np.ones(len(x)); Byz = np.ones(len(x)) 
    Bxz = np.ones(len(x)); Bxy = np.ones(len(x))
    wAyz = np.ones((len(x), 6)); wAxz = np.ones((len(x), 6))
    wAxy = np.ones((len(x), 6)); wByz = np.ones((len(x), 6))
    wBxz = np.ones((len(x), 6)); wBxy = np.ones((len(x), 6))
    
    ###Projectors###
    PAyz = np.array([[1,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],
                     [0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0]])
    PAxz = np.array([[0,0,0,0,0,0],[0,1,0,0,0,0],[0,0,0,0,0,0],
                     [0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0]])
    PAxy = np.array([[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,1,0,0,0],
                     [0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0]])
    PByz = np.array([[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],
                     [0,0,0,1,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0]])
    PBxz = np.array([[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],
                     [0,0,0,0,0,0],[0,0,0,0,1,0],[0,0,0,0,0,0]])
    PBxy = np.array([[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],
                     [0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,1]])
    #TB submatrix
    def A(i):
        A = np.array([[-mu, complex(0,l) + f6[i], -l],
                      [-complex(0,l) + f6[i], -mu, complex(0,l)],
                      [-l, -complex(0,l), -mu + f4[i] + f5[i]]])
        return A
    #TB submatrix
    def B(i): 
        B = np.array([[t2 * fx[i] + t1 * fy[i], 0, 0],
                      [0, t1 * fx[i] + t2 * fy[i], 0],
                      [0, 0, t3 * (fx[i] + fy[i])]])
        return B
    #Tight binding Hamiltonian
    def H(i):
        C1 = np.concatenate((A(i), B(i)), 1)
        C2 = np.concatenate((B(i), A(i)), 1)
        H  = np.concatenate((C1, C2), 0)
        return H
    #Diagonalization of symmetric Hermitian matrix on k-mesh
    for i in range(len(x)):
        eval, evec = la.eigh(H(i))
        eval = np.real(eval)
        Ayz[i] = eval[0]; Axz[i] = eval[1]; Axy[i] = eval[2]
        Byz[i] = eval[3]; Bxz[i] = eval[4]; Bxy[i] = eval[5]
        en_values = (Ayz[i], Axz[i], Axy[i], Byz[i], Bxz[i], Bxy[i])
        n = 0
        for en_value in en_values:
            wAyz[i, n] = np.real(np.sum(np.conj(evec[:, n]) * (PAyz * evec[:, n]))) 
            wAxz[i, n] = np.real(np.sum(np.conj(evec[:, n]) * (PAxz * evec[:, n]))) 
            wAxy[i, n] = np.real(np.sum(np.conj(evec[:, n]) * (PAxy * evec[:, n]))) 
            wByz[i, n] = np.real(np.sum(np.conj(evec[:, n]) * (PByz * evec[:, n]))) 
            wBxz[i, n] = np.real(np.sum(np.conj(evec[:, n]) * (PBxz * evec[:, n]))) 
            wBxy[i, n] = np.real(np.sum(np.conj(evec[:, n]) * (PBxy * evec[:, n]))) 
    #        plt.plot(x[i], en_value, 'o', ms=1,
    #                 color=[wAxz[i, n] + wBxz[i, n] + wAyz[i, n] + wByz[i, n], 0, wAxy[i, n] + wBxy[i, n]])
            wz = wAyz[i, n] + wAxz[i, n] + wByz[i, n] + wBxz[i, n]
            wxy = wAxy[i, n] + wBxy[i, n]
            w = wz - wxy
            val, _val = utils.find(en, en_value)
            spec[_val, i] = w
            n += 1
    bndstr = (Ayz, Axz, Axy, Byz, Bxz, Bxy)
    spec = gaussian_filter(spec, sigma=3, mode='constant')
    return en, spec, bndstr   

def FDsl(x, p0, p1, p2, p3, p4):
    """
    Fermi Dirac Function sloped
    p3 + (p2 + p4 * x) * (np.exp((x - p1) / p0) + 1) ** -1
    """
    return p3 + (p2 + p4 * x) * (np.exp((x - p1) / p0) + 1) ** -1

def poly1(x, p0, p1):
    """
    Polynomial first order
    p0 + p1 * x
    """
    return p0 + p1 * x

def poly2(x, p0, p1, p2, p3):
    """
    Polynomial second order
    p1 + p2 * (x - p0) + p3 * (x - p0)**2 
    """
    return p1 + p2 * (x - p0) + p3 * (x - p0)**2 

def lor(x, p0, p1, p2, 
        p3, p4, p5):
    """
    Single lorentzians on a quadratic background
    """
    return (p2 / (1 + ((x - p0) / p1) ** 2) + 
            p3 + p4 * x + p5 * x ** 2)

def lorHWHM(x, p0, p1, p2, 
        p3, p4, p5):
    """
    Single lorentzians on a quadratic background HWHM version
    """
    return (p2 / (np.pi * p1 * (1 + ((x - p0) / p1) ** 2)) +
            p3 + p4 * x + p5 ** 2)
    
def lor2(x, p0, p1, 
         p2, p3, 
         p4, p5, 
         p6, p7, p8):
    """
    Two lorentzians on a quadratic background
    """
    return (p4 / (1 + ((x - p0) / p2) ** 2) + 
            p5 / (1 + ((x - p1) / p3) ** 2) +
            p6 + p7 * x + p8 * x ** 2)

def lor4(x, p0, p1, p2, p3, 
         p4, p5, p6, p7, 
         p8, p9, p10, p11, 
         p12, p13, p14):
    """
    Four lorentzians on a quadratic background
    """
    return (p8 / (1 + ((x - p0) / p4)  ** 2) + 
            p9 / (1 + ((x - p1) / p5)  ** 2) +
            p10 / (1 + ((x - p2) / p6)  ** 2) +
            p11 / (1 + ((x - p3) / p7)  ** 2) +
            p12 + p13 * x + p14 * x ** 2)
    
def lor6(x, p0, p1, p2, p3, p4, p5, 
         p6, p7, p8, p9, p10, p11, 
         p12, p13, p14, p15, p16, p17, 
         p18, p19, p20):
    """
    Six lorentzians on a quadratic background
    """
    return (p12 / (1 + ((x - p0) / p6)  ** 2) + 
            p13 / (1 + ((x - p1) / p7)  ** 2) +
            p14 / (1 + ((x - p2) / p8)  ** 2) +
            p15 / (1 + ((x - p3) / p9)  ** 2) +
            p16 / (1 + ((x - p4) / p10) ** 2) +
            p17 / (1 + ((x - p5) / p11) ** 2) +
            p18 + p19 * x + p20 * x ** 2)

def lor7(x, p0, p1, p2, p3, p4, p5, p6,
         p7, p8, p9, p10, p11, p12, p13, 
         p14, p15, p16, p17, p18, p19, p20,
         p21, p22, p23):
    """
    Seven lorentzians on a quadratic background
    """
    return (p14 / (1 + ((x - p0) / p7)  ** 2) + 
            p15 / (1 + ((x - p1) / p8)  ** 2) +
            p16 / (1 + ((x - p2) / p9)  ** 2) +
            p17 / (1 + ((x - p3) / p10) ** 2) +
            p18 / (1 + ((x - p4) / p11) ** 2) +
            p19 / (1 + ((x - p5) / p12) ** 2) +
            p20 / (1 + ((x - p6) / p13) ** 2) +
            p21 + p22 * x + p23 * x ** 2)
    
def lor8(x, p0, p1, p2, p3, p4, p5, p6, p7, 
         p8, p9, p10, p11, p12, p13, p14, p15, 
         p16, p17, p18, p19, p20, p21, p22, p23, 
         p24, p25, p26):
    """
    Eight lorentzians on a quadratic background
    """
    return (p16 / (1 + ((x - p0) / p8)  ** 2) + 
            p17 / (1 + ((x - p1) / p9)  ** 2) +
            p18 / (1 + ((x - p2) / p10) ** 2) +
            p19 / (1 + ((x - p3) / p11) ** 2) +
            p20 / (1 + ((x - p4) / p12) ** 2) +
            p21 / (1 + ((x - p5) / p13) ** 2) +
            p22 / (1 + ((x - p6) / p14) ** 2) +
            p23 / (1 + ((x - p7) / p15) ** 2) +
            p24 + p25 * x + p26 * x ** 2)
    
def gauss2(x, p0, p1, p2, p3, p4, p5, p6, p7, p8):
    """
    Two gaussians on a quadratic background
    """
    return (p4 * np.exp(-(x - p0) ** 2 / (2 * p2 ** 2)) + 
            p5 * np.exp(-(x - p1) ** 2 / (2 * p3 ** 2)) +
            p6 + p7 * x + p8 * x ** 2)    
    
def FL_simple(x, p0, p1, p2, p3, p4, p5):
    """
    Fermi liquid quasiparticle with simple self energy
    """
    ReS = p0 * x
    ImS = p1 + p2 * x ** 2;

    return (p4 * 1 / np.pi * ImS / ((x - ReS - p3) ** 2 + ImS ** 2) * 
            (np.exp((x - 0) / p5) + 1) ** -1)

    
def gauss_mod(x, p0, p1, p2, p3, p4, p5):
    """
    Modified Gaussian
    """
    return (p0 * np.exp(-.5 * ((-x + p1) / p2) ** 2) * 
            (p3 * special.erf((-x + p4) / p5) + 1))

def Full_simple(x, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9):
    """
    Spectral function simple FL + incoherent weight as exp. mod. Gaussian
    """
    return (FL_simple(x, p0, p1, p2, p3, p4, p5) +
            p9 * exponnorm.pdf(-x, K=p6, loc=p7, scale = p8) *
            FDsl(x, p0=1e-3, p1=0, p2=1, p3=0, p4=0))
    
def Full_mod(x, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11):
    """
    Spectral function simple FL + incoherent weight as exp. mod. Gaussian
    """
    return (FL_simple(x, p0, p1, p2, p3, p4, p5) +
            gauss_mod(x, p6, p7, p8, p9, p10, p11))

