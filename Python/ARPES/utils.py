#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 09:57:01 2018

@author: ilikecarbs

%%%%%%%%%%%%%%%%%%%%%
        utils
%%%%%%%%%%%%%%%%%%%%%

**Useful helper functions**

.. note::
        To-Do:
            -
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as la
import utils
from scipy import special
from scipy.ndimage.filters import gaussian_filter
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.cm as cm
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


def Shirley(EDC):
    """returns b

    **Generates a Shirley background from given edc**

    Args
    ----
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


"""
%%%%%%%%%%%%%%%%%%%%%
     Colormaps
%%%%%%%%%%%%%%%%%%%%%
"""


def rainbow_light():
    filepath = '/Users/denyssutter/Documents/PhD/data/rainbow_light.dat'
    data = np.loadtxt(filepath)
    colors = np.array([(i[0], i[1], i[2]) for i in data])

    # Normalize the colors
    colors /= colors.max()

    # Build the colormap
    rainbow_light = LinearSegmentedColormap.from_list('rainbow_light', colors,
                                                      N=len(colors))
    return rainbow_light


def rainbow_light_2():
    filepath = '/Users/denyssutter/Documents/PhD/data/rainbow_light_2.dat'
    data = np.loadtxt(filepath)
    colors = np.array([(i[0], i[1], i[2]) for i in data])

    # Normalize the colors
    colors /= colors.max()

    # Build the colormap
    rainbow_light_2 = LinearSegmentedColormap.from_list('rainbow_light',
                                                        colors, N=len(colors))
    return rainbow_light_2


def orbitals():
    colors = np.zeros((100, 3))
    for i in range(100):
        colors[i, :] = [i/100, 0, 1-i/100]

    # Normalize the colors
    colors /= colors.max()

    # Build the colormap
    orbitals = LinearSegmentedColormap.from_list('orbitals', colors,
                                                 N=len(colors))
    return orbitals


rainbow_light = rainbow_light()
cm.register_cmap(name='rainbow_light', cmap=rainbow_light)

rainbow_light_2 = rainbow_light_2()
cm.register_cmap(name='rainbow_light_2', cmap=rainbow_light_2)

orbitals = orbitals()
cm.register_cmap(name='orbitals', cmap=orbitals)


"""
%%%%%%%%%%%%%%%%%%%%%
   Tight Binding
%%%%%%%%%%%%%%%%%%%%%
"""


def paramSRO():
    """returns param

    **Parameter set of TB model Sr2RuO4 arXiv:1212.3994v1**

    Args
    ----

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


def paramSRO_opt():
    """returns param

    **Parameter set after optimization starting from paramSRO**

    Args
    ----

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

    param = dict([('t1', .12898992), ('t2', .00046937), ('t3', .07199656),
                  ('t4', .04071875), ('t5', .00886822), ('t6', 0),
                  ('mu', .09534531), ('so', .032)])
    return param


def paramCSRO20():
    """returns param

    **parameter set of TB model D. Sutter et al. :-)**

    Args
    ----

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
                  ('t5', .012), ('t6', 0), ('mu', .084), ('so', .036)])
    return param


def paramCSRO20_opt():
    """returns param

    **parameter set after optimization starting from param CSRO20**

    Args
    ----

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

    param = dict([('t1', .09335122), ('t2', .00889674), ('t3', .05688837),
                  ('t4', .03258921), ('t5', .00754219), ('t6', 0),
                  ('mu', .06510367), ('so', .03607035)])
    return param


def paramCSRO30():
    """returns param

    **Parameter test set CSRO30**

    Args
    ----

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


def paramCSRO_fit():
    """returns param

    **Parameter from fit**

    Args
    ----

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
#    From 3000 iteration set
#    param = dict([('t1', .09201014), ('t2', .00913753), ('t3', .06603581),
#                  ('t4', .03799015), ('t5', .00984912), ('t6', 0),
#                  ('mu', .07022563), ('so', .03937185)])

#    From 4000 iteration set
#    param = dict([('t1', .08836273), ('t2', .00913917), ('t3', .05625457),
#                  ('t4', .03287336), ('t5', .00755447), ('t6', 0),
#                  ('mu', .06325721), ('so', .036)])

    # used for figure20
#    param = dict([('t1', .09979339), ('t2', .00778938), ('t3', .0742577),
#                  ('t4', .04302189), ('t5', .01156417), ('t6', 0),
#                  ('mu', .08059366), ('so', .04042714)])
    # testing GX-fit
#    param = dict([('t1', .09013866), ('t2', .00886533), ('t3', .07392306),
#                  ('t4', .03953298), ('t5', .01798494), ('t6', 0),
#                  ('mu', .07997456), ('so', .04042714)])
#    # testing all dispersions 10k
#    param = dict([('t1', .08505056), ('t2', .01004666), ('t3', .07186227),
#                  ('t4', .04297814), ('t5', .01180308), ('t6', 0),
#                  ('mu', .06981258), ('so', .04156431)])
    # low SO
#    param = dict([('t1', .10362879), ('t2', .00468736), ('t3', .07764697),
#                  ('t4', .04163935), ('t5', .01655971), ('t6', 0),
#                  ('mu', .08132872), ('so', .04561293)])
    # lr 1e-4 start from SRO (all disperions and FS)
    param = dict([('t1', .08765864), ('t2', .00950954), ('t3', .07530941),
                  ('t4', .04488273), ('t5', .0128184), ('t6', 0),
                  ('mu', .0724995+.001), ('so', .04298228)])
    # lr 5e-4 start from SRO (all disperions and FS)
#    param = dict([('t1', .10115535), ('t2', .00604792), ('t3', .0775029),
#                  ('t4', .04252476), ('t5', .01576043), ('t6', 0),
#                  ('mu', .07990218), ('so', .04574906)])
    # lr 1e-3 start from SRO (all disperions and FS)
#    param = dict([('t1', .11267502), ('t2', .00580496), ('t3', .0825573),
#                  ('t4', .0444382), ('t5', .01699491), ('t6', 0),
#                  ('mu', .08552713), ('so', .05123512)])
    # high SO
#    param = dict([('t1', .20983586), ('t2', .02027852), ('t3', .15528096),
#                  ('t4', .0903955), ('t5', .02389806), ('t6', 0),
#                  ('mu', .14948591), ('so', .106)])
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
        ----
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
        plt.figure('SRO_TB', clear=True)
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
            plt.figure('SRO_vertices', clear=True)
            # Placeholders vertices
            VX = ()
            VY = ()
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
                    plt.subplot(1, 3, n_bnd+1)
                    plt.plot(v_x, v_y)
                    plt.axis('equal')
                    plt.text(v_x[0], v_y[0], str(i))
                    plt.show()

                VX = VX + (V_x,)
                VY = VY + (V_y,)

                if proj:  # Do projection for all vertices
                    for j in range(len(V_x)):  # Loop over all vertices
                        for i in range(len(V_x[j])):  # Loop over k-points

                            #  Find values and diagonalize
                            x_val, x_idx = utils.find(x, V_x[j][i])
                            y_val, y_idx = utils.find(y, V_y[j][i])
                            val_proj, vec_proj = la.eigh(H(x_idx, y_idx))
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
                            w = np.tanh(10 * (wz - wxy))

                            # Build Fermi surface
                            kx_val, kx_idx = utils.find(kx, V_x[j][i])
                            ky_val, ky_idx = utils.find(ky, V_y[j][i])
                            FS[kx_idx, ky_idx] = w

            # Blur for visual effect
            FS = gaussian_filter(FS, sigma=10, mode='constant')
            self.kx = kx
            self.ky = ky
            self.FS = FS
            self.VX = VX
            self.VY = VY
        self.bndstr = dict([('yz', yz), ('xz', xz), ('xy', xy)])

    def SRO_folded(self, param=paramSRO(), e0=0, vert=False, proj=False):
        """returns self.bndstr, self.kx, self.ky, self.FS

        **Calculates folded band structure from 3 band tight binding model**

        Args
        ----
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

        q = np.pi / a

        # Hopping terms
        fyz = (- 2 * t2 * np.cos((X + Y) * a) -
               2 * t1 * np.cos((X - Y) * a))
        fxz = (- 2 * t1 * np.cos((X + Y) * a) -
               2 * t2 * np.cos((X - Y) * a))
        fxy = (- 2 * t3 * (np.cos((X + Y) * a) + np.cos((X - Y) * a)) -
               4 * t4 * (np.cos((X + Y) * a) * np.cos((X - Y) * a)) -
               2 * t5 * (np.cos(2 * (X + Y) * a) +
                         np.cos(2 * (X - Y) * a)))
        off = - 4 * t6 * (np.sin((X + Y) * a) * np.sin((X - Y) * a))

        fyz_q = (- 2 * t2 * np.cos((X + Y - q) * a) -
                 2 * t1 * np.cos((X - Y - q) * a))
        fxz_q = (- 2 * t1 * np.cos((X + Y - q) * a) -
                 2 * t2 * np.cos((X - Y - q) * a))
        fxy_q = (- 2 * t3 * (np.cos((X + Y - q) * a) +
                             np.cos((X - Y - q) * a)) -
                 4 * t4 * (np.cos((X + Y - q) * a) *
                           np.cos((X - Y - q) * a)) -
                 2 * t5 * (np.cos(2 * (X + Y - q) * a) +
                           np.cos(2 * (X - Y - q) * a)))
        off_q = - 4 * t6 * (np.sin((X + Y - q) * a) * np.sin((X - Y - q) * a))

        # Placeholders energy eigenvalues
        yz = np.ones((len(x), len(y)))
        xz = np.ones((len(x), len(y)))
        xy = np.ones((len(x), len(y)))
        yz_q = np.ones((len(x), len(y)))
        xz_q = np.ones((len(x), len(y)))
        xy_q = np.ones((len(x), len(y)))

        # Tight binding Hamiltonian
        def H(i, j):
            H = np.array([[fyz[i, j] - mu, off[i, j] + complex(0, so), -so],
                          [off[i, j] - complex(0, so), fxz[i, j] - mu,
                           complex(0, so)],
                          [-so, -complex(0, so), fxy[i, j] - mu]])
            return H

        def H_q(i, j):
            H = np.array([[fyz_q[i, j] - mu, off_q[i, j] + complex(0, so),
                           -so],
                          [off_q[i, j] - complex(0, so), fxz_q[i, j] - mu,
                           complex(0, so)],
                          [-so, -complex(0, so), fxy_q[i, j] - mu]])
            return H

        # Diagonalization of symmetric Hermitian matrix on k-mesh
        for i in range(len(x)):
            for j in range(len(y)):
                val = la.eigvalsh(H(i, j))
                val_q = la.eigvalsh(H_q(i, j))
                val = np.real(val)
                val_q = np.real(val_q)
                yz[i, j] = val[0]
                xz[i, j] = val[1]
                xy[i, j] = val[2]
                yz_q[i, j] = val_q[0]
                xz_q[i, j] = val_q[1]
                xy_q[i, j] = val_q[2]

        # Band structure
        bndstr = (yz, xz, xy, yz_q, xz_q, xy_q)

        # Placeholder for contours
        C = ()

        # Projectors
        Pyz = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]])
        Pxz = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
        Pxy = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 1]])

        # Generate contours C
        n = 0
        plt.figure('SRO_TB', clear=True)
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
            plt.figure('SRO_vertices', clear=True)
            # Placeholders vertices
            VX = ()
            VY = ()
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
                    plt.subplot(1, 3, n_bnd+1)
                    plt.plot(v_x, v_y)
                    plt.axis('equal')
                    plt.text(v_x[0], v_y[0], str(i))
                    plt.show()

                VX = VX + (V_x,)
                VY = VY + (V_y,)

                if proj:  # Do projection for all vertices
                    for j in range(len(V_x)):  # Loop over all vertices
                        for i in range(len(V_x[j])):  # Loop over k-points

                            #  Find values and diagonalize
                            x_val, x_idx = utils.find(x, V_x[j][i])
                            y_val, y_idx = utils.find(y, V_y[j][i])
                            val_proj, vec_proj = la.eigh(H(x_idx, y_idx))
                            val_proj = np.real(val_proj)
                            if any(x == n for x in [0, 1, 3]):
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
                                w = np.tanh(10 * (wz - wxy))

                                # Build Fermi surface
                                kx_val, kx_idx = utils.find(kx, V_x[j][i])
                                ky_val, ky_idx = utils.find(ky, V_y[j][i])
                                FS[kx_idx, ky_idx] = w

            # Blur for visual effect
            FS = gaussian_filter(FS, sigma=10, mode='constant')
            self.kx = kx
            self.ky = ky
            self.FS = FS
            self.VX = VX
            self.VY = VY
        self.bndstr = dict([('yz', yz), ('xz', xz), ('xy', xy),
                            ('yz_q', yz_q), ('xz_q', xz_q), ('xy_q', xy_q)])

    def CSRO(self, param=paramCSRO_fit(), e0=0, vert=False, proj=True):
        """returns self.bndstr, self.kx, self.ky, self.FS

        **Calculates band structure from 6 band tight binding model**

        Args
        ----
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
        plt.figure('CSRO_TB', clear=True)
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
            plt.figure('CSRO_vertices', clear=True)

            # Placeholders vertices
            VX = ()
            VY = ()

            if proj:  # Also project eigenbasis onto orbitals from here

                # Placeholders for projected FS
                kx = np.linspace(np.min(x), np.max(x), 1000)
                ky = np.linspace(np.min(y), np.max(y), 1000)
                FS = np.zeros((len(kx), len(ky)))

            for n_bnd in range(len(bndstr)):
                p = C[n_bnd].collections[0].get_paths()
                p = np.asarray(p)
                plt.figure('CSRO_vertices')

                V_x = ()
                V_y = ()

                # Get vertices and plot them
                for i in range(len(p)):
                    v = p[i].vertices
                    v_x = v[:, 0]
                    v_y = v[:, 1]
                    V_x = V_x + (v_x,)
                    V_y = V_y + (v_y,)
                    plt.subplot(2, 3, n_bnd+1)
                    plt.plot(v_x, v_y)
                    plt.axis('equal')
                    plt.text(v_x[0], v_y[0], str(i))
                    plt.show()

                VX = VX + (V_x,)
                VY = VY + (V_y,)

                if proj:  # Do projection for all vertices
                    for j in range(len(V_x)):  # Loop over all vertices
                        for i in range(len(V_x[j])):  # Loop over k-points

                            # Find values and diagonalize
                            x_val, x_idx = utils.find(x, V_x[j][i])
                            y_val, y_idx = utils.find(y, V_y[j][i])
                            val_proj, vec_proj = la.eigh(H(x_idx, y_idx))
                            val_proj = np.real(val_proj)

                            # orbital weights
                            wAyz = np.real(
                                    np.sum(
                                            np.conj(vec_proj[:, n_bnd]) *
                                            (PAyz * vec_proj[:, n_bnd])))
                            wAxz = np.real(
                                    np.sum(
                                            np.conj(vec_proj[:, n_bnd]) *
                                            (PAxz * vec_proj[:, n_bnd])))
                            wAxy = np.real(
                                    np.sum(
                                            np.conj(vec_proj[:, n_bnd]) *
                                            (PAxy * vec_proj[:, n_bnd])))
                            wByz = np.real(
                                    np.sum(
                                            np.conj(vec_proj[:, n_bnd]) *
                                            (PByz * vec_proj[:, n_bnd])))
                            wBxz = np.real(
                                    np.sum(
                                            np.conj(vec_proj[:, n_bnd]) *
                                            (PBxz * vec_proj[:, n_bnd])))
                            wBxy = np.real(
                                    np.sum(
                                            np.conj(vec_proj[:, n_bnd]) *
                                            (PBxy * vec_proj[:, n_bnd])))

                            # Total out-of-plane weight
                            wz = wAyz + wAxz + wByz + wBxz

                            # Total in-plane weight
                            wxy = wAxy + wBxy

                            # Weight for divergence colorscale
                            w = np.tanh(10*(wz - wxy))

                            # Build Fermi surface
                            kx_val, kx_idx = utils.find(kx, V_x[j][i])
                            ky_val, ky_idx = utils.find(ky, V_y[j][i])
                            FS[kx_idx, ky_idx] = w + 0

            # Blur for visual effect
            FS = gaussian_filter(FS, sigma=10, mode='constant')
            self.kx = kx
            self.ky = ky
            self.FS = FS
            self.VX = VX
            self.VY = VY

        self.bndstr = dict([('Ayz', Ayz), ('Axz', Axz), ('Axy', Axy),
                            ('Byz', Byz), ('Bxz', Bxz), ('Bxy', Bxy)])

    def single(self, param):
        """returns self.bndstr, self.kx, self.ky, self.FS

        **Calculates single band tight binding band structure**

        Args
        ----
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


def CSRO_eval(x, y, param=paramCSRO_fit()):
    """returns bndstr

    **Calculates band structure along kx, ky**

    Args
    ----
    :param:     TB parameters
    :x:         kx array
    :y:         ky array

    Return
    ------
    :bndstr:    eigenenergies band structure
    """

    a = np.pi

    # Load TB parameters
    t1 = param['t1']  # Nearest neighbour for out-of-plane orbitals large
    t2 = param['t2']  # Nearest neighbour for out-of-plane orbitals small
    t3 = param['t3']  # Nearest neighbour for dxy orbitals
    t4 = param['t4']  # Next nearest neighbour for dxy orbitals
    t5 = param['t5']  # Next next nearest neighbour for dxy orbitals
    t6 = param['t6']  # Off diagonal matrix element
    mu = param['mu']  # Chemical potential
    so = param['so']  # spin orbit coupling

    # Hopping terms
    fx = -2 * np.cos((x + y) / 2 * a)
    fy = -2 * np.cos((x - y) / 2 * a)
    f4 = -2 * t4 * (np.cos(x * a) + np.cos(y * a))
    f5 = -2 * t5 * (np.cos((x + y) * a) + np.cos((x - y) * a))
    f6 = -2 * t6 * (np.cos(x * a) - np.cos(y * a))

    # Placeholders energy eigenvalues
    Ayz = np.ones(len(x))
    Axz = np.ones(len(x))
    Axy = np.ones(len(x))
    Byz = np.ones(len(x))
    Bxz = np.ones(len(x))
    Bxy = np.ones(len(x))

    # TB submatrix
    def A(i):
        A = np.array([[-mu, complex(0, so) + f6[i], -so],
                      [-complex(0, so) + f6[i], -mu, complex(0, so)],
                      [-so, -complex(0, so), -mu + f4[i] + f5[i]]])
        return A

    # TB submatrix
    def B(i):
        B = np.array([[t2 * fx[i] + t1 * fy[i], 0, 0],
                      [0, t1 * fx[i] + t2 * fy[i], 0],
                      [0, 0, t3 * (fx[i] + fy[i])]])
        return B

    # Tight binding Hamiltonian
    def H(i):
        C1 = np.concatenate((A(i), B(i)), 1)
        C2 = np.concatenate((B(i), A(i)), 1)
        H = np.concatenate((C1, C2), 0)
        return H

    # Diagonalization of symmetric Hermitian matrix on k-mesh
    for i in range(len(x)):
        val, vec = la.eigh(H(i))
        val = np.real(val)
        Ayz[i] = val[0]
        Axz[i] = val[1]
        Axy[i] = val[2]
        Byz[i] = val[3]
        Bxz[i] = val[4]
        Bxy[i] = val[5]

    bndstr = (Ayz, Axz, Axy, Byz, Bxz, Bxy)

    return bndstr


def CSRO_eval_proj(x, y, param=paramCSRO_fit()):
    """returns en_tb, int_tb, bndstr

    **Calculates band structure along kx, ky, orbitally projected**

    Args
    ----
    :param:     TB parameters
    :x:         kx array
    :y:         ky array

    Return
    ------
    :en_tb:     binding energy of tight binding model
    :int_tb:    intensities of tight binding model
    :bndstr:    eigenenergies band structure
    """

    a = np.pi

    # Load TB parameters
    t1 = param['t1']  # Nearest neighbour for out-of-plane orbitals large
    t2 = param['t2']  # Nearest neighbour for out-of-plane orbitals small
    t3 = param['t3']  # Nearest neighbour for dxy orbitals
    t4 = param['t4']  # Next nearest neighbour for dxy orbitals
    t5 = param['t5']  # Next next nearest neighbour for dxy orbitals
    t6 = param['t6']  # Off diagonal matrix element
    mu = param['mu']  # Chemical potential
    so = param['so']  # spin orbit coupling

    # Hopping terms
    fx = -2 * np.cos((x + y) / 2 * a)
    fy = -2 * np.cos((x - y) / 2 * a)
    f4 = -2 * t4 * (np.cos(x * a) + np.cos(y * a))
    f5 = -2 * t5 * (np.cos((x + y) * a) + np.cos((x - y) * a))
    f6 = -2 * t6 * (np.cos(x * a) - np.cos(y * a))

    # Placeholders energy eigenvalues
    Ayz = np.ones(len(x))
    Axz = np.ones(len(x))
    Axy = np.ones(len(x))
    Byz = np.ones(len(x))
    Bxz = np.ones(len(x))
    Bxy = np.ones(len(x))
    wAyz = np.ones((len(x), 6))
    wAxz = np.ones((len(x), 6))
    wAxy = np.ones((len(x), 6))
    wByz = np.ones((len(x), 6))
    wBxz = np.ones((len(x), 6))
    wBxy = np.ones((len(x), 6))

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

    # TB submatrix
    def A(i):
        A = np.array([[-mu, complex(0, so) + f6[i], -so],
                      [-complex(0, so) + f6[i], -mu, complex(0, so)],
                      [-so, -complex(0, so), -mu + f4[i] + f5[i]]])
        return A

    # TB submatrix
    def B(i):
        B = np.array([[t2 * fx[i] + t1 * fy[i], 0, 0],
                      [0, t1 * fx[i] + t2 * fy[i], 0],
                      [0, 0, t3 * (fx[i] + fy[i])]])
        return B

    # Tight binding Hamiltonian
    def H(i):
        C1 = np.concatenate((A(i), B(i)), 1)
        C2 = np.concatenate((B(i), A(i)), 1)
        H = np.concatenate((C1, C2), 0)
        return H

    # Placeholders energy, spectra
    en_tb = np.linspace(-.65, .3, 500)
    int_tb = np.zeros((len(en_tb), len(x)))

    # Diagonalization of symmetric Hermitian matrix on k-mesh
    for i in range(len(x)):
        val, vec = la.eigh(H(i))
        val = np.real(val)
        Ayz[i] = val[0]
        Axz[i] = val[1]
        Axy[i] = val[2]
        Byz[i] = val[3]
        Bxz[i] = val[4]
        Bxy[i] = val[5]
        en_vals = (Ayz[i], Axz[i], Axy[i], Byz[i], Bxz[i], Bxy[i])

        # Project on orbitals
        n = 0
        for en_val in en_vals:

            # orbital weights
            wAyz[i, n] = np.real(np.sum(np.conj(vec[:, n]) *
                                        (PAyz * vec[:, n])))
            wAxz[i, n] = np.real(np.sum(np.conj(vec[:, n]) *
                                        (PAxz * vec[:, n])))
            wAxy[i, n] = np.real(np.sum(np.conj(vec[:, n]) *
                                        (PAxy * vec[:, n])))
            wByz[i, n] = np.real(np.sum(np.conj(vec[:, n]) *
                                        (PByz * vec[:, n])))
            wBxz[i, n] = np.real(np.sum(np.conj(vec[:, n]) *
                                        (PBxz * vec[:, n])))
            wBxy[i, n] = np.real(np.sum(np.conj(vec[:, n]) *
                                        (PBxy * vec[:, n])))

            # Total out-of-plane weight
            wz = wAyz[i, n] + wAxz[i, n] + wByz[i, n] + wBxz[i, n]

            # Total in-plane weight
            wxy = wAxy[i, n] + wBxy[i, n]

            # Weight for divergence colorscale
            w = np.tanh(10 * (wz - wxy))

            # Build band structure
            en_tb_val, en_tb_idx = utils.find(en_tb, en_val)
            int_tb[en_tb_idx, i] = w
            n += 1

    bndstr = (Ayz, Axz, Axy, Byz, Bxz, Bxy)

    # Blur for visual effect
    int_tb = gaussian_filter(int_tb, sigma=3, mode='constant')

    return en_tb, int_tb, bndstr


def cost(Kx, Ky, En, t1, t2, t3, t4, t5, t6, mu, so):
    """returns J

    **Calculates the cost of the model**

    Args
    ----
    :Kx:    kx of all sheets
    :Ky:    ky of all sheets
    :En:    band energy
    - t1:   Nearest neighbour for out-of-plane orbitals large
    - t2:   Nearest neighbour for out-of-plane orbitals small
    - t3:   Nearest neighbour for dxy orbitals
    - t4:   Next nearest neighbour for dxy orbitals
    - t5:   Next next nearest neighbour for dxy orbitals
    - t6:   Off diagonal matrix element
    - mu:   Chemical potential
    - so:   spin orbit coupling

    Return
    ------
    :J:     cost
    """

    # prepare parallelization
    num_cores = multiprocessing.cpu_count()
    inputs = range(len(Kx))

    def J_eval(k):
        J = 0
        # extract k's
        kx = Kx[k]
        ky = Ky[k]
        en = En[k]
        a = np.pi

        # hopping terms
        fx = -2 * np.cos((kx + ky) / 2 * a)
        fy = -2 * np.cos((kx - ky) / 2 * a)
        f4 = -2 * t4 * (np.cos(kx * a) + np.cos(ky * a))
        f5 = -2 * t5 * (np.cos((kx + ky) * a) + np.cos((kx - ky) * a))
        f6 = -2 * t6 * (np.cos(kx * a) - np.cos(ky * a))

        # TB submatrix
        def A(i):
            A = np.array([[-mu, complex(0, so) + f6[i], -so],
                          [-complex(0, so) + f6[i], -mu, complex(0, so)],
                          [-so, -complex(0, so), -mu + f4[i] + f5[i]]])
            return A

        # TB submatrix
        def B(i):
            B = np.array([[t2 * fx[i] + t1 * fy[i], 0, 0],
                          [0, t1 * fx[i] + t2 * fy[i], 0],
                          [0, 0, t3 * (fx[i] + fy[i])]])
            return B

        # Tight binding Hamiltonian
        def H(i):
            C1 = np.concatenate((A(i), B(i)), 1)
            C2 = np.concatenate((B(i), A(i)), 1)
            H = np.concatenate((C1, C2), 0)
            return H

        # calculate eigenvalues and cost J
        for j in range(len(kx)):
            val = la.eigvalsh(H(j))
            val = np.real(val)

            j = min(abs(en[j] - val))

            # regularization
            if any(x == k for x in np.arange(0, 16, 1)):
                j *= 2.4  # 1: no reg.
            J += j

        return J

    # parallel computation
    J = np.sum(
            np.array(
                    Parallel(n_jobs=num_cores)
                    (delayed(J_eval)(i) for i in inputs)))

    return J


def d_cost(Kx, Ky, En, P, d):
    """returns J

    **Calculates the cost of the model**

    Args
    ----
    :Kx:    kx of all sheets
    :Ky:    ky of all sheets
    :En:    band energy
    :P:     P[0]..P[7] correspond to t1..so
    :d:     derivative with respect to parameter d
    - t1:   Nearest neighbour for out-of-plane orbitals large
    - t2:   Nearest neighbour for out-of-plane orbitals small
    - t3:   Nearest neighbour for dxy orbitals
    - t4:   Next nearest neighbour for dxy orbitals
    - t5:   Next next nearest neighbour for dxy orbitals
    - t6:   Off diagonal matrix element
    - mu:   Chemical potential
    - so:   spin orbit coupling

    Return
    ------
    :dJ:    derivative of cost
    """

    eps = 1e-9

    P_p = np.copy(P)
    P_n = np.copy(P)
    P_p[d] += eps
    P_n[d] -= eps
    dJ = (cost(Kx, Ky, En, *P_p) - cost(Kx, Ky, En, *P_n)) / (2 * eps)

    return dJ


def cost_deriv(Kx, Ky, En, P):
    """returns dJ

    **Calculates the cost gradients of the model**

    Args
    ----
    :Kx:    kx of all sheets
    :Ky:    ky of all sheets
    :En:    band energy
    :P:     P[0]..P[7] correspond to t1..so
    - t1:   Nearest neighbour for out-of-plane orbitals large
    - t2:   Nearest neighbour for out-of-plane orbitals small
    - t3:   Nearest neighbour for dxy orbitals
    - t4:   Next nearest neighbour for dxy orbitals
    - t5:   Next next nearest neighbour for dxy orbitals
    - t6:   Off diagonal matrix element
    - mu:   Chemical potential
    - so:   spin orbit coupling

    Return
    ------
    :DJ:    gradient of cost w.r.t. parameters
    """

    # parallelize
    num_cores = multiprocessing.cpu_count()
    inputs = range(8)  # not including so update

    DJ = np.array(Parallel(n_jobs=num_cores)(delayed(d_cost)(Kx, Ky, En, P, i)
                  for i in inputs))
#    DJ = np.append(DJ, 0)
    if DJ[-1] > 0:
        DJ[-1] = 0
    return DJ


def optimize_TB(Kx, Ky, En, it_max, P):
    """returns it, J, param

    **Optimizes the model and returns the cost and parameters**

    Args
    ----
    :Kx:        kx of all sheets
    :Ky:        ky of all sheets
    :En:    band energy
    :it_max:    maximum of iterations
    :P:         TB initial parameters

    Return
    ------
    :it:        iteratons
    :J:         cost
    :param:     optimized parameters
    """

    J = np.array([])  # cost
    it = np.array([])  # iterations
    m = np.zeros(P.size)  # initial parameter
    v = np.zeros(P.size)  # initial parameter
    beta1 = .9  # parameter Adam optimizer
    beta2 = .999  # parameter Adam optimizer
    epsilon = 1e-8  # preventing from dividing by zero
    alpha = 1e-4  # external learning rate

    # start optimizing
    start_time = time.time()

    try:
        while True:
            for i in range(it_max):
                it = np.append(it, i)
                J = np.append(J, cost(Kx, Ky, En, *P))  # cost
                DJ = cost_deriv(Kx, Ky, En, P)  # gradient
                lr = alpha * np.sqrt((1 - beta2) / (1 - beta1))  # rate

                # update parameters
                m = beta1 * m + (1 - beta1) * DJ
                v = beta2 * v + ((1 - beta2) * DJ) * DJ
                P = P - lr * m / (np.sqrt(v) + epsilon)
                if P[-1] > .106:
                    P[-1] = .106
                if P[-1] < .032:
                    P[-1] = .032

                # display every 10 iterations
                if np.mod(i, 10) == 0:
                    print('iteration nr. ' + str(i))
                    print("--- %s seconds ---" % (time.time() - start_time))
                    print("SO = " + str(P[-1]))
                    print("J = " + str(J[-1]))
    except KeyboardInterrupt:
        pass
    # build up dictionary
    param = dict([('t1', P[0]), ('t2', P[1]), ('t3', P[2]), ('t4', P[3]),
                  ('t5', P[4]), ('t6', P[5]), ('mu', P[6]), ('so', P[7])])

    # visualize result
    tb = utils.TB(a=np.pi, kbnd=1, kpoints=100)  # Initialize
    tb.CSRO(param=param, e0=0, vert=False, proj=False)
    plt.show()

    return it, J, P


def ang2k(angdg, Ekin, lat_unit=False, a=5.33, b=5.33, c=11,
          V0=0, thdg=0, tidg=0, phidg=0):
        """returns k, k_V0

        **Converts detector angles into k-space**

        Args
        ----
        :angdg:     detector angles in degrees
        :Ekin:      photon kinetic energy
        :lat_unit:  lattice units used (Boolean)
        :a, b, c:   lattice parameters
        :V0:        inner potential
        :thdg:      manipulator angle theta in degrees
        :tidg:      manipulator angle tilt in degrees
        :phidg:     manipulator angle phi in degrees

        Return
        ------
        :k:        k-vector
        :k_V0:     k-vector (with inner potential)
        """

        hbar = 6.58212e-16  # eV * s
        me = 5.68563e-32  # eV * s^2 / Angstrom^2
        ang = np.pi * angdg / 180
        th = np.pi * thdg / 180
        ti = np.pi * tidg / 180
        phi = np.pi * phidg / 180

        # Rotation matrices
        Ti = np.array([
                [1, 0, 0],
                [0, np.cos(ti), np.sin(ti)],
                [0, -np.sin(ti), np.cos(ti)]
                ])
        Phi = np.array([
                [np.cos(phi), -np.sin(phi), 0],
                [np.sin(phi), np.cos(phi), 0],
                [0, 0, 1]
                ])
        Th = np.array([
                [np.cos(th), 0, -np.sin(th)],
                [0, 1, 0],
                [np.sin(th), 0, np.cos(th)]
                ])

        # Norm of k-vector
        k_norm = np.sqrt(2 * me * Ekin) / hbar
        k_norm_V0 = np.sqrt(2 * me * (Ekin + V0)) / hbar

        # Placeholders
        kv = np.ones((3, 1))
        kv_V0 = np.ones((3, 1))

        # Build k-vector
        kv = np.array([k_norm * np.sin(ang), 0 * ang, k_norm * np.cos(ang)])
        kv_V0 = np.array([k_norm * np.sin(ang), 0 * ang,
                          np.sqrt(k_norm_V0**2 - (k_norm * np.sin(ang)**2))])
        k = np.matmul(Phi, np.matmul(Ti, np.matmul(Th, kv)))
        k_V0 = np.matmul(Phi, np.matmul(Ti, np.matmul(Th, kv_V0)))

        if lat_unit:  # lattice units
            k *= np.array([a / np.pi, b / np.pi, c / np.pi])
            k_V0 *= np.array([a / np.pi, b / np.pi, c / np.pi])

        return k, k_V0

        print('\n ~ Angles converted into k-space',
              '\n', '==========================================')


def partial_deriv(x, y, f):
    """returns F

    **Derivatives of 2-dimensional data**

    Args
    ----
    :x:     x-axis
    :y:     y-axis
    :f:     data (2-dim)

    Return
    ------
    :F:     Dictionary of derivatives
    """

    # Placeholders
    fx = np.empty_like(f)
    fxx = np.empty_like(f)
    fxy = np.empty_like(f)
    fy = np.empty_like(f)
    fyy = np.empty_like(f)
    fyx = np.empty_like(f)

    # calculate derivatives
    for i in range(len(y)):
        fx[0:-1, i] = np.diff(f[:, i]) / np.diff(x)
        fx[-1, i] = (f[-1, i] - f[-2, i]) / (x[-1] - x[-2])
    for i in range(len(y)):
        fxx[0:-1, i] = np.diff(fx[:, i]) / np.diff(x)
        fxx[-1, i] = (fx[-1, i] - fx[-2, i]) / (x[-1] - x[-2])
    for i in range(len(x)):
        fy[i, 0:-1] = np.diff(f[i, :]) / np.diff(y)
        fy[i, -1] = (f[i, -1] - f[i, -2]) / (y[-1] - y[-2])
    for i in range(len(x)):
        fyy[i, 0:-1] = np.diff(fy[i, :]) / np.diff(y)
        fyy[i, -1] = (fy[i, -1] - fy[i, -2]) / (y[-1] - y[-2])
    for i in range(len(y)):
        fyx[0:-1, i] = np.diff(fy[:, i]) / np.diff(x)
        fyx[-1, i] = (fy[-1, i] - fy[-2, i]) / (x[-1] - x[-2])
    for i in range(len(x)):
        fxy[i, 0:-1] = np.diff(fx[i, :]) / np.diff(y)
        fxy[i, -1] = (fx[i, -1] - fx[i, -2]) / (y[-1] - y[-2])
    for i in range(len(y)):
        fyx[:, i] = np.gradient(fy[:, i])

    # build up dictionary
    F = {'fx': fx, 'fxx': fxx, 'fxy': fxy,
         'fy': fy, 'fyy': fyy, 'fyx': fyx}

    return F


def curvature_equiv(x, y, f, C0):
    """returns C

    **Curvature with two equivalent axes**

    Args
    ----
    :x:     x-axis
    :y:     y-axis
    :f:     data (2-dim)
    :C0:    curvature parameter

    Return
    ------
    :C:     Curvature map
    """

    # derivatives
    F = partial_deriv(x, y, f)

    # unpack partial derivaatives
    fx = F['fx']
    fxx = F['fxx']
    fxy = F['fxy']
    fy = F['fy']
    fyy = F['fyy']
#    fyx = F['fyx']  # not needed

    # nominator terms
    nom_1 = (C0 + fx ** 2) * fyy
    nom_2 = 2 * fx * fy * fxy
    nom_3 = (C0 + fy ** 2) * fxx

    # denominator term
    denom = (C0 + fx ** 2 + fyy ** 2) ** (3 / 2)

    C = (nom_1 - nom_2 + nom_3) / denom

    return C


def curvature_inequiv(x, y, f, Cx, Cy):
    """returns C

    **Curvature with two inequivalent axes**

    Args
    ----
    :x:     x-axis
    :y:     y-axis
    :f:     data (2-dim)
    :Cx:    curvature parameter for x-direction
    :Cy:    curvature parameter for y-direction

    Return
    ------
    :C:     Curvature map
    """

    # derivatives
    F = partial_deriv(x, y, f)

    # unpack partial derivaatives
    fx = F['fx']
    fxx = F['fxx']
    fxy = F['fxy']
    fy = F['fy']
    fyy = F['fyy']
#    fyx = F['fyx']  # not needed

    # nominator terms
    nom_1 = (1 + Cx * fx ** 2) * Cy * fyy
    nom_2 = 2 * Cx * Cy * fx * fy * fxy
    nom_3 = (1 + Cy * fy ** 2) * Cx * fxx

    # denominator term
    denom = (1 + Cx * fx ** 2 + Cy * fyy ** 2) ** (3 / 2)

    C = (nom_1 - nom_2 + nom_3) / denom

    return C


"""
%%%%%%%%%%%%%%%%%%%%%
  Useful functions
%%%%%%%%%%%%%%%%%%%%%
"""


def FDsl(x, *p):
    """returns FDsl

    **Fermi Dirac function on a sloped**

    Args
    ----
    :x:     energy axis
    :p0:    kB * T
    :p1:    EF
    :p2:    Amplitude
    :p3:    Constant background
    :p4:    Slope

    Return
    ------
    :FDsl:  Fermi Dirac function on a sloped background
    """

    FDsl = p[3] + (p[2] + p[4] * x) * (np.exp((x - p[1]) / p[0]) + 1) ** -1

    return FDsl


def poly_n(x, n, *p):
    """returns poly_n

    **Polynomial n-th order**

    Args
    ----
    :x:       x
    :n:       order
    :p[n]:    coefficients

    Return
    ------
    :poly_n:  polynomial n-th order
    """

    poly_n = 0

    # Loop over orders
    for i in range(n+1):
        poly_n += p[i] * x ** i

    return poly_n


def lor_n(x, n, *p):
    """returns lor_n

    **n Lorentzians on a quadratic background**

    Args
    ----
    :x:          momentum
    :n:          number of Lorentzians
    :p[0:n-1]:   center
    :p[n:2*n-1]: HWHM
    :p[2*n:-4]:  amplitudes

    :p[-3]:      constant
    :p[-2]:      slope
    :p[-1]:      quadratic

    Return
    ------
    :lor_n:        n Lorentzians
    """

    lor_n = 0

    # Loop over Lorentzians
    for i in range(n):
        lor_n += (p[i+2*n] / (np.pi * p[i+n] *
                  (1 + ((x - p[i]) / p[i+n]) ** 2)))
    lor_n += p[-3] + p[-2] * x + p[-1] * x ** 2

    return lor_n


def gauss_n(x, n, *p):
    """returns gauss_n

    **n Gaussians on a quadratic background**

    Args
    ----
    :x:          momentum axis
    :n:          number of Gaussians
    :p[0:n-1]:   center
    :p[n:2*n-1]: width
    :p[2*n:-4]:  amplitudes

    :p[-3]:      constant
    :p[-2]:      slope
    :p[-1]:      quadratic

    Return
    ------
    :gauss_n:      n Gaussians
    """

    gauss_n = 0

    # Loop over Gaussians
    for i in range(n):
        gauss_n += p[i+2*n] * np.exp(-(x - p[i]) ** 2 / (2 * p[i+n] ** 2))

    gauss_n += p[-3] + p[-2] * x + p[-1] * x ** 2

    return gauss_n


def FL_spectral_func(x, *p):
    """returns FL_spectral_func

    **Saturated Fermi liquid quasiparticle with Fermi Dirac function**

    Args
    ----
    :x:         energy axis
    :p[0]:      slope coefficient for real part of self energy
    :p[1]:      constant coefficient for imaginary part of self energy
    :p[2]:      quadratic coefficient for imaginary part of self energy
    :p[3]:      excitation energy
    :p[4]:      amplitude
    :p[5]:      kB * T

    Return
    ------
    :FL_spectral_func:    Saturated Fermi liquid model
    """

    ReS = p[0] * x
    ImS = p[1] + p[2] * x ** 2

    FL_spectral_func = (p[4] * 1 / np.pi * ImS /
                        ((x - ReS - p[3]) ** 2 + ImS ** 2) *
                        (np.exp((x - 0) / p[5]) + 1) ** -1)

    return FL_spectral_func


def gauss_mod(x, *p):
    """returns gauss_mod

    **Exponentially modified Gaussian**

    Args
    ----
    :x:         energy axis
    :p[0]:      amplitude Gaussian
    :p[1]:      center Gaussian
    :p[2]:      width Gaussian
    :p[3]:      amplitude error function
    :p[4]:      center error function
    :p[5]:      width error function

    Return
    ------
    :gauss_mod:   Exponentially modified Gaussian
    """

    gauss_mod = (p[0] * np.exp(-.5 * ((-x + p[1]) / p[2]) ** 2) *
                 (p[3] * special.erf((-x + p[4]) / p[5]) + 1))

    return gauss_mod


def Full_spectral_func(x, *p):
    """returns Full_spectral_func

    **Spectral function simple FL + incoherent weight as exp. mod. Gaussian**

    Args
    ----
    :x:         energy axis
    :p[0]:      slope coefficient for real part of self energy
    :p[1]:      constant coefficient for imaginary part of self energy
    :p[2]:      quadratic coefficient for imaginary part of self energy
    :p[3]:      excitation energy
    :p[4]:      amplitude
    :p[5]:      kB * T
    :p[6]:      amplitude Gaussian
    :p[7]:      center Gaussian
    :p[8]:      width Gaussian
    :p[9]:      amplitude error function
    :p[10]:     center error function
    :p[11]:     width error function

    Return
    ------
    :Full_spectral_func:   Fermi liquid spectral function + modified Gaussian
    """

    Full_spectral_func = (FL_spectral_func(x, *p[:6]) +
                          gauss_mod(x, *p[-6:]))

    return Full_spectral_func


"""
%%%%%%%%%%%%%%%%%%%%%
  Wrapper functions
%%%%%%%%%%%%%%%%%%%%%
"""


def poly_1(x, *p):
    """returns poly_1

    **wrapper function of poly_n with n=1**

    Args
    ----
    :x:          momentum axis
    :p[0]:       constant
    :p[1]:       slope

    Return
    ------
    poly_1       polynomial first order
    """

    poly_1 = poly_n(x, 1, *p)

    return poly_1


def poly_2(x, *p):
    """returns poly_2

    **wrapper function of poly_n with n=2**

    Args
    ----
    :x:          momentum axis
    :p[0]:       constant
    :p[1]:       slope
    :p[2]:       quadratic part

    Return
    ------
    :poly_2:       polynomial second order
    """

    poly_2 = poly_n(x, 2, *p)

    return poly_2


def lor(x, *p):
    """returns lor

    **wrapper function of lor_n with n=1**

    Args
    ----
    :x:      momentum
    :n:      number of Lorentzians
    :p[0]:   center
    :p[1]:   HWHM
    :p[2]:   amplitudes

    :p[3]:   constant
    :p[4]:   slope
    :p[5]:   quadratic

    Return
    ------
    :lor:      single Lorentzian
    """

    lor = lor_n(x, 1, *p)

    return lor


def lor_2(x, *p):
    """returns lor_2

    **wrapper function of lor_n with n=2**

    Args
    ----
    :x:          momentum
    :n:          number of Lorentzians
    :p[0:1]:     center
    :p[2:3]:     HWHM
    :p[4:5]:     amplitudes

    :p[-3]:      constant
    :p[-2]:      slope
    :p[-1]:      quadratic

    Return
    ------
    :lor_2:      2 Lorentzians
    """

    lor_2 = lor_n(x, 2, *p)

    return lor_2


def lor_4(x, *p):
    """returns lor_4

    **wrapper function of lor_n with n=4**

    Args
    ----
    :x:          momentum
    :n:          number of Lorentzians
    :p[0:3]:     center
    :p[4:7]:     HWHM
    :p[8:-4]:    amplitudes

    :p[-3]:      constant
    :p[-2]:      slope
    :p[-1]:      quadratic

    Return
    ------
    :lor_4:      4 Lorentzians
    """

    lor_4 = lor_n(x, 4, *p)

    return lor_4


def lor_6(x, *p):
    """returns lor_6

    **wrapper function of lor_n with n=6**

    Args
    ----
    :x:          momentum
    :n:          number of Lorentzians
    :p[0:5]:     center
    :p[6:11]:    HWHM
    :p[12:-4]:   amplitudes

    :p[-3]:      constant
    :p[-2]:      slope
    :p[-1]:      quadratic

    Return
    ------
    :lor_6:      6 Lorentzians
    """

    lor_6 = lor_n(x, 6, *p)

    return lor_6


def lor_7(x, *p):
    """returns lor_7

    **wrapper function of lor_n with n=7**

    Args
    ----
    :x:          momentum
    :n:          number of Lorentzians
    :p[0:6]:     center
    :p[7:12]:    HWHM
    :p[13:-4]:   amplitudes

    :p[-3]:      constant
    :p[-2]:      slope
    :p[-1]:      quadratic

    Return
    ------
    :lor_7:      7 Lorentzians
    """

    lor_7 = lor_n(x, 7, *p)

    return lor_7


def lor_8(x, *p):
    """returns lor_8

    **wrapper function of lor_n with n=8**

    Args
    ----
    :x:          momentum
    :n:          number of Lorentzians
    :p[0:7]:     center
    :p[8:15]:    HWHM
    :p[16:-4]:   amplitudes

    :p[-3]:      constant
    :p[-2]:      slope
    :p[-1]:      quadratic

    Return
    ------
    :lor_8:      8 Lorentzians
    """

    lor_8 = lor_n(x, 8, *p)

    return lor_8


def gauss_2(x, *p):
    """returns gauss_2

    **wrapper function of gauss_n with n=2**

    Args
    ----
    :x:          momentum axis
    :n:          number of Gaussians
    :p[0:2]:     center
    :p[3:5]:     width
    :p[6:8]:     amplitudes

    :p[-3]:      constant
    :p[-2]:      slope
    :p[-1]:      quadratic

    Return
    ------
    :gauss_2:    2 Gaussians
    """

    gauss_2 = gauss_n(x, 2, *p)

    return gauss_2
