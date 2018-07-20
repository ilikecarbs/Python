#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 10 13:20:31 2018

@author: ilikecarbs

%%%%%%%%%%%%%%%%%%%%
        ARPES
%%%%%%%%%%%%%%%%%%%%

**Data Loader and data manipulation of ARPES data files.**

Implemented:
___________
    - **Analysis**:     Methods superclass
    - **DLS**:          Diamond Light Source, i05 beamline
    - **ALS**:          Advanced Light Source, Maestro beamline
    - **SIS**:          Swiss Light Source, SIS beamline
    - **Bessy**:        Helmholtz institute Bessy II, 1^3 beamline
    - **CASS**:         Synchrotron Soleil, Cassiopée beamline

Methods (self.*)
----------
:file: file number that is analysed
:mat: material under study
:year: year of experiment
:sample: sample number
:folder: path of sample number
:filename: full file name
:path: full path

:ang: detector angles
:angs: broadcasted detector angle
:en: energy
:ens: broadcasted energy
:int: intensity
:eint: errors on intensity
:hv: photon energy
:pol: manipulator polar angles (if available)
:pols: broadcasted polar angles (if available)

.. note::
    convention of broadcasting dimensions: *.size
        - 3-dimensional (pol*, ang*, en*)
        - 3-dimensional (hv*, ang*, en*)
        - 2-dimensional (ang*, en*)
"""

import os
import h5py
import numpy as np
from astropy.io import fits
from igor import binarywave
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

import utils
import utils_plt
import utils_math


class Analysis:
    """**Methods superclass**

    Class with all methods for basic data analysis.
    These methods will be passed by to the base data loaders
    from different beamlines and are called from utils.py.

    """

    def gold(self, Ef_ini):
        """Generates gold file

        **Fits and generates gold file for normalization.**

        Args
        ----------
        :Ef_ini: initial guess of Fermi level

        Return
        ------
        Saves data files into the current sample folder

        :Ef_file.dat:       array with fitted Fermi energies
        :norm_file.dat:     total intensity per channel for normalization
        """

        # Change these parameters to tune fitting
        # anchor points for poly fit of Fermi energies
        bnd = 50
        # Takes initial fit parameters for Fermi function until this value.
        # From there on, the initial parameters are the ones from last fit.
        ch = 300

        plt.figure('gold')
        plt.subplot(211)
        enval, inden = utils.find(self.en, Ef_ini-0.12)
        plt.plot(self.en[inden:], self.int[ch, inden:], 'bo', markersize=3)

        # initial guess
        p_ini_FDsl = [.001, Ef_ini, np.max(self.int[ch, :]), 20, 0]

        # Placeholders
        Ef = np.zeros(len(self.ang))
        norm = np.zeros(len(self.ang))

        # Fit loop
        for i in range(0, len(self.ang)):
            try:
                p_FDsl, c_FDsl = curve_fit(utils_math.FDsl, self.en[inden:],
                                           self.int[i, inden:], p_ini_FDsl)
            except RuntimeError:
                print("Error - convergence not reached")

            # Plots data at this particular channel
            if i == ch:
                plt.plot(self.en[inden:], utils_math.FDsl(self.en[inden:],
                         *p_FDsl), 'r-')
            Ef[i] = p_FDsl[1]  # Fit parameter
            norm[i] = sum(self.int[i, :])  # Fit parameters

        # Fit Fermi level fits with a polynomial
        p_ini_poly2 = [Ef[ch], 0, 0, 0]
        p_poly2, c_poly2 = curve_fit(utils_math.poly2, self.ang[bnd:-bnd],
                                     Ef[bnd:-bnd], p_ini_poly2)
        Ef_fit = utils_math.poly2(self.ang, *p_poly2)

        # Save data
        os.chdir(self.folder)
        np.savetxt(''.join(['Ef_', str(self.file), '.dat']), Ef_fit)
        np.savetxt(''.join(['norm_', str(self.file), '.dat']), norm)
        os.chdir('/Users/denyssutter/Documents/library/Python')

        # Plot data
        plt.subplot(212)
        plt.plot(self.ang, Ef, 'bo')
        plt.plot(self.ang, Ef_fit, 'r-')
        plt.ylim(Ef[ch]-5, Ef[ch]+5)
        self.plt_spec()
        plt.show()

    def norm(self, gold):
        """Normalizes data

        .. seealso:: - in utils.py
        """

        en_norm, int_norm, eint_norm = utils.norm(self, gold)
        self.en_norm = en_norm
        self.int_norm = int_norm
        self.eint_norm = eint_norm
        print('\n ~ Data normalized',
              '\n', '==========================================')

    def shift(self, gold):
        """Shifts the energy to Fermi level, but no intensity normalization

        .. seealso:: - in utils.py
        """

        en_shift, int_shift = utils.shift(self, gold)
        self.en_norm = en_shift
        self.int_norm = int_shift
        print('\n ~ Only energy normalized',
              '\n', '==========================================')

    def flatten(self, norm=False):
        """Flatten the spectra (Divide EDC by its sum per channel)

        .. seealso:: - in utils.py
        """

        int_flat = utils.flatten(self, norm)
        if norm:
            self.int_norm = int_flat
        else:
            self.int = int_flat
        print('\n ~ Spectra flattened',
              '\n', '==========================================')

    def FS_flatten(self, ang=True):
        """returns self.map

        **For every angle, the signal in the other angular channel is
        divided by its sum. Data get flattened in this fashion.**

        Args
        ----------
        :ang:      'True': iterates through self.ang, 'False': self.pol

        Return
        ------
        :self.map:  Flattened Fermi surface map
        """

        map_flat = np.zeros(self.map.shape)
        if ang:
            for i in range(self.ang.size):
                map_flat[:, i] = np.divide(self.map[:, i], np.sum(
                        self.map[:, i]))
        else:
            for i in range(self.pol.size):
                map_flat[i, :] = np.divide(self.map[i, :], np.sum(
                        self.map[i, :]))

        self.map = map_flat
        print('\n ~ FS flattened',
              '\n', '==========================================')

    def bkg(self, norm=False):
        """returns self.int, self.int_norm, self.eint, self.eint_norm

        **For every energy, the minimum signal is subtracted for all angles**

        Args
        ----------
        :norm:      'True': self.int_norm is manipulated, 'False': self.int

        Return
        ------
        :self.int:          intensity background subtracted
        :self.int_norm:     normalized intensity background subtracted
        :self.eint:         errors on new intensity
        :self.eint_norm:    errors on new normalized intensity
        """

        if norm:
            int_bkg = self.int_norm
            eint_bkg = self.eint_norm
        else:
            int_bkg = self.int
            eint_bkg = self.eint

        # Subtract background
        for i in range(self.en.size):
            int_bkg[:, i] = int_bkg[:, i] - np.min(int_bkg[:, i])
            eint_bkg[:, i] = eint_bkg[:, i] - np.min(eint_bkg[:, i])

        if norm:
            self.int_norm = int_bkg
            self._norm = eint_bkg
        else:
            self.int = int_bkg
            self.eint = eint_bkg
        print('\n ~ Background subtracted',
              '\n', '==========================================')

    def restrict(self, bot=0, top=1, left=0, right=1):
        """returns self.ang, self.angs, self.pol, self.pols, self.en, self.ens,
        self.en_norm, self.int, self.int_norm, self.eint, self.eint_norm

        **If files are too large or if it is convenient to do so,
        cropt the data files to a smaller size.**

        Args
        ----------
        :bot:       set bottom crop boundary from 0..1
        :top:       set top crop boundary from 0..1
        :left:      set left crop boundary from 0..1
        :right:     set right crop boundary from 0..1

        Return
        ------
        New data variables:
            - self.ang
            - self.angs
            - self.pol (if available)
            - self.pols (if available)
            - self.en
            - self.ens
            - self.en_norm (if available)
            - self.int
            - self.int_norm (if available)
            - self.eint
            - self.eint_norm (if available)
        """

        # For 2-dimensional spectra
        if self.int.ndim == 2:
            d1, d2 = self.int.shape

            # Get indices
            val, _bot = utils.find(range(d2), bot * d2)
            val, _top = utils.find(range(d2), top * d2)
            val, _left = utils.find(range(d1), left * d1)
            val, _right = utils.find(range(d1), right * d1)

            # Restrict spectra
            self.en = self.en[_bot:_top]
            self.ens = self.ens[_left:_right, _bot:_top]
            self.ang = self.ang[_left:_right]
            self.angs = self.angs[_left:_right, _bot:_top]
            self.int = self.int[_left:_right, _bot:_top]
            self.eint = self.eint[_left:_right, _bot:_top]
            try:
                self.en_norm = self.en_norm[_left:_right, _bot:_top]
                self.int_norm = self.int_norm[_left:_right, _bot:_top]
                self.eint_norm = self.eint_norm[_left:_right, _bot:_top]
            except AttributeError:
                pass

        # For 3-dimensional data
        elif self.int.ndim == 3:
            d1, d2 = self.int.shape[1], self.int.shape[0]

            # Get indices
            val, _bot = utils.find(range(d2), bot * d2)
            val, _top = utils.find(range(d2), top * d2)
            val, _left = utils.find(range(d1), left * d1)
            val, _right = utils.find(range(d1), right * d1)

            # Restrict spectra
            self.pol = self.pol[_bot:_top]
            self.pols = self.pols[_bot:_top, _left:_right, :]
            self.ens = self.ens[_bot:_top, _left:_right, :]
            self.ang = self.ang[_left:_right]
            self.angs = self.angs[_bot:_top, _left:_right, :]
            self.int = self.int[_bot:_top, _left:_right, :]
            self.int = self.int[_bot:_top, _left:_right, :]
            try:
                self.en_norm = self.en_norm[_bot:_top, _left:_right, :]
                self.int_norm = self.int_norm[_bot:_top, _left:_right, :]
                self.eint_norm = self.eint_norm[_bot:_top, _left:_right, :]
            except AttributeError:
                pass

        print('\n ~ Spectra restricted',
              '\n', '==========================================')

    def ang2k(self, angdg, Ekin, lat_unit=False, a=5.33, b=5.33, c=11,
              V0=0, thdg=0, tidg=0, phidg=0):
        """returns self.k, self.k_V0, self.kxs, self.kx_V0s,
        self.kys, self.ky_V0s, self.lat_unit

        **Converts detector angles into k-space**

        Args
        ----------
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

        :self.k:        k-vector
        :self.k_V0:     k-vector (with inner potential)
        :self.kxs:      kx-vector broadcasted
        :self.kx_V0s:   kx-vector broadcasted (with inner potential)
        :self.kys:      ky-vector broadcasted
        :self.ky_V0s:   ky-vector broadcasted (with inner potential)
        :self.lat_unit: reciprocal lattice units (boolean)
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
        kv = np.ones((3, len(ang)))
        kv_V0 = np.ones((3, len(ang)))

        # Build k-vector
        kv = np.array([k_norm * np.sin(ang), 0 * ang, k_norm * np.cos(ang)])
        kv_V0 = np.array([k_norm * np.sin(ang), 0 * ang,
                          np.sqrt(k_norm_V0**2 - (k_norm * np.sin(ang)**2))])
        k = np.matmul(Phi, np.matmul(Ti, np.matmul(Th, kv)))
        k_V0 = np.matmul(Phi, np.matmul(Ti, np.matmul(Th, kv_V0)))

        if lat_unit:  # lattice units
            k *= np.array([[a / np.pi], [b / np.pi], [c / np.pi]])
            k_V0 *= np.array([[a / np.pi], [b / np.pi], [c / np.pi]])

        self.k = k
        self.k_V0 = k_V0
        self.lat_unit = lat_unit
        self.kxs = np.transpose(
                np.broadcast_to(k[0], (self.en.size, self.ang.size)))
        self.kx_V0s = np.transpose(
                np.broadcast_to(k_V0[0], (self.en.size, self.ang.size)))
        self.kys = np.transpose(
                np.broadcast_to(k[1], (self.en.size, self.ang.size)))
        self.ky_V0s = np.transpose(
                np.broadcast_to(k_V0[1], (self.en.size, self.ang.size)))
        print('\n ~ Angles converted into k-space',
              '\n', '==========================================')

    def ang2kFS(self, angdg, Ekin, lat_unit=False, a=5.33, b=5.33, c=11,
                V0=0, thdg=0, tidg=0, phidg=0):
        """returns self.kx, self.ky, self.kx_V0, self.ky_V0

        **Converts detector angles into k-space for a FS map**

        Args
        ----------
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

        :self.kx:        reciprocal space vector in x
        :self.ky:        reciprocal space vector in y
        :self.kx_V0:     reciprocal space vector in x (with inner potential)
        :self.ky_V0:     reciprocal space vector in y (with inner potential)
        """

        # Placeholders
        kx = np.ones((self.pol.size, self.ang.size))
        ky = np.ones((self.pol.size, self.ang.size))
        kx_V0 = np.ones((self.pol.size, self.ang.size))
        ky_V0 = np.ones((self.pol.size, self.ang.size))

        # Loop over tilt angles and build up kx, ky
        for i in range(self.pol.size):
            self.ang2k(angdg, Ekin, lat_unit, a, b, c, V0, thdg,
                       self.pol[i]-tidg, phidg)
            kx[i, :] = self.k[0, :]
            ky[i, :] = self.k[1, :]
            kx_V0[i, :] = self.k_V0[0, :]
            ky_V0[i, :] = self.k_V0[1, :]

        self.kx = kx
        self.ky = ky
        self.kx_V0 = kx_V0
        self.ky_V0 = ky_V0
        self.lat_unit = lat_unit
        print('\n ~ Angles converted into k-space for Fermi surface',
              '\n', '==========================================')

    def FS(self, e=0, ew=0.05, norm=False):
        """returns self.map

        **For a given energy e, extract a Fermi surface map,
        integrated to e-ew**

        Args
        ----------
        :e:     energy at which to cut the data
        :ew:    energy window from e downwards

        Return
        ------

        :self.map: Fermi surface map
        """

        FSmap = np.zeros((self.pol.size, self.ang.size))
        if norm:
            for i in range(self.ang.size):
                e_val, e_ind = utils.find(self.en_norm[0, i, :], e)
                ew_val, ew_ind = utils.find(self.en_norm[0, i, :], e-ew)
                FSmap[:, i] = np.sum(self.int_norm[:, i, ew_ind:e_ind], axis=1)
        else:
            e_val, e_ind = utils.find(self.en, e)
            ew_val, ew_ind = utils.find(self.en, e-ew)
            FSmap = np.sum(self.int[:, :, ew_ind:e_ind], axis=2)

        self.map = FSmap
        print('\n ~ Constant energy map extracted',
              '\n', '==========================================')

    def plt_spec(self, norm=False, v_max=1):
        """Plot ARPES spectrum

        .. seealso:: - in utils.py
        """

        utils_plt.plt_spec(self, norm, v_max)

    def plt_FS_polcut(self, norm=False, p=0, pw=0.1):
        """Spectral cut through Fermi surface

        .. seealso:: - in utils.py
        """

        utils_plt.plt_FS_polcut(self, norm, p, pw)

    def plt_FS(self, coord=False, v_max=1):
        """Plot Fermi surface

        .. seealso:: - in utils.py
        """
        utils_plt.plt_FS(self, coord, v_max=1)

    def plt_FS_all(self, coord=False, norm=False):
        """Plot constant energy maps from top energy to bottom energy
        in 0.1 eV binding energy steps.

        .. seealso:: - in utils.py
        """

        utils_plt.plt_FS_all(self, coord, norm)

    def plt_hv(self, v_max=1):
        """Plot all spectra of hv scan

        .. seealso:: - in utils.py
        """

        utils_plt.plt_hv(self, v_max)


class DLS(Analysis):
    """
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
              Data loader DLS
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    """

    def __init__(self, file, mat, year, sample):
        # Define directories
        folder = ''.join(['/Users/denyssutter/Documents/Denys/', str(mat),
                          '/Diamond', str(year), '/', str(sample), '/'])
        filename = ''.join(['i05-', str(file), '.nxs'])
        path = folder + filename
        print('\n ~ Initializing Diamond Light Source data file: \n {}'
              .format(path),
              '\n', '==========================================')

        # Read meta data
        f = h5py.File(path, 'r')  # Read file with h5py reader
        data_list = list(f['/entry1/analyser/data'])
        ang = np.array(list(f['/entry1/analyser/angles']))
        en = np.array(list(f['/entry1/analyser/energies']))
        photon = np.array(list(f['/entry1/instrument/monochromator/energy']))

        # Try if polar angles available
        try:
            pol = np.array(list(f['/entry1/analyser/sapolar']))
            data = np.array(data_list)[:, :, :]

            self.pol = pol
            self.ens = np.broadcast_to(en, (pol.size, ang.size, en.size))
            self.angs = np.transpose(
                            np.broadcast_to(
                                    ang, (pol.size, en.size, ang.size)),
                            (0, 2, 1))
            self.pols = np.transpose(
                            np.broadcast_to(
                                    pol, (ang.size, en.size, pol.size)),
                            (2, 0, 1))
        except KeyError:
            print('- No polar angles available \n')
            data = np.asarray(data_list)[0, :, :]
            self.ens = np.broadcast_to(en, (ang.size, en.size))
            self.angs = np.transpose(np.broadcast_to(ang, (en.size, ang.size)))

        self.file = file
        self.mat = mat
        self.year = year
        self.sample = sample
        self.path = path
        self.filename = filename
        self.folder = folder
        self.ang = ang
        self.en = en
        self.int = data
        self.eint = np.sqrt(data)
        self.hv = photon

        print('\n ~ Initialization complete. Data has {} dimensions'.format(
                len(np.shape(self.int))),
              '\n', '==========================================')
        super(DLS, self)


class ALS(Analysis):
    """
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
              Data loader ALS
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    """

    def __init__(self, file, mat, year, sample):
        # Define directories
        folder = ''.join(['/Users/denyssutter/Documents/Denys/', str(mat),
                          '/ALS', str(year), '/', str(sample), '/'])
        filename = ''.join([str(year), file, '.fits'])
        path = folder + filename
        print('\n ~ Initializing Advanced Light Source data file: \n {}'
              .format(path),
              '\n', '==========================================')

        # Read meta data
        f = fits.open(path)
        hdr = f[0].header  # load header
        meta_data = f[1].data
        mode = hdr['NM_0_0']  # scan mode
        px_per_en = hdr['SSPEV_0']  # pixels per energy
        e_i = hdr['SSX0_0']  # initial energy
        e_f = hdr['SSX1_0']  # final energy
        a_i = hdr['SSY0_0']  # initial angle
        a_f = hdr['SSY1_0']  # final angle
        Ef = hdr['SSKE0_0']  # Fermi energy (not sure it is correct every time)

        # Some parameters
        ang_per_px = 0.193  # Angled per pixel
        binning = 4  # I think this is the correct binning
        npol = meta_data.size  # number of polar cuts
        (nen, nang) = meta_data[0][-1].shape  # number of energy, angle steps
        data = np.zeros((npol, nang, nen))  # Placeholder

        # Build up data
        en = (np.arange(e_i, e_f, 1.) - Ef) / px_per_en
        ang = np.arange(a_i, a_f, 1.) * ang_per_px / binning
        pol = np.arange(0, npol, 1.)
        if mode == 'Beta':
            for i in range(npol):
                pol[i] = meta_data[i][1]
                data[i, :, :] = np.transpose(meta_data[i][-1])
                self.ens = np.broadcast_to(en, (pol.size, ang.size, en.size))
                self.angs = np.transpose(np.broadcast_to(
                                ang, (pol.size, en.size, ang.size)), (0, 2, 1))
                self.pols = np.transpose(np.broadcast_to(
                                pol, (ang.size, en.size, pol.size)), (2, 0, 1))

        self.file = file
        self.mat = mat
        self.year = year
        self.sample = sample
        self.path = path
        self.folder = folder
        self.pol = pol
        self.ang = ang
        self.en = en
        self.int = data
        self.eint = np.sqrt(data)
        self.hv = meta_data[0][-4]

        print('\n ~ Initialization complete. Data has {} dimensions'.format(
                len(np.shape(self.int))),
              '\n', '==========================================')
        super(ALS, self)


class SIS(Analysis):
    """
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
              Data loader SIS
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    """

    def __init__(self, file, mat, year, sample):
        # Define directories
        folder = ''.join(['/Users/denyssutter/Documents/Denys/', str(mat),
                          '/SIS', str(year), '/', str(sample), '/'])
        filename = ''.join([str(file), '.h5'])
        path = folder + filename
        print('\n ~ Initializing SIS data file: \n {}'.format(path),
              '\n', '==========================================')

        # Read meta data
        f = h5py.File(path, 'r')
        data = np.array((f['Electron Analyzer/Image Data']))
        if data.ndim == 3:
            self.int = np.transpose(data, (2, 1, 0))
            d1, d2, d3 = data.shape
        elif data.ndim == 2:
            self.int = np.transpose(data)
            d1, d2 = data.shape
        e_i, de = data.attrs['Axis0.Scale']  # initial energy, energy step
        a_i, da = data.attrs['Axis1.Scale']  # initial angle, angle step

        # Build up data
        en = np.arange(e_i, e_i + d1 * de, de)
        ang = np.arange(a_i, a_i + d2 * da, da)
        hv = np.array(f['Other Instruments/hv'])
        pol = np.array(f['Other Instruments/Tilt'])
        # Cases of scan modes#
        if pol.size > 1 & hv.size == 1:
            print('- Fermi surface map \n')
            self.ens = np.broadcast_to(en, (pol.size, ang.size, en.size))
            self.angs = np.transpose(
                            np.broadcast_to(
                                    ang, (pol.size, en.size, ang.size)),
                            (0, 2, 1))
            self.pols = np.transpose(
                            np.broadcast_to(
                                    pol, (ang.size, en.size, pol.size)),
                            (2, 0, 1))
        elif hv.size > 1 & pol.size == 1:
            print('- Photon energy scan \n')
            self.ens = np.broadcast_to(en, (hv.size, ang.size, en.size))
            self.angs = np.transpose(
                            np.broadcast_to(ang, (hv.size, en.size, ang.size)),
                            (0, 2, 1))
            self.hvs = np.transpose(
                            np.broadcast_to(hv, (ang.size, en.size, hv.size)),
                            (2, 0, 1))
        elif pol.size == 1 & hv.size == 1:
            print('- No polar angles available \n- No photon energy scan \n')
            self.ens = np.broadcast_to(en, (ang.size, en.size))
            self.angs = np.transpose(np.broadcast_to(ang, (en.size, ang.size)))

        self.file = file
        self.mat = mat
        self.sample = sample
        self.year = year
        self.filename = filename
        self.path = path
        self.folder = folder
        self.ang = ang
        self.en = en
        self.eint = np.sqrt(data)
        self.hv = hv
        self.pol = pol
        print('\n ~ Initialization complete. Data has {} dimensions'.format(
                len(np.shape(self.int))),
              '\n', '==========================================')
        super(SIS, self)


class Bessy(Analysis):
    """
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
             Data loader Bessy
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    .. warning::
        - This data loader is quickly done and not properly implemented.
        - Only good enough for the plots I needed to make.
    """

    def __init__(self, file, mat, year, sample):
        # Define directories
        folder = ''.join(['/Users/denyssutter/Documents/Denys/', str(mat),
                          '/Bessy', str(year), '/', str(sample), '/'])
        path_ang = ''.join(['Ca_', str(file), 'ang', '.dat'])
        path_pol = ''.join(['Ca_', str(file), 'pol', '.dat'])
        path_en = ''.join(['Ca_', str(file), 'en', '.dat'])

        # Build up data
        ang = np.loadtxt(folder + path_ang)
        en = np.loadtxt(folder + path_en)
        if file == 8:
            pol = np.loadtxt(folder + path_pol)
            self.pol = pol
        print('\n ~ Initializing Bessy data file: \n {}'
              .format(folder+str(file)),
              '\n', '==========================================')

        # This file is a map and treated differently
        if file == 8:
            data = np.zeros((len(pol), len(ang), len(en)))
            for k in range(self.pol.size):
                filename = ''.join(['Ca_', str(file), 'int',
                                    str(k + 1), '.dat'])
                path = folder + filename
                data[k, :, :] = np.transpose(np.loadtxt(path))
            print('- Fermi surface map \n')
            self.ens = np.broadcast_to(en, (pol.size, ang.size, en.size))
            self.angs = np.transpose(
                            np.broadcast_to(
                                    ang, (pol.size, en.size, ang.size)),
                            (0, 2, 1))
            self.pols = np.transpose(
                            np.broadcast_to(
                                    pol, (ang.size, en.size, pol.size)),
                            (2, 0, 1))
        else:
            filename = ''.join(['Ca_', str(file), 'int', '.dat'])
            path = folder + filename
            data = np.transpose(np.loadtxt(path))
            print('- No polar angles available \n')
            self.ens = np.broadcast_to(en, (ang.size, en.size))
            self.angs = np.transpose(np.broadcast_to(ang, (en.size, ang.size)))

        self.filename = filename
        self.path = path
        self.file = file
        self.mat = mat
        self.year = year
        self.sample = sample
        self.folder = folder
        self.ang = ang
        self.en = en
        self.int = data
        self.eint = np.sqrt(data)
        print('\n ~ Initialization complete. Data has {} dimensions'.format(
                len(np.shape(self.int))),
              '\n', '==========================================')
        super(Bessy, self)


class CASS(Analysis):
    """
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            Data loader Cassiopee
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    """

    def __init__(self, file, mat, year, mode):
        # Define directories
        folder = ''.join(['/Users/denyssutter/Documents/Denys/', str(mat),
                          '/CASS', str(year), '/'])

        # txt file
        if mode == 'cut_txt':
            filename = ''.join([str(file), '.txt'])
            path = folder + filename

            # Build up data
            with open(path, 'r') as f:
                for i, line in enumerate(f.readlines()):
                    if line.startswith('Dimension 1 scale='):
                        en = line.split('=')[-1].split()
                        en = np.array(en, dtype=float)
                    elif line.startswith('Dimension 2 scale='):
                        ang = line.split('=')[-1].split()
                        ang = np.array(ang, dtype=float)
                    elif line.startswith('Excitation Energy'):
                        hv = float(line.split('=')[-1])
                    elif 'Data' in line:
                        break
            data = np.loadtxt(path, skiprows=i+1)[:, 1:]
            data = np.transpose(data)
            self.ens = np.broadcast_to(en, (ang.size, en.size))
            self.angs = np.transpose(np.broadcast_to(ang, (en.size, ang.size)))

        # Igro ibw file
        elif mode == 'cut_ibw':
            filename = ''.join([str(file), '.ibw'])
            path = folder + filename
            wave = binarywave.load(path)['wave']
            data = np.array([wave['wData']])[0]
            data = np.transpose(data)

            # Load meta data
            header = wave['wave_header']
            nDim = header['nDim']
            steps = header['sfA']
            starts = header['sfB']
            en = np.linspace(starts[0], starts[0] + nDim[0] * steps[0],
                             nDim[0])
            ang = np.linspace(starts[1], starts[1] + nDim[1] * steps[1],
                              nDim[1])
            # Convert `note`, which is a bytestring of ASCII characters that
            # contains some metadata, to a list of strings
            note = wave['note']
            note = note.decode('ASCII').split('\r')
            # Now the extraction fun begins. Most lines are of the form
            # `Some-kind-of-name=some-value`
            metadata = dict()
            for line in note:
                try:
                    name, val = line.split('=')
                except ValueError:
                    continue
                metadata.update({name: val})
            hv = metadata['Excitation Energy']
            self.ens = np.broadcast_to(en, (ang.size, en.size))
            self.angs = np.transpose(np.broadcast_to(ang, (en.size, ang.size)))

        # Fermi Surface mode
        if mode == 'FSM':
            sub_folder = ''.join([folder, str(file), '/'])
            n_scans = (len([name for name in os.listdir(sub_folder)
                            if os.path.isfile(
                                    os.path.join(sub_folder, name))])) / 2
            n_scans = np.int(n_scans)
            pol = np.zeros(n_scans)
            for scan in range(n_scans):
                filename = ''.join(['FS_', str(scan+1), '_ROI1_.txt'])
                info = ''.join(['FS_', str(scan+1), '_i.txt'])
                path_info = sub_folder + info
                path = sub_folder + filename
                print('\n ~ Initializing Cassiopee data file: \n {}'
                      .format(path),
                      '\n', '==========================================')

                # Build pol
                with open(path_info) as f:
                    for line in f.readlines():
                        if 'theta' in line:
                            theta = line.strip('theta (deg):')
                            pol[scan] = np.float32(theta.split())
                data_txt = np.genfromtxt(path, skip_header=44, delimiter='\t')
                if scan == 0:
                    en = data_txt[:, 0]

                    # Build ang
                    with open(path) as f:
                        for line in f.readlines():
                            if 'Dimension 2 scale' in line:
                                ang_raw = line.strip('Dimension 2 scale=')
                                ang_raw = ang_raw.split()
                                ang = np.array(ang_raw, dtype=np.float32)
                    data = np.zeros((n_scans, len(ang), len(en)))
                data[scan] = np.transpose(data_txt[:, 1:])
            self.pol = pol
            self.ens = np.broadcast_to(en, (pol.size, ang.size, en.size))
            self.angs = np.transpose(
                            np.broadcast_to(
                                    ang, (pol.size, en.size, ang.size)),
                            (0, 2, 1))
            self.pols = np.transpose(
                            np.broadcast_to(
                                    pol, (ang.size, en.size, pol.size)),
                            (2, 0, 1))

        # hv scan mode
        elif mode == 'hv':
            sub_folder = ''.join([folder, str(file), '/'])
            n_scans = (len([name for name in os.listdir(sub_folder)
                            if os.path.isfile(
                                    os.path.join(sub_folder, name))])) / 2
            n_scans = np.int(n_scans)
            hv = np.zeros(n_scans)
            for scan in range(n_scans):
                filename = ''.join(['hv_', str(scan + 1), '_ROI1_.txt'])
                info = ''.join(['hv_', str(scan + 1), '_i.txt'])
                path_info = sub_folder + info
                path = sub_folder + filename
                print('\n ~ Initializing Cassiopee data file: \n {}'
                      .format(path),
                      '\n', '==========================================')

                # Build up hv
                with open(path_info) as f:
                    for line in f.readlines():
                        if 'hv' in line:
                            hv_raw = line.strip(('hv (eV):'))
                            try:
                                hv[scan] = np.float32(hv_raw.split())
                            except ValueError:
                                break
                data_txt = np.genfromtxt(path, skip_header=44, delimiter='\t')
                if scan == 0:
                    en = data_txt[:, 0]

                    # Build angle
                    with open(path) as f:
                        for line in f.readlines():
                            if 'Dimension 2 scale' in line:
                                ang_raw = line.strip('Dimension 2 scale=')
                                ang_raw = ang_raw.split()
                                ang = np.array(ang_raw, dtype=np.float32)
                    data = np.zeros((n_scans, len(ang), len(en)))
                data[scan] = np.transpose(data_txt[:, 1:])
            self.hv = hv
            self.ens = np.broadcast_to(en, (hv.size, ang.size, en.size))
            self.angs = np.transpose(
                            np.broadcast_to(ang, (hv.size, en.size, ang.size)),
                            (0, 2, 1))
            self.hvs = np.transpose(
                            np.broadcast_to(hv, (ang.size, en.size, hv.size)),
                            (2, 0, 1))

        self.file = file
        self.mat = mat
        self.mode = mode
        self.path = path
        self.filename = filename
        self.folder = folder
        self.en = en
        self.ang = ang
        self.int = data
        self.eint = np.sqrt(data)
        print('\n ~ Initialization complete. Data has {} dimensions'.format(
                len(np.shape(self.int))),
              '\n', '==========================================')

        super(CASS, self)
