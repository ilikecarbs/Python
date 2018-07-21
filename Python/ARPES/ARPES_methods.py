#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 20 22:44:11 2018

@author: ilikecarbs

%%%%%%%%%%%%%%%%%%%%
    APRES_methods
%%%%%%%%%%%%%%%%%%%%

**Methods superclass for ARPES data manipulation.**

.. note::
        To-Do:
            - Method to cut through Fermi surfaces
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import matplotlib.cm as cm

import utils


# Set standard fonts
rainbow_light = utils.rainbow_light
cm.register_cmap(name='rainbow_light', cmap=rainbow_light)
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.serif'] = ['Computer Modern Roman']
plt.rc('font', **{'family': 'serif', 'serif': ['STIXGeneral']})
font = {'family': 'serif',
        'style': 'normal',
        'color':  [0, 0, 0],
        'weight': 'ultralight',
        'size': 12,
        }
kwargs_spec = {'cmap': cm.ocean_r}


class Methods:
    """
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
             Methods superclass
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    **Class with all methods for basic data analysis.
    These methods will be passed by to the base data loaders
    from different beamlines and are called from utils.py.**
    """

    def gold(self, Ef_ini):
        """Generates gold file

        **Fits and generates gold file for normalization.**

        Args
        ----
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
                p_FDsl, c_FDsl = curve_fit(utils.FDsl, self.en[inden:],
                                           self.int[i, inden:], p_ini_FDsl)
            except RuntimeError:
                print("Error - convergence not reached")

            # Plots data at this particular channel
            if i == ch:
                plt.plot(self.en[inden:], utils.FDsl(self.en[inden:],
                         *p_FDsl), 'r-')
            Ef[i] = p_FDsl[1]  # Fit parameter
            norm[i] = sum(self.int[i, :])  # Fit parameters

        # Fit Fermi level fits with a polynomial
        p_ini_poly2 = [Ef[ch], 0, 0, 0]
        p_poly2, c_poly2 = curve_fit(utils.poly2, self.ang[bnd:-bnd],
                                     Ef[bnd:-bnd], p_ini_poly2)
        Ef_fit = utils.poly2(self.ang, *p_poly2)

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
        """returns self.en_norm, self.int_norm, self.eint_norm, self.gold

        **Normalizes data intensity and shifts to Fermi level**

        Args
        ----
        :gold:      gold file used

        Return
        ------
        :self.en_norm:      shifted energy
        :self.int_norm:     normalized intensities
        :self.eint_norm:    corresponding errors
        :self.gold:         gold file used
        """

        # Test if there is a gold file
        try:
            os.chdir(self.folder)
            Ef = np.loadtxt(''.join(['Ef_', str(gold), '.dat']))
            norm = np.loadtxt(''.join(['norm_', str(gold), '.dat']))
            os.chdir('/Users/denyssutter/Documents/library/Python')

            # Placeholders
            en_norm = np.ones(self.ens.shape)
            int_norm = np.ones(self.int.shape)
            eint_norm = np.ones(self.eint.shape)

            # Take dimensionality of data file into consideration
            if np.size(self.int.shape) == 2:
                for i in range(self.angs.shape[0]):
                    en_norm[i, :] = self.en - Ef[i]
                    int_norm[i, :] = np.divide(self.int[i, :], norm[i])
                    eint_norm[i, :] = np.divide(self.eint[i, :], norm[i])
            elif np.size(self.int.shape) == 3:
                for i in range(self.angs.shape[1]):
                    en_norm[:, i, :] = self.en - Ef[i]
                    int_norm[:, i, :] = np.divide(self.int[:, i, :], norm[i])
                    eint_norm[:, i, :] = np.divide(self.eint[:, i, :], norm[i])

        except OSError:
            os.chdir('/Users/denyssutter/Documents/library/Python')
            print('- No gold files: {}'.format(self.gold), '\n')

        self.gold = gold
        self.en_norm = en_norm
        self.int_norm = int_norm
        self.eint_norm = eint_norm
        print('\n ~ Data normalized',
              '\n', '==========================================')

    def shift(self, gold):
        """returns self.en_norm, self.gold

        **Shifts energy with by the Fermi level**

        Args
        ----
        :gold:      gold file used

        Return
        ------
        :self.en_norm:  shifted energy
        :self.gold:     gold file used
        """

        # Test if there is a gold file in the folder
        try:
            os.chdir(self.folder)
            Ef = np.loadtxt(''.join(['Ef_', str(gold), '.dat']))
            os.chdir('/Users/denyssutter/Documents/library/Python')

            # Placeholder
            en_shift = np.ones(self.ens.shape)

            # Take dimensionality of data file into consideration
            if np.size(self.int.shape) == 2:
                for i in range(self.angs.shape[0]):
                    en_shift[i, :] = self.en - Ef[i]
            elif np.size(self.int.shape) == 3:
                for i in range(self.angs.shape[1]):
                    en_shift[:, i, :] = self.en - Ef[i]

        except OSError:
            os.chdir('/Users/denyssutter/Documents/library/Python')
            print('- No gold files: {}'.format(self.gold), '\n')

        self.gold = gold
        self.en_norm = en_shift
        print('\n ~ Energies shifted',
              '\n', '==========================================')

    def flatten(self):
        """returns self.int, self.eint, self.int_norm, self.eint_norm

        **For every angle, the signal is divided by its total intensity**

        Args
        ----

        Return
        ------
        :self.int:          flattened intensity
        :self.eint:         corresponding error
        :self.int_norm:     flattened normalized intensity
        :self.eint_rnom:    corresponding error
        """

        for i in range(self.ang.size):
                self.int[i, :] = np.divide(
                                            self.int[i, :],
                                            np.sum(self.int[i, :]))
                self.eint[i, :] = np.divide(
                                            self.eint[i, :],
                                            np.sum(self.eint[i, :]))

        # Tests if it can flatten self.int_norm
        try:
            for i in range(self.ang.size):
                self.int_norm[i, :] = np.divide(
                                            self.int_norm[i, :],
                                            np.sum(self.int_norm[i, :]))
                self.eint_norm[i, :] = np.divide(
                                            self.eint_norm[i, :],
                                            np.sum(self.eint_norm[i, :]))
        except AttributeError:
            pass

        print('\n ~ Spectra flattened',
              '\n', '==========================================')

    def FS_flatten(self, ang=True):
        """returns self.map

        **For every angle, the signal in the other angular channel is
        divided by its sum. Data gets flattened in this fashion.**

        Args
        ----
        :ang:      'True': iterates through self.ang, 'False': self.pol

        Return
        ------
        :self.map:  Flattened Fermi surface map
        """

        if ang:
            for i in range(self.ang.size):
                self.map[:, i] = np.divide(self.map[:, i], np.sum(
                        self.map[:, i]))
        else:
            for i in range(self.pol.size):
                self.map[i, :] = np.divide(self.map[i, :], np.sum(
                        self.map[i, :]))

        print('\n ~ Fermi surface flattened',
              '\n', '==========================================')

    def bkg(self):
        """returns self.int, self.int_norm, self.eint, self.eint_norm

        **For every energy, the minimum signal is subtracted for all angles**

        Args
        ----
        :norm:      'True': self.int_norm is manipulated, 'False': self.int

        Return
        ------
        :self.int:          intensity background subtracted
        :self.int_norm:     normalized intensity background subtracted
        :self.eint:         errors on new intensity
        :self.eint_norm:    errors on new normalized intensity
        """

        try:
            for i in range(self.en.size):
                self.int_norm[:, i] = (self.int_norm[:, i] -
                                       np.min(self.int_norm[:, i]))
                self.eint_norm[:, i] = (self.eint_norm[:, i] -
                                        np.min(self.eint_norm[:, i]))
        except AttributeError:
            pass

        for i in range(self.en.size):
            self.int[:, i] = (self.int[:, i] - np.min(self.int[:, i]))
            self.eint[:, i] = (self.eint[:, i] - np.min(self.eint[:, i]))

        print('\n ~ Background subtracted',
              '\n', '==========================================')

    def restrict(self, bot=0, top=1, left=0, right=1):
        """returns self.ang, self.angs, self.pol, self.pols, self.en, self.ens,
        self.en_norm, self.int, self.int_norm, self.eint, self.eint_norm

        **If files are too large or if it is convenient to do so,
        cropt the data files to a smaller size.**

        Args
        ----
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

    def FS(self, e=0, ew=0.05):
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
        try:
            for i in range(self.ang.size):
                e_val, e_ind = utils.find(self.en_norm[0, i, :], e)
                ew_val, ew_ind = utils.find(self.en_norm[0, i, :], e-ew)
                FSmap[:, i] = np.sum(self.int_norm[:, i, ew_ind:e_ind], axis=1)
        except AttributeError:
            e_val, e_ind = utils.find(self.en, e)
            ew_val, ew_ind = utils.find(self.en, e-ew)
            FSmap = np.sum(self.int[:, :, ew_ind:e_ind], axis=2)

        self.map = FSmap
        print('\n ~ Constant energy map extracted',
              '\n', '==========================================')

    def plt_spec(self, v_max=1):
        """returns ARPES plot

        **Plots ARPES spectrum**

        Args
        ----
        :v_max:    contrast of plot: 0..1

        Return
        ------
        - ARPES spectrum plot (normalized if available)
        """

        # Create figure
        fig = plt.figure(('spec_' + str(self.filename)), figsize=(5, 5),
                         clear=True)
        ax = fig.add_axes([0.2, 0.2, 0.6, 0.6])
        ax.tick_params(direction='in', length=1.5, width=.5, colors='k')

        # Plot normalized data if available
        try:
            c0 = ax.contourf(self.angs, self.en_norm, self.int_norm, 150,
                             **kwargs_spec,
                             vmax=v_max*np.max(self.int_norm))
            ax.plot([np.min(self.angs), np.max(self.angs)], [0, 0], 'k:')

        except AttributeError:
            c0 = ax.contourf(self.angs, self.ens, self.int, 150, **kwargs_spec,
                             vmax=v_max*np.max(self.int))

        # Labels and other stuff
        ax.set_xlabel('Angles', fontdict=font)
        ax.set_ylabel(r'$\omega$', fontdict=font)
        pos = ax.get_position()
        cax = plt.axes([pos.x0+pos.width+0.01, pos.y0, 0.03, pos.height])
        cbar = plt.colorbar(c0, cax=cax, ticks=None)
        cbar.set_ticks([])
        fig.show()
        print('\n ~ Plot ARPES spectrum',
              '\n', '==========================================')

    def plt_FS(self, v_max=1):
        """returns FS plot

        **Plots Fermi surface**

        Args
        ----
        :v_max:    contrast of plot: 0..1

        Return
        ------
        - Fermi surface plot (with appropriate coordinate transformation)
        """

        # Plot FS only if D.map exists
        try:
            print('D.map exists with shape {}'.format(self.map.shape))

            # Create figure
            fig = plt.figure(('FS' + str(self.file)), figsize=(8, 8),
                             clear=True)
            ax = fig.add_axes([0.2, 0.2, 0.6, 0.6])
            ax.tick_params(direction='in', length=1.5, width=.5, colors='k')

            # Plots in kx and ky if available
            try:
                c0 = ax.contourf(self.kx, self.ky, self.map, 150,
                                 **kwargs_spec, vmax=v_max*np.max(self.map))

                # Plot in appropriate units
                if self.lat_unit:
                    ax.set_xlabel(r'$k_x \, (\pi / a)$', fontdict=font)
                    ax.set_ylabel(r'$k_y \, (\pi / y)$', fontdict=font)
                else:
                    ax.set_xlabel(r'$k_x \, (\mathrm{\AA}^{-1})$',
                                  fontdict=font)
                    ax.set_ylabel(r'$k_y \, (\mathrm{\AA}^{-1})$',
                                  fontdict=font)

            except AttributeError:
                c0 = ax.contourf(self.ang, self.pol, self.map,
                                 150, **kwargs_spec,
                                 vmax=v_max*np.max(self.map))
                ax.set_xlabel('Detector angles', fontdict=font)
                ax.set_ylabel('Polar angles', fontdict=font)

            # Some other stuff
            ax.grid(alpha=.5)
            ax.axis('equal')
            pos = ax.get_position()
            cax = plt.axes([pos.x0+pos.width+0.01, pos.y0, 0.03, pos.height])
            cbar = plt.colorbar(c0, cax=cax, ticks=None)
            cbar.set_ticks([])
            cbar.set_clim(np.min(self.map), np.max(self.map))
            fig.show()

        except AttributeError:
            print('No FS available. Use .FS method before using .plt_FS!')

        print('\n ~ Plot Fermi surface',
              '\n', '==========================================')

    def plt_FS_all(self):
        """returns plot of all FS

        **Plots constant energy maps at all energies**

        Args
        ----
        :v_max:    contrast of plot: 0..1

        Return
        ------
        - Fermi surface plot (with appropriate coordinate transformation)
        """

        # Define energy range for all the maps
        try:
            en_range = np.arange(
                    np.min(self.en_norm) + .2, np.max(self.en_norm), .1)

        except AttributeError:
            en_range = np.arange(np.min(self.en) + .2, np.max(self.en), .1)

        # Create figure
        fig = plt.figure(('FS_all' + str(self.file)), figsize=(8, 8),
                         clear=True)

        # number of maps
        n_maps = np.ceil(np.sqrt(len(en_range)))
        n = 0
        for en in en_range:
            n += 1
            self.FS(e=en, ew=.1)
            ax = fig.add_subplot(n_maps, n_maps, n)
            ax.tick_params(direction='in', length=1.5, width=.5, colors='k')

            # Try to plot in k-space
            try:
                ax.contourf(self.kx, self.ky, self.map, 150, **kwargs_spec)

                # set appropriate labels for lattice units
                # also only label the border subplots
                if self.lat_unit:
                    if np.mod(n - 1 + n_maps, n_maps) == 0:
                        ax.set_ylabel(r'$k_y \, (\pi / b)$', fontdict=font)
                        ax.set_yticks(np.arange(-10, 10, 1))
                    else:
                        ax.set_yticks(np.arange(-10, 10, 1), [])
                    if n > n_maps ** 2 - n_maps - (n_maps ** 2 -
                                                   len(en_range)):
                        ax.set_xlabel(r'$k_x \, (\pi / a)$', fontdict=font)
                        ax.set_xticks(np.arange(-10, 10, 1))
                    else:
                        ax.set_xticks(np.arange(-10, 10, 1), [])
                else:
                    if np.mod(n - 1 + n_maps, n_maps) == 0:
                        ax.set_ylabel(r'$k_y \, (\mathrm{\AA}^{-1})$',
                                      fontdict=font)
                        ax.set_yticks(np.arange(-10, 10, 1))
                    else:
                        ax.set_yticks(np.arange(-10, 10, 1), [])
                    if n > n_maps ** 2 - n_maps - (n_maps ** 2 -
                                                   len(en_range)):
                        ax.set_xlabel(r'$k_x \, (\mathrm{\AA}^{-1})$',
                                      fontdict=font)
                        ax.set_xticks(np.arange(-10, 10, 1))
                    else:
                        ax.set_xticks(np.arange(-10, 10, 1), [])

                ax.set_xlim(np.max(self.kx), np.min(self.kx))
                ax.set_ylim(np.max(self.ky), np.min(self.ky))

            except AttributeError:
                ax.contourf(self.ang, self.pol, self.map, 150, **kwargs_spec)
                ax.set_xlim(np.max(self.ang), np.min(self.ang))
                ax.set_ylim(np.max(self.pol), np.min(self.pol))

            # Titles and stuff
            ax.set_title('energy = ' + str(np.round(en, 2)) + 'eV')
            ax.grid(alpha=.5)
            ax.axis('equal')
        fig.show()
        print('\n ~ Plot all constant energy maps',
              '\n', '==========================================')

    def plt_hv(self, v_max=1):
        """returns plot of hv scan

        **Plots ARPES spectra at all photon energies at hv scan**

        Args
        ----
        :v_max:    contrast of plot: 0..1

        Return
        ------
        - hv scan plot
        """

        # number of spectra
        n = np.ceil(np.sqrt(self.hv.size))

        # Create figure
        fig = plt.figure(('hv scan' + str(self.filename)),
                         figsize=(10, 10), clear=True)

        # Loops over all spectra
        for i in range(self.hv.size):
            ax = fig.add_subplot(n, n, i+1)
            ax.tick_params(direction='in', length=1.5, width=.5, colors='k')
            ax.contourf(self.ang, self.en, np.transpose(self.int[i, :, :]),
                        150, **kwargs_spec,
                        vmax=v_max*np.max(self.int[i, :, :]))
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title('hv = ' + str(np.round(self.hv[i], 0)) + ' eV')
        fig.show()
        print('\n ~ Plot photon energy scan',
              '\n', '==========================================')