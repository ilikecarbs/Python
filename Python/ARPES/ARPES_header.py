#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 10 13:20:31 2018

@author: ilikecarbs

%%%%%%%%%%%%%%%%%%%%%%%%%%%
        ARPES_header
%%%%%%%%%%%%%%%%%%%%%%%%%%%

**Data Loader ARPES data files.**

Implemented:
___________
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
    Todo
        - proper implementation of Bessy header
"""

import os
import h5py
import numpy as np
from astropy.io import fits
from igor import binarywave

from ARPES_methods import Methods  # Methods superclass


class DLS(Methods):
    """
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
              Data loader DLS
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    """

    def __init__(self, file, mat, year, sample):
        # Define directories
        folder = ''.join(['/Users/denyssutter/Documents/2_physics/DATA/',
                          str(mat), '/Diamond', str(year), '/', str(sample), '/'])
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


class ALS(Methods):
    """
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
              Data loader ALS
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    """

    def __init__(self, file, mat, year, sample):
        # Define directories
        folder = ''.join(['/Users/denyssutter/Documents/2_physics/DATA/',
                          str(mat), '/ALS', str(year), '/', str(sample), '/'])
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


class SIS(Methods):
    """
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
              Data loader SIS
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    """

    def __init__(self, file, mat, year, sample):
        # Define directories
        folder = ''.join(['/Users/denyssutter/Documents/2_physics/DATA/',
                          str(mat), '/SIS', str(year), '/', str(sample), '/'])
        filename = ''.join([str(file), '.h5'])
        path = folder + filename
        print('\n ~ Initializing SIS data file: \n {}'.format(path),
              '\n', '==========================================')

        # Read meta data
        f = h5py.File(path, 'r')
        data_meta = f['Electron Analyzer/Image Data']
        data = np.array(data_meta)
        if data.ndim == 3:
            self.int = np.transpose(data, (2, 1, 0))
            d1, d2, d3 = data.shape
        elif data.ndim == 2:
            self.int = np.transpose(data)
            d1, d2 = data.shape
        e_i, de = data_meta.attrs['Axis0.Scale']  # initial energy, energy step
        a_i, da = data_meta.attrs['Axis1.Scale']  # initial angle, angle step

        # Build up data
        en = np.arange(e_i, e_i + (d1) * de -de/2, de)
        ang = np.arange(a_i, a_i + (d2) * da, da)
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


class Bessy(Methods):
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
        folder = ''.join(['/Users/denyssutter/Documents/2_physics/DATA/',
                          str(mat), '/Bessy', str(year), '/', str(sample), '/'])
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


class CASS(Methods):
    """
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            Data loader Cassiopee
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    """

    def __init__(self, file, mat, year, mode):
        # Define directories
        folder = ''.join(['/Users/denyssutter/Documents/2_physics/DATA/',
                          str(mat), '/CASS', str(year), '/'])

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
