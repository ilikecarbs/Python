#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 10 13:20:31 2018

@author: denyssutter

%%%%%%%%%%%%%%%%%%%%
        ARPES
%%%%%%%%%%%%%%%%%%%%

Content:
Data Loader and data manipulation ARPES data files.

"""
import os
os.chdir('/Users/denyssutter/Documents/library/Python/ARPES')
import h5py
import numpy as np
import utils
import utils_plt
from astropy.io import fits
from igor import binarywave

class Analysis:  
    """
    Class with all methods for basic data analysis.
    These methods will be passed by to the base data loaders
    from different beamlines and are called from utils.py.
    """  
    def gold(self, Ef_ini):
        """
        %%%%%%%%%%%%%%%
        Fits and generates gold file.
        %%%%%%%%%%%%%%%
            args:
                - Ef_ini:   Initial guess of Fermi energy
        ---------------
            output:
                - Fitted gold spectra, saved in self.folder
        ---------------
        """
        utils.gold(self, Ef_ini)
        
    def norm(self, gold):
        """
        Normalize data with a gold spectrum
        ---------------
            args: 
                - gold:   gold spectra used for normalization
        ---------------
            output:
                - self.en_norm (shifted energy to Fermi level)
                - self.int_norm (normalized intensity)
                - self.eint_norm (errors on normalized intensity)
        ---------------
        """
        en_norm, int_norm, eint_norm = utils.norm(self, gold)
        self.en_norm = en_norm
        self.int_norm = int_norm
        self.eint_norm = eint_norm
        print('\n ~ Data normalized',
              '\n', '==========================================')   
        
    def shift(self, gold):
        """
        %%%%%%%%%%%%%%%
        Shift to the Fermi level without normalizing intensity
        %%%%%%%%%%%%%%%
            args: 
                - gold:   gold spectra used for normalization
        ---------------
            output:
                - self.en_norm (shifted energy to Fermi level)
                - self.int_norm (= self.int)
                - self.eint_norm (errors on intensity)
        ---------------
        """
        en_shift, int_shift = utils.shift(self, gold)
        self.en_norm = en_shift
        self.int_norm = int_shift
        print('\n ~ Only energy normalized',
              '\n', '==========================================')   
        
    def flatten(self, norm=False): 
        """
        %%%%%%%%%%%%%%%
        Flatten the spectra (Divide EDC by its sum per channel)
        %%%%%%%%%%%%%%%
            args: 
                - norm:     'False': flattens self.int
                            'True': flattens self.int_norm
        ---------------
            output:
                - self.int (flattened self.int)
                - self.int_norm (flattened self.int_norm)
                - self.eint_norm (errors on normalized intensity)
        ---------------
        """
        int_flat = utils.flatten(self, norm)
        if norm == False:
            self.int = int_flat
        elif norm == True:
            self.int_norm = int_flat
        print('\n ~ Spectra flattened',
              '\n', '==========================================')   
    
    def FS_flatten(self, ang=True): 
        """
        %%%%%%%%%%%%%%%
        Flattens Fermi surface (Divide for every angle by its sum)
        %%%%%%%%%%%%%%%
            args: 
                - ang:      'True': flattens for every self.ang
                            'False': flattens for every self.pol
        ---------------
            output:
                - self.map (flattened self.map)
        ---------------
        """
        map_flat = utils.FS_flatten(self, ang)
        self.map = map_flat
        print('\n ~ FS flattened',
              '\n', '==========================================') 
        
    def bkg(self, norm=False): 
        """
        %%%%%%%%%%%%%%%
        Subtracts a background EDC. Takes minimum of every MDC
        and appends it to the background EDC.
        %%%%%%%%%%%%%%%
        ---------------
            args: 
                - norm:     'False': subtracts background for self.int
                            'True': subtracts background for self.int_norm
        ---------------
            output:
                - self.int (background subtracted self.int)
                - self.int_norm (background subtracted self.int_norm)
        ---------------
        """
        int_bkg = utils.bkg(self, norm)
        if norm == True:
            self.int_norm = int_bkg
        elif norm == False:
            self.int = int_bkg
        print('\n ~ Background subtracted',
              '\n', '==========================================')    
        
    def restrict(self, bot = 0, top = 1, left = 0, right = 1):
        """
        %%%%%%%%%%%%%%%
        Crops data file to reduce size.
        %%%%%%%%%%%%%%%
        ---------------
            args: 
                - bot:  from 0..1 (0..100%) cuts first axis, bot < top!
                - top:  from 0..1 (0..100%) cuts first axis, bot < top!
                - left:  from 0..1 (0..100%) cuts first axis, left < right!
                - right:  from 0..1 (0..100%) cuts first axis, left < right!
        ---------------
            output:
                Following methods are now restricted:
                    - self.en, self.ens, self.en_norm
                    - self.ang, self.angs
                    - self.pol, self.pols
                    - self.int, self.eint, self.int_norm, self.eint_norm
        ---------------
        """
        (en_restr, ens_restr, en_norm_restr, ang_restr, angs_restr, 
         pol_restr, pols_restr, int_restr, eint_restr, int_norm_restr, 
         eint_norm_restr) = utils.restrict(self, bot, top, left, right)
        self.en = en_restr
        self.ens = ens_restr
        self.ang = ang_restr
        self.angs = angs_restr
        self.pol = pol_restr
        self.pols = pols_restr
        self.en_norm = en_norm_restr
        self.int = int_restr
        self.eint = eint_restr
        self.int_norm = int_norm_restr
        self.eint_norm = eint_norm_restr
        print('\n ~ Spectra restricted',
              '\n', '==========================================')  
        
    def ang2k(self, angdg, Ekin, lat_unit=False, a=5.33, b=5.33, c=11, 
              V0=0, thdg=0, tidg=0, phidg=0):     
        """
        %%%%%%%%%%%%%%%
        Conversion angle to reciprocal space. 
        %%%%%%%%%%%%%%%
            args: 
                - angd:  detector angle in degrees
                - Ekin:  photon kinetic energy
                - lat_unit:     'False': k-space units in inverse Angstrom
                                'True': k-space units in lattice units
                - a, b, c: lattice units
                - V0: inner potential
                - thdg: manipulator angle theta in degrees
                - tidg: manipulator angle tilt in degrees
                - phi: manipulator angle phi in degreees
        ---------------
            output:
                - self.k: 3-dim k-vector
                - self.k_V0: 3-dim k-vector with V0 != 0
                - self.lat_unit: status in what units k is calculated
                - self.kxs: broadcasted kx-projection
                - self.kys: broadcasted ky-projection
                - self.kx_V0s: broadcasted kx-projection with V0!=0
                - self.ky_V0s: broadcasted ky-projection with V0!=0
        ---------------
        """
        k, k_V0 = utils.ang2k(self, angdg, Ekin, lat_unit, a, b, c, 
                          V0, thdg, tidg, phidg)
        self.k = k
        self.k_V0 = k_V0
        self.lat_unit = lat_unit
        self.kxs = np.transpose(np.broadcast_to(k[0], 
                                               (self.en.size, self.ang.size)))
        self.kx_V0s = np.transpose(np.broadcast_to(k_V0[0], 
                                               (self.en.size, self.ang.size)))
        self.kys = np.transpose(np.broadcast_to(k[1], 
                                               (self.en.size, self.ang.size)))
        self.ky_V0s = np.transpose(np.broadcast_to(k_V0[1], 
                                               (self.en.size, self.ang.size)))  
        print('\n ~ Angles converted into k-space',
              '\n', '==========================================')  
        
    def ang2kFS(self, angdg, Ekin, lat_unit=False, a=5.33, b=5.33, c=11, 
                V0=0, thdg=0, tidg=0, phidg=0):   
        """
        %%%%%%%%%%%%%%%
        Conversion angle to reciprocal space for a Fermi Surface
        %%%%%%%%%%%%%%%
            args: 
                - angd:  detector angle in degrees
                - Ekin:  photon kinetic energy
                - lat_unit:     'False': k-space units in inverse Angstrom
                                'True': k-space units in lattice units
                - a, b, c: lattice units
                - V0: inner potential
                - thdg: manipulator angle theta in degrees
                - tidg: manipulator angle tilt in degrees
                - phi: manipulator angle phi in degreees
        ---------------
            output:
                - self.kx: reciprocal space vector kx
                - self.ky: reciprocal space vector ky
                - self.kx_V0: reciprocal space vector kx with V0!=0
                - self.ky_V0: reciprocal space vector ky with V0!=0
        ---------------
        """
        kx, ky, kx_V0, ky_V0 = utils.ang2kFS(self, angdg, Ekin, lat_unit, 
                                             a, b, c, V0, thdg, tidg, phidg)
        self.kx = kx
        self.ky = ky
        self.kx_V0 = kx_V0
        self.ky_V0 = ky_V0
        self.lat_unit = lat_unit
        print('\n ~ Angles converted into k-space for Fermi surface',
              '\n', '==========================================')  
        
    def FS(self, e=0, ew=0.05, norm=False): 
        """
        %%%%%%%%%%%%%%%
        Extract Fermi Surface map
        %%%%%%%%%%%%%%%
            args: 
                - e: energy at wich the map is extracted
                - ew: integration window from e downwards in energy
                - norm:     'False': energy and intensity not normalized
                            'True': energy and intensity are normalized
        ---------------
            output:
                - self.map: Fermi surface map
        ---------------
        """
        FSmap = utils.FS(self, e, ew, norm)
        self.map = FSmap
        print('\n ~ Constant energy map extracted',
              '\n', '==========================================')  
        
    def plt_spec(self, norm=False, v_max=1):
        """
        %%%%%%%%%%%%%%%
        Plot ARPES spectrum
        %%%%%%%%%%%%%%%
            args:
                - norm:     'False': energy and intensity not normalized
                            'True': energy and intensity are normalized
                - v_max: 0..1, vmax in contourf (contrast)
        ---------------
            output:
                - self.plt_spec: plots ARPES spectrum
        ---------------
        """
        utils_plt.plt_spec(self, norm, v_max)
    
    def plt_FS_polcut(self, norm=False, p=0, pw=0.1):
        """
        %%%%%%%%%%%%%%%
        Spectral cut through Fermi surface
        %%%%%%%%%%%%%%%
            args:
                - norm:     'False': energy and intensity not normalized
                            'True': energy and intensity are normalized
                - p: polar angle self.pol to cut through
                - pw: integration window from p downwards in polar angle
        ---------------
            output:
                - self.plt_FS_polcut: plots ARPES spectrum through FS
        ---------------
        """
        utils_plt.plt_FS_polcut(self, norm, p, pw)
        
    def plt_FS(self, coord=False, v_max=1):
        """
        %%%%%%%%%%%%%%%
        Plot Fermi surface
        %%%%%%%%%%%%%%%
            args:
                - coord:    'False': x and y axis in angles
                            'True': x and y axis in k-space
                - v_max: 0..1, vmax in contourf (contrast)
        ---------------
            output:
                - self.plt_FS: plots Fermi surface
        ---------------
        """
        utils_plt.plt_FS(self, coord, v_max=1)
    
    def plt_FS_all(self, coord=False, norm=False):
        """
        %%%%%%%%%%%%%%%
        Plot constant energy maps from top energy to bottom energy 
        in 0.1 eV binding energy steps.
        %%%%%%%%%%%%%%%
            args:
                - coord:    'False': x and y axis in angles
                            'True': x and y axis in k-space
                - norm:     'False': energy and intensity not normalized
                            'True': energy and intensity are normalized
        ---------------
            output:
                - self.plt_FS_all: plots all maps
        ---------------
        """
        utils_plt.plt_FS_all(self, coord, norm)
        
    def plt_hv(self, v_max=1):
        """
        %%%%%%%%%%%%%%%
        Plot all spectra of hv-scan
        %%%%%%%%%%%%%%%
            args:
                - v_max: 0..1, vmax in contourf (contrast)
        ---------------
            output:
                - self.plt_hv: plots all hv spectra
        ---------------
        """
        utils_plt.plt_hv(self, v_max)


class DLS(Analysis):
    """
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Data from Diamond Light Source %%
    %% Beamline: i05                  %%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
        methods: self.*
        
            *file: file number that is analysed
            *mat: material under study
            *year: year of experiment
            *sample: sample number
            *folder: path of sample number
            *filename: full file name
            *path: full path 
            
            *ang: detector angles
            *angs: broadcasted detector angle
            *en: energy
            *ens: broadcasted energy
            *int: intensity
            *eint: errors on intensity
            *hv: photon energy
            *pol: manipulator polar angles (if available)
            *pols: broadcasted polar angles (if available)
            
            convention of broadcasting dimensions: *.size
            3-dimensional (pol*, ang*, en*)
            3-dimensional (hv*, ang*, en*)
            2-dimensional (ang*, en*)
    ---------------
    """  
    def __init__(self, file, mat, year, sample):  #Load Data file
        folder = ''.join(['/Users/denyssutter/Documents/Denys/',str(mat),
                          '/Diamond',str(year),'/',str(sample),'/'])
        filename = ''.join(['i05-',str(file),'.nxs'])
        path = folder + filename
        print('\n ~ Initializing Diamond Light Source data file: \n {}'.format(self.path), 
              '\n', '==========================================')
        f = h5py.File(path,'r') #Read file with h5py reader
        intensity_list  = list(f['/entry1/analyser/data']) 
        ang = np.array(list(f['/entry1/analyser/angles']))
        en = np.array(list(f['/entry1/analyser/energies']))
        photon = np.array(list(f['/entry1/instrument/monochromator/energy']))
        try: #Try if polar angles available
            pol = np.array(list(f['/entry1/analyser/sapolar']))
            self.pol = pol
            intensity = np.array(intensity_list)[:, :, :]
            ###generate methods###
            self.ens   = np.broadcast_to(en, (pol.size, ang.size, en.size))
            self.angs  = np.transpose(
                            np.broadcast_to(ang, (pol.size, en.size, ang.size)),
                                (0, 2, 1))
            self.pols  = np.transpose(
                            np.broadcast_to(pol, (ang.size, en.size, pol.size)),
                                (2, 0, 1))
        except KeyError:
            print('- No polar angles available \n')
            intensity = np.asarray(intensity_list)[0, :, :]
            ###generate methods###
            self.ens = np.broadcast_to(en, (ang.size, en.size))
            self.angs = np.transpose(np.broadcast_to(ang, (en.size, ang.size)))
        ###generate methods###
        self.file = file
        self.mat = mat
        self.path = path
        self.filename = filename
        self.folder = folder
        self.ang = ang
        self.en = en
        self.int = intensity
        self.eint = np.sqrt(intensity)
        self.hv = photon
        print('\n ~ Initialization complete. Data has {} dimensions'.format(
                len(np.shape(self.int))),
              '\n', '==========================================')  
        super(DLS, self)
    
class ALS(Analysis):  
    """
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Data from Advanced Light Source %%
    %% Beamline: Maestro               %%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
        methods: self.*
        
            *file: file number that is analysed
            *mat: material under study
            *year: year of experiment
            *sample: sample number
            *folder: path of sample number
            *filename: full file name
            *path: full path 
            
            *ang: detector angles
            *angs: broadcasted detector angle
            *en: energy
            *ens: broadcasted energy
            *int: intensity
            *eint: errors on intensity
            *hv: photon energy
            *pol: manipulator polar angles (if available)
            *pols: broadcasted polar angles (if available)
            
            convention of broadcasting dimensions: *.size
            3-dimensional (pol*, ang*, en*)
            3-dimensional (hv*, ang*, en*)
            2-dimensional (ang*, en*)
    ---------------
    """    
    def __init__(self, file, mat, year, sample):  #Load Data file
        folder = ''.join(['/Users/denyssutter/Documents/Denys/',str(mat),
                          '/ALS',str(year),'/',str(sample),'/'])
        filename = ''.join([str(year),file,'.fits'])
        path = folder + filename
        print('\n ~ Initializing Advanced Light Source data file: \n {}'.format(self.path), 
              '\n', '==========================================')
        ###Read meta data###
        f = fits.open(path)
        hdr = f[0].header #load header
        meta_data = f[1].data
        mode = hdr['NM_0_0']  #scan mode
        px_per_en = hdr['SSPEV_0'] #pixels per energy
        e_i = hdr['SSX0_0'] #initial energy
        e_f = hdr['SSX1_0'] #final energy
        a_i = hdr['SSY0_0'] #initial angle
        a_f = hdr['SSY1_0'] #final angle
        Ef = hdr['SSKE0_0'] #Fermi energy (not sure it is correct every time)
        ###Some parameters###
        ang_per_px = 0.193 #Angled per pixel
        binning = 4 #I think this is the correct binning
        npol = meta_data.size #number of polar cuts
        (nen, nang) = meta_data[0][-1].shape #number of energy, angle steps
        data = np.zeros((npol, nang, nen)) #Placeholder
        en = (np.arange(e_i, e_f, 1.) - Ef) / px_per_en
        ang = np.arange(a_i, a_f, 1.) * ang_per_px / binning
        pol = np.arange(0, npol, 1.)
        ###Build up data###
        if mode == 'Beta':
            for i in range(npol):
                pol[i] = meta_data[i][1]
                data[i, :, :] = np.transpose(meta_data[i][-1])
                self.ens = np.broadcast_to(en, (pol.size, ang.size, en.size))
                self.angs  = np.transpose(np.broadcast_to(
                                ang, (pol.size, en.size, ang.size)), (0, 2, 1))
                self.pols  = np.transpose(np.broadcast_to(
                                pol, (ang.size, en.size, pol.size)), (2, 0, 1))
        ###generate methods###
        self.file = file
        self.mat = mat
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
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Data from Swiss Light Source %%
    %% Beamline: SIS                %%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
        methods: self.*
        
            *file: file number that is analysed
            *mat: material under study
            *year: year of experiment
            *sample: sample number
            *folder: path of sample number
            *filename: full file name
            *path: full path 
            
            *ang: detector angles
            *angs: broadcasted detector angle
            *en: energy
            *ens: broadcasted energy
            *int: intensity
            *eint: errors on intensity
            *hv: photon energy
            *pol: manipulator polar angles (if available)
            *pols: broadcasted polar angles (if available)
            
            convention of broadcasting dimensions: *.size
            3-dimensional (pol*, ang*, en*)
            3-dimensional (hv*, ang*, en*)
            2-dimensional (ang*, en*)
    ---------------
    """    
    def __init__(self, file, mat, year, sample):  #Load Data file
        folder = ''.join(['/Users/denyssutter/Documents/Denys/',str(mat),
                          '/SIS',str(year),'/',str(sample),'/'])
        filename = ''.join([str(file),'.h5'])
        path = folder + filename
        print('\n ~ Initializing SIS data file: \n {}'.format(self.path), 
              '\n', '==========================================')
        f = h5py.File(path,'r')
        data  = np.array((f['Electron Analyzer/Image Data'])) 
        if data.ndim == 3:
            self.int = np.transpose(data, (2, 1, 0))
            d1, d2, d3 = data.shape
        elif data.ndim == 2:
            self.int = np.transpose(data)
            d1, d2 = data.shape
        e_i, de = data.attrs['Axis0.Scale'] #initial energy, energy step
        a_i, da = data.attrs['Axis1.Scale'] #initial angle, angle step
        en = np.arange(e_i, e_i + d1 * de, de)
        ang = np.arange(a_i, a_i + d2 * da, da)
        hv = np.array(f['Other Instruments/hv'])
        pol  = np.array(f['Other Instruments/Tilt'])
        ###Cases of scan modes###
        if pol.size > 1 & hv.size == 1:
            print('- Fermi surface map \n')
            self.ens   = np.broadcast_to(en, (pol.size, ang.size, en.size))
            self.angs  = np.transpose(
                            np.broadcast_to(ang, (pol.size, en.size, ang.size)),
                                (0, 2, 1))
            self.pols  = np.transpose(
                            np.broadcast_to(pol, (ang.size, en.size, pol.size)),
                                (2, 0, 1))
        elif hv.size > 1 & pol.size == 1:
            print('- Photon energy scan \n')
            self.ens   = np.broadcast_to(en, (hv.size, ang.size, en.size))
            self.angs  = np.transpose(
                            np.broadcast_to(ang, (hv.size, en.size, ang.size)),
                                (0, 2, 1))
            self.hvs  = np.transpose(
                            np.broadcast_to(hv, (ang.size, en.size, hv.size)),
                                (2, 0, 1))
        elif pol.size == 1 & hv.size == 1:
            print('- No polar angles available \n- No photon energy scan \n')
            self.ens = np.broadcast_to(en, (ang.size, en.size))
            self.angs = np.transpose(np.broadcast_to(ang, (en.size, ang.size)))
        ###generate methods###
        self.file = file
        self.mat = mat
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
    %%%%%%%%%%%%%%%%%%%%%
    %% Data from Bessy %%
    %% Beamline: 1^3   %%
    %%%%%%%%%%%%%%%%%%%%%
    
        methods: self.*
        
            *file: file number that is analysed
            *mat: material under study
            *year: year of experiment
            *sample: sample number
            *folder: path of sample number
            *filename: full file name
            *path: full path 
            
            *ang: detector angles
            *angs: broadcasted detector angle
            *en: energy
            *ens: broadcasted energy
            *int: intensity
            *eint: errors on intensity
            *hv: photon energy
            *pol: manipulator polar angles (if available)
            *pols: broadcasted polar angles (if available)
            
            convention of broadcasting dimensions: *.size
            3-dimensional (pol*, ang*, en*)
            3-dimensional (hv*, ang*, en*)
            2-dimensional (ang*, en*)
    ---------------
    """     
    def __init__(self, file, mat, year, sample):  #Load Data file
        self.file = file
        self.mat = mat
        folder = ''.join(['/Users/denyssutter/Documents/Denys/',str(mat),
                          '/Bessy',str(year),'/',str(sample),'/'])
        
        path_ang = ''.join(['Ca_',str(file),'ang','.dat'])
        path_pol = ''.join(['Ca_',str(file),'pol','.dat'])
        path_en = ''.join(['Ca_',str(file),'en','.dat'])
        ang = np.loadtxt(folder + path_ang)
        en = np.loadtxt(folder + path_en)
        if file == 8:
            pol = np.loadtxt(folder + path_pol)
            self.pol = pol
        self.folder = folder
        print('\n ~ Initializing Bessy data file: \n {}'.format(folder+str(file)), 
              '\n', '==========================================')
        self.ang = ang
        self.en = en
        if file == 8:
            intensity = np.zeros((len(pol), len(ang), len(en)))
            for k in range(self.pol.size):
                path_int = ''.join(['Ca_',str(file),'int',str(k + 1),'.dat'])
                intensity[k, :, :] = np.transpose(np.loadtxt(folder + path_int))
            print('- Fermi surface map \n')
            self.ens   = np.broadcast_to(en, (pol.size, ang.size, en.size))
            self.angs  = np.transpose(
                            np.broadcast_to(ang, (pol.size, en.size, ang.size)),
                                (0, 2, 1))
            self.pols  = np.transpose(
                            np.broadcast_to(pol, (ang.size, en.size, pol.size)),
                                (2, 0, 1))
            self.int = intensity
            self.eint = np.sqrt(intensity)
        else:
            path_int = ''.join(['Ca_',str(file),'int','.dat'])
            intensity = np.transpose(np.loadtxt(folder + path_int))
            print('- No polar angles available \n')
            self.ens = np.broadcast_to(en, (ang.size, en.size))
            self.angs = np.transpose(np.broadcast_to(ang, (en.size, ang.size)))
            self.int = intensity
            self.eint = np.sqrt(intensity)
        self.filename = path_int
        print('\n ~ Initialization complete. Data has {} dimensions'.format(
                len(np.shape(self.int))),
              '\n', '==========================================')  
        super(Bessy, self)
        
class CASS(Analysis):
    """
    Data from Soleil
    Beamline: Cassiopee, modes = {'FSM', 'hv', 'cut_txt', 'cut_ibw'}
    """    
    def __init__(self, file, mat, year, mode):  #Load Data file
        self.file = file
        self.mat = mat
        self.mode = mode
        folder = ''.join(['/Users/denyssutter/Documents/Denys/',str(mat),
                          '/CASS',str(year),'/'])
        if mode == 'cut_txt':
            filename = ''.join([str(file),'.txt'])
            path = folder + filename
            with open(path, 'r') as f :
                for i,line in enumerate(f.readlines()) :
                    if line.startswith('Dimension 1 scale=') :
                        en = line.split('=')[-1].split()
                        en = np.array(en, dtype=float)
                    elif line.startswith('Dimension 2 scale=') :
                        ang = line.split('=')[-1].split()
                        ang = np.array(ang, dtype=float)
                    elif line.startswith('Excitation Energy') :
                        hv = float(line.split('=')[-1])
                    elif 'Data' in line:
                        break
            intensity = np.loadtxt(path, skiprows=i+1)[:,1:]
            intensity = np.transpose(intensity)
            self.ens = np.broadcast_to(en, (ang.size, en.size))
            self.angs = np.transpose(np.broadcast_to(ang, (en.size, ang.size)))
        elif mode == 'cut_ibw':
            filename = ''.join([str(file),'.ibw'])
            path = folder + filename
            wave = binarywave.load(path)['wave']
            intensity = np.array([wave['wData']])[0]
            intensity = np.transpose(intensity)
            # The `header` contains some metadata
            header = wave['wave_header']
            nDim = header['nDim']
            steps = header['sfA']
            starts = header['sfB']
            en = np.linspace(starts[0], starts[0] + nDim[0] * steps[0], nDim[0])
            ang = np.linspace(starts[1], starts[1] + nDim[1] * steps[1], nDim[1])
            # Convert `note`, which is a bytestring of ASCII characters that 
            # contains some metadata, to a list of strings
            note = wave['note']
            note = note.decode('ASCII').split('\r')
            # Now the extraction fun begins. Most lines are of the form 
            # `Some-kind-of-name=some-value`
            metadata = dict()
            for line in note :
                # Split at '='. If it fails, we are not in a line that contains 
                # useful information
                try :
                    name, val = line.split('=')
                except ValueError :
                    continue
                metadata.update({name: val})
            hv = metadata['Excitation Energy']
            self.en = en
            self.ang = ang
            self.ens = np.broadcast_to(en, (ang.size, en.size))
            self.angs = np.transpose(np.broadcast_to(ang, (en.size, ang.size)))
        if mode == 'FSM':
            sub_folder = ''.join([folder,str(file),'/'])
            n_scans = (len([name for name in os.listdir(sub_folder)\
                               if os.path.isfile(os.path.join(sub_folder, name))])) / 2
            n_scans = np.int(n_scans)
            pol = np.zeros(n_scans)
            for scan in range(n_scans):
                filename = ''.join(['FS_', str(scan + 1),'_ROI1_.txt'])
                info = ''.join(['FS_', str(scan + 1),'_i.txt'])
                path_info = sub_folder + info
                path = sub_folder + filename
                self.path = path
                print('\n ~ Initializing Cassiopee data file: \n {}'.format(self.path), 
                      '\n', '==========================================')
                with open(path_info) as f: #read theta
                    for line in f.readlines():
                        if 'theta' in line:
                            theta = line.strip('theta (deg):')
                            pol[scan] = np.float32(theta.split())
                data = np.genfromtxt(path, skip_header = 44, delimiter='\t')
                if scan == 0:
                    en = data[:, 0]
                    with open(path) as f: #read angle
                        for line in f.readlines():
                            if 'Dimension 2 scale' in line:
                                ang_raw = line.strip('Dimension 2 scale=')
                                ang_raw = ang_raw.split()
                                ang = np.array(ang_raw, dtype=np.float32)  
                    intensity = np.zeros((n_scans, len(ang), len(en)))
                intensity[scan] = np.transpose(data[:, 1:])
            self.pol = pol
            self.ens   = np.broadcast_to(en, (pol.size, ang.size, en.size))
            self.angs  = np.transpose(
                            np.broadcast_to(ang, (pol.size, en.size, ang.size)),
                                (0, 2, 1))
            self.pols  = np.transpose(
                            np.broadcast_to(pol, (ang.size, en.size, pol.size)),
                                (2, 0, 1))
        elif mode == 'hv':
            sub_folder = ''.join([folder,str(file),'/'])
            n_scans = (len([name for name in os.listdir(sub_folder)\
                               if os.path.isfile(os.path.join(sub_folder, name))])) / 2
            n_scans = np.int(n_scans)
            hv = np.zeros(n_scans)
            for scan in range(n_scans):
                filename = ''.join(['hv_', str(scan + 1),'_ROI1_.txt'])
                info = ''.join(['hv_', str(scan + 1),'_i.txt'])
                path_info = sub_folder + info
                path = sub_folder + filename
                self.path = path
                print('\n ~ Initializing Cassiopee data file: \n {}'.format(self.path), 
                      '\n', '==========================================')
                with open(path_info) as f:
                    for line in f.readlines():
                        if 'hv' in line:
                            hv_raw = line.strip(('hv (eV):'))
                            try:
                                hv[scan] = np.float32(hv_raw.split())
                            except ValueError:   
                                 print('')
                data = np.genfromtxt(path, skip_header = 44, delimiter='\t')
                if scan == 0:
                    en = data[:, 0]
                    with open(path) as f: #read angle
                        for line in f.readlines():
                            if 'Dimension 2 scale' in line:
                                ang_raw = line.strip('Dimension 2 scale=')
                                ang_raw = ang_raw.split()
                                ang = np.array(ang_raw, dtype=np.float32)  
                    intensity = np.zeros((n_scans, len(ang), len(en)))
                intensity[scan] = np.transpose(data[:, 1:])
            self.hv = hv
            self.ens   = np.broadcast_to(en, (hv.size, ang.size, en.size))
            self.angs  = np.transpose(
                            np.broadcast_to(ang, (hv.size, en.size, ang.size)),
                                (0, 2, 1))
            self.hvs  = np.transpose(
                            np.broadcast_to(hv, (ang.size, en.size, hv.size)),
                                (2, 0, 1))
        self.filename = filename
        self.folder = folder
        self.en = en
        self.ang = ang
        self.int = intensity
        self.eint = np.sqrt(intensity)        
        print('\n ~ Initialization complete. Data has {} dimensions'.format(
                len(np.shape(self.int))),
              '\n', '==========================================')  
                
        super(CASS, self)