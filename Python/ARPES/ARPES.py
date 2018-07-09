#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 10 13:20:31 2018

@author: denyssutter
"""
import os
os.chdir('/Users/denyssutter/Documents/library/Python/ARPES')
import h5py
import numpy as np
import utils as u
import utils_plt as uplt
from astropy.io import fits

class DLS:
    """
    Data from Diamond Light Source
    Beamline: i05
    """    
    def __init__(self, file, mat, year, sample):  #Load Data file
        self.file = file
        self.mat = mat
        folder = ''.join(['/Users/denyssutter/Documents/Denys/',str(mat),
                          '/Diamond',str(year),'/',str(sample),'/'])
        filename = ''.join(['i05-',str(file),'.nxs'])
        path = folder + filename
        self.path = path
        self.folder = folder
        print('\n ~ Initializing Diamond Light Source data file: \n {}'.format(self.path), 
              '\n', '==========================================')
        f = h5py.File(path,'r')
        intensity  = list(f['/entry1/analyser/data'])
        ang = np.array(list(f['/entry1/analyser/angles']))
        self.ang = ang
        en = np.array(list(f['/entry1/analyser/energies']))
        self.en = en
        try: 
            pol = np.array(list(f['/entry1/analyser/sapolar']))
            self.pol = pol
            self.int  = np.array(intensity)[:,:,:]
            self.ens   = np.broadcast_to(en, (pol.size, ang.size, en.size))
            self.angs  = np.transpose(
                            np.broadcast_to(ang, (pol.size, en.size, ang.size)),
                                (0, 2, 1))
            self.pols  = np.transpose(
                            np.broadcast_to(pol, (ang.size, en.size, pol.size)),
                                (2, 0, 1))
        except KeyError:
            print('- No polar angles available \n')
            self.int = np.asarray(intensity)[0, :, :]
            self.ens = np.broadcast_to(en, (ang.size, en.size))
            self.angs = np.transpose(np.broadcast_to(ang, (en.size, ang.size)))
        photon = list(f['/entry1/instrument/monochromator/energy'])
        self.hv = np.array(photon)
        print('\n ~ Initialization complete. Data has {} dimensions'.format(
                len(np.shape(self.int))),
              '\n', '==========================================')  
                
    def norm(self, gold):  #Normalize Data file with gold
        en_norm, int_norm = u.norm(self, gold)
        self.en_norm = en_norm
        self.int_norm = int_norm
        print('\n ~ Data normalized',
              '\n', '==========================================')   
        
    def shift(self, gold): #Flatten spectra
        en_shift, int_shift = u.shift(self, gold)
        self.en_norm = en_shift
        self.int_shift = int_shift
        print('\n ~ Only energy normalized',
              '\n', '==========================================')   
        
    def flatten(self, norm=False): #Flatten spectra
        int_flat = u.flatten(self, norm)
        self.int_flat = int_flat
        print('\n ~ Spectra flattened',
              '\n', '==========================================')   
    
    def FS_flatten(self, ang=True): #Flatten FS
        map_flat = u.FS_flatten(self, ang)
        self.map_flat = map_flat
        print('\n ~ FS flattened',
              '\n', '==========================================') 
        
    def restrict(self, bot = 0, top = 1, left = 0, right = 1): #restrict spectrum
        (en_restr, ens_restr, en_norm_restr, ang_restr, angs_restr, pol_restr, 
         pols_restr, int_restr, int_norm_restr) = u.restrict(
                 self, bot, top, left, right)
        self.en = en_restr
        self.ens = ens_restr
        self.ang = ang_restr
        self.angs = angs_restr
        self.pol = pol_restr
        self.pols = pols_restr
        self.en_norm = en_norm_restr
        self.int = int_restr
        self.int_norm = int_norm_restr
        print('\n ~ Spectra restricted',
              '\n', '==========================================')  
        
    def ang2k(self, angdg, Ekin, lat_unit=False, a=5.33, b=5.33, c=11, 
              V0=0, thdg=0, tidg=0, phidg=0):      
        k, k_V0 = u.ang2k(self, angdg, Ekin, lat_unit, a, b, c, 
                          V0, thdg, tidg, phidg)
        self.k = k
        self.k_V0 = k_V0
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
        kx, ky, kx_V0, ky_V0 = u.ang2kFS(self, angdg, Ekin, lat_unit, a, b, c, 
                                         V0, thdg, tidg, phidg)
        self.kx = kx
        self.ky = ky
        self.kx_V0 = kx_V0
        self.ky_V0 = ky_V0
        print('\n ~ Angles converted into k-space for Fermi surface',
              '\n', '==========================================')  
        
    def FS(self, e=0, ew=0, norm=False): #Extract Constant Energy Map
        FSmap = u.FS(self, e, ew, norm)
        self.map = FSmap
        print('\n ~ Constant energy map extracted',
              '\n', '==========================================')  
        
    def plt_spec(self, norm=False):
        uplt.plt_spec(self, norm)
    
    def plt_FS_polcut(self, norm=False, p=0, pw=0):
        uplt.plt_FS_polcut(self, norm, p, pw)
        
    def plt_FS(self, coord=False):
        uplt.plt_FS(self, coord)

    def plt_hv(self, a=0, aw=0):
        uplt.plt_hv(self, a, aw)
        
class ALS:  
    """
    Data from Advanced Light Source
    Beamline: Maestro
    """    
    def __init__(self, file, mat, year, sample):  #Load Data file
        self.file = file
        self.mat = mat
        folder = ''.join(['/Users/denyssutter/Documents/Denys/',str(mat),
                          '/ALS',str(year),'/',str(sample),'/'])
        filename = ''.join([str(year),file,'.fits'])
        path = folder + filename
        self.path = path
        self.folder = folder
        print('\n ~ Initializing Advanced Light Source data file: \n {}'.format(self.path), 
              '\n', '==========================================')
        f = fits.open(path)
        hdr = f[0].header
        data = f[1].data
        mode = hdr['NM_0_0']
        px_per_en = hdr['SSPEV_0']
        e_i = hdr['SSX0_0']
        e_f = hdr['SSX1_0']
        a_i = hdr['SSY0_0']
        a_f = hdr['SSY1_0']
        Ef = hdr['SSKE0_0']
        ang_per_px = 0.193
        binning = 4
        ###Creating Placeholders
        npol = data.size
        (nen, nang) = data[0][-1].shape
        intensity = np.zeros((npol, nang, nen))
        en = (np.arange(e_i, e_f, 1.) - Ef) / px_per_en
        ang = np.arange(a_i, a_f, 1.) * ang_per_px / binning
        pol = np.arange(0, npol, 1.)
        ###Build up data
        if mode == 'Beta':
            for i in range(npol):
                pol[i] = data[i][1]
                intensity[i, :, :] = np.transpose(data[i][-1])
                self.ens = np.broadcast_to(en, (pol.size, ang.size, en.size))
                self.angs  = np.transpose(np.broadcast_to(
                                ang, (pol.size, en.size, ang.size)), (0, 2, 1))
                self.pols  = np.transpose(np.broadcast_to(
                                pol, (ang.size, en.size, pol.size)), (2, 0, 1))
            self.int = intensity
        self.pol = pol
        self.ang = ang
        self.en = en
        self.hv = data[0][-4]
        print('\n ~ Initialization complete. Data has {} dimensions'.format(
                len(np.shape(self.int))),
              '\n', '==========================================')  
                
    def norm(self, gold):  #Normalize Data file with gold
        en_norm, int_norm = u.norm(self, gold)
        self.en_norm = en_norm
        self.int_norm = int_norm
        print('\n ~ Data normalized',
              '\n', '==========================================')   
        
    def shift(self, gold): #Flatten spectra
        en_shift, int_shift = u.shift(self, gold)
        self.en_norm = en_shift
        self.int_norm = int_shift
        print('\n ~ Only energy normalized',
              '\n', '==========================================')   
        
    def flatten(self, norm=False): #Flatten spectra
        int_flat = u.flatten(self, norm)
        self.int_flat = int_flat
        print('\n ~ Spectra flattened',
              '\n', '==========================================')   
    
    def FS_flatten(self, ang=True): #Flatten FS
        map_flat = u.FS_flatten(self, ang)
        self.map_flat = map_flat
        print('\n ~ FS flattened',
              '\n', '==========================================') 
        
    def restrict(self, bot = 0, top = 1, left = 0, right = 1): #restrict spectrum
        (en_restr, ens_restr, en_norm_restr, ang_restr, angs_restr, pol_restr, 
         pols_restr, int_restr, int_norm_restr) = u.restrict(
                 self, bot, top, left, right)
        self.en = en_restr
        self.ens = ens_restr
        self.ang = ang_restr
        self.angs = angs_restr
        self.pol = pol_restr
        self.pols = pols_restr
        self.en_norm = en_norm_restr
        self.int = int_restr
        self.int_norm = int_norm_restr
        print('\n ~ Spectra restricted',
              '\n', '==========================================')   
        
    def ang2k(self, angdg, Ekin, lat_unit=False, a=5.33, b=5.33, c=11, 
              V0=0, thdg=0, tidg=0, phidg=0):      
        k, k_V0 = u.ang2k(self, angdg, Ekin, lat_unit, a, b, c, 
                          V0, thdg, tidg, phidg)
        self.k = k
        self.k_V0 = k_V0
        self.ks = np.transpose(np.broadcast_to(k[0], 
                                               (self.en.size, self.ang.size)))
        self.k_V0s = np.transpose(np.broadcast_to(k_V0[0], 
                                               (self.en.size, self.ang.size)))
            
        print('\n ~ Angles converted into k-space',
              '\n', '==========================================')  
        
    def ang2kFS(self, angdg, Ekin, lat_unit=False, a=5.33, b=5.33, c=11, 
                V0=0, thdg=0, tidg=0, phidg=0):   
        kx, ky, kx_V0, ky_V0 = u.ang2kFS(self, angdg, Ekin, lat_unit, a, b, c, 
                                         V0, thdg, tidg, phidg)
        self.kx = kx
        self.ky = ky
        self.kx_V0 = kx_V0
        self.ky_V0 = ky_V0
        print('\n ~ Angles converted into k-space for Fermi surface',
              '\n', '==========================================')  
        
    def FS(self, e=0, ew=0, norm=False): #Extract Constant Energy Map
        FSmap = u.FS(self, e, ew, norm)
        self.map = FSmap
        print('\n ~ Constant energy map extracted',
              '\n', '==========================================')  
        
    def plt_spec(self, norm=False):
        uplt.plt_spec(self, norm)
        
    def plt_FS_polcut(self, norm=False, p=0, pw=0):
        uplt.plt_FS_polcut(self, norm, p, pw)
        
    def plt_FS(self, coord=False):
        uplt.plt_FS(self, coord)
        
    def plt_hv(self, a=0, aw=0):
        uplt.plt_hv(self, a, aw)
        
class SIS:  
    """
    Data from Swiss light source
    Beamline: SIS
    """    
    def __init__(self, file, mat, year, sample):  #Load Data file
        self.file = file
        self.mat = mat
        folder = ''.join(['/Users/denyssutter/Documents/Denys/',str(mat),
                          '/SIS',str(year),'/',str(sample),'/'])
        filename = ''.join([str(file),'.h5'])
        path = folder + filename
        self.path = path
        self.folder = folder
        print('\n ~ Initializing SIS data file: \n {}'.format(self.path), 
              '\n', '==========================================')
        f = h5py.File(path,'r')
        data  = (f['Electron Analyzer/Image Data'])
        intensity = np.array(data)
        self.int = intensity
        if data.ndim == 3:
            self.int = np.transpose(intensity, (2, 1, 0))
            d1, d2, d3 = data.shape
        elif data.ndim == 2:
            self.int = np.transpose(intensity)
            d1, d2 = data.shape
        e_i, de = data.attrs['Axis0.Scale']
        a_i, da = data.attrs['Axis1.Scale']
        en = np.arange(e_i, e_i + d1 * de, de)
        ang = np.arange(a_i, a_i + d2 * da, da)
        self.ang = ang
        self.en = en
        hv = np.array(f['Other Instruments/hv'])
        pol  = np.array(f['Other Instruments/Tilt'])
        self.hv = hv
        self.pol = pol
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
        print('\n ~ Initialization complete. Data has {} dimensions'.format(
                len(np.shape(self.int))),
              '\n', '==========================================')  
                
    def norm(self, gold):  #Normalize Data file with gold
        en_norm, int_norm = u.norm(self, gold)
        self.en_norm = en_norm
        self.int_norm = int_norm
        print('\n ~ Data normalized',
              '\n', '==========================================')   
        
    def shift(self, gold): #Flatten spectra
        en_shift, int_shift = u.shift(self, gold)
        self.en_norm = en_shift
        self.int_norm = int_shift
        print('\n ~ Only energy normalized',
              '\n', '==========================================')   
        
    def flatten(self, norm=False): #Flatten spectra
        int_flat = u.flatten(self, norm)
        self.int_flat = int_flat
        print('\n ~ Spectra flattened',
              '\n', '==========================================')
        
    def FS_flatten(self, ang=True): #Flatten FS
        map_flat = u.FS_flatten(self, ang)
        self.map_flat = map_flat
        print('\n ~ FS flattened',
              '\n', '==========================================')   
        
    def restrict(self, bot=0, top=1, left=0, right=1): #restrict spectrum
        (en_restr, ens_restr, en_norm_restr, ang_restr, angs_restr, pol_restr, 
         pols_restr, int_restr, int_norm_restr) = u.restrict(
                 self, bot, top, left, right)
        self.en = en_restr
        self.ens = ens_restr
        self.ang = ang_restr
        self.angs = angs_restr
        self.pol = pol_restr
        self.pols = pols_restr
        self.en_norm = en_norm_restr
        self.int = int_restr
        self.int_norm = int_norm_restr
        print('\n ~ Spectra restricted',
              '\n', '==========================================')  
 
    def FS_restrict(self, bot=0, top=1, left=0, right=1): #restrict FS
        (ang_restr, angs_restr, pol_restr, pols_restr, map_restr) = u.FS_restrict(
                 self, bot, top, left, right)
        self.ang = ang_restr
        self.angs = angs_restr
        self.pol = pol_restr
        self.pols = pols_restr
        self.map = map_restr
        print('\n ~ FS restricted',
              '\n', '==========================================')  
        
    def ang2k(self, angdg, Ekin, lat_unit=False, a=5.33, b=5.33, c=11, 
              V0=0, thdg=0, tidg=0, phidg=0):      
        k, k_V0 = u.ang2k(self, angdg, Ekin, lat_unit, a, b, c, 
                          V0, thdg, tidg, phidg)
        self.k = k
        self.k_V0 = k_V0
        self.ks = np.transpose(np.broadcast_to(k[0], 
                                               (self.en.size, self.ang.size)))
        self.k_V0s = np.transpose(np.broadcast_to(k_V0[0], 
                                               (self.en.size, self.ang.size)))
            
        print('\n ~ Angles converted into k-space',
              '\n', '==========================================')  
        
    def ang2kFS(self, angdg, Ekin, lat_unit=False, a=5.33, b=5.33, c=11, 
                V0=0, thdg=0, tidg=0, phidg=0):   
        kx, ky, kx_V0, ky_V0 = u.ang2kFS(self, angdg, Ekin, lat_unit, a, b, c, 
                                         V0, thdg, tidg, phidg)
        self.kx = kx
        self.ky = ky
        self.kx_V0 = kx_V0
        self.ky_V0 = ky_V0
        print('\n ~ Angles converted into k-space for Fermi surface',
              '\n', '==========================================')  
        
    def FS(self, e=0, ew=0, norm=False): #Extract Constant Energy Map
        FSmap = u.FS(self, e, ew, norm)
        self.map = FSmap
        print('\n ~ Constant energy map extracted',
              '\n', '==========================================')  
        
    def plt_spec(self, norm=False):
        uplt.plt_spec(self, norm)
    
    def plt_FS_polcut(self, norm=False, p=0, pw=0):
        uplt.plt_FS_polcut(self, norm, p, pw)
        
    def plt_FS(self, coord=False):
        uplt.plt_FS(self, coord)
        
    def plt_hv(self, a=0, aw=0):
        uplt.plt_hv(self, a, aw)
        
class Bessy:  
    """
    Data from Bessy
    Beamline: 1^3
    """    
    def __init__(self, file, mat, year, sample):  #Load Data file
        self.file = file
        self.mat = mat
        folder = ''.join(['/Users/denyssutter/Documents/Denys/',str(mat),
                          '/Bessy',str(year),'/',str(sample),'/'])
        path_int = ''.join(['Ca_',str(file),'int','.dat'])
        path_ang = ''.join(['Ca_',str(file),'ang','.dat'])
        path_en = ''.join(['Ca_',str(file),'en','.dat'])
        intensity = np.loadtxt(folder + path_int)
        ang = np.loadtxt(folder + path_ang)
        en = np.loadtxt(folder + path_en)
        self.folder = folder
        print('\n ~ Initializing Bessy data file: \n {}'.format(folder+path_int), 
              '\n', '==========================================')
        self.int = np.transpose(intensity)
        self.ang = ang
        self.en = en
        self.ens = np.broadcast_to(en, (ang.size, en.size))
        self.angs = np.transpose(np.broadcast_to(ang, (en.size, ang.size)))
        print('\n ~ Initialization complete. Data has {} dimensions'.format(
                len(np.shape(self.int))),
              '\n', '==========================================')  
                
    def norm(self, gold):  #Normalize Data file with gold
        en_norm, int_norm = u.norm(self, gold)
        self.en_norm = en_norm
        self.int_norm = int_norm
        print('\n ~ Data normalized',
              '\n', '==========================================')   
        
    def shift(self, gold): #Flatten spectra
        en_shift, int_shift = u.shift(self, gold)
        self.en_norm = en_shift
        self.int_norm = int_shift
        print('\n ~ Only energy normalized',
              '\n', '==========================================')   
        
    def flatten(self, norm=False): #Flatten spectra
        int_flat = u.flatten(self, norm)
        self.int_flat = int_flat
        print('\n ~ Spectra flattened',
              '\n', '==========================================')
        
    def FS_flatten(self, ang=True): #Flatten FS
        map_flat = u.FS_flatten(self, ang)
        self.map_flat = map_flat
        print('\n ~ FS flattened',
              '\n', '==========================================')   
        
    def restrict(self, bot=0, top=1, left=0, right=1): #restrict spectrum
        (en_restr, ens_restr, en_norm_restr, ang_restr, angs_restr, pol_restr, 
         pols_restr, int_restr, int_norm_restr) = u.restrict(
                 self, bot, top, left, right)
        self.en = en_restr
        self.ens = ens_restr
        self.ang = ang_restr
        self.angs = angs_restr
        self.pol = pol_restr
        self.pols = pols_restr
        self.en_norm = en_norm_restr
        self.int = int_restr
        self.int_norm = int_norm_restr
        print('\n ~ Spectra restricted',
              '\n', '==========================================')  
 
    def FS_restrict(self, bot=0, top=1, left=0, right=1): #restrict FS
        (ang_restr, angs_restr, pol_restr, pols_restr, map_restr) = u.FS_restrict(
                 self, bot, top, left, right)
        self.ang = ang_restr
        self.angs = angs_restr
        self.pol = pol_restr
        self.pols = pols_restr
        self.map = map_restr
        print('\n ~ FS restricted',
              '\n', '==========================================')  
        
    def ang2k(self, angdg, Ekin, lat_unit=False, a=5.33, b=5.33, c=11, 
              V0=0, thdg=0, tidg=0, phidg=0):      
        k, k_V0 = u.ang2k(self, angdg, Ekin, lat_unit, a, b, c, 
                          V0, thdg, tidg, phidg)
        self.k = k
        self.k_V0 = k_V0
        self.ks = np.transpose(np.broadcast_to(k[0], 
                                               (self.en.size, self.ang.size)))
        self.k_V0s = np.transpose(np.broadcast_to(k_V0[0], 
                                               (self.en.size, self.ang.size)))
            
        print('\n ~ Angles converted into k-space',
              '\n', '==========================================')  
        
    def ang2kFS(self, angdg, Ekin, lat_unit=False, a=5.33, b=5.33, c=11, 
                V0=0, thdg=0, tidg=0, phidg=0):   
        kx, ky, kx_V0, ky_V0 = u.ang2kFS(self, angdg, Ekin, lat_unit, a, b, c, 
                                         V0, thdg, tidg, phidg)
        self.kx = kx
        self.ky = ky
        self.kx_V0 = kx_V0
        self.ky_V0 = ky_V0
        print('\n ~ Angles converted into k-space for Fermi surface',
              '\n', '==========================================')  
        
    def FS(self, e=0, ew=0, norm=False): #Extract Constant Energy Map
        FSmap = u.FS(self, e, ew, norm)
        self.map = FSmap
        print('\n ~ Constant energy map extracted',
              '\n', '==========================================')  
        
    def plt_spec(self, norm=False):
        uplt.plt_spec(self, norm)
    
    def plt_FS_polcut(self, norm=False, p=0, pw=0):
        uplt.plt_FS_polcut(self, norm, p, pw)
        
    def plt_FS(self, coord=False):
        uplt.plt_FS(self, coord)
        
    def plt_hv(self, a=0, aw=0):
        uplt.plt_hv(self, a, aw)
        