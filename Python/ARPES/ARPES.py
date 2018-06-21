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


class DLS:
    """
    ARPES Data handling from i05 beamline at Diamond Light Source
    
    input:  - file number
            - material
            - year
            - sample
            
    """

    def __init__(self, file, mat, year, sample): #Load Data file
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
        
        f = h5py.File(path,'r');
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
            self.int = np.asarray(intensity)[0,:,:]
            self.ens = np.broadcast_to(en, (ang.size, en.size))
            self.angs = np.transpose(np.broadcast_to(ang, (en.size, ang.size)))
        photon = list(f['/entry1/instrument/monochromator/energy'])
        self.hv = np.array(photon)
        
        print('\n ~ Initialization complete. Data has {} dimensions'.format(len(np.shape(self.int))),
              '\n', '==========================================')
        
    def norm(self, gold): #Normalize Data file with gold
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
        
    def restrict(self, bot = 0, top = 1, left = 0, right = 1): #restrict spectrum
        (en_restr, ens_restr, en_norm_restr, ang_restr, angs_restr,
         int_restr, int_norm_restr) = u.restrict(self, bot, top, left, right)
        self.en = en_restr
        self.ens = ens_restr
        self.ang = ang_restr
        self.angs = angs_restr
        self.en_norm = en_norm_restr
        self.int = int_restr
        self.int_norm = int_norm_restr
        print('\n ~ Spectra restricted',
              '\n', '==========================================')   
        
    def ang2k(self, angdg, Ekin, a, b, c=11, V0=0, thdg=0, tidg=0, phidg=0):      
        k, k_V0 = u.ang2k(self, angdg, Ekin, a, b, c, V0, thdg, tidg, phidg)
        self.k = k
        self.k_V0 = k_V0
        
        self.ks = np.transpose(np.broadcast_to(k[0], 
                                               (self.en.size, self.ang.size)))
        self.k_V0s = np.transpose(np.broadcast_to(k_V0[0], 
                                               (self.en.size, self.ang.size)))
            
        print('\n ~ Angles converted into k-space for Fermi surface',
              '\n', '==========================================')  
        
    def ang2kFS(self, angdg, Ekin, a, b, c=11, V0=0, thdg=0, tidg=0, phidg=0):   
        kx, ky, kx_V0, ky_V0 = u.ang2kFS(self, angdg, Ekin, a, b, c, 
                                         V0, thdg, tidg, phidg)
        self.kx = kx
        self.ky = ky
        self.kx_V0 = kx_V0
        self.ky_V0 = ky_V0
        print('\n ~ Angles converted into k-space',
              '\n', '==========================================')  
        
    def FS(self, e, ew=0, norm=False): #Extract Constant Energy Map
        FSmap = u.FS(self, e, ew, norm)
        self.map = FSmap
        print('\n ~ Constant energy map extracted',
              '\n', '==========================================')  
        
    def plt_spec(self, norm=False):
        uplt.plt_spec(self, norm)
            
    def plt_FS(self, coord=False):
        uplt.plt_FS(self, coord)

