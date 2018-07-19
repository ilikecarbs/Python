#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 09:57:01 2018

@author: denyssutter

%%%%%%%%%%%%%%%%%%%%
        utils
%%%%%%%%%%%%%%%%%%%%

Content:
utilities for data manipulation

"""
import os
os.chdir('/Users/denyssutter/Documents/library/Python/ARPES')
import numpy as np
import utils_math
import utils_plt
import ARPES
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
   
 
def find(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx

def Shirley(en, edc):
    N = len(en)
    A = 1e-5
    B = np.ones((N))
    B[-1] = edc[-1]
    B[-2] = edc[-1]
    it = 10
    for k in range(it):
        for i in range(N - 2):
            SUM = 0.
            for j in np.arange(i + 1, N):
                SUM += (edc[j] - B[j])    
            B[i] = B[-1] + A * SUM
        A = A * (1. + (edc[0] - B[0]) / edc[0])
    return B

def norm(self, gold): 
    """
    Normalize Data
    """
    self.gold = gold
    try:
        os.chdir(self.folder)
        Ef = np.loadtxt(''.join(['Ef_',str(gold),'.dat']))
        norm = np.loadtxt(''.join(['norm_',str(gold),'.dat']))
        os.chdir('/Users/denyssutter/Documents/library/Python')
        en_norm = np.ones(self.ens.shape)
        int_norm = np.ones(self.int.shape)
        eint_norm = np.ones(self.eint.shape)
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
        print('- No gold files: {}'.format(self.gold),'\n')
    return en_norm, int_norm, eint_norm

def shift(self, gold): 
    """
    Normalize energy file with gold
    """
    self.gold = gold
    try:
        os.chdir(self.folder)
        Ef = np.loadtxt(''.join(['Ef_',str(gold),'.dat']))
        os.chdir('/Users/denyssutter/Documents/library/Python')
        en_shift = np.ones(self.ens.shape)
        int_shift = np.ones(self.int.shape)
        if np.size(self.int.shape) == 2:
            for i in range(self.angs.shape[0]):
                en_shift[i, :] = self.en - Ef[i]
                int_shift[i, :] = self.int[i, :]
        elif np.size(self.int.shape) == 3:
            for i in range(self.angs.shape[1]):
                en_shift[:, i, :] = self.en - Ef[i]
                int_shift[:, i, :] = self.int[:, i, :]
    except OSError:
        os.chdir('/Users/denyssutter/Documents/library/Python')
        print('- No gold files: {}'.format(self.gold),'\n')    
    return en_shift, int_shift

def flatten(self, norm): 
    """
    Flatten spectra
    """
    if norm == True:
        int_flat = self.int_norm
    elif norm == False:
        int_flat = self.int  
    elif norm == 'shift':
        int_flat = self.int_shift
    if int_flat.ndim == 2:
        for i in range(int_flat.shape[0]):
            int_flat[i, :] = np.divide(int_flat[i, :], np.sum(int_flat[i, :]))           
    if int_flat.ndim == 3:
        for i in range(int_flat.shape[1]):
            int_flat[:, i, :] = np.divide(int_flat[:, i, :], 
                                    np.sum(int_flat[:, i, :]))   
    return int_flat
  
def FS_flatten(self, ang): 
    """
    Flatten FS
    """
    map_flat = np.zeros(self.map.shape)
    if ang == True:
        for i in range(self.ang.size):
            map_flat[:, i] = np.divide(self.map[:, i],
                                np.sum(self.map[:, i]))    
    elif ang == False:
        for i in range(self.pol.size):
            map_flat[i, :] = np.divide(self.map[i, :],
                                np.sum(self.map[i, :]))    
    return map_flat

def bkg(self, norm):
    """
    Subtract background
    """
    if norm == True:
        int_bkg = self.int_norm
    elif norm == False:
        int_bkg = self.int
    for i in range(self.en.size):
        int_bkg[:, i] = int_bkg[:, i] - np.min(int_bkg[:, i])
    return int_bkg
    
    
def restrict(self, bot, top, left, right):
    if self.int.ndim == 2:
        d1, d2 = self.int.shape
        val, _bot = find(range(d2), bot * d2)
        val, _top = find(range(d2), top * d2)
        val, _left = find(range(d1), left * d1)
        val, _right = find(range(d1), right * d1)
        en_restr = self.en[_bot:_top]
        ens_restr = self.ens[_left:_right, _bot:_top]
        ang_restr = self.ang[_left:_right]
        angs_restr = self.angs[_left:_right, _bot:_top]
        en_norm_restr = self.en_norm[_left:_right, _bot:_top]
        int_restr = self.int[_left:_right, _bot:_top]
        eint_restr = self.eint[_left:_right, _bot:_top]
        int_norm_restr = self.int_norm[_left:_right, _bot:_top]
        eint_norm_restr = self.eint_norm[_left:_right, _bot:_top]
        pol_restr = 0
        pols_restr = 0
    elif self.int.ndim == 3:
        d1, d2 = self.int.shape[1], self.int.shape[0]
        val, _bot = find(range(d2), bot * d2)
        val, _top = find(range(d2), top * d2)
        val, _left = find(range(d1), left * d1)
        val, _right = find(range(d1), right * d1)
        pol_restr = self.pol[_bot:_top]
        pols_restr = self.pols[_bot:_top, _left:_right, :]
        en_restr = self.en
        ens_restr = self.ens[_bot:_top, _left:_right]
        ang_restr = self.ang[_left:_right]
        angs_restr = self.angs[_bot:_top, _left:_right, :]
        en_norm_restr = self.en_norm[_bot:_top, _left:_right, :]
        int_restr = self.int[_bot:_top, _left:_right, :]
        eint_restr = self.int[_bot:_top, _left:_right, :]
        int_norm_restr = self.int[_bot:_top, _left:_right, :]
        eint_norm_restr = self.int[_bot:_top, _left:_right, :]
    return (en_restr, ens_restr, en_norm_restr, ang_restr, 
            angs_restr, pol_restr, pols_restr,
            int_restr, eint_restr, int_norm_restr, eint_norm_restr)

def FS_restrict(self, bot, top, left, right):
    d1, d2 = self.map.shape
    val, _bot = find(range(d1), bot * d1)
    val, _top = find(range(d1), top * d1)
    val, _left = find(range(d2), left * d2)
    val, _right = find(range(d2), right * d2)
    pol_restr = self.pol[_bot:_top]
    pols_restr = self.pols[_bot:_top, _left:_right, :]
#        en_restr = self.en
#        ens_restr = self.ens[_bot:_top, _left:_right]
    ang_restr = self.ang[_left:_right]
    angs_restr = self.angs[_bot:_top, _left:_right, :]
#        en_norm_restr = self.en_norm[_bot:_top, _left:_right, :]
    map_restr = self.int[_bot:_top, _left:_right, :]
    return (ang_restr, angs_restr, pol_restr, pols_restr, map_restr)

def ang2k(self, angdg, Ekin, lat_unit, a, b, c, V0, thdg, tidg, phidg):     
    """
    Transformation from angles to k-space
    """
    hbar = 6.58212*10**-16; #eV * s
    me = 5.68563*10**-32; #eV * s^2 / Angstrom^2
    ang = np.pi * angdg / 180
    th = np.pi * thdg / 180
    ti = np.pi * tidg / 180
    phi = np.pi * phidg / 180
    #Rotation matrices
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
    #Build k-vector
    k_norm = np.sqrt(2 * me * Ekin) / hbar  #norm of k-vector
    k_norm_V0 = np.sqrt(2 * me * (Ekin + V0)) / hbar  #norm of k-vector 
    kv = np.ones((3, len(ang)))  #Placeholder
    kv_V0 = np.ones((3, len(ang)))  #Placeholder
    kv = np.array([k_norm * np.sin(ang), 0*ang, k_norm * np.cos(ang)])
    kv_V0 = np.array([k_norm * np.sin(ang), 0*ang, 
                      np.sqrt(k_norm_V0**2 - (k_norm * np.sin(ang)**2))])
    k_vec = np.matmul(Phi, np.matmul(Ti, np.matmul(Th, kv)))
    k_vec_V0 = np.matmul(Phi, np.matmul(Ti, np.matmul(Th, kv_V0)))
    if lat_unit == True: #lattice units
        k_vec *= np.array([[a / np.pi], [b / np.pi], [c / np.pi]])    
    return k_vec, k_vec_V0

def ang2kFS(self, angdg, Ekin, lat_unit, a, b, c, V0, thdg, tidg, phidg):
    """
    Transformation angles to k-space for Fermi surfaces
    """
    kx = np.ones((self.pol.size, self.ang.size))
    ky = np.ones((self.pol.size, self.ang.size))
    kx_V0 = np.ones((self.pol.size, self.ang.size))
    ky_V0 = np.ones((self.pol.size, self.ang.size))
    for i in range(self.pol.size):
        k, k_V0 = ang2k(self, angdg, Ekin, lat_unit, a, b, c, V0, thdg, 
                        self.pol[i]-tidg, phidg)
        kx[i, :] = k[0, :]
        ky[i, :] = k[1, :]
        kx_V0[i, :] = k_V0[0, :]
        ky_V0[i, :] = k_V0[1, :]
    return kx, ky, kx_V0, ky_V0

def FS(self, e, ew, norm): #Extract Constant Energy Map
    """
    Extracts Constant Energy Map, Integrated from e to e-ew
    """
    FSmap = np.zeros((self.pol.size, self.ang.size))
    if norm == True:
        for i in range(self.ang.size):
            e_val, e_ind = find(self.en_norm[0, i, :], e)
            ew_val, ew_ind = find(self.en_norm[0, i, :], e-ew)
            FSmap[:, i] = np.sum(self.int_norm[:, i, ew_ind:e_ind], axis=1)
    elif norm == False:
        e_val, e_ind = find(self.en, e)
        ew_val, ew_ind = find(self.en, e-ew)
        FSmap = np.sum(self.int[:, :, ew_ind:e_ind], axis=2)
    return FSmap

def gold(self, Ef_ini):
    """
    Generates Files for Normalization
    """
    bnd = 1
    ch = 100
    plt.figure('gold')
    plt.subplot(211)
    enval, inden = find(self.en, Ef_ini-0.12)
    plt.plot(self.en[inden:],self.int[ch,inden:],'bo',markersize=3)
    p1_ini = [.001, Ef_ini, np.max(self.int[ch, :]), 20, 0]
    Ef   = np.zeros(len(self.ang))
    norm = np.zeros(len(self.ang))
    for i in range(0,len(self.ang)):
        try:
            popt, pcov = curve_fit(utils_math.FDsl, self.en[inden:], 
                                   self.int[i,inden:], p1_ini)
        except RuntimeError:
            print("Error - convergence not reached")
        if i==ch:
            plt.plot(self.en[inden:], utils_math.FDsl(self.en[inden:], 
                     popt[0], popt[1], popt[2], popt[3], popt[4]),'r-')
        Ef[i]   = popt[1]
        norm[i] = sum(self.int[i,:])
        
    pini_poly2 = [Ef[ch], 0, 0, 0]
    #bounds_poly2 = ([-1, Ef[300]-1, -np.inf, -1], [1, Ef[300]+1, np.inf, 1])
    popt, pcov = curve_fit(utils_math.poly2, self.ang[bnd:-bnd], 
                           Ef[bnd:-bnd], pini_poly2)
    Ef_fit = utils_math.poly2(self.ang, popt[0], popt[1], popt[2], popt[3])
    os.chdir(self.folder)
    np.savetxt(''.join(['Ef_',str(self.file),'.dat']),Ef_fit)
    np.savetxt(''.join(['norm_',str(self.file),'.dat']),norm)
    os.chdir('/Users/denyssutter/Documents/library/Python')
    plt.subplot(212)
    plt.plot(self.ang, Ef, 'bo')
    plt.plot(self.ang, Ef_fit, 'r-')
    plt.ylim(Ef[ch]-5, Ef[ch]+5)
    #label='fit: a=%0.2f, b=%5.2f, c=%5.2f' % tuple(popt))
    self.plt_spec()
    plt.show()
        