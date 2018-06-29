#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 09:57:01 2018

@author: denyssutter
"""
import os
os.chdir('/Users/denyssutter/Documents/library/Python/ARPES')
import numpy as np
import utils_math as umath
import ARPES
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
   
 
def find(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx

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
        if np.size(self.int.shape) == 2:
            for i in range(self.angs.shape[0]):
                en_norm[i, :] = self.en - Ef[i]
                int_norm[i, :] = np.divide(self.int[i, :], norm[i])
        elif np.size(self.int.shape) == 3:
            for i in range(self.angs.shape[1]):
                en_norm[:, i, :] = self.en - Ef[i]
                int_norm[:, i, :] = np.divide(self.int[:, i, :], norm[i])
    except OSError:
        os.chdir('/Users/denyssutter/Documents/library/Python')
        print('- No gold files: {}'.format(self.gold),'\n')
    return en_norm, int_norm

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
    int_flat = self.int
    if norm == True:
        int_flat = self.int_norm
    elif norm == False:
        int_flat = self.int  
    elif norm == 'shift':
        int_flat = self.int_shift
    for i in range(int_flat.shape[0]):
        int_flat[i, :] = np.divide(int_flat[i, :], np.sum(int_flat[i, :]))           
    return int_flat
  
def restrict(self, bot, top, left, right):
    m, n = self.int.shape
    val, _bot = find(range(n), bot * n)
    val, _top = find(range(n), top * n)
    val, _left = find(range(m), left * m)
    val, _right = find(range(m), right * m)
    en_restr = self.en[_bot:_top]
    ens_restr = self.ens[_left:_right, _bot:_top]
    ang_restr = self.ang[_left:_right]
    angs_restr = self.angs[_left:_right, _bot:_top]
    en_norm_restr = self.en_norm[_left:_right, _bot:_top]
    int_restr = self.int[_left:_right, _bot:_top]
    int_norm_restr = self.int_norm[_left:_right, _bot:_top]
    return (en_restr, ens_restr, en_norm_restr, ang_restr, 
            angs_restr, int_restr, int_norm_restr)

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
                        self.pol[i], phidg)
        kx[i, :] = k[0, :]
        ky[i, :] = k[1, :]
        kx_V0[i, :] = k_V0[0, :]
        ky_V0[i, :] = k_V0[1, :]
    return kx, ky, kx_V0, ky_V0

def FS(self, e, ew, norm): #Extract Constant Energy Map
    """
    Extracts Constant Energy Map, Integrated from e to e-ew
    """
    if norm == True:
        e_val, e_ind = find(self.en_norm[0, 0, :], e)
        ew_val, ew_ind = find(self.en_norm[0, 0, :], e-ew)
        FSmap = np.sum(self.int_norm[:, :, e_ind:-1:ew_ind], axis=2)
    elif norm == False:
        e_val, e_ind = find(self.en, e)
        ew_val, ew_ind = find(self.en, e-ew)
        FSmap = np.sum(self.int[:, :, e_ind:-1:ew_ind], axis=2)
    return FSmap

def gold(file, mat, year, sample, Ef_ini, BL):
    """
    Generates Files for Normalization
    """
    if BL == 'DLS':
        D = ARPES.DLS(file, mat, year, sample)
    bnd = 150
    ch = 300
    plt.figure(5001)
    plt.subplot(211)
    enval, inden = find(D.en, Ef_ini-0.12)
    plt.plot(D.en[inden:],D.int[ch,inden:],'bo',markersize=3)
    p1_ini = [.001, Ef_ini, np.max(D.int[ch, :]), 20, 0]
    Ef   = np.zeros(len(D.ang))
    norm = np.zeros(len(D.ang))
    for i in range(0,len(D.ang)):
        try:
            popt, pcov = curve_fit(umath.FDsl, D.en[inden:], 
                                   D.int[i,inden:], p1_ini)
        except RuntimeError:
            print("Error - convergence not reached")
        if i==ch:
            plt.plot(D.en[inden:], umath.FDsl(D.en[inden:], 
                     popt[0], popt[1], popt[2], popt[3], popt[4]),'r-')
        Ef[i]   = popt[1]
        norm[i] = sum(D.int[i,:])
        
    pini_poly2 = [Ef[ch], 0, 0, 0]
    #bounds_poly2 = ([-1, Ef[300]-1, -np.inf, -1], [1, Ef[300]+1, np.inf, 1])
    popt, pcov = curve_fit(umath.poly2, D.ang[bnd:-bnd], 
                           Ef[bnd:-bnd], pini_poly2)
    Ef_fit = umath.poly2(D.ang, popt[0], popt[1], popt[2], popt[3])
    os.chdir(D.folder)
    np.savetxt(''.join(['Ef_',str(file),'.dat']),Ef_fit)
    np.savetxt(''.join(['norm_',str(file),'.dat']),norm)
    os.chdir('/Users/denyssutter/Documents/library/Python')
    plt.subplot(212)
    plt.plot(D.ang, Ef, 'bo')
    plt.plot(D.ang, Ef_fit, 'r-')
    plt.ylim(Ef[ch]-5, Ef[ch]+5)
    #label='fit: a=%0.2f, b=%5.2f, c=%5.2f' % tuple(popt))
    plt.show()
        