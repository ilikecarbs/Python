#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 09:57:01 2018

@author: ilikecarbs

%%%%%%%%%%%%%%%%%%%%
        utils
%%%%%%%%%%%%%%%%%%%%

**Useful helper functions, mainly used for ARPES.py**

.. note::
        To-Do:
            -

"""
import os
import numpy as np
import utils_math
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def find(array, val):
    """Returns array[_val], _val

    **Searches entry in array closest to val.**

    Args
    ----------
    :array: entry value in array
    :_val: index of entry

    Return
    ------
    :array[_val]:   entry value in array
    :_val:          index of entry

    """
    array = np.asarray(array)
    _val = (np.abs(array - val)).argmin()
    return array[_val], _val


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
