#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 14:37:57 2018

@author: denyssutter
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from utils_ext import rainbow_light


rainbow_light = rainbow_light()
cm.register_cmap(name='rainbow_light', cmap=rainbow_light)

def plt_spec(self, norm):

    if norm == True:
        k = self.angs
        en = self.en_norm
        dat = self.int_norm
    elif norm == False:
        k = self.ang
        en = self.en
        dat = np.transpose(self.int)
       
    plt.figure(1000)
    plt.clf()
    plt.pcolormesh(k, en, dat, cmap = cm.Greys)
    
    if norm == True:
        plt.plot([np.min(k), np.max(k)], [0, 0], 'k:')
    plt.xlabel('$k_x$')   
    plt.show()
    
    
def plt_FS(self, coord=False):
    if coord == True:
        kx = self.kx
        ky = self.ky
        dat = self.map
    elif coord == False:
        kx = self.ang
        ky = self.pol
        dat = self.map
       
    plt.figure(2000)
    plt.clf()
    plt.pcolormesh(kx, ky, dat, cmap = rainbow_light)
    plt.colorbar()
    plt.show()


def plt_cont_TB_simple(self, e0):
    bndstr = self.bndstr
    coord = self.coord   
    X = coord['X']; Y = coord['Y']   
    en = bndstr['en']
    plt.contour(X, Y, en, levels = e0)
  
def plt_cont_TB_SRO(self, e0):
    bndstr = self.bndstr
    coord = self.coord   
    X = coord['X']; Y = coord['Y']   
    xz = bndstr['xz']; yz = bndstr['yz']; xy = bndstr['xy']
    
    plt.subplot(231)
    plt.contour(X, Y, xz, colors = 'black', linestyles = ':', levels = e0)
    plt.subplot(232)
    plt.contour(X, Y, yz, colors = 'black', linestyles = ':', levels = e0)
    plt.subplot(233)
    plt.contour(X, Y, xy, colors = 'black', linestyles = ':', levels = e0)
  
def plt_cont_TB_CSRO20(self, e0):   
    bndstr = self.bndstr
    coord = self.coord   
    X = coord['X']; Y = coord['Y']   
    Axz = bndstr['Axz']; Ayz = bndstr['Ayz']; Axy = bndstr['Axy']
    Bxz = bndstr['Bxz']; Byz = bndstr['Byz']; Bxy = bndstr['Bxy']

    plt.subplot(231)
    plt.contour(X, Y, Axz, levels = e0)
    plt.subplot(232)
    plt.contour(X, Y, Ayz, levels = e0)
    plt.subplot(233)
    plt.contour(X, Y, Axy, levels = e0)
    plt.subplot(234)
    plt.contour(X, Y, Bxz, levels = e0)
    plt.subplot(235)
    plt.contour(X, Y, Byz, levels = e0)
    plt.subplot(236)
    plt.contour(X, Y, Bxy, levels = e0)
        
