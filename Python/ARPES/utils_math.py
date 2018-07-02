#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 15 13:53:54 2018

@author: denyssutter
"""

import numpy as np
from numpy import linalg as la
import utils_plt as uplt 

def paramSRO():
    """
    Parameter set of TB model Sr2RuO4 arXiv:1212.3994v1
    """
    param = dict([('t1', .145), ('t2', .016), ('t3', .081), ('t4', .039),
              ('t5', .005), ('t6', 0), ('mu', .122), ('l', .032)])
    return param
        
def paramCSRO20():
    """
    Parameter set of TB model D. Sutter et al. :-)
    """
    param = dict([('t1', .115), ('t2', .002), ('t3', .071), ('t4', .039),
              ('t5', .012), ('t6', 0), ('mu', .084), ('l', .037)])
    return param

        
class TB:
    """
    Tight binding models
    """
    def __init__(self, a = np.pi, kbnd = 1, kpoints = 100):
        self.a = a
        x = np.linspace(-kbnd, kbnd, kpoints)
        y = np.linspace(-kbnd, kbnd, kpoints)
        [X, Y] = np.meshgrid(x,y)
        self.coord = dict([('x', x), ('y', y), ('X', X), ('Y', Y)])
  
    def SRO(self, param):
        #Load TB parameters
        t1 = param['t1']; t2 = param['t2']; t3 = param['t3']
        t4 = param['t4']; t5 = param['t5']; t6 = param['t6']
        mu = param['mu']; l = param['l']
        coord = self.coord
        a = self.a
        x = coord['x']; y = coord['y']; X = coord['X']; Y = coord['Y']
        #Hopping terms
        fyz = - 2 * t2 * np.cos(X * a) - 2 * t1 * np.cos(Y * a)
        fxz = - 2 * t1 * np.cos(X * a) - 2 * t2 * np.cos(Y * a)
        fxy = - 2 * t3 * (np.cos(X * a) + np.cos(Y * a)) - \
                4 * t4 * (np.cos(X * a) * np.cos(Y * a)) - \
                2 * t5 * (np.cos(2 * X * a) + np.cos(2 * Y * a))
        off = - 4 * t6 * (np.sin(X * a) * np.sin(Y * a))
        #Placeholders energy eigenvalues
        yz = np.ones((len(x), len(y))); xz = np.ones((len(x), len(y)))
        xy = np.ones((len(x), len(y)))
        #Tight binding Hamiltonian
        def H(i,j):
            H = np.array([[fyz[i,j] - mu, off[i,j] + complex(0,l), -l],
                          [off[i,j] - complex(0,l), fxz[i,j] - mu, complex(0,l)],
                          [-l, -complex(0,l), fxy[i,j] - mu]])
            return H
        #Diagonalization of symmetric Hermitian matrix on k-mesh
        for i in range(len(x)):
            for j in range(len(y)):
                eval = la.eigvalsh(H(i,j))
                eval = np.real(eval)
                yz[i,j] = eval[0]; xz[i,j] = eval[1]; xy[i,j] = eval[2]
        bndstr = dict([('yz', yz), ('xz', xz), ('xy', xy)])
        self.bndstr = bndstr
        
    def CSRO(self, param):
        #Load TB parameters
        t1 = param['t1']; t2 = param['t2']; t3 = param['t3']
        t4 = param['t4']; t5 = param['t5']; t6 = param['t6']
        mu = param['mu']; l = param['l']
        coord = self.coord
        a = self.a
        x = coord['x']; y = coord['y']; X = coord['X']; Y = coord['Y']
        #Hopping terms
        fx = -2 * np.cos((X + Y) / 2 * a)
        fy = -2 * np.cos((X - Y) / 2 * a)
        f4 = -2 * t4 * (np.cos(X * a) + np.cos(Y * a))
        f5 = -2 * t5 * (np.cos((X + Y) * a) + np.cos((X - Y) * a))
        f6 = -2 * t6 * (np.cos(X * a) - np.cos(Y * a))
        #Placeholders energy eigenvalues
        Ayz = np.ones((len(x), len(y))); Axz = np.ones((len(x), len(y)))
        Axy = np.ones((len(x), len(y))); Byz = np.ones((len(x), len(y))) 
        Bxz = np.ones((len(x), len(y))); Bxy = np.ones((len(x), len(y)))
        #TB submatrix
        def A(i,j):
            A = np.array([[-mu, complex(0,l) + f6[i,j], -l],
                          [-complex(0,l) + f6[i,j], -mu, complex(0,l)],
                          [-l, -complex(0,l), -mu + f4[i,j] + f5[i,j]]])
            return A
        #TB submatrix
        def B(i,j): 
            B = np.array([[t2 * fx[i,j] + t1 * fy[i,j], 0, 0],
                          [0, t1 * fx[i,j] + t2 * fy[i,j], 0],
                          [0, 0, t3 * (fx[i,j] + fy[i,j])]])
            return B
        #Tight binding Hamiltonian
        def H(i,j):
            C1 = np.concatenate((A(i,j), B(i,j)), 1)
            C2 = np.concatenate((B(i,j), A(i,j)), 1)
            H  = np.concatenate((C1, C2), 0)
            return H
        #Diagonalization of symmetric Hermitian matrix on k-mesh
        for i in range(len(x)):
            for j in range(len(y)):
                eval = la.eigvalsh(H(i,j))
                eval = np.real(eval)
                Ayz[i,j] = eval[0]; Axz[i,j] = eval[1]; Axy[i,j] = eval[2]
                Byz[i,j] = eval[3]; Bxz[i,j] = eval[4]; Bxy[i,j] = eval[5]
        bndstr = dict([('Ayz', Ayz), ('Axz', Axz), ('Axy', Axy),
                       ('Byz', Byz), ('Bxz', Bxz), ('Bxy', Bxy)])
        self.bndstr = bndstr
                
    def simple(self, param):
        t1 = param['t1']; t2 = param['t2']; t3 = param['t3']
        t4 = param['t4']; t5 = param['t5']; mu = param['mu']   
        coord = self.coord
        a = self.a
        X = coord['X']; Y = coord['Y']
        
        en = - mu - \
            2 * t1 * (np.cos(X * a) + np.cos(Y * a)) - \
            4 * t2 * (np.cos(X * a) * np.cos(Y * a)) - \
            2 * t3 * (np.cos(2 * X * a) + np.cos(2 * Y * a)) - \
            4 * t4 * (np.cos(2 * X * a) * np.cos(Y * a) + \
                      np.cos(X * a) * np.cos(2 * Y * a)) - \
            4 * t5 * (np.cos(2 * X * a) * np.cos(2 * Y * a))
            
        bndstr = dict([('en', en)])
        self.bndstr = bndstr
  
    def plt_cont_TB_SRO(self, e0=0):
        uplt.plt_cont_TB_SRO(self, e0)
        
    def plt_cont_TB_CSRO20(self, e0=0): 
        uplt.plt_cont_TB_CSRO20(self, e0)
        
    def plt_cont_TB_simple(self, e0=0):
        uplt.plt_cont_TB_simple(self, e0)
        
        

def FDsl(x, p0, p1, p2, p3, p4):
    """
    Fermi Dirac Function sloped
    p3 + (p2 + p4 * x) * (np.exp((x - p1) / p0) + 1) ** -1
    """
    return p3 + (p2 + p4 * x) * (np.exp((x - p1) / p0) + 1) ** -1

def poly2(x, p0, p1, p2, p3):
    """
    Polynomial second order
    p1 + p2 * (x - p0) + p3 * (x - p0)**2 
    """
    return p1 + p2 * (x - p0) + p3 * (x - p0)**2 
