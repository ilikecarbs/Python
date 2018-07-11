#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 15 13:53:54 2018

@author: denyssutter
"""

import numpy as np
from numpy import linalg as la
import utils_plt as uplt 
from scipy.stats import exponnorm
from scipy import special
from uncertainties import ufloat
import uncertainties.umath as umath

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

def lor(x, p0, p1, p2, 
        p3, p4, p5):
    """
    Single lorentzians on a quadratic background
    """
    return (p2 / (1 + ((x - p0) / p1) ** 2) + 
            p3 + p4 * x + p5 * x ** 2)

def lorHWHM(x, p0, p1, p2, 
        p3, p4, p5):
    """
    Single lorentzians on a quadratic background
    """
    return (p2 / (np.pi * p2 / 2 * (1 + ((x - p0) / p1 * 2) ** 2)) +
            p3 + p4 * x + p5 ** 2)
    
def lor2(x, p0, p1, 
         p2, p3, 
         p4, p5, 
         p6, p7, p8):
    """
    Two lorentzians on a quadratic background
    """
    return (p4 / (1 + ((x - p0) / p2) ** 2) + 
            p5 / (1 + ((x - p1) / p3) ** 2) +
            p6 + p7 * x + p8 * x ** 2)

def lor4(x, p0, p1, p2, p3, 
         p4, p5, p6, p7, 
         p8, p9, p10, p11, 
         p12, p13, p14):
    """
    Four lorentzians on a quadratic background
    """
    return (p8 / (1 + ((x - p0) / p4)  ** 2) + 
            p9 / (1 + ((x - p1) / p5)  ** 2) +
            p10 / (1 + ((x - p2) / p6)  ** 2) +
            p11 / (1 + ((x - p3) / p7)  ** 2) +
            p12 + p13 * x + p14 * x ** 2)
    
def lor6(x, p0, p1, p2, p3, p4, p5, 
         p6, p7, p8, p9, p10, p11, 
         p12, p13, p14, p15, p16, p17, 
         p18, p19, p20):
    """
    Six lorentzians on a quadratic background
    """
    return (p12 / (1 + ((x - p0) / p6)  ** 2) + 
            p13 / (1 + ((x - p1) / p7)  ** 2) +
            p14 / (1 + ((x - p2) / p8)  ** 2) +
            p15 / (1 + ((x - p3) / p9)  ** 2) +
            p16 / (1 + ((x - p4) / p10) ** 2) +
            p17 / (1 + ((x - p5) / p11) ** 2) +
            p18 + p19 * x + p20 * x ** 2)
    
def lor8(x, p0, p1, p2, p3, p4, p5, p6, p7, 
         p8, p9, p10, p11, p12, p13, p14, p15, 
         p16, p17, p18, p19, p20, p21, p22, p23, 
         p24, p25, p26):
    """
    Eight lorentzians on a quadratic background
    """
    return (p16 / (1 + ((x - p0) / p8)  ** 2) + 
            p17 / (1 + ((x - p1) / p9)  ** 2) +
            p18 / (1 + ((x - p2) / p10) ** 2) +
            p19 / (1 + ((x - p3) / p11) ** 2) +
            p20 / (1 + ((x - p4) / p12) ** 2) +
            p21 / (1 + ((x - p5) / p13) ** 2) +
            p22 / (1 + ((x - p6) / p14) ** 2) +
            p23 / (1 + ((x - p7) / p15) ** 2) +
            p24 + p25 * x + p26 * x ** 2)
    
def gauss2(x, p0, p1, p2, p3, p4, p5, p6, p7, p8):
    """
    Two gaussians on a quadratic background
    """
    return (p4 * np.exp(-(x - p0) ** 2 / (2 * p2 ** 2)) + 
            p5 * np.exp(-(x - p1) ** 2 / (2 * p3 ** 2)) +
            p6 + p7 * x + p8 * x ** 2)    
    
def FL_simple(x, p0, p1, p2, p3, p4, p5):
    """
    Fermi liquid quasiparticle with simple self energy
    """
    ReS = p0 * x
    ImS = p1 + p2 * x ** 2;

    return (p4 * 1 / np.pi * ImS / ((x - ReS - p3) ** 2 + ImS ** 2) * 
            (np.exp((x - 0) / p5) + 1) ** -1)
    
def uFL_simple(x, p0, p1, p2, p3, p4, p5,
               ep0, ep1, ep2, ep3, ep4, ep5):
    """
    Fermi liquid quasiparticle with simple self energy
    """
    ReS = ufloat(p0, ep0) * x
    ImS = ufloat(p1, ep1) + ufloat(p2, ep2) * x ** 2;

    return (ufloat(p4, ep4) * 1 / np.pi * 
            ImS / ((x - ReS - ufloat(p3, ep3)) ** 2 + ImS ** 2) * 
            (umath.exp((x - 0) / ufloat(p5, ep5)) + 1) ** -1)
    
def gauss_mod(x, p0, p1, p2, p3, p4, p5):
    """
    Modified Gaussian
    """
    return (p0 * np.exp(-.5 * ((-x + p1) / p2) ** 2) * 
            (p3 * special.erf((-x + p4) / p5) + 1))

def Full_simple(x, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9):
    """
    Spectral function simple FL + incoherent weight as exp. mod. Gaussian
    """
    return (FL_simple(x, p0, p1, p2, p3, p4, p5) +
            p9 * exponnorm.pdf(-x, K=p6, loc=p7, scale = p8) *
            FDsl(x, p0=1e-3, p1=0, p2=1, p3=0, p4=0))
    
def Full_mod(x, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11):
    """
    Spectral function simple FL + incoherent weight as exp. mod. Gaussian
    """
    return (FL_simple(x, p0, p1, p2, p3, p4, p5) +
            gauss_mod(x, p6, p7, p8, p9, p10, p11))

