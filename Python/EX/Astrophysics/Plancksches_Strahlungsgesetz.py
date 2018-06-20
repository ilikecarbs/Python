# -*- coding: utf-8 -*-
from numpy import *
from pylab import *

therm   =   1.419   *   10**32
lap     =   1.615   *   10**-35
tick    =   .538297e-43
marb    =   .217716e-7

me      =   4.184e-23
mb      =   7.688e-20
alf     =   1/137.0359990

def cmb():
    T   =   2.725/therm
    nu  =   linspace(1e10, 6e11, 100)*tick
    S_nu    =   (4*pi)*nu**3/(exp(2*pi*nu/(T))-1)

    S_nu    *=  marb/tick**2
    plot(nu/(lap*100), S_nu/1e-20)
    xlabel('frequency in cm$^{-1}$')
    ylabel('Spectrum in Mj/(sr)')
    
cmb()
show()

