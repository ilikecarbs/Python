# -*- coding: utf-8 -*-

from numpy import sqrt, linspace, vectorize
from scipy.integrate import quad
from pylab import *


zrange  =   linspace(0., 2, 1000)

H0  =   1./13.7
Om  =   0.3
Og  =   0.
OL  =   0.7
acmb=   1./1100.
zcmb=   1./acmb-1.



def horintegrand(z):
    return 1./(H0*(1+z)**2*sqrt(Om*(1+z)**3 + Og*(1+z)**4 + OL))
    
#Compute horizon.
def horizon(x,y):
    return quad(horintegrand, x,y)[0]
    


print horizon(0,zcmb)/horizon(zcmb,1)




show()