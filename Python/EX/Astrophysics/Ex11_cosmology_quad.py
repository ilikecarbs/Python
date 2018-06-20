# -*- coding: utf-8 -*-

from numpy import sqrt, linspace, vectorize
from scipy.integrate import quad
from pylab import *


zrange  =   linspace(0., 2, 1000)

H0  =   1./13.7
Om  =   0.3
Og  =   0
OL  =   0.7


#The integrand for lookback time.
def lbtintegrand(z):
    return 1./(H0*(1+z)*sqrt(Om*(1+z)**3 + Og*(1+z)**4 + OL))
    
#Compute the lookback time.
def lbtime(z):
    return quad(lbtintegrand, 0.,z)[0]
    
vlbtime =   vectorize(lbtime)

#The integrand for distances
def dintegrand(z):
    return 1./(H0*sqrt(Om*(1+z)**3 + Og*(1+z)**4 + OL))
    
#Compute the distances    
def angdist(z):
    return 1./(1+z)*quad(dintegrand, 0, z)[0]

vangdist    =   vectorize(angdist)

def lumdist(z):
    return (1+z)*quad(dintegrand,0,z)[0]

vlumdist    =   vectorize(lumdist)

xlabel('z')
ylabel('Gyrs')

title('concordance cosmology')
plot(zrange, vlbtime(zrange), label='$t$')
plot(zrange, vangdist(zrange), label='$d_A$')
plot(zrange, vlumdist(zrange), label='$d_L$')
legend(loc='upper left')

show()