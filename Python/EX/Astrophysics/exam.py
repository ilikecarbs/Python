# -*- coding: utf-8 -*-

from numpy import sqrt, linspace, vectorize
from scipy.integrate import quad
from pylab import *


zrange  =   linspace(0., 5., 1000)#z=0 is a=1 (now), z=5 is a=1/6 (past)

#Omega matter, Omega gamma, Omega lambda
Om  =   0.3
Og  =   0
OL  =   0.7


#The integrand for distances
def dintegrandconc(z):
    return 1./(sqrt(Om*(1+z)**3 + Og*(1+z)**4 + OL))
  
#The Omega factors must add up to unity. Einstein-de-Sitter universe is matter-dominated. No OL or Og.  
def dintegrandeinstein(z):
    return 1./(sqrt(1*(1+z)**3))

#Integral (8.27) from the script for luminosity distances
def lumdistconc(z):
    return (1+z)*quad(dintegrandconc,0,z)[0] #only need the zero-components (the value)

def lumdisteinstein(z):
    return (1+z)*quad(dintegrandeinstein,0,z)[0]

vlumdistconc    =   vectorize(lumdistconc) #gives me an array
vlumdisteinstein=   vectorize(lumdisteinstein)

xlabel('redshift $z$')
ylabel('luminosity distance $d_{lum}$')

title('luminosity distance against redshift')
plot(zrange, vlumdistconc(zrange), label='Concordance-model')
plot(zrange, vlumdisteinstein(zrange), label='Einstein-de-Sitter universe')
legend(loc='upper left')

show()