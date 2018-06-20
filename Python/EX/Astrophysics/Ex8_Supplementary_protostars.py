# -*- coding: utf-8 -*-

from pylab import *
from numpy import *

marb    =   2.177e-8
me      =   4.184e-23 * marb
mb      =   7.688e-20 * marb
ML      =   mb**-2
R_Jup   =   14/(me * mb)
M_Jup   =   .5e-3 * ML

M   =   linspace(0, 2* ML, 10000)
R   =   3.5/(me * mb)* (M/ML)**(-1./3.)

loglog(M/M_Jup, R/R_Jup, 'r')
plot(1, 1, 'bs')
grid()


show()