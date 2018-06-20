# -*- coding: utf-8 -*-

from numpy import sqrt, linspace
from scipy.integrate import odeint
from pylab import *
#Define the initial conditions for each of the four ODEs
a       =   20
tend    =   2*a**2

inic    =   [a,0,0,1/sqrt(a)]

#Times to evaluate the ODEs. 800 times from 0 to 100 (inclusive).
t       =   linspace(0,tend,10*tend)

#The derivative function.
def f(z,time):
    l   =   z[0]*z[2] + z[1]*z[3]
    r   =   sqrt(z[0]**2 + z[1]**2)
    
    return [ z[2] -2.*l*z[0]/r**3,
             z[3] -2.*l*z[1]/r**3,
             - z[0]/(r*(r-2.)**2) + l*(2.*z[2] - 3.*z[0]*l/r**2)/r**3,
             - z[1]/(r*(r-2.)**2) + l*(2.*z[3] - 3.*z[1]*l/r**2)/r**3
             ]
             
#Compute the ODE
res =   odeint(f, inic, t)

#Plot the results
plot(res[:,0], res[:,1], label='$p_y = \sqrt{M/a}$')
cir =   Circle((0,0), a, color='r', fill=False)
gca().add_patch(cir)
legend(loc='upper left')
axis('equal')
xlim(-a,a)
ylim(-a,a)

#savefig('Schw_plots/circ_20.pdf', format='pdf')

show()

