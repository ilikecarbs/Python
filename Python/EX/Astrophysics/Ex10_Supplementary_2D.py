# -*- coding: utf-8 -*-

from pylab import *
from numpy import *

'''
                       a              e               I                L            long.peri.      long.node.
                     AU, AU/Cy     rad, rad/Cy     deg, deg/Cy      deg, deg/Cy      deg, deg/Cy     deg, deg/Cy
-----------------------------------------------------------------------------------------------------------
'''
Mercury   =array([0.38709927   ,   0.20563593   ,   7.00497902   ,   252.25032350  ,   77.45779628   ,  48.33076593])

Venus     =array([0.72333566   ,   0.00677672    ,  3.39467605   ,   181.97909950  ,  131.60246718   ,  76.67984255])

EM        =array([1.00000261   ,   0.01671123   ,  -0.00001531   ,   100.46457166  ,  102.93768193   ,   0.0])

Mars      =array([1.52371034   ,   0.09339410   ,   1.84969142     ,  -4.55343205  ,  -23.94362959   ,  49.55953891])

Jupiter   =array([5.20288700   ,   0.04838624  ,    1.30439695    ,   34.39644051   ,  14.72847983   , 100.47390909])

Saturn    =array([9.53667594   ,   0.05386179   ,   2.48599187    ,   49.95424423    , 92.59887831  ,  113.66242448])

Uranus   =array([19.18916464   ,   0.04725744   ,   0.77263783   ,   313.23810451 ,   170.95427630   ,  74.01692503])

Neptune  =array([30.06992276   ,   0.00859048   ,   1.77004347    ,  -55.12002969  ,   44.96476227  ,  131.78422574])

Pluto    =array([39.48211675    ,  0.24882730   ,  17.14001206    ,  238.92903833  ,  224.06891629  ,  110.30393684])


def orbits(data):
    eta =   linspace(0,2*pi,100)
    x, y = data[0]*(cos(eta)-data[1]), data[0]*(sqrt(1-data[1]**2)*sin(eta))
    omega   =   (data[5]-data[4])
    cs,sn   =   cos(omega), sin(omega)
    x,y     =   x*cs - y*sn, x*sn + y*cs
    I       =   data[2]
    y       =   y*cos(I)
    Omega   =   data[4]
    cs, sn  =   cos(Omega), sin(Omega)
    x,y     =   x*cs - y*sn, x*sn + y*cs
    plot(x,y)

orbits(Mercury)
orbits(Venus)
orbits(EM)
orbits(Mars)
orbits(Jupiter)
orbits(Saturn)
orbits(Uranus)
orbits(Neptune)
orbits(Pluto)
plot(0,0, 'yo')
    
show()