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

from mpl_toolkits.mplot3d import Axes3D

def orbits(data):
    eta =   linspace(0,2*pi,100)
    w = (data[5]-data[4])*pi/180
    O = data[4]*pi/180
    I = data[2]*pi/180
    Rzw = matrix([[cos(w), sin(w), 0],
                  [-sin(w), cos(w), 0],
                  [0,        0,    1]])
    RxI = matrix([[1, 0, 0],
                  [0, cos(I), sin(I)],
                  [0, -sin(I), cos(I)]])
    RzO = matrix([[cos(O), sin(O), 0],
                  [-sin(O), cos(O), 0],
                  [0,        0,    1]])
    x = zeros(len(eta))
    y = zeros(len(eta))
    z = zeros(len(eta))
    
    for i in frange(0, len(eta)-1):
        r = array([data[0]*(cos(eta[i])-data[1]), data[0]*(sqrt(1-data[1]**2)*sin(eta[i])), 0])   # x, y, z
        rot = r*Rzw*RxI*RzO
        x[i] = rot[0,0]
        y[i] = rot[0,1]
        z[i] = rot[0,2]
    return x, y, z


fig = figure()
ax  = Axes3D(fig)
#ax.plot(x, y, z, '.')
x,y,z = orbits(Mercury)
ax.plot(x,y,z)
x,y,z=orbits(Venus)
ax.plot(x,y,z)
x,y,z = orbits(EM)
ax.plot(x,y,z)
x,y,z = orbits(Mars)
ax.plot(x,y,z)
x,y,z = orbits(Jupiter)
ax.plot(x,y,z)
x,y,z=orbits(Saturn)
ax.plot(x,y,z)
x,y,z= orbits(Uranus)
ax.plot(x,y,z)
x,y,z=orbits(Neptune)
ax.plot(x,y,z)
x,y,z=orbits(Pluto)
ax.plot(x,y,z)
ax.plot(array([0]),array([0]), array([0]), 'yo')
  
show()