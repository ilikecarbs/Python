# -*- coding: utf-8 -*-


from scipy.optimize import brentq
from pylab import *
from numpy import *
from matplotlib.animation import FuncAnimation


def eccentric_anomaly(t,e):
    t   =   t % (2*pi) - pi
    def kepler(eta):
        return eta - e*sin(eta) - t
    return brentq(kepler, -pi, pi)

t   =   linspace(-pi,pi,20)
eta =   0*t

def sky_position(t):
    e       =   0.9
    for i in range(len(t)):
        eta[i]  =   eccentric_anomaly(t[i],e)
    x,y     =   cos(eta)-e, sqrt(1-e*e)*sin(eta)
    omega   =   65*pi/180
    cs,sn   =   cos(omega), sin(omega)
    x,y     =   x*cs - y*sn, x*sn + y*cs
    I       =   10*pi/180
    y       =   y*cos(I)
    Omega   =   -40*pi/180
    cs, sn  =   cos(Omega), sin(Omega)
    x,y     =   x*cs - y*sn, x*sn + y*cs
    return x,y
    
fig     =   figure()
panel   =   fig.add_subplot(1,1,1)

def frame(n):
    global t
    t   +=  0.03
    x,y  =  sky_position(t)
    panel.clear()
    R    =  2.2
    panel.set_xlim(-R,R)
    panel.set_ylim(-R,R)
    panel.set_aspect('equal')
    panel.xaxis.set_visible(False)
    panel.yaxis.set_visible(False)
    panel.scatter(0,0,marker='+', color='black')
    panel.scatter(x,y,color='red')
    
    
dummy   =   FuncAnimation(fig,frame,range(10000), interval=100)
show()