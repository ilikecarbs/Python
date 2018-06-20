# -*- coding: utf-8 -*-
"""
Schiefer Wurf
-------------------------
Wir betrachten einen Schiefen Wurf, der durch folgende
Gleichungen beschreiben wird:
      dx/dt = v_x
      dy/dt = v_y
    dv_x/dt = 0
    dv_y/dt = -g
Benutze geeignete Anfangsbedingungen x0, y0, v_x0 und v_y0
    
Verwende die Euler Metode um die Bahn des Schiefen Wurfes zu
berechnen und plotte diese.
""" 

from pylab import plot, show
from numpy import array
from matplotlib.mlab import frange

def schiefer_wurf():
    x   = 0
    y   = 0
    v_x = 20.
    v_y = 20.
    x_vals = [x]
    y_vals = [y]    
    dt  = 0.1    
    while y>=0:
        x   +=   v_x*dt
        y   +=   v_y*dt
        v_y +=     0*dt
        v_y += -9.81*dt
        x_vals.append(x)
        y_vals.append(y)
    plot(x_vals, y_vals, 'b.')
    show()



'''
Verwende nun eine Methode in dem Solver Script um die Differentialgleichungen
des Schiefen Wurfes zu lösen. Implementiere dazu die Funktion dXdt(X, t)
'''

def dXdt(X, t):
    """ Differential-Gleichung zum Schiefen Wurf

    Parameters
    ----------
    X : array of float
        Vektor mit (x, y, v_x, v_y) zum Zeitpunkt t
    t : float
        Zeitpunkt 
    
    Returns
    -------
    dXdt : array of float
        Änderungen der Komponenten von X zum Zeitpunkt t
        
    """
    g    = 9.81
    dx   = X[2]
    dy   = X[3]
    dv_x = 0
    dv_y = -g
    return array([dx, dy, dv_x, dv_y])


def plot_wurf(h=0.1):
    '''
    Berechne mehrere Näherungeslösungen für diess Beispiel und plotte dann
    y als Funktion von x. Verwende für jede Näherung eine andere Schrittlänge
    aus dem Interval [0.1, 2]. 
    Welchen Effekt siehst du? Welche Lösungen sind brauchbar?
    '''
    from solvers_sol import euler
    from solvers_sol import rungekutta
    x1 = euler(dXdt, y0 = array([0., 0., 20., 20.]), t=frange(0, 4, h))
    x2 = rungekutta(dXdt, y0 = array([0., 0., 20., 20.]), t=frange(0, 4, h))
    plot(x1[:,0],x1[:,1], 'b.')
    plot(x2[:,0],x2[:,1], 'r.')
    show()




if __name__ == "__main__":
    #schiefer_wurf()
    plot_wurf(0.1)
