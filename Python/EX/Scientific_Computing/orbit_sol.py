# -*- coding: utf-8 -*-
"""
Planetare Umlaufbahnen
-------------------------
Die Differentialgleichungen für ein Teilchen i im Gravitationspotential lauten:

    dxi/dt = v_xi
    dyi/dt = v_yi
    dv_xi/dt = - Sum_(j!=i) (G mj /rij^3 * rij_x)
    dv_yi/dt = - Sum_(j!=i) (G mj /rij^3 * rij_y)
    
    mit rij = ri - rj
    
Verwende die Runge Kutta 4 Methode um gegenbene Anfangsbedingungen für das 3-Teilchen
System Sonne, Erde und Jupiter zu integrieren. 

Implementiere dann die leapfrog methode und untersuche die Unterschiede zur Runge Kutta Methode.
    

""" 

from pylab import plot, show
from numpy import array
from matplotlib.mlab import frange
import math



def dXdt(X, t):
    """ Differential-Gleichung für N-Teilchen system

    Parameters
    ----------
    X : array of float
        Vektor mit (x0, y0, vx0, vy0, m0, x1, y1, vx1, vy1, m1, ... , x(n-1), y(n-1), vx(n-1), vy(n-1), m(n-1)) zum Zeitpunkt t
    t : float
        Zeitpunkt 
    
    Returns
    -------
    dXdt : array of float
        Änderungen der Komponenten von X zum Zeitpunkt t
        
    """
    result = [0] * len(X) 
    N = len(X) / 5
    G = 1
    
    for i in range(N):
        dx   = X[i*5 + 2]
        dy   = X[i*5 + 3]
        
        ax = 0.0
        ay = 0.0
        
        xi = X[i*5]
        yi = X[i*5 + 1]
        mi = X[i*5 + 4]
        for j in range(N):
            if (i != j):
	       
                xj = X[j*5]
                yj = X[j*5 + 1]       
                mj = X[j*5 + 4]
	       
                xij = xj - xi
                yij = yj - yi
	       
                r2 = xij * xij + yij * yij
                r = math.sqrt(r2)
                ir3 = 1.0 / (r * r2)
                s = G * mj * ir3
	       
                ax += s * xij
                ay += s * yij
        dv_x = ax
        dv_y = ay
        
        result[i*5] = dx
        result[i*5 + 1] = dy
        result[i*5 + 2] = dv_x
        result[i*5 + 3] = dv_y
        result[i*5 + 4] = 0.   
        
    return array(result)
  
  
def leapfrog(func, y0=array([0.]), t=array([0, 0.1, 0.2])):
    """
    Implementiere die leapfrog Methode
    
    Parameters
    ----------
    func : fuction
        Python-funktion welche die Differential-Gleichung beschreibt
    y0 : array of float
        Startwerte für die Parameter der Differential-Gleichung
        y0 = (x0, y0, vx0, vy0, m0, x1, y1, vx1, vy1, m1, ... , x(n-1), y(n-1), vx(n-1), vy(n-1), m(n-1))
    t : array of float
        Zeitpunkte für die numerische Integration
    
    Returns
    -------
    y_list : arrays of float
        Liste mit den Näherungs-Werten von y, ein Zeitschritt pro Zeile

    """
    y = y0
    y_list = [1*y]
    
    N = len(y0) / 5
    
    #first half step
    h = t[1]-t[0]
    dy = 0.5 * h * func(y, t[0])
    for i in range(N):
        y[i*5 +2] += dy[i*5 + 2]
        y[i*5 +3] += dy[i*5 + 3]
    
    for k in range(len(t)-1):
        h = t[k+1]-t[k]
        for i in range(N):
            y[i*5] += h *y[i*5 + 2]
            y[i*5 + 1] += h *y[i*5 + 3]
        dy = h*func(y, t[k+1])
        for i in range(N):
            y[i*5 + 2] += dy[i*5 + 2]
            y[i*5 + 3] += dy[i*5 + 3]
        y_list.append(1*y)
        
    return array(y_list)
    


def plot_orbit(h=0.1):
    '''
    Integriere die gegebenen Anfangsbedingungen sowohl mit Runge Kutta 4 und mit
    Leapfrog. Untersuche die stabilität des Systems nach einige Umläufen.
    
    Die Masse der Erde beträgt 3.0024584e-6 Sonnenmassen.
    die Masse des Jupiters  beträgt 0.0009542 Sonnenmassen.
    '''

    from solvers_sol import rungekutta4
    from solvers_sol import rungekutta2
    from solvers_sol import euler
    x1 = leapfrog(dXdt, y0 = array([1., 0., 0., 1., 3.0024584e-6, 0., 0., 0., -0.00041, 1., 5.203, 0., 0., 0.43, 0.0009542 ]), t=frange(0, 1620, h))
    plot(x1[:,0],x1[:,1], 'b,')
    plot(x1[:,5],x1[:,6], 'g,')
    plot(x1[:,10],x1[:,11], 'b,')

    x2 = rungekutta4(dXdt, y0 = array([1., 0., 0., 1., 3.0024584e-6, 0., 0., 0., -0.00041, 1., 5.203, 0., 0., 0.43, 0.0009542 ]), t=frange(0, 1620, h))
    plot(x2[:,0],x2[:,1], 'r,')
    plot(x2[:,5],x2[:,6], 'g,')
    plot(x2[:,10],x2[:,11], 'r.')
    show()




if __name__ == "__main__":

    plot_orbit(0.3)

