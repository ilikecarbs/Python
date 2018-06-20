# -*- coding: utf-8 -*-
"""
Jäger und Beute
---------------------------
Das Lotka-Voltera System beschreibt zwei Populationen von Tieren.
Population y sind Jäger, Population x deren Beute. Gibt es viele 
Beutetier so vermehren sich die Jäger und reduzieren damit die
Population der Beutetiere. Gibt es zuwenig Beutetiere, so verhungern
die Jäger und die Beutetiere können sich wieder stärker vermehren.
Die entsprechenden Differenzialgleichungen sind:
    dx/dt = \alpha*x-\beta*xy
    dy/dt = \gamma*xy-\delta*y

@author: 
@with:   
""" 

from numpy import array
from matplotlib.mlab import frange
from pylab import show, plot, semilogy

def dPdt(P, t):
    """ Differential-Gleichung für Lokta-Volterra
    
    Verwende einen deiner Solver um 
    die Entwicklung dieses Beispiels zu berechnen.

    Parameters
    ----------
    P : array of float
        Vektor der Beiden Populationen (X, Y) zum Zeitpunkt t
    t : float
        Zeitpunkt 
    
    Returns
    -------
    dPdt : array of float
        Änderungen der Komponenten von P zum Zeitpunkt t
        
    """
    alpha  = 2
    beta   = 0.1
    gamma  = 0.01
    delta  = 0.5
    dx = alpha*P[0]-beta*P[0]*P[1]
    dy = gamma*P[0]*P[1]-delta*P[1]
    return array([dx, dy])
    
def dPdt_rescaled(P, t):
    a  = 2/0.5
    dx = a*P[0]-P[0]*P[1]
    dy = P[0]*P[1]-P[1]
    return array([dx, dy])

def plot_populationen(h = 0.01):
    """ Zeichnet die Lösung für das Jäger - Beute Model
    
    Zeichen den Bestand beider Populationen gegen die zeit in eine Grafik.
    Evtl. ist es nützlich die Plots mit *semilogy* zu erstellen.
    
    """    
    
    from solvers_sol import euler
    from solvers_sol import rungekutta2
    from solvers_sol import rungekutta4
    tt=frange(0,30,h)
        
    P1 = euler(dPdt, y0 = array([100.,20.]), t=tt)
    P2 = rungekutta2(dPdt, y0 = array([100.,20.]), t=tt)
    P3 = rungekutta4(dPdt, y0 = array([100.,20.]), t=tt)

    semilogy(tt, P1[:,0], ',g')
    semilogy(tt, P1[:,1], ',r')
    
    semilogy(tt, P3[:,0], 'g')
    semilogy(tt, P3[:,1], 'r')


if __name__ == "__main__":
    plot_populationen(0.01)
    plot_populationen(0.001)
    show()
