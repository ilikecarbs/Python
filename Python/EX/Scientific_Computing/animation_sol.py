# -*- coding: utf-8 -*-
"""
Aufgabe zum Thema Animationen: 

Aendere den Beispiel-Code unten so ab, dass die animation als *.mp4 file 
gespeichert wird.

Aendere den Beispiel-Code mit dem Doppelpendel in ein einfaches freies Pendel um.
Aendere den Beispiel-Code mit dem Doppelpendel in ein Pendel an einem Rad um.

Verwende deine eigene Runge Kutta Methode anstelle von integrate.odeint um die 
Gleichungen zu lösen.


Quellenangabe:
Dieser Code basiert auf folgendem Beispiel zu matplotlib.animation:
    http://matplotlib.sourceforge.net/examples/animation/double_pendulum_animated.html
Der Original Code produziert dieselbe Animation auf einem etwas anderen Weg:
    Es wird ein Grafik-Objekt (line) erzeugt und für jeden Frame werden dann
    die Punkte dieses Objekts aktualisiert.

"""

from numpy import sin, cos, pi, array
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import matplotlib.animation as animation

G =  9.8 # acceleration due to gravity, in m/s^2
L1 = 1.0 # length of pendulum 1 in m
L2 = 1.0 # length of pendulum 2 in m
M1 = 1.0 # mass of pendulum 1 in kg
M2 = 1.0 # mass of pendulum 2 in kg
b = 2.0  #radius des Rades
omega = 2*pi #Winkelgeschwidigkeit

def derivs(state, t): # Bewegungsgleichungen Doppelpendel:

    dydx = np.zeros_like(state)
    dydx[0] = state[1]

    del_ = state[2]-state[0]
    den1 = (M1+M2)*L1 - M2*L1*cos(del_)*cos(del_)
    dydx[1] = (M2*L1*state[1]*state[1]*sin(del_)*cos(del_)
               + M2*G*sin(state[2])*cos(del_) + M2*L2*state[3]*state[3]*sin(del_)
               - (M1+M2)*G*sin(state[0]))/den1

    dydx[2] = state[3]

    den2 = (L2/L1)*den1
    dydx[3] = (-M2*L2*state[3]*state[3]*sin(del_)*cos(del_)
               + (M1+M2)*G*sin(state[0])*cos(del_)
               - (M1+M2)*L1*state[1]*state[1]*sin(del_)
               - (M1+M2)*G*sin(state[2]))/den2

    return dydx


def freies_pendel(X, t):
    """ Berechnet die Ableitungen für ein freies Pendel
    
    Für ein freies Pendel gilt
      H = 1/2*m*v^2 - m*g*l*cos(alpha)
    Dabei ist v die Geschwindigkeit des Gewichts mit Masse m
    l ist die Länge und alpha die Auslenkung des Pendels.
    Unsere beiden Koordinaten sind v und alpha. Mit Hamiltons
    Formeln erhalten wir daraus:
      d(alpha)/dt = v/l
            dv/dt = -g*sin(alpha)
    Definiere die entsprechenden Funktion und implementiere 
    dann die *zeichenen*-Funktion. 
    
    """
    return array([X[1]/L1, -G*sin(X[0])])

def pendel_an_rad(X, t):
    """ Berechnet die Ableitungen für ein angetriebenes Pendel 
    
    Wir bestigen unser freies Pendel nun an einem Rad, das 
    sich mit konstanter Geschwindigkeit dreht. Dieses System 
    wird durch folgende Gleichungen beschrieben:
      d(alpha)/dt = v/l
            dv/dt = -g*sin(alpha) - b/l*omgea^2*sin(alpha-omega*t)
    Wobei b der Radius und omega die Wingkelgeschwindigkeit des
    Rades sind.
    
    """
    e = b/L1*omega*omega
    return array([X[1]/L1, -G*sin(X[0])-e*sin(X[0]-omega*t)])

# create a time array from 0..100 sampled at 0.1 second steps
dt = 0.05
t = np.arange(0.0, 20, dt)

# th1 and th2 are the initial angles (degrees)
# w10 and w20 are the initial angular velocities (degrees per second)
th1 = 120.0
w1 = 0.0
th2 = -10.0
w2 = 0.0

rad = pi/180

# initial state, Doppelpendel:
#state = np.array([th1, w1, th2, w2])*pi/180.
##einfaches Pendels oder Pendel am Rad
state = np.array([th1, w1])*pi/180.


# integrate your ODE using scipy.integrate.
from solvers_sol import rungekutta
#y = integrate.odeint(derivs, state, t)
#y = integrate.odeint(freies_pendel, state, t)
#y = integrate.odeint(pendel_an_rad, state, t)
y = rungekutta(pendel_an_rad, state, t)

## Doppelpendel
#x1 = L1*sin(y[:,0])
#y1 = -L1*cos(y[:,0])
#x2 = L2*sin(y[:,2]) + x1
#y2 = -L2*cos(y[:,2]) + y1

##Pendel am Rad
cx = b*sin(omega*t)
cy = -1*b*cos(omega*t)
x1 = cx+L1*sin(y[:,0])
y1 = 1*(cy-L1*cos(y[:,0]))
##


fig = plt.figure()

def init():
## Doppelpendel    
#    L = L1+L2
## einfaches Penel
#    L = L1
## Pendel am Rad
    L  = b + L1
    plt.plot([-L,L], [-L,L], ',')

def animate(i):
#    Doppelpendel    
#    thisx = [0, x1[i], x2[i]]
#    thisy = [0, y1[i], y2[i]]
##einfaches Pendel
#    thisx = [0, x1[i]]
#    thisy = [0, y1[i]]
##Pendel am Rad
    thisx = [0,cx[i], x1[i]]
    thisy = [0,cy[i], y1[i]]

    return plt.plot(thisx, thisy, 'bo-', lw=2)

ani = animation.FuncAnimation(fig, animate, np.arange(1, len(y)),
    interval=60, blit=True, init_func=init())
#ani.save('double_pendulum.mp4', fps=15, clear_temp=True)

plt.show()
