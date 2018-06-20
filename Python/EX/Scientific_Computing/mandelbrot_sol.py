# -*- coding: utf-8 -*-
"""
Die Mandelbrot-Menge ist eines der bekanntesten Fraktale. Wie sie definiert 
ist, wurde in der Vorlesung gezeigt.

1) Zeichne die Mandelbrot-Menge. 
2) Passe deinen Code so an, dass du beliebige Ausschnitte der Menge 
   zeichnen kannst.
3) Zusatzaufgabe: Passe deinen Code so an, dass für Punkte ausserhalb der
   Mandelbrot Menge (d.h. Divergenzpunkte) die Divergenz-Geschwindigkeit farbig
   dargestellt wird.
   
   
Tipp
-----
Zur Berechnung der Mandelbrot-Menge musst bestimmen ob die Reihe z=z**2 + c
konvergiert. Sobald der Absolutbetrag einer Reihe grösser als zwei wird, ist 
schon sicher, dass diese Reihe divergiert. Die Anzahl Iterationen, die es dafür
braucht, ist ein einfaches Mass für die Divergenz-Geschwindigkeit. Bleibt der
Betrag auch nach 500 Iterationen kleiner als 2, kann man annehmen, dass diese
Reihe konvergiert, dh. ihr Startpunkt ist in der Mandelbrot-Menge.

@author: nchiapol & diemand
"""
import matplotlib.cm as cm
from math import *
from pylab import *

limit      = 2.0
N          = 500
nx         = 640
ny         = 520

def pcn(c, n):
    if n == 0:
        return 0j+c
    z = pcn(c,n-1)
    if abs(z) >= limit:
        return limit
    return z**2+c

def check(c):
    ret = pcn(c, N)
    if abs(ret) < limit:
        return True
    return False

def run(wd = [-2, 1, -1, 1]):
    x_steps = linspace(wd[0], wd[1], nx)
    y_steps = linspace(wd[2], wd[3], ny)
    x_arr = []
    y_arr = []
    for x in x_steps:
        for y in y_steps:
            in_set = check(x+1j*y)
            if in_set:
                x_arr.append(x)
                y_arr.append(y)
    f1 = figure("Mandelbrot set, black and white " + str(wd))
    plot(x_arr, y_arr,'k,')
    axis('equal')
    return f1
    
def get_color(c):
    iter = 0
    z = c
    while (iter < N) and (abs(z) < 2):
        z = z**2+c
        iter += 1 
    return iter

def run_color(wd = [-2, 1, -1, 1]):
    x_steps = linspace(wd[0], wd[1], nx)
    y_steps = linspace(wd[2], wd[3], ny)
    img = zeros([ny,nx])
    for i in range(nx):
        for j in range(ny):
            img[j,i] = get_color(x_steps[i]+1j*y_steps[j])            
    f2 = figure("Mandelbrot set, color " + str(wd))
    imshow(img, cmap=cm.jet,extent=[wd[0],wd[1],wd[2],wd[3]], origin='lower')
    return f2

    
if __name__ == "__main__":
   #run()
   run([-0.75, -0.65, 0.325, 0.38])
   #run_color()   
   run_color([-0.75, -0.65, 0.325, 0.38])
   show()
    