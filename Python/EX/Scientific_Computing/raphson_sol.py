# -*- coding: utf-8 -*-
"""
Mit dem Newton-Raphson Iterations-Verfahren können Nullstellen 
einer beliebigen Funktion bestimmt werden. Verwende das 
Newton-Raphson Verfahren um folgende Aufgaben zu lösen.

@author:   
""" 

from pylab import *

def parabola_zero():
    """ Berechnet die Nullstelle von f(x) = (x-3)**2.
    
    - Definiere zwei Hilfsfunktionen, eine für f(x) und eine für die 
      Ableitung f_prime(x). (Berechne die Ableitung auf Papier).
    - Brich die Iteration ab, wenn f(x) < epsilon ist
      (für ein sinnvoll gewähltes epsilon)
    - runde dein Resultat auf 2 signifikante Stellen.
    
    Returns
    -------
    x : double
       x-Wert der Nullstelle
    
    >>> parabel_zero()
    3.0
    
    """
    def f(x):
        return 1.*x*x-6*x+9

    def f_prime(x):
        return 2.*x-6

    x       = 0
    epsilon = 1e-5
    while abs(f(x)) > epsilon:
        x -= f(x)/f_prime(x)

#    x_vals = frange(0,6, 0.1)
#    plot(x_vals,f(x_vals))
#    show()

    return round(x, 2)

def raphson(f, x):
    """ Berechnet die Nullstelle der funktion f in der Nähe von x.
   
    - Definiere eine Hilfsfunktion, welche die Ableitung mit Hilfe des 
      Differenten-Quotienten bestimmt.
    - Berechne das Resultat so genau wie möglich. (Iteriere, bis du 
      bei der Berechnung des neuen x-Werts einen Underfolw erhältst.)
    

    Parameters
    ----------
    f : function
       Funktion deren Nullstelle bestimmt werden soll
    x : double
       Startwert für die Iteration

    Returns
    -------
    x : double
       x-Wert der Nullstelle
    
    >>> round(raphson(cos, 1), 10)
    1.5707963268
    
    """
    def slope(f, x):
        delta = 1e-8
        return (f(x+delta)-f(x))/delta

    x_old = [x+1, x+1]  # verhindert "oszillationen"
    while x_old[0] != x and x_old[1] != x:
        x_old[0] = x_old[1]
        x_old[1] = x
        x -= f(x)/slope(f, x)
    
    return x
    
def n_root(a, n=2):
    """ Berechnet die n-te Wurzle von a.
   
    Wenn du deine Funktion verschiebst, kannst du mit dem Newton-Raphson-
    Verfahren den x-Wert für beliebige Funktionswerte f(x) bestimmen.
    Verwende das um, die n-te Wurzel von x zu berechnen.
    Implementiere deine Lösung so, dass es auch für negativen a ein möglichst
    sinnvolles Resultat liefert.
    Tipp: verwende die raphson-Funktion oben

    Parameters
    ----------
    a : double
       Wurzelbasis
    n : int 
       Wurzelexponent

    Returns
    -------
    x : double
       Wurzel des Eingabewerts
    
    >>> n_root(16)
    4.0
    >>> n_root(8,3)
    2.0
    
    """
    if a < 0 and n % 2 == 0:
        print("imaginäre Lösung")
    
    def func(x, p1=n, p2=a):           # erklären: Werte via default übergeben
        return pow(x,p1)-p2
    
    return raphson(func, a/2.)
      

def polynom_zero():
    
    """ Zusatzaufgabe: Berechne die Nullstelle von f(x) = x**3 - 2*x + 2.
    
    - Definiere zwei Hilfsfunktionen, eine für f(x) und eine für die 
      Ableitung f_prime(x). (Berechne die Ableitung auf Papier).
    - Brich nach 10 Iterationen ab, und speichere die Resultate für 400 
      Startwerte x0 zwischen -2 und 2
    - Plotte diese Resultate als Funktion vom Startwert x0.
    - Dasselbe mit 11 Iterationen, und im selben Plot darstellen.    
        
    """
    pass    
    
    
    def f(x):
        return 1.*x*x*x-2.*x+2.

    def f_prime(x):
        return 3.*x*x-2.
        
    x0 =  frange(-2., 2., 0.01)
    x10 = zeros(size(x0))
    x11 = zeros(size(x0))    

    for i in range(size(x0)):
        x = x0[i]    
        for j in range(10):
            x -= f(x)/f_prime(x)  
        x10[i] = x
        x -= f(x)/f_prime(x)
        x11[i] = x
    
    plot(x0,x10,'bo')
    plot(x0,x11,'r.')
    show()
    return
    
if __name__ == "__main__":
    print(parabola_zero())
    print(raphson(cos, 1))
    print(('%.60f' % n_root(3, 2)))  # erklären: String formattieren
    polynom_zero()