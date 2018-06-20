# -*- coding: utf-8 -*-
# pylint: disable-msg=C0103
"""
In der Vorlesung hast du 2 verschiedene Methoden kennengelernt um 
Differentialgleichungen zu lösen. 
- Euler
- Runge-Kutta

Implementiere diese Metoden als solver Funtionen, die von einen 
anderen Python Skript aufgerufen werden können.



Python-Hinweis
--------------
Wenn du ein Array zu einer Liste hinzufügtst, wird manchmal keine
Kopie erstellt. Wenn du das Array veränderst, ändert sich
auch der Eintrag in der Liste. Um das zu verhindern, kannst du 
das Array bei beim Hinzufügen *1 rechnen:
>>> x = array([42])
>>> ohne_mal = [x]
>>> mit_mal = [1*x]
>>> x[0] = -1
>>> print ohne_mal[0][0]
-1
>>> print mit_mal[0][0]
42


@author: 
@with:   
""" 

from numpy import array

def euler(func, y0=array([0.]), t=array([0, 0.1, 0.2])):
    """ berechnet eine Näherungs-Lösung für die Differential-Gleichung func
    
    Implementiere die Euler Methode
    
    Parameters
    ----------
    func : fuction
        Python-funktion welche die Differential-Gleichung beschreibt
    y0 : array of float
        Startwerte für die Parameter der Differential-Gleichung
    t : array of float
        Zeitpunkte für die numerische Integration
    
    Returns
    -------
    y_list : arrays of float
        Liste mit den Näherungs-Werten von y, ein Zeitschritt pro Zeile

    """
    y = y0
    y_list = [1*y] # or x.copy() but 1*x works for float and array
    for i in range(len(t)-1):
        h = t[i+1]-t[i]
        dy = h*func(y, t[i])
        y += dy
        y_list.append(1*y)
    return array(y_list)


def rungekutta2(func, y0, t):
    y = y0
    y_list = [1*y] # or x.copy() but 1*x works for float and array
    for i in range(len(t)-1):
        h = t[i+1]-t[i]
        y += h*func(y+0.5*h*func(y, t[i]), t[i] + 0.5*h)
        y_list.append(1*y)
    return array(y_list)
    
def rungekutta4(func, y0, t):
    y = y0
    y_list = [1*y] # or x.copy() but 1*x works for float and array
    for i in range(len(t)-1):
        h = t[i+1]-t[i]
        f1 = func(y, t[i])
        f2 = func(y+f1*h/2, t[i] + 0.5*h)
        f3 = func(y+f2*h/2, t[i] + 0.5*h)
        f4 = func(y+f3*h, t[i] + h)
        y += (f1 + 2*f2 + 2*f3 + f4)*h/6.
        y_list.append(1*y)
    return array(y_list)

def rungekutta(func, y0=array([0.]), t=array([0, 0.1, 0.2])):
    """ berechnet eine Näherungs-Lösung für die Differential-Gleichung func
    
    Implementiere eine der beiden Runge-Kutta Methoden und untersuche 
    deren Effekt auf auf die Beispiele.
    
    Parameters
    ----------
    func : fuction
        Python-funktion welche die Differential-Gleichung beschreibt
    y0 : array of float
        Startwerte für die Parameter der Differential-Gleichung
    t : array of float
        Zeitpunkte für die numerische Integration
    
    Returns
    -------
    y_list : arrays of float
        Liste mit den Näherungs-Werten von y, ein Zeitschritt pro Zeile
    
    
    """
    return rungekutta4(func, y0, t)
