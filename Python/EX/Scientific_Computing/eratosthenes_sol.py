# -*- coding: utf-8 -*-
"""

Das Sieb des Eratosthenes ist ein seit der Antike bekanntes Verfahren
um Primzahlen zu finden. Im Folgenden implementieren wir dieses 
Verfahren um Primzahlen zu finden. Das zweite Verfahren folgt 
direkt der Definition einer Primzahl und probiert einfach aus ob
ein Kandidat durch kleinere Zahlen (Primzahlen genügt) teilbar ist.
  
Was sind die Vor- und Nachteile des direkten Verfahrens gegenüber
dem Sieb des Eratosthenes? Wie schnell finden die beiden Methoden alle
Primzahlen kleiner als 100? Und alle kleiner als 100'000?

Überprüfe den Primzahlensatz #Primzahlen(<n) ~ n /ln n mit einer Grafik.

Berechne Riemannsche Zeta-Funktion zeta(2) mit der Euler-Produkt Formel
und vergleiche das Resultat mit (pi**2)/6.

@author: 
@with:   
""" 

from numpy import pi
import timeit as ti
from functools import reduce

def sieb(n_max):
    """ Bestimmt alle Primzahlen < n_max
    
    Implementiere das Sieb in seiner ursprünglichen Form:
    http://de.wikipedia.org/wiki/Sieb_des_Eratosthenes
    
    Tipps:
    - Erstelle eine Liste mit n_max Einträgen mit dem Wert True
      ([True, True, ... ]). Der Index eines Listen-Elements ist 
      deine Zahl, der Wert des Elements gibt an ob es sich um 
      eine Primzahl handeln kann.
    - Verwende eine zweite Liste um laufend gefundene Primzahlen 
      zu speichern.
    
    Parameters
    ----------
    n_max : int
        Obergrenze für die Primzahlsuche
    
    Returns
    -------
    primes : list of int
        Liste der gefundenen Primzahlen
    
    Examples
    ---------
    >>> simple(10)
    [2, 3, 5, 7]
    
    """
    sieve = [True]*n_max
    sieve[0:1] = [False]
    
    primes = []
    for n_test in range(2,n_max):
        if sieve[n_test]:
            primes.append(n_test)
            not_prime = 2*n_test
            while not_prime < n_max:
                sieve[not_prime] = False
                not_prime += n_test
    return primes

def direkt(n_primes):
    """ Berechnet die ersten n_primes Primzahlen.
   
    Implementiere den folgenden Algorithmus:
    - Erstelle eine Listen in der du deine Primzahlen speicherst
    - Nimm n = 2, 3, 4, ... und teste jeweils ob sich n durch 
      eine der bisher gefundenen Primzahlen teilen lässt. 
    - Falls nicht hast du einen nächste Primzahl gefunden. Füge 
      sie zur Liste hinzu.

    Parameters
    ----------
    n_primes : int
        Anzahl von Primzahlen die du bestimmen möchtest.

    Returns
    -------
    primes : list of int
        Liste der gefundenen Primzahlen
    
    Examples
    ---------
    >>> direkt(4)
    [2, 3, 5, 7]
    
    """
    sieve = [2]
    n = 3 
    while len(sieve)<n_primes:
        is_prime = True
        for p in sieve:
            if p*p > n:
                break
            if n % p == 0:
              is_prime = False
              break
        if is_prime:
            sieve += [n]            
        n += 2
    return sieve


def simple_draw(primes):
    """ Erstellt eine Grafik der Primzahl p_k gegen k*log(p_k)
    
    Versuche Array-Operationen und nicht for-loops zu verwenden.
    
    Parameters
    ----------
    primes : list of int
        Liste der ersten n Primzahlen
    
    """
    from numpy import array, log
    from pylab import axis, plot, xlabel, ylabel, show
    
    x = array(primes)
    y = array(range(x.size))*log(x)

    mx = max(y[-1], x[-1])
    axis(xmin=0,ymin=0,xmax=mx,ymax=mx)
    plot(x,y,'b.')
    xlabel('P_k')
    ylabel('k ln(P_k)')
    show()

def zeta(z, primes):
    """ Bestimmt \zeta(z) mit dem Euler-Produkt basierend auf den übergebenen Primzahlen

    Parameters
    ----------
    z : int
        Argument der Zeta-Funktion
    primes : list of int
        Liste der ersten n Primzahlen

    Returns
    -------
    zeta : double
        Wert von \zeta(z)
        
    Examples
    ---------
    >>> round(zeta(2, sieb(1000)), 2)
    1.64
    
    """
    zeta = reduce(lambda p,t: p/(1.-t**(-z)), primes,1)
    return zeta
        
if __name__ == "__main__":
    nmax = 100000
    sim = sieb(nmax)
    pf= len(sim)
    #print sim
    print(pf)
    print(ti.timeit("sieb(nmax)",setup="from __main__ import sieb, nmax",number=10))
    #print direkt(pf)
    print(ti.timeit("direkt(pf)",setup="from __main__ import direkt, pf",number=10))

   # simple_draw(direkt(100))
    print(zeta(2,sieb(1000)))
    print((pi**2)/6)
