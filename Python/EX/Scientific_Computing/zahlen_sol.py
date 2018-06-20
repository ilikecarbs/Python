# -*- coding: utf-8 -*-
"""
Aufgaben zur Darstellung von Zahlen 

@author: 
"""

import sys ## u.a. verschiedene Informationen über die Zahlen-Darstellungen

""" Integer 
Implementiere die beiden Hilfsfunktionen unten und teste dann folgende 
Dinge:
    - Vergleiche den Wert von largest_int() mit sys.maxint 
    - Was passiert, wenn du 1 zu largest_int() addierst?
    - Welches ist der kleinste (stärkst negative) Integer? 
      (verwende largest_int() und dein Wissen über die Darstellung von int)
"""

def show_int(x, vspace=True):
    """ gibt Informationen zu einem Integer aus 

    Parameters
    ----------
    x : int
       Input
    vspace : boolean
       wenn True: zusätzliche Leerzeile vor Ausgabe
    
    """
    if type(x) == str:
        binary  = x
        numeric = int(x, 2)
    else:
        binary  = int_binary(x)
        numeric = x
    if vspace:
        print
    print numeric
    print binary
    print type(numeric)

def int_binary(x):
    """ wandelt x in seine Binär-Darstellung um 

    Parameters
    ----------
    x : int
       Input
       
    Returns
    -------
    s : string
       Binär-Darstellung von x
    
    """
    size_int   = int.bit_length(sys.maxint)+1
    if type(x) == long:
        size_int *= 2
    b_string = "" 
    for i in range(size_int):
        b_string = str(x >> i & 0b1) + b_string
    return '0b'+b_string

def size_of_int():
    """ Bestimmt die Anzahl Bits eines normalen Integers
    
    Addiere so lange Potenzen von 2 bis deine Zahl nicht mehr 
    type() 'int', sondern 'long' hat
    
    Returns
    -------
    i : int
       Anzahl Bits eines int (32, 64)
    
    """
    x     = 0b1 
    i     = 1
    while type(x) == int:
#        x = x << 1  ## fancy way
        x += 2**(i)
        i += 1
    return i

def largest_int():
    """ Berechnet die grösste noch als int darstellbare Zahl 
    
    Returns
    -------
    x : int
       Grösster Integer
       
    >>> largest_int() == sys.maxint
    True
    
    """
    size_int = size_of_int()
    largest = '0b0'+'1'*(size_int-1)
    return int(largest, 2)



""" Darstellungen in verschiedenen Basen """

def new_base(v, b, explain = False):
    """ berechnet Darstellung eines Int in neuer Basis
    
    Mögliches Vorgehen
    ------------------
    1) bestimme das e für den grössten benötigten b**e Term
    2) beginnend mit dem grössten e berechne n in n * b**e 
    3) ersetze v durch den noch nicht dargestellten Rest

    Parameter
    ---------
    v : int 
       Input
    b : int
       gewünschte Basis

    Returns
    -------
    in_base : string
       Darstellung von v in Basis b
    
    """
    if b > 10 or b < 2 or type(b) != int:
        print "Basis nicht unterstützt"
        return
    order = 0
    while v % b**order != v:
        order += 1
    order -= 1
    in_base = ""
    for o in range(order,-1,-1):
        factor = b**o
        in_base += str(v/factor)
        if explain:
            in_base += " * "+str(b)+"**"+str(o)
            if o != 0:
                in_base += " + "
        v %= factor
    return in_base



""" Floats
Implementiere die folgende Funtionen und demonstriere damit die Probleme und 
Grenzen der Gleitkomma-Darstellung.

"""

def show_float(f, vspace=True):
    """ gibt Informationen zu einem float aus 

    Parameters
    ----------
    x : float
       Input
    vspace : boolean
       wenn True: zusätzliche Leerzeile vor Ausgabe
    
    """
    size_of_double = 64
    if vspace:
        print 
    print f
    print ('%0.40f' % f)
    f_hex = f.hex()
    sign  = '0'
    start = 4
    if f_hex[0] == '-':
        sign = '1'
        start += 1
    
    fraction = f_hex[start:start+13]
    size_of_frac = sys.float_info.mant_dig-1
    frac = int_binary(eval('0x'+fraction))[-size_of_frac:]
    
    exponent = f_hex[start+14:]
    i_exp = int(exponent)
    bias = abs(sys.float_info.min_exp)+2  ## +2 due to stupid min_exp definition  
                                          ## http://forum.dlang.org/thread/hlsgd0$1i2f$1@digitalmars.com
    if i_exp == -1*bias+1:
        exp = '0'
    else:
        exp = bin(i_exp+bias)[2:]
    while len(exp) < (size_of_double-size_of_frac-1):
        exp = "0"+exp
    print sign, exp, frac
        

def size_of_mantissa():
    """ bestimmt die Anzahl Bits der Mantissa 
      
    Returns
    -------
    x : int
       Grösste der Mantissa
       
    >>> size_of_mantissa() == sys.float_info.mant_dig
    True
    
    """
    f = 1.
    i = 1
    while f != f+1.:
        i += 1
        f *= 2
    return i-1
    """
    0b0.001  * 2**3 =  1
    --
    0b1.000  * 2**4 = 16
    0b0.0001 * 2**4 =  1
    --
    """
    
def int_add_check():
    """ berechnet die Integer um f = 2**size_of_mantissa()
    
    Für d in [-n,n] berechne f+d und kontrolliere die Resultate.
    
    """
    print 
    print "Addition um int-Grenze"    
    f = 2.**size_of_mantissa()
    show_float(f, False)
    for d in range(-5,5):
        inc = f+d
        print ('%2d: %.1f' % (d, inc)), (inc-f) == d
    
def one_plus_epsilon():
    """ findet das grösste epsilon = 1/2**n so dass 1 + epsilon = 1
    
    Returns
    -------
    epsilon : float
       epsilon = 1/2**n so dass 1 + epsilon = 1
    
    """
    epsilon = 1.
    while 1 != 1+epsilon:
        epsilon /= 2
    return epsilon*2
    
def zero_epsilon():
    """ findet das grösste epsilon, so dass epsilon/2 == 0 
      
    Returns
    -------
    epsilon : float
       kleinster darstellbarer float
      
    """
    epsilon = 1.
    while 0 != epsilon/2:
        epsilon /= 2
    return epsilon
    
def size_of_exponent():
    """ bestimmt die Anzahl Bits des Exponenten
    
    berechne 2.**exponent für Exponenten, die in Basis 2 eine immer 
    grössere Anzahl Stellen benötigen. Kann der Exponent nicht mehr
    dargestellt werden, wird ein OverflowError erzeugt. Du kannst 
    diesen mit try ... except ... abfangen.
    
    Returns
    -------
    i : int
       Anzahl Stellen des Exponenten
    
    """
    i = 0
    i_exp = 0
    while True:
        print 
        try:
            f = 2.**i_exp
            show_float(f)
        except OverflowError:
            break            
#        i_exp += 2**i
        i_exp = 2**i
        i += 1
    return i

def compare_decimal():
    """ Illustriert den Effekt des decimal-Moduls 
    
    Zeige, dass mit decimal.Dezimal die Probleme der float 
    nicht auftreten.
    """
    from decimal import Decimal
    zero_e = zero_epsilon()
    one_e  = one_plus_epsilon()
    d_ze = Decimal(zero_e)
    d_oe = Decimal(one_e)
    print "normal float:"
    print "e:  ", zero_e
    print "e/2:", zero_e/2
    print "1+e:", 1+one_e
    print "decimal float"
    print "e:  ", d_ze
    print "e/2:", d_ze/2
    print "1+e:", 1+d_oe


if __name__ == "__main__":
## das ist ein 64-bit system
#    print size_of_int()    
## kontrolliere den grössten int
#    print
#    print sys.maxint
#    show_int( largest_int(), False )
#    print "+1", type(largest_int()+1)  # long
## kontrolliere den kleinsten int
#    show_int( -1*largest_int()-1 )    
#    print "+1", type( -1*largest_int()-2 )
## python int(b, 2) interpetieret b als unsigned int -> verwendet long
#    print int_binary(-1)
#    show_int( int(int_binary(-1), 2) )

## neue Basis
#    print
#    n = 11236
#    b = 5
#    neu = new_base(n, b)
##    neu = new_base(n, b, True)
#    print n, "in Basis", b, ":", neu

## float Grössen
    print size_of_mantissa()
    print sys.float_info.mant_dig 
    print size_of_exponent()
## kontrolliere ab wann +1 nicht mehr funktioniert
#    int_add_check()
## binär-Darstellung verschiedener interessanter Zahlen
#    show_float(zero_epsilon())    
#    show_float(1.25)
#    show_float(one_plus_epsilon())
#    show_float(one_plus_epsilon()+1)
#    show_float(one_plus_epsilon()/2+1)
## tests mit decimal-Module:
    compare_decimal()

