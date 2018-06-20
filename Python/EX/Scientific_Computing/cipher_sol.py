# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 16:50:06 2013

@author: nchiapol
"""

ciphertext = "GROIK LOTMG TYOIN FARGT MCKOR KTYOK YGYYY INUTR GTMKH KOONX KXYIN CKYZK XGSAL KXATJ NGZZK TOINZ YFAZN ATJGY HAINJ GYONX KYINC KYZKX RGYMK LOKRO NXTOI NZJKT TKYCG XKTCK JKXHO RJKXT UINMK YVXIN KJGXO TATJC GYTZF KTHIN KXJGI NZKGR OIKUN TKHOR JKXAT JMKYV XINK"

from pylab import *

def get_index(letter):
    """ Bildet Buchstaben auf die Zaheln von 0 bis 26 ab 
    
    >>> get_index('E')
    4
    """

    return ord(letter)-ord('A')

def do_shift(letter, n):
    """ Rotiert einen Buchstaben um n Zeichen 

    >>> do_shift('A', 1)
    'B'
    """
    return chr( (get_index(letter) + n) % 26 + ord('A'))

def calc_stats(message):
    letter_stats = zeros(26)
    for letter in message:
        if letter != " ":
            letter_stats[get_index(letter)] += 1
    return letter_stats
        
def get_shift(message):
    letter_stats = calc_stats(message)
    E = argmax(letter_stats)
    return E-get_index('E')
    

def decrypt(message, shift):
    cleartext = ""
    for letter in message:
        if letter != " ":
            cleartext += do_shift(letter, -1*shift)
        else:
            cleartext += letter
    return cleartext

if __name__ == "__main__":
    key       = get_shift(ciphertext)
    cleartext = decrypt(ciphertext, key)
    print(cleartext)
