# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 16:13:07 2013

@author: nchiapol
"""

from pylab import *

def calc_center(a, b):
    return (array(a)+array(b))/2.

def run():
    corners = [(0,0), (10,0), (5,sqrt(3)*5)]
    point = rand(2)*10
    
    x_values = []
    y_values = []
    while len(x_values) < 1e5:
#    for _ in range(100000):
        point = calc_center(point, corners[randint(3)])
        x_values.append(point[0])
        y_values.append(point[1])

    plot(x_values, y_values, 'k,')
    show()

if __name__ == "__main__":
    run()
