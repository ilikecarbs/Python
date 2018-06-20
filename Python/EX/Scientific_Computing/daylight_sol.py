# -*- coding: utf-8 -*-
"""

Today we will make a contour plot of the solar radiation on the world right now!

1)  Write the function get_time() to get the number of seconds since this
    year's Spring Equinox.

2)  Write the functions rot_x, _y, and _z that take an angle w and returns
    the appropriate rotation matrix.

3)  Write the function tageslicht that takes a given latitude and longitude 
    (in radians) on the world, and time (in seconds since the Spring Equinox 
    of the current year), and returns the sun's intensity between -1 and 1. 
    Positive intensity means daylight, while negative means darkness.

4)  Use the function plot_map to load the image of the world map provided. Use
    the contour function to plot contours of sunlight over this word map. Make
    sure to only draw contours for where it is day. Note: you need to specify  
    the length and width of the plot in pixels for this to work with the image.

5)  Optional: plot the sunlight for an arbitrary time.


@author:
@with:  

"""
import matplotlib.pyplot as plt
from matplotlib.image import imread
from pylab import imshow, contour, show
from time import time
from numpy import pi, arange, meshgrid, matrix, zeros, cos, sin, linspace

s_per_hour = 60*60
s_per_day  = 24*s_per_hour
s_per_year = 365.242*s_per_day
inclination = 23.5/180*pi
equinox13 = 1363780800  # Referenzzeit, time.time() um 12:00 UTC am 20. März 2013, 

def plot_map(bildname):
    """ Lädt die Karte 
    
    Returns
    -------
    map_dimensions : tuple of 2 ints (height, width)
        Grösse der geladenen Karte in Pixeln

    """
    fig2 = plt.figure()
    ax = fig2.add_subplot(1,1,1)
    img = imread(bildname)
    ax.imshow(img,extent=[-180,180,-90,90])
    ax.set_xticks(linspace(-180,180,13))
    ax.set_yticks(linspace(-90,90,7))
    return img.shape[0:2]

def tageslicht(longitude, latitude, t):
    """ Bestimmt die Sonneneinstrahlung für den gegeben Punkt
    
    Parameters
    ----------
    latitude : 
        Geografische Breite des gewünschten Punktes
    longitude : 
        Geografische Länge des gewünschten Punktes
    t : int
        Zeit in Sekunden, zu der Einstrahlung berechnet werden soll
        
    

    Returns
    -------
    sun_value : double
        Sonneneinstrahlung an diesem Punkt
    
    """
    delta = 1.*t/s_per_year*2*pi
    hour  = (t % s_per_day) / s_per_hour 
    sidereal_time = longitude+delta+hour/12.*pi
    sun_vec = matrix([[1., 0, 0]])
    # rotate
    sun_vec *= rot_y(latitude)
    sun_vec *= rot_z(sidereal_time)
    sun_vec *= rot_x(inclination)
    sun_vec *= rot_z(-delta)
    # return x-component
    return sun_vec[0,0]

def get_time():
    return (int(time()) - equinox13) % s_per_year

def rot_x(w):
    return matrix([[1,       0,      0],
                   [0,  cos(w), sin(w)],
                   [0, -sin(w), cos(w)]])

def rot_y(w):
    return matrix([[  cos(w), 0, -sin(w)],
                   [      0,  1,       0],
                   [  sin(w), 0,  cos(w)]])

def rot_z(w):
    return matrix([[  cos(w),  sin(w), 0],
                   [ -sin(w),  cos(w), 0],
                   [       0,      0,  1]])

t = get_time()    
N = 50
x = linspace(-pi,pi,N)
y = linspace(-pi/2,pi/2,N)
z = zeros((N,N))
for i in range(N):
    for j in range(N):
        z[j,i] = tageslicht(x[i],y[j], t)

map_dim = plot_map("plate-carree.png")

#Transformiert auf die Pixeldarstellung der Kartenprojektion
#x = linspace(0,map_dim[1],N)
#y = linspace(map_dim[0],0,N)
x = linspace(-180, 180,N)
y = linspace(-90,90,N)

lev = linspace(0,1,12)
contour(x,y,z,lev)
show()

