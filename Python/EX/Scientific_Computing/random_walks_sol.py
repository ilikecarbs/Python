# -*- coding: utf-8 -*-
"""
(1) Erzeuge 20'000 eindimensionale Random Walks als Funktion der Zeit t.
    Jeder Walk startet bei 0.0 und ist 100 Schritte lang. Die Schritte sind
    
        a) step = +1.0 oder step = -1.0, mit je 50% Wahrscheinlichkeit.
        
        b) step = np.random.normal(0.0,1.0), dh. Normalverteilt um 0.0 mit
            Standartabweichung sigma=1.0
            
    Plotte die ersten 50 Walks als Beispiele und im selben Plot jeweils auch
    sigma*sqrt(t), wobei t von 0 bis 100 geht.

    
(2) Die Endpunkte der 20'000 random walks von 1) werden gespeichert und als
    Histogram geplotted (mit matplotlib.pyplot.hist).
    
    Plotte darueber auch die zu diesem Histogram passende Normalverteilung.

Hinweis zur Darstellung der Lösung: Die vier Plots von 1) und 2), jeweils
Varianten a) und b), können mit subplot(2,2,1) bis subplot(2,2,4) in einer
figure dargestellt werden. Verwende title('abc..'), xlabel('abc..') und
ylabel ('abc..') um jeden subplot klar zu beschriften.

""" 
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from numpy import pi,arange
from pylab import *


N = 100
N2 = 20000
t = arange(N)
a = np.zeros(N)
b = np.zeros(N)
endpoints_a = np.zeros(N2)
endpoints_b = np.zeros(N2)
sigma = 1.0;
figure(figsize=(10.0,10.0))
for j in range(N2):
    for i in range(1,N):      
        step = np.random.normal(0.0,sigma)
        b[i] = b[i-1] + step    
        if(step > 0): step = sigma
        else: step = -sigma
        a[i] = a[i-1] + step
    if(j<50):
        subplot(221)
        plot(t,a)
        subplot(222)
        plot(t,b)
    endpoints_a[j] = a[N-1]
    endpoints_b[j] = b[N-1]

subplot(221)
plot(t,sigma*np.sqrt(t),'ro',t,-sigma*np.sqrt(t),'ro')
xlabel('step')
ylabel('position')
title('random walks a)')
subplot(222)
plot(t,sigma*np.sqrt(t),'ro',t,-sigma*np.sqrt(t),'ro')    
xlabel('step')
ylabel('position')
title('random walks b)')

subplot(223)
# Bin-Grenzen erzeugen: [-40.5,-38.5,...,38.5,40.5] 
dh = 2.0
edges   = mlab.frange(-40.5,40.5,dh)
plt.hist(endpoints_a,edges)
x = (arange(200) - 100)*0.5
s = sigma*np.sqrt(N)
plot(x, dh*N2/(np.sqrt(2.0*pi)*s)*np.exp(-0.5*x*x/(s*s)) )
xlabel('final position')
ylabel('number of paths')

subplot(224)
plt.hist(endpoints_b,edges)
plot(x, dh*N2/(np.sqrt(2.0*pi)*s)*np.exp(-0.5*x*x/(s*s)) )
xlabel('final position')
ylabel('number of paths')
show()
