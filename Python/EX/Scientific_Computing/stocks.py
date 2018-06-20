# -*- coding: utf-8 -*-
"""
    Wie weit verhalten sich Aktienkurse (resp. deren Logarithmen) wie random
    walks? Die Datei sp.txt enthält Informationen des S&P 500 Index von 1950
    bis 2012 (von finance.yahoo.com). Man kann die Schlussstaende so darstellen:

    data = np.loadtxt('sp.txt', delimiter=',',comments='#',usecols=(6,) )
    plot(data)

    Achtung: Die Werte sind Rückwärts in der Zeit sortiert!
    
    Betrache jetzt die relativen Aenderungen (Intervall = 1 Tag):
    
    N = len(data)
    d = np.zeros(N-1)
    for i in range(N-1):
        d[i] = data[i]/data[i+1]
    tdays_per_yr = 252
    t = -1.0*arange(N)/tdays_per_yr
    plot(2013.0+t,data)
        
    1) Plotte den S&P 500 seit 1950. Berechene und plotte seine relativen
    Ein-Tages-Aenderungen d. Berechne sigma_1day, die Standartabweichung dieser
    relativen Aenderungen. Plotte ein Histogram von log(d) und die Normalverteilung
    mit demselben sigma_1day.
    
    2) Bereche und plotte sigma_1day fuer jedes einzelne Jahr (etwa 252 Handelstage) 
    und vergleiche die Werte mit dem langjaehrigen Ergebnis von 1).    
    
    3) Untersuche wie sich die Standartabweichung der Werte aendert, wenn das
    Intervall schrittweise auf mehrere hundert Tage erhoeht. Plotte zum Vergleich
    auch sqrt(t/1day)*sigma_1day.

    4) Berechene und plotte seine relativen 10-Tages-Aenderungen d_10. Plotte ein
    Histogram von log(d_10) und die Normalverteilung mit sigma = sqrt(10)*sigma_1day.
    
    Nicht-technische Diskusionen dieser Fragen findet man z. B. in:     
    Nassim Nicolas Taleb, "The Black Swan", 2007
    Emanuel Derman, "My Life as a Quant: Reflections on Physics and Finance", 2004
    
""" 
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from numpy import pi,arange
from pylab import *

tdays_per_yr = 252
data = np.loadtxt('sp.txt', delimiter=',',comments='#',usecols=(6,) )
N = len(data)
d = np.zeros(N-1)
t = -1.0*arange(N)/tdays_per_yr
for i in range(N-1):
    d[i] = data[i]/data[i+1]

sigma = np.std(log(d)) 
mean_1day = np.mean(log(d))
print(sigma,mean_1day)

# Aufgabe 1)
figure(figsize=(16,10))
subplot(231)
plot(2013.0+t,data)
yscale('log')
title('1) S&P 500 daily close')
xlabel('year')
ylabel('daily close (ajdusted for dividends)')

subplot(232)
# ....
title('1) relative changes')
xlabel('year')
ylabel('d (relative change from previous day)')

subplot(233)
# ...
yscale('log')
title('1) distr. of 1-day relative changes')
ylabel('P(log(d))')
xlabel('log(d)')

# Aufgabe 2)
subplot(234)
# ....
title('2) 1-year standard deviations')
ylabel('sigma_1yr')
xlabel('year')
    
# Aufgabe 3)
subplot(235)
# ...
title('3) Standard deviation vs. period lenght')
xlabel('L = lenght of period [days]')
ylabel('sigma_L')

# Aufgabe 4)
subplot(236)
# ...
yscale('log')
title('4) distr. of 10-day relative changes')
ylabel('$P(log(d_{10})$')
xlabel('$log(d_{10}))$')
show()