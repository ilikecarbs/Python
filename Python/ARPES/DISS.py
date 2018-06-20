#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 15:14:29 2018

@author: denyssutter
"""
import os
os.chdir('/Users/denyssutter/Documents/library/Python/ARPES')
import utils_plt_DISS as up
import matplotlib.pyplot as plt


"""
fig1: DFT plot Ca2RuO4: figure 3 of Nature Comm.
fig2: DMFT pot Ca2RuO4: figure 3 of Nature Comm.
fig3: DFT plot orbitally selective Mott scenario
fig4: DFT plot uniform gap scnenario
"""
#[7974,8048,7993,8028]



up.fig3()



#%%
plt.savefig(
'/Users/denyssutter/Documents/PhD/PhD_Denys/Chapter_Ca214/Figs/Raster/fig3.png', 
dpi = 300,bbox_inches="tight")

#%%
import os
os.chdir('/Users/denyssutter/Documents/library/Python/ARPES')

from ARPES import DLS
import utils as ut
file = 47974
gold = 48000
mat = 'Ca2RuO4'
year = 2016
sample = 'T10'

D = DLS(file, mat, year, sample)
D.norm(gold)
D.ang2k(D.ang, Ekin=65-4.5, a=3.89, b=3.89, c=11, V0=0, thdg=0, tidg=0, phidg=0)

#%%
ut.gold(gold, mat, year, sample, Ef_ini=60.4, BL='DLS')


#%%
D.plt_spec(norm = True)
A = np.array([1])