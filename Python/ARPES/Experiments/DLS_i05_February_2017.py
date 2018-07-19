#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 19 21:11:16 2018

@author: denyssutter
"""

import os
os.chdir('/Users/denyssutter/Documents/library/Python/ARPES')
import ARPES
import matplotlib.pyplot as plt

path = '/Users/denyssutter/Documents/PhD/data/Experiments/DLS_i05_February_2017/'

"""
%%%%%%%%%%%%%%%%%%%%
      CSRO30
%%%%%%%%%%%%%%%%%%%%
"""
#%%
"""
Gold
"""
###file S30001-S30007 are .ibw files! -> mode='cut_ibw'
os.chdir('/Users/denyssutter/Documents/library/Python/ARPES')
mat = 'CSRO30'
year = 2017
file = 62492
sample = 'S13'
D = ARPES.DLS(file, mat, year, sample)
D.gold(Ef_ini=67.38)
#%%
"""
High stat cuts
"""
os.chdir('/Users/denyssutter/Documents/library/Python/ARPES')
mat = 'CSRO30'
year = 2017
sample = 'S13'
gold_22 = 62455
files = [62488, 62470, 62449, 62444] #72eV, the rest 22eV
golds = [62492, gold_22, gold_22, gold_22]
for i in range(len(files)):
    D = ARPES.DLS(files[i], mat, year, sample)
    D.norm(golds[i])
#    D.flatten(norm=False)
#    D.bkg(norm=False)
    D.plt_spec(norm=True, v_max=.7)
#    plt.savefig((path + str(D.file) + '_bkg_norm.png'), 
#                dpi = 300,bbox_inches="tight")

#%%
"""
Fermi Surface maps
"""
###High temperature T=230K###
###90eV###
os.chdir('/Users/denyssutter/Documents/library/Python/ARPES')
mat = 'CSRO30'
year = 2017
file = 'S3_FSM_fine_hv90_T230'
sample = 'S13'
D = ARPES.CASS(file, mat, year, mode)
D.norm(gold='S30005')
D.ang2kFS(D.ang, Ekin=90-4.5, lat_unit=False, a=1, b=1, c=1, 
          V0=0, thdg=-6, tidg=24.5, phidg=-0)
D.plt_FS_all(coord=True, norm=True)
plt.savefig((path + str(D.file) + '.png'), 
                dpi = 600,bbox_inches="tight")
