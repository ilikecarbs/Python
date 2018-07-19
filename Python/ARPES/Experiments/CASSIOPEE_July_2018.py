#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 19 15:10:01 2018

@author: denyssutter
"""
import os
os.chdir('/Users/denyssutter/Documents/library/Python/ARPES')
import ARPES
import matplotlib.pyplot as plt

path = '/Users/denyssutter/Documents/PhD/data/Experiments/CASSIOPEE_July_2018/'

"""
%%%%%%%%%%%%%%%%%%%%
      CaMn2Sb2
%%%%%%%%%%%%%%%%%%%%
"""
#%%
"""
Gold
"""
###file S30001-S30007 are .ibw files! -> mode='cut_ibw'
os.chdir('/Users/denyssutter/Documents/library/Python/ARPES')
mat = 'CaMn2Sb2'
year = 2018
file = 'S30021'
mode = 'cut_txt'

D = ARPES.CASS(file, mat, year, mode)
D.gold(Ef_ini=71.4)
#%%
"""
High stat cuts
"""
###High temperature T=230K###
###90eV###
os.chdir('/Users/denyssutter/Documents/library/Python/ARPES')
mat = 'CaMn2Sb2'
year = 2018
files = ['S30008', 'S30010', 'S30011'] #CR, LH, LV
golds = ['S30005', 'S30006', 'S30007']
mode = 'cut_txt'

for i in range(len(files)):
    D = ARPES.CASS(files[i], mat, year, mode)
    D.norm(golds[i])
    D.flatten(norm=True)
    D.bkg(norm=True)
    D.plt_spec(norm=True, v_max=.5)
    plt.savefig((path + str(D.file) + '_bkg_norm.png'), 
                dpi = 300,bbox_inches="tight")
#%%
###Low temperature T=65K###
###90eV###
os.chdir('/Users/denyssutter/Documents/library/Python/ARPES')
mat = 'CaMn2Sb2'
year = 2018
files = ['S30012', 'S30013', 'S30014'] #CR, LH, LV
golds = ['S30023', 'S30024', 'S30025']
mode = 'cut_txt'

for i in range(len(files)):
    D = ARPES.CASS(files[i], mat, year, mode)
    D.norm(golds[i])
    D.flatten(norm=True)
    D.bkg(norm=True)
    D.plt_spec(norm=True, v_max=.5)
    plt.savefig((path + str(D.file) + '_bkg_norm.png'), 
                dpi = 300,bbox_inches="tight")
#%%
###Low temperature T=65K###
###75eV###
os.chdir('/Users/denyssutter/Documents/library/Python/ARPES')
mat = 'CaMn2Sb2'
year = 2018
files = ['S30016', 'S30017', 'S30018'] #CR, LH, LV
golds = ['S30021', 'S30020', 'S30019']
mode = 'cut_txt'

for i in range(len(files)):
    D = ARPES.CASS(files[i], mat, year, mode)
    D.norm(golds[i])
    D.flatten(norm=True)
    D.bkg(norm=True)
    D.plt_spec(norm=True, v_max=.5)
    plt.savefig((path + str(D.file) + '_bkg_norm.png'), 
                dpi = 300,bbox_inches="tight")
#%%
"""
Fermi Surface maps
"""
###High temperature T=230K###
###90eV###
os.chdir('/Users/denyssutter/Documents/library/Python/ARPES')
mat = 'CaMn2Sb2'
year = 2018
file = 'S3_FSM_fine_hv90_T230'
mode = 'FSM'
D = ARPES.CASS(file, mat, year, mode)
D.norm(gold='S30005')
D.ang2kFS(D.ang, Ekin=90-4.5, lat_unit=False, a=1, b=1, c=1, 
          V0=0, thdg=-6, tidg=24.5, phidg=-0)
D.plt_FS_all(coord=True, norm=True)
plt.savefig((path + str(D.file) + '.png'), 
                dpi = 600,bbox_inches="tight")
#%%
###Low temperature T=65K###
###90eV###
os.chdir('/Users/denyssutter/Documents/library/Python/ARPES')
mat = 'CaMn2Sb2'
year = 2018
file = 'S3_FSM_fine_hv90_T65'
mode = 'FSM'
D = ARPES.CASS(file, mat, year, mode)
D.norm(gold='S30023')
D.ang2kFS(D.ang, Ekin=90-4.5, lat_unit=False, a=1, b=1, c=1, 
          V0=0, thdg=-6, tidg=24.5, phidg=-0)
D.plt_FS_all(coord=True, norm=True)
plt.savefig((path + str(D.file) + '.png'), 
                dpi = 600,bbox_inches="tight")
#%%
###Low temperature T=65K###
###75eV###
os.chdir('/Users/denyssutter/Documents/library/Python/ARPES')
mat = 'CaMn2Sb2'
year = 2018
file = 'S3_FSM_fine_hv75_T65'
mode = 'FSM'
D = ARPES.CASS(file, mat, year, mode)
D.norm(gold='S30021')
D.ang2kFS(D.ang, Ekin=90-4.5, lat_unit=False, a=1, b=1, c=1, 
          V0=0, thdg=-6, tidg=24.5, phidg=-0)
D.plt_FS_all(coord=True, norm=True)
plt.savefig((path + str(D.file) + '.png'), 
                dpi = 600,bbox_inches="tight")
