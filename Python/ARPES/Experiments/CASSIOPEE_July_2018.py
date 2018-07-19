#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 19 15:10:01 2018

@author: denyssutter
"""
import os
import ARPES

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
###High temperature###
os.chdir('/Users/denyssutter/Documents/library/Python/ARPES')
mat = 'CaMn2Sb2'
year = 2018
files = ['S30008', 'S30010', 'S30011']
golds = ['S30005', 'S30006', 'S30007']
mode = 'cut_txt'

for i in range(len(files)):
    D = ARPES.CASS(files[i], mat, year, mode)
    D.norm(golds[i])
    D.plt_spec(norm=True)
    
    