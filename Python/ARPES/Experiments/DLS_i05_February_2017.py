#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 19 21:11:16 2018

@author: denyssutter
"""

import os
import ARPES
import matplotlib.pyplot as plt

os.chdir('/Users/denyssutter/Documents/library/Python/ARPES')
path = ('/Users/denyssutter/Documents/PhD/data/Experiments/' +
        'DLS_i05_February_2017/')

"""
%%%%%%%%%%%%%%%%%%%%
      CSRO30
%%%%%%%%%%%%%%%%%%%%
"""

mat = 'CSRO30'
year = '2017'
sample = 'S13'

# %%
"""
Gold
"""

file = 62492

D = ARPES.DLS(file, mat, year, sample)
D.gold(Ef_ini=67.38)

# %%
"""
High stat cuts
"""

gold_22 = 62455
files = [62488, 62470, 62449, 62444]  # 72eV, the rest 22eV
golds = [62492, gold_22, gold_22, gold_22]

for i in range(len(files)):
    D = ARPES.DLS(files[i], mat, year, sample)
    D.norm(golds[i])
    D.flatten()
    D.bkg()
    D.plt_spec(v_max=.7)
    plt.savefig((path + str(D.file) + '_bkg_norm.png'),
                dpi=300, bbox_inches="tight")

# %%
"""
Fermi Surface maps
"""
