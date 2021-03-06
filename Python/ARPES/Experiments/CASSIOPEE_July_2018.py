#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 19 15:10:01 2018

@author: ilikecarbs

%%%%%%%%%%%%%%%%%%%%%
 CASSIOPEE July 2018
%%%%%%%%%%%%%%%%%%%%%

**ARPES experiment**

.. note::
        To-Do:
            -

"""
import os
import ARPES
import matplotlib.pyplot as plt

os.chdir('/Users/denyssutter/Documents/library/Python/ARPES')
path = '/Users/denyssutter/Documents/PhD/data/Experiments/CASSIOPEE_July_2018/'

"""
%%%%%%%%%%%%%%%%%%%%
      CaMn2Sb2
%%%%%%%%%%%%%%%%%%%%
"""

mat = 'CaMn2Sb2'
year = 2018
mode = 'cut_txt'

# %%
"""
Gold
"""
# file S30001-S30007 are .ibw files! -> mode='cut_ibw'
file = 'S30021'

D = ARPES.CASS(file, mat, year, mode)
D.gold(Ef_ini=71.4)
# %%
"""
High stat cuts
"""
# High temperature T=230K
# 90eV

files = ['S30008', 'S30010', 'S30011']  # CR, LH, LV
golds = ['S30005', 'S30006', 'S30007']
mode = 'cut_txt'

# %%
# T=110K
# 90eV

files = ['S30041', 'S30042', 'S30043']  # CR, LH, LV
golds = ['S30005', 'S30006', 'S30007']


# %%
# T=100K
# 90eV

files = ['S30036', 'S30038', 'S30037']  # CR, LH, LV
golds = ['S30005', 'S30006', 'S30007']

# %%
# Low temperature T=65K
# 90eV

files = ['S30012', 'S30013', 'S30014']  # CR, LH, LV
golds = ['S30023', 'S30024', 'S30025']

# %%
# Low temperature T=65K
# 75eV

files = ['S30016', 'S30017', 'S30018']  # CR, LH, LV
golds = ['S30021', 'S30020', 'S30019']

# %%
# T=90K
# 90eV

files = ['S30032', 'S30034', 'S30033']  # CR, LH, LV
golds = ['S30023', 'S30024', 'S30025']

# %%

for i in range(len(files)):
    D = ARPES.CASS(files[i], mat, year, mode)
    D.norm(golds[i])
    D.flatten()
    D.bkg()
    D.plt_spec(v_max=.5)
    plt.savefig((path + str(D.file) + '_bkg_norm.png'),
                dpi=300, bbox_inches="tight")

# %%
"""
Fermi Surface maps
"""
# High temperature T=230K
# 90eV

file = 'S3_FSM_fine_hv90_T230'
gold = 'S30005'
mode = 'FSM'

# %%
# T=120K
# 90eV

file = 'S3_FSM_fine_hv90_T120'
gold = 'S30005'
mode = 'FSM'

# %%
# T=110K
# 90eV

file = 'S3_FSM_fine_hv90_T110'
gold = 'S30005'
mode = 'FSM'

# %%
# T=90
# 90eV

file = 'S3_FSM_fine_hv90_T90'
gold = 'S30023'
mode = 'FSM'

# %%
# Low temperature T=75K
# 90eV

file = 'S3_FSM_fine_hv90_T75'
gold = 'S30023'
mode = 'FSM'

# %%
# Low temperature T=65K
# 90eV

file = 'S3_FSM_fine_hv90_T65'
gold = 'S30023'
mode = 'FSM'

# %%
# Low temperature T=65K
# 75eV

file = 'S3_FSM_fine_hv75_T65'
gold = 'S30021'
mode = 'FSM'


# %%

D = ARPES.CASS(file, mat, year, mode)
D.norm(gold='S30023')
D.ang2kFS(D.ang, Ekin=90-4.5, lat_unit=False, a=1, b=1, c=1,
          V0=0, thdg=-6, tidg=24.5, phidg=-0)
D.plt_FS_all()
plt.savefig((path + str(D.file) + '.png'),
            dpi=600, bbox_inches="tight")

# %%
"""
Photon Energy Scans
"""
# High temperature T=230K

file = 'S3_hv50_hv100_T230'
mode = 'hv'

# %%
# Low temperature T=65

file = 'S3_hv50_hv100_T65'
mode = 'hv'

# %%

D = ARPES.CASS(file, mat, year, mode)
D.plt_hv(v_max=.5)
plt.savefig((path + str(D.file) + '.png'),
            dpi=600, bbox_inches="tight")
