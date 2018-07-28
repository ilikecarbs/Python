#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Created on Tue Jun 19 15:14:29 2018

@author: ilikecarbs

%%%%%%%%%%%%%%%%%%%%%
        main
%%%%%%%%%%%%%%%%%%%%%

**Plotting figures for dissertation**

.. note::
        To-Do:
            -
"""

import PhD_chapter_CRO as CRO
import PhD_chapter_CSRO as CSRO


# %%
"""
----------  Ca2RuO4 Figures   ----------
CRO.fig1:   DFT plot: figure 3 of Nature Comm.
CRO.fig2:   (L): DMFT plot: figure 3 of Nature Comm.
CRO.fig3:   DFT plot: orbitally selective Mott scenario
CRO.fig4:   DFT plot: uniform gap scnenario
CRO.fig5:   Experimental Data of Nature Comm.
CRO.fig6:   Constant energy map CaRuO4 of alpha branch
CRO.fig7:   Photon energy dependence Ca2RuO4: figure 2 of Nature Comm.
CRO.fig8:   Polarization dependence Ca2RuO4: figure 2 of Nature Comm.
CRO.fig9:   (L): DMFT plot dxy/dxz,yz: figure 4 of Nature Comm.
CRO.fig10:  (L): DFT plot: spaghetti and spectral representation
CRO.fig11:  Multiplet analysis Ca2RuO4
CRO.fig12:  Constant energy maps oxygen band -5.2eV
CRO.fig13:  Constant energy maps alpha band -0.5eV
CRO.fig14:  Constant energy maps gamma band -2.4eV

----------  Ca1.8Sr0.2RuO4 Figures ----------
CSRO.fig1:  Experimental data: Figure 1 CSRO20 paper
CSRO.fig2:  Experimental PSI data: Figure 2 CSCRO20 paper
CSRO.fig3:  (L): Polarization and orbital characters. Figure 3 in paper
CSRO.fig4:  (L): Temperature dependence. Figure 4 in paper
CSRO.fig5:  (L): Analysis Z epsilon band (load=True)
CSRO.fig6:  Analysis MDC's beta band (load=True)
CSRO.fig7:  Background subtraction
CSRO.fig8:  Extraction LDA Fermi velocity
CSRO.fig9:  ReSigma vs ImSigma (load=True)
CSRO.fig10: Quasiparticle Z
CSRO.fig11: Tight binding model CSRO
CSRO.fig12: Tight binding model SRO
CSRO.fig13: TB along high symmetry directions, orbitally resolved
CSRO.fig14: (L): TB and density of states (N=3.7)
CSRO.fig15: DMFT Fermi surface
CSRO.fig16: (L): DMFT bandstructure calculation
CSRO.fig17: (L): LDA bandstructure calculation
CSRO.fig18: CSRO30 Experimental band structure
CSRO.fig19: CSRO30 Gamma - S cut epsilon pocket
CSRO.fig20: (L): Fit Fermi surface (it_max=3000 -> 70min, load=True)


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Which figure do you desire?
            enter below:
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

"""

# %%


CRO.fig1()


# %%
import os
import utils
import numpy as np

save_dir = '/Users/denyssutter/Documents/PhD/PhD_Denys/Figs/'
data_dir = '/Users/denyssutter/Documents/PhD/data/'
home_dir = '/Users/denyssutter/Documents/library/Python/ARPES'


versions = ('SRO', 'CSRO20', 'CSRO30', 'fit')

for version in versions:
    it, J, P = CSRO.fig20(print_fig=True, load=False, version=version)

    # build up dictionary
    param = dict([('t1', P[0]), ('t2', P[1]), ('t3', P[2]), ('t4', P[3]),
                  ('t5', P[4]), ('t6', P[5]), ('mu', P[6]), ('so', P[7])])

    kbnd = 1  # boundaries
    tb = utils.TB(a=np.pi, kbnd=kbnd, kpoints=5000)
    tb.CSRO(param=param, e0=0, vert=False, proj=False)

    # load data
    bndstr = tb.bndstr
    coord = tb.coord
    X = coord['X']
    Y = coord['Y']
    Axz = bndstr['Axz']
    Ayz = bndstr['Ayz']
    Axy = bndstr['Axy']
    Bxz = bndstr['Bxz']
    Byz = bndstr['Byz']
    Bxy = bndstr['Bxy']

    os.chdir(data_dir)
    Axz_dos = np.savetxt('Data_CSRO20_Axz_kpts_5000_' + str(version) + '.dat', Axz)
    Ayz_dos = np.savetxt('Data_CSRO20_Ayz_kpts_5000_' + str(version) + '.dat', Ayz)
    Axy_dos = np.savetxt('Data_CSRO20_Axy_kpts_5000_' + str(version) + '.dat', Axy)
    Bxz_dos = np.savetxt('Data_CSRO20_Bxz_kpts_5000_' + str(version) + '.dat', Bxz)
    Byz_dos = np.savetxt('Data_CSRO20_Byz_kpts_5000_' + str(version) + '.dat', Byz)
    Bxy_dos = np.savetxt('Data_CSRO20_Bxy_kpts_5000_' + str(version) + '.dat', Bxy)
    os.chdir(home_dir)
# %%
