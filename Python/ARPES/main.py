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

import matplotlib.pyplot as plt
import matplotlib.cm as cm

import utils
import PhD_chapter_CRO as CRO
import PhD_chapter_CSRO as CSRO


rainbow_light = utils.rainbow_light
cm.register_cmap(name='rainbow_light', cmap=rainbow_light)
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.serif'] = ['Computer Modern Roman']
plt.rc('font', **{'family': 'serif', 'serif': ['STIXGeneral']})
font = {'family': 'serif',
        'style': 'normal',
        'color':  [0, 0, 0],
        'weight': 'ultralight',
        'size': 12,
        }

# %%
"""
---------  Ca2RuO4 Figures   ---------
CROfig1:   DFT plot Ca2RuO4: figure 3 of Nature Comm.
CROfig2:   (L): DMFT pot Ca2RuO4: figure 3 of Nature Comm.
CROfig3:   DFT plot orbitally selective Mott scenario
CROfig4:   DFT plot uniform gap scnenario
CROfig5:   Experimental Data of Nature Comm.
CROfig6:   Constant energy map CaRuO4 of alpha branch
CROfig7:   Photon energy dependence Ca2RuO4: figure 2 of Nature Comm.
CROfig8:   Polarization dependence Ca2RuO4: figure 2 of Nature Comm.
CROfig9:   (L): DMFT plot Ca2RuO4 dxy/dxz,yz: figure 4 of Nature Comm.
CROfig10:  (L): DFT plot Ca2RuO4: spaghetti and spectral representation
CROfig11:  Multiplet analysis Ca2RuO4
CROfig12:  Constant energy maps oxygen band -5.2eV
CROfig13:  Constant energy maps alpha band -0.5eV
CROfig14:  Constant energy maps gamma band -2.4eV

---------  Ca1.8Sr0.2RuO4 Figures ---------
CSROfig1:  Experimental data: Figure 1 CSRO20 paper
CSROfig2:  Experimental PSI data: Figure 2 CSCRO20 paper
CSROfig3:  (L): Polarization and orbital characters. Figure 3 in paper
CSROfig4:  (L): Temperature dependence. Figure 4 in paper
CSROfig5:  (L): Analysis Z epsilon band (load=True)
CSROfig6:  Analysis MDC's beta band (load=True)
CSROfig7:  Background subtraction
CSROfig8:  Extraction LDA Fermi velocity
CSROfig9:  ReSigma vs ImSigma (load=True)
CSROfig10: Quasiparticle Z
CSROfig11: Tight binding model CSRO
CSROfig12: Tight binding model SRO
CSROfig13: TB along high symmetry directions, orbitally resolved
CSROfig14: (L): TB and density of states
CSROfig15: DMFT FS
CSROfig16: (L): DMFT bandstructure calculation
CSROfig17: (L): LDA bandstructure calculation
CSROfig18: CSRO30 Experimental band structure
CSROfig19: CSRO30 Gamma - S cut epsilon pocket

---------  To-Do ---------

CSRO: TB with cuts
CSRO: Symmetrization
CSRO: FS area counting
CSRO: kz dependence
CSRO: TB specific heat



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Which figure do you desire?
            enter below:
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

"""


CRO.fig12()


# %%

CSRO.fig1()


