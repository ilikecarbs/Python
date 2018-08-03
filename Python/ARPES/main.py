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
CSRO.fig21: Fermi surface extraction points
CSRO.fig22: Tight binding model folded SRO

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Which figure do you desire?
            enter below:
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

"""

# %%


CRO.fig1()


# %%

#CSRO.fig20(load=False)
CSRO.fig22()

# %%
