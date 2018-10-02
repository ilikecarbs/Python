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

import PhD_chapter_Concepts as CON
import PhD_chapter_CRO as CRO
import PhD_chapter_CSRO as CSRO


# %%
"""
----------  Concept Figures   ----------
CON.fig1:   Photoemission principle DOS
CON.fig2:   Electron transmission
CON.fig3:   EDC / MDC
CON.fig4:   Experimental Setup
CON.fig5:   (L): eg, t2g orbitals
CON.fig6:   Manipulator angles
CON.fig7:   Mirror plane
CON.fig8:   Data normalization
CON.fig9:   Analyzer energies
CON.fig10:  Analyzer setup
CON.fig11:  Laue + Crystal
CON.fig12:  Inelastic mean free path
CON.fig13:  Sr2RuO4 model
CON.fig14:  Fermi liquid scattering scheme

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
CRO.fig15:  Schematic DOS OSMP
CRO.fig16:  Schematic DOS band / Mott
CRO.fig17:  Schematic DOS uniform gap
CRO.fig18:  Oxygen bands + DFT
CRO.fig19:  Schematic DOS Mott-Hubbard

----------  Ca1.8Sr0.2RuO4 Figures ----------
CSRO.fig1:  Experimental data FS + cuts
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
CSRO.fig14: (L): TB and density of states (N=3.80 \pm .11)
CSRO.fig15: DMFT Fermi surface
CSRO.fig16: (L): DMFT bandstructure calculation
CSRO.fig17: (L): LDA bandstructure calculation
CSRO.fig18: CSRO30 Experimental band structure
CSRO.fig19: CSRO30 Gamma - S cut epsilon pocket
CSRO.fig20: (L): Fit Fermi surface (it_max=3000 -> 70min, load=True)
CSRO.fig21: Fermi surface extraction points
CSRO.fig22: Tight binding model folded SRO
CSRO.fig23: Fit dispersions + FS
CSRO.fig24: Alternative CSRO.fig1
CSRO.fig25: self energy + Z + DOS
CSRO.fig26: Fermi surface counting CSRO (3.84 \pm .14)
CSRO.fig27: Fermi surface counting CSRO unfolded
CSRO.fig28: Fermi surface counting SRO
CSRO.fig29: Fermi surface folded CSRO
CSRO.fig30: xFig1 2nd version
CSRO.fig31: xFig3 self energy

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Which figure do you desire?
            enter below:
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

"""

# %%

CON.fig14()

# %%

CRO.fig19(print_fig=True)

# %%
# CSRO.fig20(load=False)
CSRO.fig22(print_fig=True)

# %%
