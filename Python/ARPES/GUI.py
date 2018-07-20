#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 20:39:55 2018

@author: denyssutter

%%%%%%%%%%%%%%%%%%%%
        GUI
%%%%%%%%%%%%%%%%%%%%

**In development, many many things to do...**

.. note::
        To-Do:
            -

"""

from pyqtgraph.Qt import QtGui, QtCore, QtWidgets
import pyqtgraph as pg 

#def FS_GUI(D): 
    # Interpret image data as row-major instead of col-major
pg.setConfigOptions(imageAxisOrder='row-major')

app = QtCore.QCoreApplication.instance()
if app is None:
    app = QtWidgets.QApplication(sys.argv)
    
## Create window with two ImageView widgets
mw = QtGui.QMainWindow()
mw.resize(1500,800)
mw.setWindowTitle('pyqtgraph example: DataSlicing')
cw = QtGui.QWidget()
mw.setCentralWidget(cw)
l = QtGui.QGridLayout()
cw.setLayout(l)

imv1 = pg.ImageView()
imv2 = pg.ImageView()
l.addWidget(imv1, 0, 0, 0, 1)
l.addWidget(imv2, 0, 1, 0, 1)
mw.show()
data = np.transpose(D.int_norm,(2,0,1))
roi = pg.LineSegmentROI([[10, 64], [120,64]], pen='r')
imv1.addItem(roi)

def update():
    global data, imv1, imv2
    d = roi.getArrayRegion(data, imv1.imageItem, axes=(0,1))
    imv2.setImage(d)
    
roi.sigRegionChanged.connect(update)

## Display the data
imv1.setImage(data, scale = (1, 6))
imv1.setHistogramRange(-0.01, 0.01)
imv1.setLevels(-0.003, 0.003)

#imv1.scale(0.2, 0.2)

update()

## Start Qt event loop unless running in interactive mode.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
        