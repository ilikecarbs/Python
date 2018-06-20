# -*- coding: utf-8 -*-

"""

Today we will explore eigenvectors and their matrices.

1)  Write the function *plot2dvec* that plots any vector in two dimensions. 
    As input it takes the vector itself, and the style of the plot. 
    Place the vector’s tail at (0,0) and use a marker for the head.

2)  Write the function *eigenplot* that plots the eigen vectors of a 
    2-dimensional matrix, A, by making use of *plot2dvec*. For each 
    eigenvector of A, plot:
        a) the eigenvector itself
        b) the product of the eigenvector and its eigenvalue
        c) the image of matrix A and the eigenvector
        Make use of different colours and markers to make each object 
        clear and different.

3)  Write the function *unitplot* that takes a matrix as its input. It will 
    then plot, using *plot2dvec* without lines,
        a) The unit circle
        b) The dot product of the unit circle and the matrix A.
 
4)  Use the functions eigenplot and unitplot on the following two matrices. 
    Use a separate figure for each matrix.
        a) The symmetric matrix: 
                (0.5 -1)
            A = (-1   3)
        b) The asymmetric matrix:
                (0.5 1)
            B = (-1  3)
    What differences do you find?

Hints:
    *The numpy function np.linalg.eig will give you the normalized eigenvectors 
    and eigenvalues of any matrix.
    *Make use of plt.axis('equal') for your plots so that the unit circle 
    actually looks like a circle.
    *Every time you want to start a new figure, use plot.figure(). You only 
    need to call plt.show() once at the very end of your program.      

@author:
@with:

"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

def plot2dvec(vec, c='b', m='o', l='-'):
    """ Plots a vector with its tail at the origin

    Input
    ----------
    vec:    array
            vector be plotted
    c:      string
            colour of plot
    m:      string
            marker of vector head
    l:      string
            linestyle of vector
  
    """
    plt.plot([vec[0],0], [vec[1],0], c=c, marker='None', ls=l)
    plt.plot(vec[0], vec[1], c=c, marker=m, ls=l)

def eigenplot(A):
    w, v = np.linalg.eig(A)
    plot2dvec(v[:,0])
    plot2dvec(w[0]*v[:,0],m='*')
    plot2dvec(np.dot(A,v[:,0]),m='p');

    plot2dvec(v[:,1],c='r')
    plot2dvec(w[1]*v[:,1],c='r',m='*')
    plot2dvec(np.dot(A,v[:,1]),c='r',m='p');

def unitplot(A):
    for angel in mlab.frange(0,2*np.pi,0.1):
        x = np.cos(angel)
        y = np.sin(angel)
        plot2dvec([x,y],'g','.','None')
        plot2dvec(np.dot(A,[x,y]),'c','.','None')


plt.axis('equal')
plt.grid()

A=np.array([[0.5, -1],[-1, 3]])
eigenplot(A)
unitplot(A)

plt.figure()
plt.axis('equal')
plt.grid()

B=np.array([[0.5,1],[-1,3]])
eigenplot(B)
unitplot(B)

plt.show()
