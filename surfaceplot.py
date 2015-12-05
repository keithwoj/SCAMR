# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 17:01:18 2015

@author: wojciechowskik
"""
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def surfaceplot(data,f):
    
    fig = plt.figure()
    ax = Axes3D(fig)
    f = f.flatten()
    x = data[:,0]
    y = data[:,1]
    
    if x.shape == f.shape:
        ax.plot_trisurf(x,y,f)
        ax.scatter3D(x,y,f,c='r')
    else:
        NameError('Surface and axes must have same dimension.')
        