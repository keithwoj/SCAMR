# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 15:51:04 2015

@author: wojciechowskik
"""
from matplotlib.pylab import array
import rbf
import halton

N = 13
Np = 33

x = halton.halton(3,N)
y = halton.halton(5,N)

x = (2*x - (min(x)+max(x)))/(max(x)-min(x))
y = (2*y - (min(y)+max(y)))/(max(y)-min(y))

d = array([x,y]).T

f = rbf.testfunction(d)

xp = halton.halton(7,Np)
yp = halton.halton(11,Np)

xp = (2*xp - (min(xp)+max(xp)))/(max(xp)-min(xp))
yp = (2*yp - (min(yp)+max(yp)))/(max(yp)-min(yp))

p = array([xp,yp]).T

fp = rbf.rbfinterp(d,f,p,'mq',shapeparm=2.4)

rbf.surfaceplot(p,fp)
 