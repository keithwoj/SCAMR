# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 17:41:47 2015

@author: Keith J. Wojciechowski (KJW)
"""

from numpy import sum, sqrt, diff, trapz

# Discrete L2-norm on grid x
def normd(u,x,rule = 'even'):
    """
    normd(u,x,rule)
    
    Numerical approximation to the L2 norm:
    Discrete L2 norm = sqrt{sum_j^n (|u_j|^2 h_j)} where h_j = x_{j+1} - x_j
    
    u = 1D numpy array
    x = 1D grid (usually constructed with linspace)
    rule = string: 'right' = right-hand rule, 'left' = left-hand rule
                   'even' = evenly-spaced nodes
    """
    
    if rule.lower() == 'right':
        h = abs(diff(x))
        return sqrt(sum(h*abs(u[1:])**2))
    elif rule.lower() == 'left':
        h = abs(diff(x))
        return sqrt(sum(h*abs(u[0:-1])**2))
    elif rule.lower() == 'even':
        h = max(abs(diff(x)))
        return sqrt(h*sum(abs(u)**2))
    else:
        print "Error: Rule not in catalog."
        return 0