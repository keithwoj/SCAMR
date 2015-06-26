# -*- coding: utf-8 -*-
"""
Module for creating differentiation matrices
Created on Thu Jun 25 14:37:43 2015

@author: Keith J. Wojciechowski (KJW)

CONSTRUCTORS:

D = diff1lag(x)

Input:
        x -- data grid (nodes)
        
Output:
        D -- global differentiation matrix
        
Notes:
        First-order global (uses all grid points) differentiation matrix
        
Edits:
        25 JUN 2015:
        First working version diff1lag (KJW)
        Unit test added, f(x) = exp(sin(pi*x)) (KJW)
        
Differentiation Matrix Constructors:
        yi = difflag(x)
        diffmat.difflag?? to see the unit test and an example
"""
from pylab import *
import tools

def diff1lag(x):
    # Depends on interp1.weight(x) - Barycentric-Lagrange interpolation weights    
    import interp1

    # Initialize the weights and differentiation matrix
    w = interp1.weight(x)
    n = len(x)
    D = zeros((n,n))
    
    for i in xrange(n):
        jindx = where(arange(n)!=i)        # array of indices != diagonal entry
        dx = x[i] - x[jindx]               # x_i - x_j where i != j
        # off-diagonal entry: D_ij = L_j'(x_i) = (w_j/w_i)/(x_i - x_j)
        D[i,jindx] = w[jindx]/(w[i]*dx)    # off-diagonal entries
        # diagonal entry: D_ii = -sum_{i!=j}L_j'(x_i)
        D[i,i] = -sum(D[i,jindx])          # row sum of off-diagonal entries
    return D

def testf(x):
    u = exp(sin(pi*x))
    du = pi*cos(pi*x)*u
    
    return du, u

def test():
    for n in 2**arange(4,7):
        #Create Chebyshev points (of the second kind) to interpolate
        x = cos(pi*linspace(0,n,n)/n)
        D = diff1lag(x)
        du_true, u = testf(x)
        du = dot(D,u)
        err = tools.normd(du-du_true,x,'right')
        print("Discrete L2-error is %20.15e for n = %4d") % (err,n)

if __name__ == "__main__":
    print "Running Test"
    test()