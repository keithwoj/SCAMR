# -*- coding: utf-8 -*-
"""
Module for creating differentiation matrices
Created on Thu Jun 25 14:37:43 2015

@author: Keith J. Wojciechowski (KJW)
        
Differentiation Matrix Constructors:

        D = diff1lag(x)     ** First-order differentiation matrix
        diffmat.diff1lag?? for help
        
        D = diff2lag(x)     ** Second-order differentiation matrix
        diffmat.diff2lag?? for help
"""
from pylab import *
import tools

def diff1lag(x):
    """
    D = diff1lag(x)    
    
    Notes:
        * Global (i.e. uses all grid points) differentiation matrices
        * Constructed using Barycentric-Lagrange interpolation
        * Berrut, J. and Trefethen L.N., "Barycentric Lagrange Interpolation"
          SIAM Review, Vol 46, No. 3, pp 501-517 (2004)

    Input:
        x -- data grid (nodes)
        
    Output:
        D -- global differentiation matrix
        
    Edits:
        25 JUN 2015:
        First working version diff1lag (KJW)
        Unit test added, f(x) = exp(sin(pi*x)) (KJW)
    """
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
        D[i,jindx] = w[jindx]/w[i]/dx    # off-diagonal entries
        # diagonal entry: D_ii = -sum_{i!=j}L_j'(x_i)
        D[i,i] = -sum(D[i,jindx])          # row sum of off-diagonal entries
    return D

def diff2lag(x):
    """
    D = diff2lag(x)    
    
    Notes:
        * Global (i.e. uses all grid points) differentiation matrix
        * Constructed using Barycentric-Lagrange interpolation
        * Berrut, J. and Trefethen L.N., "Barycentric Lagrange Interpolation"
          SIAM Review, Vol 46, No. 3, pp 501-517 (2004)

    Input:
        x -- data grid (nodes)
        
    Output:
        D -- global differentiation matrix
        
    Edits:
        01 JUL 2015:
        First working version diff2lag (KJW)
        Unit test added, f(x) = exp(sin(pi*x)) (KJW)
    """    
    # Depends on interp1.weight(x) - Barycentric-Lagrange interpolation weights    
    import interp1

    # Initialize the weights and differentiation matrix
    w = interp1.weight(x)
    n = len(x)
    D = diff1lag(x)
    
    for i in xrange(n):
        jindx = where(arange(n)!=i)        # array of indices != diagonal entry
        dxij = x[i] - x[jindx]               # x_i - x_j where i != j
        # off-diagonal entry: D_ij = L_j''(x_i)
        D[i,jindx] = -2.*D[i,jindx]*(sum(w[jindx]/w[i]/dxij)+1/dxij)
        # diagonal entry: D_ii = -sum_{i!=j}L_j''(x_i)
        D[i,i] = -sum(D[i,jindx])          # row sum of off-diagonal entries
    return D

'''
                    ***** Unit Tests *****
'''

def testf(x,order=1):
    u = exp(sin(pi*x))
    if order == 1:
        du = pi*cos(pi*x)*u
        return du, u
    elif order == 2:
        d2u = pi**2*(-sin(pi*x) + cos(pi*x)**2)*u
        return d2u, u

def test_diff1lag():
    for n in 2**arange(4,7):
        #Create Chebyshev points (of the second kind) to interpolate
        x = cos(pi*linspace(0,n,n)/n)
        # First-order differentiation matrix against first-order derivative
        D = diff1lag(x)
        du_true, u = testf(x)
        du = dot(D,u)
        # Discrete L2-norm -- right-hand rule
        err = tools.normd(du-du_true,x,'right')
        print("Discrete L2-error is %20.15e for n = %4d") % (err,n)

def test_diff2lag():
    for n in 2**arange(4,7):
        #Create Chebyshev points (of the second kind) to interpolate
        x = cos(pi*linspace(0,n,n)/n)
        # Second-order differentiation matrix against second-order derivative        
        D = diff2lag(x)
        du_true, u = testf(x,2)
        du = dot(D,u)
        # Discrete L2-norm -- right-hand rule
        err = tools.normd(du-du_true,x,'right')
        print("Discrete L2-error is %20.15e for n = %4d") % (err,n)

if __name__ == "__main__":
    print "Running Test"
    test_diff1lag()
    test_diff2lag()