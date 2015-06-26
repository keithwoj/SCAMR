# -*- coding: utf-8 -*-
"""
Module for implementing 1-D interpolation
Created on Wed Jun 24 16:39:11 2015

@author: Keith J. Wojciechowski (KJW)

The functions for this module adopt the following convention

yi = INTERPOLATOR(x,y,xi)

Input:
        x -- data grid (nodes)
        y -- data to be interpolated
        xi -- interpolation grid
        
Output:
        yi -- interpolant
        
Notes:
        
Edits:
        24 JUN 2015:
        First working version barylag (KJW)
        Unit test added, f(x) = 1/(1+(5*x)**2) (KJW)
        
Interpolators:
        yi = barylag(x,y,xi)
        interp1.barylag?? to see the unit test and an example
"""

from pylab import *
import tools

# Barycentric weights
def weight(x):
    # Given the nodes, compute the weights. An advantage to this method over,
    # say Newton interpolation, is that the weights depend only on the grid and
    # NOT on the data. So these weights will work for a variety of functions.
    # Requires O(n^2) flops but interpolation only requires O(n) flops.
    n = len(x)
    # Compute the difference matrix
    # x[0] - x[0] x[1] - x[0] x[2] - x[0] ... x[n] - x[0]
    # x[0] - x[1] x[1] - x[1] x[2] - x[1] ... x[n] - x[1]
    #                       ...
    # x[0] - x[n] x[1] - x[n] x[2] - x[n] ... x[n] - x[n]
    #
    # Notice that the diagonal entries are 0 so replace them with 1's
    
    # Replicate the interval over n columns
    X = outer(x,ones(n))
    # Compute the difference matrix as defined above
    dX = X.T - X + eye(n)
    
    # Barycentric weights are 1/product(columns)
    return 1/prod(dX,axis=0)

# Barycentric-Lagrange interpolation
def barylag(x,y,xi):
    # Initialize dimensions and storage allocation for output
    nx = len(x)
    nxi = len(xi)
    yi = zeros(nxi)
    # Obtain the Barycentric weights
    w = weight(x)
    
    # The interpolant is defined
    #   p(x) = sum_{j=0}^n(y_j*w_j/(x-x_j))/sum_{j=0}^n(w_j/(x-x_j))
    for k in xrange(nxi):
        fv = w/(xi[k]-x)
        yi[k] = sum(y*fv)/sum(fv)

    return yi

# Test function, Runge's example
def testf(x):
    return 1/(1+(5*x)**2)

def test():
    for n in 2**arange(4,7)-1:
        #Create Chebyshev points (of the second kind) to interpolate
        x = cos(pi*linspace(0,n,n)/n)
        y = testf(x)
        # Run the test
        ni = 2*(n+1)
        xi = linspace(-1+1./n**2,1-1./n**2,ni)
        yi = barylag(x,y,xi)
        ytrue = testf(xi)
        err = tools.normd(yi-ytrue,xi,'right')
        print("Discrete L2-error is %20.15e for nodes = %4d, points = %4d") % (err,n,ni)
    
if __name__ == "__main__":
    print "Running Test"
    test()