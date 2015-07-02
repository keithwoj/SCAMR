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
from numpy import *
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

def diff1fd24(n,order=2,h=1):
    """
    D = diff1fd24(n,order,grid-spacing)
    
    Notes:
        * Local (i.e. uses neighboring grid points) differentiation matrix
        * Select either 2nd or 4th order accuracy
        * Uses centered-differences for the interior
        * Uses one-sided differencing for the boundaries

    Input:
        n -- number of grid points
        order -- 2nd or 4th order
        h -- grid spacing
        
    Output:
        D -- global differentiation matrix
        
    Edits:
        02 JUL 2015:
        First working version diff1fd24 (KJW)
        Unit test added, f(x) = exp(sin(pi*x)) (KJW)
    """ 
    from scipy.sparse import spdiags, lil_matrix
    e = ones(n)
    if order == 2:
        #Construct stencil weights
        stencil = array([-e/2.,e/2.])
        #Centered difference stencils are located on the off-diagonals
        diag_array = array([-1,1])
        D = spdiags(stencil,diag_array,n,n)
        D = lil_matrix(D)
        #Differentiaion at the boundary is approximated using one-sided stencil
        D[0,0:3] = array([-3./2,2.,-1./2])
        D[-1,-3:] = array([1/2.,-2.,3./2])
    elif order == 4:
        #Construct stencil weights        
        stencil = array([e/12., -2*e/3.,2*e/3.,-e/12.])
        #Centered difference stencils are located on the off-diagonals
        diag_array = array([-2,-1,1,2])
        D = spdiags(stencil,diag_array,n,n)
        D = lil_matrix(D)
        #Differentiaion at the boundary is approximated using one-sided stencil
        D[1,0:6] = array([0.,-25./12,4.,-3.,4./3,-1./4])
        D[-2,-6:] = array([1./4,-4./3,3.,-4.,25./12,0.])
        D[0,0:5] = array([-25./12,4.,-3.,4./3,-1./4])
        D[-1,-5:] = array([1./4,-4./3,3.,-4.,25./12])
    else:
        print("*** Only 2nd or 4th order approximaitons are considered.")
    return D/h

def diff2fd24(n,order=2,h=1):
    """
    D = diff2fd24(n,order,grid-spacing)
    
    Notes:
        * Local (i.e. uses neighboring grid points) differentiation matrix
        * Select either 2nd or 4th order accuracy
        * Uses centered-differences for the interior
        * Uses one-sided differencing for the boundaries

    Input:
        n -- number of grid points
        order -- 2nd or 4th order
        h -- grid spacing
        
    Output:
        D -- global differentiation matrix
        
    Edits:
        02 JUL 2015:
        First working version diff2fd24 (KJW)
        Unit test added, f(x) = exp(sin(pi*x)) (KJW)
    """     
    from scipy.sparse import spdiags, lil_matrix
    e = ones(n)
    if order == 2:
        #Construct stencil weights
        stencil = array([e,-2*e,e])
        #Centered difference stencils are located on the off-diagonals
        diag_array = array([-1,0,1])
        D = spdiags(stencil,diag_array,n,n)
        D = lil_matrix(D)
        #Differentiaion at the boundary is approximated using one-sided stencil
        D[0,0:4] = array([2.,-5.,4.,-1.])
        D[-1,-4:] = array([-1.,4.,-5.,2.])
    elif order == 4:
        #Construct stencil weights
        stencil = array([-e/12., 4*e/3.,-5*e/2.,4*e/3.,-e/12.])
        #Centered difference stencils are located on the off-diagonals
        diag_array = array([-2,-1,0,1,2])
        D = spdiags(stencil,diag_array,n,n)
        D = lil_matrix(D)
        #Differentiaion at the boundary is approximated using one-sided stencil
        D[1,0:7] = array([0.,15./4,-77./6,107./6,-13.,61./12,-5./6])
        D[-2,-7:] = array([-5./6,61./12,-13.,107./6,-77./6,15./4,0.])
        D[0,0:6] = array([15./4,-77./6,107./6,-13.,61./12,-5./6])
        D[-1,-6:] = array([-5./6,61./12,-13.,107./6,-77./6,15./4])
    else:
        print("*** Only 2nd or 4th order approximaitons are considered.")
    return D/h**2
    
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

def test_diff1fd24():
    for n in 2**arange(4,11):
        #Create an evenly-spaced grid of n points
        x = linspace(-1,1,n)
        # First-order differentiation matrix against first-order derivative        
        D2 = diff1fd24(n,2,2./(n-1))
        D4 = diff1fd24(n,4,2./(n-1))
        du_true, u = testf(x)
        du2 = D2.dot(u)
        du4 = D4.dot(u)
        # Discrete L2-norm -- right-hand rule
        err2 = tools.normd(du2-du_true,x,'even')
        print("Discrete L2-error (2nd order) is %20.15e for n = %4d") % (err2,n)
        err4 = tools.normd(du4-du_true,x,'even')
        print("Discrete L2-error (4th order) is %20.15e for n = %4d") % (err4,n)
        
def test_diff2fd24():
    for n in 2**arange(4,11):
        #Create an evenly-spaced grid of n points
        x = linspace(-1,1,n)
        # Second-order differentiation matrix against second-order derivative        
        D2 = diff2fd24(n,2,2./(n-1))
        D4 = diff2fd24(n,4,2./(n-1))
        du_true, u = testf(x,2)
        du2 = D2.dot(u)
        du4 = D4.dot(u)
        # Discrete L2-norm -- right-hand rule
        err2 = tools.normd(du2-du_true,x,'even')
        print("Discrete L2-error (2nd order) is %20.15e for n = %4d") % (err2,n)
        err4 = tools.normd(du4-du_true,x,'even')
        print("Discrete L2-error (4th order) is %20.15e for n = %4d") % (err4,n)
        
if __name__ == "__main__":
    print "Running Test"
    test_diff1lag()
    test_diff2lag()
    test_diff1fd24()
    test_diff2fd24()