# -*- coding: utf-8 -*-
"""
Created on Fri Oct 09 12:58:24 2015

@author: Keith J. Wojciechowski (KJW)

Module containing functions for constructing tools to perform 
radial basis function (RBF) interpolation and differentiation.

dmatrix constructs the distance matrix between points and centers

rbfinterp solves the collocation problem (interpolate) data to a surface (curve)

RBFs:
dmatrix is the linear RBF, f(r) = r
dmatrix(d,c)

cubic, f(r) = r^3
cubic(d,c,order=0,ep=0)

multiquadric, f(r) = sqrt(1 + (ep r)^2)
mq(d,c,order=0,ep=1)

gauss, f(r) = exp(-(ep r)^2)
gauss(d,c,order=0,ep=1)

RBF Remarks:
(1) r is the distance matrix, dmatrix(d,c) where d = data, c = centers
(2) ep is the shape parameter epsilon
(3) the interface for each RBF function is (d,c,order,ep)
    d = data, c = centers, order = derivative order, ep = shape parameter
(4) Even though an RBF does not require a shape parameter, send one anyway, this
    convention permits a general differentiation matrix interface for RBF-FD
(5) RBFs are globally supported but not all are strictly positive definite

*** TO DO: CONSTRUCT RBF-GLOBAL DIFFERENTIATION MATRICES
*** TO DO: CONSTRUCT RBF-FD DIFFERENTIATION MATRICES
*** TO DO: CONSTRUCT RBF-LA DIFFERENTIATION MATRICES

*** Unit tests for each function are contained in the module ***
"""

from pylab import array, dot, exp, linalg, linspace, norm, ones, sqrt, zeros

def dmatrix(d,c):
    # DM = dmatrix(d,c)
    #    
    # Arguments:
    # d = data, c = centers
    #
    # Typically d = c but, in general, data does not have to equal its centers
    #
    # Output DM:
    # Compute the distance matrix with entries being the distances between the
    # data and the centers.
    # The Euclidian distance matrix, DM, is the m by n matrix with entries
    #      ||d_0 - c_0|| ||d_0 - c_1|| ... ||d_0 - c_n||
    #      ||d_1 - c_0|| ||d_1 - c_1|| ... ||d_1 - c_n||
    #                       ...
    #      ||d_m - c_0|| ||d_m - c_1|| ... ||d_m - c_n||
    #
    # m = # pts, n = dim of space
    #
    # ****** ASSUMPTION: # pts >= dimension of space
    # ****** ASSUMPTION: c, d are ROW vectors, otherwise convert to row vectors
    #
    # Remark:
    # d and c are called vectors but it might be more appropriate to call
    # them matrices (or rank dim(d), rank dim(c) tensors). When called vectors
    # it is assumed that each row is a vector in the space implying the number
    # of columns is the dimension of the space and the number of rows is the
    # number of points
    #
    # Test Input:
    # Are d and c arrays of row vectors?
    # If d and c are column vectors, convert them to row vectors.
    # If d and c are square, i.e. # pts = dimension of space, notify user  
    if d.ndim > 1:    
        if d.shape[1] > d.shape[0]:
            d = d.T
        elif d.shape[1] == d.shape[0]:
            print("Assuming data is in row-vector form.")
    else:   # 1-D data, convert to 2-D data with shape (M,1)
        d = array([d]).T
    if c.ndim > 1:
        if c.shape[1] > c.shape[0]:
            c = c.T
        elif c.shape[1] == c.shape[0]:
            print("Assuming centers are in row-vector form.")
    else:   # 1-D data, convert to 2-D data with shape (N,1)
        c = array([c]).T
    # **************************************************************************
    #                               Begin Algorithm
    # **************************************************************************
    # Obtain shape of input:
    M, sd = d.shape    # M = # pts, sd = dim of data space
    N, sc = c.shape    # N = # pts, sc = dim of centers space
    #
    # Raise error if centers and data have different dimension    
    if sd != sc:
        raise NameError('Data and centers must have same dimension')
    # ********** Construct the Distance Matrix DM **********
    # Initialize the distance matrix: (data # of pts) by (centers # of pts)
    # Denote the 
    # d_0 = (d[0,0], d[0,1], ...), d_1 = (d[1,0], d[1,1], ...), etc.
    #
    # The distance matrix is the M by N matrix with entries
    #      ||d_0 - c_0|| ||d_0 - c_1|| ... ||d_0 - c_n||
    #      ||d_1 - c_0|| ||d_1 - c_1|| ... ||d_1 - c_n||
    #                       ...
    #      ||d_m - c_0|| ||d_m - c_1|| ... ||d_m - c_n||
    #
    DM = zeros((M,N))
    # Determine the distance of each point in the data-set from its center
    for i in xrange(M):
        # Compute the row ||d_i - c_0|| ||d_i - c_1|| ... ||d_i - c_n||
        DM[i,:] = ((d[i]-c)**2).sum(1)
    # Finish distance formula by taking square root of each entry
    return sqrt(DM)

def rbfinterp(d,c,s,p,func = 'dist',ep=1):
    #
    # *** TO DO: create an rbf function, allow user input for rbf type
    #
    # yp = rbfinterp(d,c,s,p)
    #
    # Use Radial Basis Functions (rbf) to interpolate
    #
    # Arguments:
    # d = data, c = centers
    # s = surface (curve) to be interpolated
    # p = evaluation points (points at which s is to be interpolated)
    # rbf = Radial Basis Function (RBF) to be used for interpolation
    #       default = Euclidian distance matrix
    #
    # Construct the rbf
    # *** TO DO: convert to a dictionary or hashmap (with derivatives, etc.)
    if func == 'dist':
        IM = dmatrix(d,c)
        EM = dmatrix(p,c)
    elif func == 'mq':
        IM = mq(d,c,0,ep)
        EM = mq(p,c,0,ep)
    elif func == 'cubic':
        IM = cubic(d,c,0,ep)
        EM = cubic(p,c,0,ep)
    elif func == 'gauss':
        IM = gauss(d,c,0,ep)
        EM = gauss(p,c,0,ep)
    else:
        raise NameError("Enter a valid RBF.")    
    #***************************************************************************
    # Linear Algebra Remarks:
    # 
    # P*a = s is a system of equations where the coefficients, a, are unknown
    # This matrix system is called the "collocation problem," i.e. What weights
    # are needed so that a linear combination of basis functions and weights
    # yeilds a point on the surface?
    # Once the weights, a, are determined, they can be used to construct the
    # interpolant.
    #
    # Summary:
    # P*a = s => a = inv(P)*s
    # EM*a = yp where yp is the interpolant and EM is the matrix with entries
    # that are known basis functions at the evaluation points.
    #
    # Since a = inv(P)*s, EM*a = yp => EM*inv(P)*s
    #
    return dot(EM,linalg.solve(IM,s))

'''
********************************************************************************
                                    RBF ZOO
********************************************************************************
'''
def cubic(d,c,order=0,ep=0):
    # cubic, f(r) = r^3
    DM = dmatrix(d,c)
    return DM**3
    
def mq(d,c,order=0,ep=1):
    # multiquadric, f(r) = sqrt(1 + (ep r)^2)
    DM = dmatrix(d,c)
    e = ep*ones(DM.shape)
    return sqrt(1+(e*DM)**2)

def gauss(d,c,order=0,ep=1):
    # gaussian, f(r) = exp(-(ep r)^2)
    DM = dmatrix(d,c)
    e = ep*ones(DM.shape)
    return exp(-(e*DM)**2)

'''
********************************************************************************
                                UNIT TESTS
********************************************************************************
'''  
def testfunction(data):
    # N-D Gaussian
    N, sd = data.shape
    f = ones((N,1))
    for i in xrange(sd):
        f = f*array([exp(-15*(data[:,i]-0.5)**2)]).T
        
    return f
    
def test_interp():
    # Testing interpolation
    x = linspace(0,1,19)    
    xp = linspace(0.01,0.99,33)
    rbf_list = ['dist','cubic','mq','gauss']
    ep_list = [0.01,0.5,1.0,1.5,2]
    for ep in ep_list:
        for ff in rbf_list:
            # 1D
            data = array([x]).T
            ctrs = data
            p = array([xp]).T
            rhs = testfunction(data)
            exact = testfunction(p)
            Pf = rbfinterp(data,ctrs,rhs,p,ff,ep)

            err = norm(Pf-exact)
        
            print "1D interp, %s, shape = %f, L2 error = %e " % (ff,ep,err)
    
            # 2D
            data = array([x,x]).T
            ctrs = data
            p = array([xp,xp]).T
            rhs = testfunction(data)
            exact = testfunction(p)
            Pf = rbfinterp(data,ctrs,rhs,p,ff,ep)
        
            err = norm(Pf-exact)
        
            print "2D interp, %s, shape = %f, L2 error = %e " % (ff,ep,err)
        
            # 3D
            data = array([x,x,x]).T
            ctrs = data
            p = array([xp,xp,xp]).T
            rhs = testfunction(data)
            exact = testfunction(p)
            Pf = rbfinterp(data,ctrs,rhs,p,ff,ep)
        
            err = norm(Pf-exact)
        
            print "3D interp, %s, shape = %f, L2 error = %e " % (ff,ep,err) 
            print "----------------------------------------------------"
            
def test_dmatrix():
    # Unit tests for the dmatrix function
    x = linspace(0,1,5)
    # Test 1D without formatting input, data is 1D, shape is (N,)
    data = x
    ctrs = data
    DM = dmatrix(data,ctrs)
    print DM
    
    # Test 1D with x in wrong orientation (dim by N pts), data is 2D array
    data = array([x])
    ctrs = data
    DM = dmatrix(data,ctrs)
    print DM    
    
    # Test 1D with x in correct orientation (N by dim pts), data is 2D array
    data = array([x]).T
    ctrs = data
    DM = dmatrix(data,ctrs)
    print DM  

    # Test 2D with x in wrong orientation (dim by N pts), data is 2D array
    data = array([x,x])
    ctrs = data
    DM = dmatrix(data,ctrs)
    print DM    

    # Test 2D with x in correct orientation (N by dim pts), data is 2D array
    data = array([x,x]).T
    ctrs = data
    DM = dmatrix(data,ctrs)
    print DM   

    # Test 3D with x in wrong orientation (dim by N pts), data is 2D array
    data = array([x,x,x])
    ctrs = data
    DM = dmatrix(data,ctrs)
    print DM    

    # Test 3D with x in correct orientation (N by dim pts), data is 2D array
    data = array([x,x,x]).T
    ctrs = data
    DM = dmatrix(data,ctrs)
    print DM