# -*- coding: utf-8 -*-
"""
Created on Fri Oct 09 12:58:24 2015

@author: Keith J. Wojciechowski (KJW)

Module containing functions for constructing tools to perform 
radial basis function (RBF) interpolation and differentiation.

Some References:
(1) Fasshauer G.E., Meshfree Approximation Methods with MATLAB, World Scientific
(2) Fornberg B., Flyer N., A Primer on Radial Basis Functions with Applications
    to the Geosciences, SIAM
--------------------------------------------------------------------------------
                                FUNCTION LIST
--------------------------------------------------------------------------------
dmatrix(d,**kwargs) builds Euclidian distance matrix between data and centers

rbfinterp(d,s,p,rbfparms) solves a collocation problem, fits surface to data

*** RBFs Included:

linear, f(r) = r
linear(d,**parms)

cubic, f(r) = r^3
cubic(d,**parms)

multiquadric, f(r) = sqrt(1 + (ep r)^2)
mq(d,**parms)

gauss, f(r) = exp(-(ep r)^2)
gauss(d,**parms)

RBF Remarks:
(1) r is the distance matrix, dmatrix(d,c) where d = data, c = centers
(2) ep is the shape parameter epsilon, may be a vector of varying parameters
(3) the interface for each RBF function is (d,c,order,ep)
    d = data, c = centers, order = derivative order, ep = shape parameter
(4) Even though an RBF does not require a shape parameter, send one anyway, this
    convention permits a general differentiation matrix interface for RBF-FD
(5) RBFs are globally supported but not all are strictly positive definite

*** TO DO: ALLOW ep TO BE A VECTOR, CONVERT ep*ones --> DM*diag(ep)
*** TO DO: CONSTRUCT RBF-GLOBAL DIFFERENTIATION MATRICES
*** TO DO: CONSTRUCT RBF-FD DIFFERENTIATION MATRICES
*** TO DO: CONSTRUCT RBF-LA DIFFERENTIATION MATRICES

*** Unit tests for each function are contained in the module ***
"""

from pylab import array, dot, exp, linalg, linspace, norm, ones, sqrt, zeros

def dmatrix(d,**kwargs):
    # DM = dmatrix(d,**kwargs)
    #    
    # Arguments:
    # d = data
    # kwargs may contain centers, c
    #     
    # Typically d = c but, in general, data does not have to equal its centers
    # as in the case of the evaluation matrix, where the d becomes the
    # evaluation points and the centers are the collocation data.
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
    
    ## **************** WHY DOES c = kwargs.get('centers',d) RETURN NONE????
    if kwargs.get('centers') is None:
        c = d
    else:
        c = kwargs.get('centers')

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

def rbfinterp(d,s,p,**rbfparms):
    #
    # yp = rbfinterp(d,s,p,**rbfparms)
    #
    # Use Radial Basis Functions (rbf) to interpolate
    #
    # Arguments:
    # d = data
    # s = surface (curve) to be interpolated
    # p = evaluation points (points at which s is to be interpolated)
    # rbfparms = 'rbf','ep', 'op'
    #       rbf = Radial Basis Function (RBF) to be used for interpolation
    #       ep = shape parameter for RBF
    #       op = which operators are needed (*** items may be a list ***)
    #          {'interp', 'first', 'second', 'div', 'grad', 'curl', 'laplacian'}
    #
    # Construct the collocation matrices:
    # bfunc = rbf
    bfunc = rbfparms.get('rbf',dmatrix)
    # op = operator
    op = rbfparms.get('op','interp')
    # ep = shape parameter
    ep = rbfparms.get('ep')
    # IM = interpolation matrix
    IM = bfunc(d, operator = op, shapeparm = ep)
    # EM = evaluation matrix
    EM = bfunc(p, centers = d, operator = op, shapeparm = ep)
    #***************************************************************************
    # Linear Algebra Remarks:
    # 
    # P*w = s is a system of equations where the coefficients, w, are unknown
    # This matrix system is called the "collocation problem," i.e. What weights
    # are needed so that a linear combination of basis functions and weights
    # yeilds a point on the surface?
    # Once the weights, w, are determined, they can be used to construct the
    # interpolant.
    #
    # Summary:
    # P*w = s => w = inv(P)*s
    # EM*w = yp where yp is the interpolant and EM is the matrix with entries
    # that are known basis functions at the evaluation points.
    #
    # Since w = inv(P)*s, EM*w = yp => EM*inv(P)*s
    #
    return dot(EM,linalg.solve(IM,s))
'''
--------------------------------------------------------------------------------
                                    RBF ZOO
--------------------------------------------------------------------------------
'''
def linear(d,**parms):
    # linear, f(r) = r
    c = parms.get('centers')
    return dmatrix(d, centers = c)
    
def cubic(d,**parms):
    # cubic, f(r) = r^3
    c = parms.get('centers')    
    #op = parms.get('operator','interp')
    DM = dmatrix(d, centers = c)
    return DM**3
    
def mq(d,**parms):
    # multiquadric, f(r) = sqrt(1 + (ep r)^2)
    c = parms.get('centers')
    #op = parms.get('operator','interp')
    ep = parms.get('shapeparm',1)
    DM = dmatrix(d, centers = c)
    e = ep*ones(DM.shape)
    return sqrt(1+(e*DM)**2)

def gauss(d,**parms):
    # gaussian, f(r) = exp(-(ep r)^2)
    c = parms.get('centers')
    #op = parms.get('operator','interp')
    ep = parms.get('shapeparm',1)
    DM = dmatrix(d, centers = c)
    e = ep*ones(DM.shape)
    return exp(-(e*DM)**2)
'''
--------------------------------------------------------------------------------
                                UNIT TESTS
--------------------------------------------------------------------------------
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
    rbf_list = [linear,cubic,mq,gauss]
    ep_list = [0.01,0.5,1.0,1.5,2]
    for ee in ep_list:
        for ff in rbf_list:
            # 1D
            d = array([x]).T
            p = array([xp]).T
            rhs = testfunction(d)
            exact = testfunction(p)
            Pf = rbfinterp(d,rhs,p,rbf = ff,ep = ee)

            err = norm(Pf-exact)
            
            print "1D interp, %s, shape = %f, L2 error = %e|" % (ff.func_name,ee,err)
    
            # 2D
            d = array([x,x]).T
            p = array([xp,xp]).T
            rhs = testfunction(d)
            exact = testfunction(p)
            Pf = rbfinterp(d,rhs,p,rbf = ff,ep = ee)
        
            err = norm(Pf-exact)
        
            print "2D interp, %s, shape = %f, L2 error = %e|" % (ff.func_name,ee,err)
        
            # 3D
            d = array([x,x,x]).T
            p = array([xp,xp,xp]).T
            rhs = testfunction(d)
            exact = testfunction(p)
            Pf = rbfinterp(d,rhs,p,rbf = ff,ep = ee)
        
            err = norm(Pf-exact)
        
            print "3D interp, %s, shape = %f, L2 error = %e|" % (ff.func_name,ee,err) 
            print "------------------------------------------------------------"
            
def test_dmatrix():
    # Unit tests for the dmatrix function
    x = linspace(0,1,5)
    # Test 1D without formatting input, data is 1D, shape is (N,)
    data = x
    DM = dmatrix(data)
    print DM
    
    # Test 1D with x in wrong orientation (dim by N pts), data is 2D array
    data = array([x])
    DM = dmatrix(data)
    print DM    
    
    # Test 1D with x in correct orientation (N by dim pts), data is 2D array
    data = array([x]).T
    DM = dmatrix(data)
    print DM  

    # Test 2D with x in wrong orientation (dim by N pts), data is 2D array
    data = array([x,x])
    DM = dmatrix(data)
    print DM    

    # Test 2D with x in correct orientation (N by dim pts), data is 2D array
    data = array([x,x]).T
    DM = dmatrix(data)
    print DM   

    # Test 3D with x in wrong orientation (dim by N pts), data is 2D array
    data = array([x,x,x])
    DM = dmatrix(data)
    print DM    

    # Test 3D with x in correct orientation (N by dim pts), data is 2D array
    data = array([x,x,x]).T
    DM = dmatrix(data)
    print DM