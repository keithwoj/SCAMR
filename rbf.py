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
-------------------------------------------------------------------------------
                                FUNCTION LIST
-------------------------------------------------------------------------------
dmatrix(d,**centers) builds Euclidian distance matrix between data and centers
d = data, **centers = centers (default = data)

rbfinterp(d,s,p,**rbfparms) solves a collocation problem, fits surface to data
d = data, s = surface, p = evaluation points, **rbfparms = see RBF Zoo below

*********************************** RBF Zoo ***********************************

polyharmonic spline, f(r) = r^m for m odd, positive integer
                     f(r) = r^m log(r) for m even, positive integer
phs(d,**parms)
**parms = centers (default = data), power 

multiquadric, f(r) = sqrt(1 + (ep r)^2)
mq(d,**parms)
**parms = centers (default = data), shapeparm

gauss, f(r) = exp(-(ep r)^2)
gauss(d,**parms)
**parms = centers (default = data), shapeparm

RBF Remarks:
(1) r is the distance matrix, dmatrix(d,c) where d = data, c = centers
(2) centers not included in argment list => centers = data
(3) shapeparm is shape parameter, epsilon, may be vector of varying parameters
(4) RBFs are globally supported but not all are strictly positive definite

*** TO DO: DMATRIX FUNCTION c = kwargs.get('centers',d) RETURNS NONE, WHY?
*** TO DO: ALLOW ep TO BE A VECTOR, CONVERT ep*ones --> DM*diag(ep)
*** TO DO: CONSTRUCT RBF-GLOBAL DIFFERENTIATION MATRICES
*** TO DO: CONSTRUCT RBF-FD DIFFERENTIATION MATRICES
*** TO DO: CONSTRUCT RBF-LA DIFFERENTIATION MATRICES

*** Unit tests for each function are contained in the module ***
"""
## from importlib import reload
from matplotlib.pylab import array, dot, exp, linalg, linspace, log, norm, ones
from matplotlib.pylab import sqrt, zeros, eye, randn, cos, pi

def dmatrix(d,**centers):
    """
    DM = dmatrix(d,**centers)
       
    Arguments:
    d = data
    *centers may contain centers, c, different from d, otherwise c = d
        
    Typically d = c but, in general, data does not have to equal its centers
    as in the case of the evaluation matrix, where the d becomes the
    evaluation points and the centers are the collocation data.
    
    Output DM:
    Compute the distance matrix with entries being the distances between the
    data and the centers.
    The Euclidian distance matrix, DM, is the m by n matrix with entries
         ||d_0 - c_0|| ||d_0 - c_1|| ... ||d_0 - c_n||
         ||d_1 - c_0|| ||d_1 - c_1|| ... ||d_1 - c_n||
                          ...
         ||d_m - c_0|| ||d_m - c_1|| ... ||d_m - c_n||
    
    m = # pts, n = dim of space
    
    ****** ASSUMPTION: # pts >= dimension of space
    ****** ASSUMPTION: c, d are ROW vectors, otherwise convert to row vectors
    
    Remark:
    d and c are called vectors but it might be more appropriate to call
    them matrices (or rank dim(d), rank dim(c) tensors). When called vectors
    it is assumed that each row is a vector in the space implying the number
    of columns is the dimension of the space and the number of rows is the
    number of points
    """
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
    if centers.get('centers') is None:
        c = d
    else:
        c = centers.get('centers')

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
    for i in range(M):
        # Compute the row ||d_i - c_0|| ||d_i - c_1|| ... ||d_i - c_n||
        DM[i,:] = ((d[i]-c)**2).sum(1)
    # Finish distance formula by taking square root of each entry
    return sqrt(DM)

def rbfinterp(d,s,p,rbf,**rbfparms):
    """
    yp = rbfinterp(data, surface, evaluation points, rbf, *rbfparms)
    
    Use Radial Basis Functions (rbf) to interpolate using Infinitely Smooth
    RBF or Polyharmonic Spline (PHS)
    
    Arguments:
                    (TYPE ARRAY SHAPE N, 2)
    d = data                                            array shape (N,2)
    s = surface (curve) to be interpolated              array shape (N,1)
    p = evaluation points (s is interpolated)           array shape (M,2)
    *rbfparms = ep', 'm'
          ep = shape parameter for RBFs                scalar / list shape (N,)
          m  = exponent for polyharmonic spline (PHS)  scalar
    
    Output:
    yp = surface interpolated at the evaluation points  array shape (M,1)
    """
    # Construct the collocation matrices:
    # ep = shape parameter
    ep = rbfparms.get('shapeparm')
    #  m = power for PHS
    m = rbfparms.get('power')
    
    zoo = {
        'linear': linear,
        'phs': phs,
        'mq': mq,
        'gauss': gauss
        }
    if rbf in zoo:
        rbf = zoo[rbf]
    else:
        raise NameError('RBF not known.')
    
    # IM = interpolation matrix
    IM = rbf(d, shapeparm = ep, power = m)
    # EM = evaluation matrix
    EM = rbf(p, centers = d, shapeparm = ep, power = m)
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
    
def phs(d,**parms):
    # phs, f(r) = r^m for m odd positive integer
    c = parms.get('centers')    
    m = parms.get('power')
    DM = dmatrix(d, centers = c)
    # Check to see if m is a positive integer
    if (m == int(m)) & (m > 0):
        if (m%2):
            #print("PHS odd m = {}".format(m))
            return DM**m            
        else:
            #print("PHS even m = {}".format(m))
            return DM**m*log(DM + 1*(DM==0))            
    else:
        raise NameError("PHS power must be a positive integer.")            
    
def mq(d,**parms):
    # multiquadric, f(r) = sqrt(1 + (ep r)^2)
    c = parms.get('centers')
    #op = parms.get('operator','interp')
    ep = parms.get('shapeparm',1)
    DM = dmatrix(d, centers = c)
    # eps_r = epsilon*r where epsilon may be an array or scalar
    eps_r = dot(ep*eye(DM.shape[0]),DM)
    return sqrt(1+(eps_r)**2)

def gauss(d,**parms):
    # gaussian, f(r) = exp(-(ep r)^2)
    c = parms.get('centers')
    #op = parms.get('operator','interp')
    ep = parms.get('shapeparm',1)
    DM = dmatrix(d, centers = c)
    # eps_r = epsilon*r where epsilon may be an array or scalar
    eps_r = dot(ep*eye(DM.shape[0]),DM)
    return exp(-(eps_r)**2)
'''
--------------------------------------------------------------------------------
                                UNIT TESTS
--------------------------------------------------------------------------------
'''  
def testfunction(data):
    # N-D Gaussian or N-D Runge Function
    N, sd = data.shape
    f = ones((N,1))
    for i in range(sd):
        #f = f*array([exp(-15*(data[:,i]-0.5)**2)]).T
        f = f*array([1./(1+(5*data[:,i])**2)]).T
        
    return f
    
def test_interp():
    # Testing interpolation
    nn = 33 # number of nodes
    ne = 65 # number of evaluation points
    x = cos(pi*(1+array(range(nn)))/(nn+1))
    xp = linspace(-1,1,ne)
    rbf_list = ['mq','gauss','phs']   
    ep_list = [3.,5.,7.,9.]
    m = 3
    for ep in ep_list:
        for ff in rbf_list:
            # 1D
            d = array([x]).T
            p = array([xp]).T
            rhs = testfunction(d)
            exact = testfunction(p)
            Pf = rbfinterp(d,rhs,p,ff,shapeparm = ep, power = m)

            err = norm(Pf-exact)
            
            print("1D interp, {}, shape = {}, L2 error = {:e}".format(ff,ep,err))
    
            # 2D
            d = array([x,x]).T
            p = array([xp,xp]).T
            rhs = testfunction(d)
            exact = testfunction(p)
            Pf = rbfinterp(d,rhs,p,ff,shapeparm = ep, power = m)
        
            err = norm(Pf-exact)
            
            print("2D interp, {}, shape = {}, L2 error = {:e}".format(ff,ep,err))
        
            # 3D
            d = array([x,x,x]).T
            p = array([xp,xp,xp]).T
            rhs = testfunction(d)
            exact = testfunction(p)
            Pf = rbfinterp(d,rhs,p,ff,shapeparm = ep, power = m)
        
            err = norm(Pf-exact)
        
            print("3D interp, {}, shape = {}, L2 error = {:e}".format(ff,ep,err))
            print("----------------------------------------------------------")
    
def test_dmatrix():
    # Unit tests for the dmatrix function
    x = linspace(0,1,5)
    # Test 1D without formatting input, data is 1D, shape is (N,)
    data = x
    DM = dmatrix(data)
    print(DM)
    
    # Test 1D with x in wrong orientation (dim by N pts), data is 2D array
    data = array([x])
    DM = dmatrix(data)
    print(DM)   
    
    # Test 1D with x in correct orientation (N by dim pts), data is 2D array
    data = array([x]).T
    DM = dmatrix(data)
    print(DM)

    # Test 2D with x in wrong orientation (dim by N pts), data is 2D array
    data = array([x,x])
    DM = dmatrix(data)
    print(DM) 

    # Test 2D with x in correct orientation (N by dim pts), data is 2D array
    data = array([x,x]).T
    DM = dmatrix(data)
    print(DM)   

    # Test 3D with x in wrong orientation (dim by N pts), data is 2D array
    data = array([x,x,x])
    DM = dmatrix(data)
    print(DM)  

    # Test 3D with x in correct orientation (N by dim pts), data is 2D array
    data = array([x,x,x]).T
    DM = dmatrix(data)
    print(DM)