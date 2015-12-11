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
from matplotlib.pylab import sqrt, zeros, eye, meshgrid, cos, pi

def diffmatrix(d,**centers):
    """
    DM = diffmatrix(d,**centers)
       
    Parameters
    --------
    d = data : array_like (N,s) where N = number of points, s = dimension
    *centers may contain centers, c, different from d, otherwise c = d
        
    Typically d = c but, in general, data does not have to equal its centers
    as in the case of the evaluation matrix, where the d becomes the
    evaluation points and the centers are the collocation data.
    
    Returns
    -------
    dr.T-cc.T : ndarray
    
    Compute the difference matrix with entries being the differences between
    data and the centers.

    d_0 = (d[0,0], d[0,1], ...), d_1 = (d[1,0], d[1,1], ...), etc.    
    
    The difference matrix is the m by n matrix with entries
         d_0 - c_0 d_0 - c_1 ... d_0 - c_n
         d_1 - c_0 d_1 - c_1 ... d_1 - c_n
                          ...
         d_m - c_0 d_m - c_1 ... d_m - c_n
    
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
    ## **************** WHY DOES c = kwargs.get('centers',d) RETURN NONE????
    if centers.get('centers') is None:
        c = d
    else:
        c = centers.get('centers')
        
    # Test Input:
    # Are d and c arrays of row vectors?
    # If d and c are column vectors, convert them to row vectors.
    # If d and c are square, i.e. # pts = dimension of space, notify user
    if centers.get('clean') is None:
        d = data_check(d)
        c = data_check(c)
    
    M, sd = d.shape
    N, sc = c.shape
    # **************************************************************************
    #                               Begin Algorithm
    # **************************************************************************
    #
    # Raise error if centers and data have different dimension    
    if sd != sc:
        raise ValueError('Data and centers must have same dimension')
    # ********** Construct the Difference Matrix **********
    dr,cc = meshgrid(d,c)
    return dr.T-cc.T

def dmatrix(d,**centers):
    """
    DM = dmatrix(d,**centers)
       
    Parameters
    --------
    d = data : array_like (N,s) where N = number of points, s = dimension
    *centers may contain centers, c, different from d, otherwise c = d
        
    Typically d = c but, in general, data does not have to equal its centers
    as in the case of the evaluation matrix, where the d becomes the
    evaluation points and the centers are the collocation data.
    
    Returns
    -------
    sqrt(DM) : ndarray
    
    Compute the distance matrix with entries being the distances between the
    data and the centers.

    DM is the Euclidian distance matrix, the m by n matrix with entries
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
    if centers.get('centers') is None:
        c = d
    else:
        c = centers.get('centers')
        
    # Test Input:
    # Are d and c arrays of row vectors?
    # If d and c are column vectors, convert them to row vectors.
    # If d and c are square, i.e. # pts = dimension of space, notify user
    if centers.get('clean') is None:
        d = data_check(d)
        c = data_check(c)
    
    M, sd = d.shape
    N, sc = c.shape
    # **************************************************************************
    #                               Begin Algorithm
    # **************************************************************************
    #
    # Raise error if centers and data have different dimension    
    if sd != sc:
        raise ValueError('Data and centers must have same dimension')
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
    
    Parameters
    ----------
    d = data : array_like (N,s)  where N = number of points, s = dimension
    s = surface (curve) to be interpolated : array_like (N,s)
    p = evaluation points (s is interpolated) : array_like (M,s)
    *rbfparms = ep', 'm'
          ep = shape parameter for RBFs : scalar or array_like (N,)
          m  = exponent for polyharmonic spline (PHS) :  scalar
    
    Returns
    -------
    yp = surface interpolated at the evaluation points :  array_like (M,s)
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
        raise NameError('RBF not known')
    
    # IM = interpolation matrix
    IM = rbf(d, 'interp', shapeparm = ep, power = m)
    # EM = evaluation matrix
    EM = rbf(p, 'interp', centers = d, shapeparm = ep, power = m)
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
    
def rbfdiff(d,rbf,op,geo,**parms):
    # ep = shape parameter
    ep = parms.get('shapeparm')
    #  m = power for PHS
    m = parms.get('power')
    
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
    
    opzoo = ['d1','d2','grad','div','laplacian','curl']
    
    if op not in opzoo:
        raise NameError('operator not known')
    
    geozoo = ['cartesian','radial','polar','cylindrical','spherical']
        
    if geo not in geozoo:
        raise NameError('unknown geometry')
    
    L = rbf(d, op, centers = d, shapeparm = ep, power = m, operator = op, geometry = geo)
    
    return L
'''
--------------------------------------------------------------------------------
                                    RBF ZOO
--------------------------------------------------------------------------------
'''
def linear(d, op = 'interp', **parms):
    # linear, f(r) = r
    c = parms.get('centers')
    return dmatrix(d, centers = c)
    
def phs(d, op = 'interp', **parms):
    # phs, f(r) = r^m for m odd positive integer
    if parms.get('centers') is None:
        c = d
    else:
        c = parms.get('centers') 
        
    m = parms.get('power')
    
    d = data_check(d)
    c = data_check(c)

    M, sc = c.shape
    N, sd = d.shape    
    
    # Construct the distance matrix
    DM = dmatrix(d, centers = c, clean = 1)
    # Check to see if m is a positive integer
    if (m == int(m)) & (m > 2):
        if (m%2):
            #print("PHS odd m = {}".format(m))
            if (op == 'interp'):
                return DM**m            
            else:
                IM = DM**m
        else:
            #print("PHS even m = {}".format(m))
            if (op == 'interp'):
                return DM**m*log(DM + 1*(DM==0))
            else:
                IM = DM**m*log(DM + 1*(DM==0))
    else:
        raise ValueError("PHS power must be a positive integer greater than 2.")

    if (op == 'd1' and geo == 'radial'):
        if (m%2):
            Lx = m*DM**(m-1)
        else:
            Lx = DM**(m-1)*(m*log(DM + 1*(DM==0)) + 1)
    if (op == 'd2' and geo == 'radial'):
        if (m%2):
            Lx = m*(m-1)*DM**(m-2)
        else:
            Lx = DM**(m-2)*((m-1)*(m*log(DM + 1*(DM==0)) + 1)+m)
    if (op == 'laplacian' and geo == 'radial'):
        if (m%2):
            Lx = m**2*DM**(m-2)
        else:
            Lx = 1

    
    return dot(Lx,linalg.inv(IM))
              
    
def mq(d, op = 'interp', **parms):
    """
    Note on differentiation of MQ RBFs:
    ||x-xj|| = sqrt{(x-xj)^2}
    => d/dx ||x-xj||^2 = d/dx (x-xj)^2 = 2(x-xj)

    d/dx sqrt(1 + ep^2||x-xj||^2) = ep^2*(x-xj)/sqrt(1+ep^2||x-xj||^2)

     d^2/dx^2 sqrt(1 + ep^2||x-xj||^2)
     = d/dx ep^2*(x-xj)/sqrt(1+ep^2||x-xj||^2)
     = ep^2(1+ep^2||x-xj||^2)^(-1/2) - ep^4(x-xj)(1+ep^2||x-xj||^2)^(-3/2)
     = [ep^2+(x-xj - 1)ep^4(x-xj)]/(1+ep^2||x-xj||^2)^(3/2)    
    """ 
    # multiquadric, f(r) = sqrt(1 + (ep r)^2)
    c = parms.get('centers')
    #op = parms.get('operator','interp')
    ep = parms.get('shapeparm',1)
    DM = dmatrix(d, centers = c)
    # eps_r = epsilon*r where epsilon may be an array or scalar
    eps_r = dot(ep*eye(DM.shape[0]),DM)
    if op == 'interp':
        return sqrt(1+(eps_r)**2)

def gauss(d, op = 'interp', **parms):
    # gaussian, f(r) = exp(-(ep r)^2)
    c = parms.get('centers')
    #op = parms.get('operator','interp')
    ep = parms.get('shapeparm',1)
    DM = dmatrix(d, centers = c)
    # eps_r = epsilon*r where epsilon may be an array or scalar
    eps_r = dot(ep*eye(DM.shape[0]),DM)
    if op == 'interp':
        return exp(-(eps_r)**2)

'''
-------------------------------------------------------------------------------
                        UTILITY FUNCTIONS
-------------------------------------------------------------------------------
'''

def data_check(d):
    if d.ndim > 1:    
        if d.shape[1] > d.shape[0]:
            d = d.T
        elif d.shape[1] == d.shape[0]:
            print("Assuming data is in row-vector form.")
    else:   # 1-D data, convert to 2-D data with shape (M,1)
        d = array([d]).T
          
    return d

def block_diag_exract(A,nb):
    N = A.shape[0]
    return [A[i*N:(i+1)*N,i*N:(i+1)*N] for i in range(nb)]

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
        ValueError('Surface and axes must have same dimension.')    
    
'''
-------------------------------------------------------------------------------
                                UNIT TESTS
-------------------------------------------------------------------------------
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