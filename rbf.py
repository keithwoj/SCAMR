# -*- coding: utf-8 -*-
"""
Created on Fri Oct 09 12:58:24 2015

@author: Keith J. Wojciechowski (KJW)
"""

from pylab import array, dot, exp, linalg, linspace, norm, ones, shape, sqrt, zeros

def dmatrix(d,c):
    # d = data, c = centers
    #
    # Typically d = c but, in general, data does not have to equal its centers
    #
    # Compute the distance matrix with entries being the distances between the
    # data and the centers.
    # The Euclidian distance matrix is the m by n matrix with entries
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
    # d and c are called vectors but it might be more appropriate to call
    # them matrices (or rank dim(d), rank dim(c) tensors). When called vectors
    # it is assumed that each row is a vector in the space implying the number
    # of columns is the dimension of the space and the number of rows is the
    # number of points
    #
    # Are d and c row vectors?
    # If d and c are column vectors, convert them to row vectors.
    # If d and c are square, i.e. # pts = dimension of space, notify user  
    if d.ndim > 1:    
        if d.shape[1] > d.shape[0]:
            d = d.T
        elif d.shape[1] == d.shape[0]:
            print("Assuming data is in row-vector form.")
        M, sd = d.shape    # M = # pts, sd = dim of data space
    else:   # 1-D data
        M = d.size
        sd = 1
    if c.ndim > 1:
        if c.shape[1] > c.shape[0]:
            c = c.T
        elif c.shape[1] == c.shape[0]:
            print("Assuming centers are in row-vector form.")
        N, sc = c.shape    # N = # pts, sc = dim of centers space
    else:   # 1-D data
        N = c.size
        sc = 1

    if sd != sc:
        raise NameError('Data and centers must be same dimension')
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
    
def testfunction(data):
    N, sd = data.shape
    p = ones((N,1))
    for i in xrange(sd):
        p = p*array([exp(-15*(data[:,i]-0.5)**2)]).T
        
    return p
    
def test_dmatrix():
    x = linspace(0,1,19)
    xp = linspace(0.01,0.99,33)    
    #data = array([x]).T
    #ep = array([xp]).T
    #data = array([x,x]).T
    #ep = array([xp,xp]).T
    data = array([x,x,x]).T
    ep = array([xp,xp,xp]).T
    ctrs = data
    IM = dmatrix(data,ctrs)
    EM = dmatrix(ep,ctrs)
    
    rhs = testfunction(data)

    Pf = dot(EM,linalg.solve(IM,rhs))
    exact = testfunction(ep)
    
    print norm(Pf-exact)
    
    return Pf, exact, ep
