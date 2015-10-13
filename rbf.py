# -*- coding: utf-8 -*-
"""
Created on Fri Oct 09 12:58:24 2015

@author: Keith J. Wojciechowski (KJW)
"""

from pylab import array, dot, exp, linalg, linspace, norm, ones, shape, sqrt, tile, zeros

def dmatrix(d,c):
    # d = data, c = centers
    #
    # Typically d = c but, in general, data does not have to equal its centers
    #
    # Compute the distance matrix with entries being the distances between the
    # data and the centers.
    # The distance matrix is the m by n matrix with entries
    #      ||d_0 - c_0|| ||d_0 - c_1|| ... ||d_0 - c_n||
    #      ||d_1 - c_0|| ||d_1 - c_1|| ... ||d_1 - c_n||
    #                       ...
    #      ||d_m - c_0|| ||d_m - c_1|| ... ||d_m - c_n||
    #
    # m = # pts, n = dim of space
    #
    # ****** ASSUMPTION: # pts >= dimension of space
    # ****** ASSUMPTION: c, d are ROW vectors, otherwise convert them
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
        M, sd = shape(d)    # M = # pts, sd = dim of data space
    else:   # 1-D data
        M = d.size
        sd = 1
    if c.ndim > 1:
        if c.shape[1] > c.shape[0]:
            c = c.T
        elif c.shape[1] == c.shape[0]:
            print("Assuming centers are in row-vector form.")
        N, sc = shape(c)    # N = # pts, sc = dim of centers space
    else:   # 1-D data
        N = c.size
        sc = 1
        
    if sd != sc:
        raise NameError('Data and centers must be same dimension')
    # Initialize the distance matrix: (data # of pts) by (centers # of pts)
    #
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
        # Notes on tile and summing by vector multiplication:
        #   (1) d[i,:] is 1 by sd
        #   (2) tile(d[i,:],(N,1)) is N by sd
        #   (3) (N,1) means tile N by 1 copies of d[i,:]
        #   (4) c is N by sc ** sc = sd otherwise ERROR
        #   (5) tile(d[i,:],(N,1)) and c are same shape => may be subtracted
        #   (6) sq dist(d_i - c_0:n) = (tile(d[i,:],(N,1)) - c)**2
        #   (7) sum each (d_i - c_0)**2, (d_i - c_1)**2, etc.
        #   i.e. ||d_i - c_j|| = sum(d_i - c_j)**2 where sum is over dims
        #   (8) summing by dot product with vector of [1, 1, ..., 1] (dim sc)
        v = dot((tile(d[i,:],(N,1))-c)**2,ones((sc,1)))
        # c and d are row vectors so these operations where completed across
        # the columns (dimension of data and centers)
        # hence, this temp vector needs to be transposed. In other words,
        # v is a column vector:
        #      ||d_i - c_0||
        #      ||d_i - c_1||
        #           ...        
        #      ||d_i - c_n||
        #
        # But DM[i,j] = ||d_i - c_j||, in other words
        #     DM[i,:] = ||d_i - c_0|| ||d_i - c_1|| ... ||d_i - c_n||
        DM[i,:] = v.T
        # Finish distance formula by taking square root of each entry
    return sqrt(DM)
    
def testfunction(x):
    return exp(-15*(x-0.5)**2)
    
def test_dmatrix():
    x = linspace(0,1,19)
    ep = linspace(0.01,0.99,32)    
    data = array([x]).transpose()
    #data = array([x,x]).transpose()
    #data = array([x,x,x]).transpose()
    ctrs = data
    IM = dmatrix(data,ctrs)
    EM = dmatrix(ep,ctrs)
    
    rhs = testfunction(x)
    
    Pf = EM* linalg.solve(IM,rhs)
    exact = testfunction(ep)
    
    print norm(Pf-exact)