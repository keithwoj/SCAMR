# -*- coding: utf-8 -*-
"""
Module for implementing exponential time-differencing (ETD) schemes
Created on Wed Jul 15 15:40:50 2015

@author: Keith J. Wojciechowski (KJW)

The functions for this module adopt the following convention

y, t = SOLVER(F,y0,tv,L,exp_comp,D,history)

Input:
        F -- Function to be integrated dy/dt = F(y,t)
        y0 -- initial value
        tv -- array [t0, tf] or [t0,t1,t2,...,tf] where tk = desired time value
        L -- coefficient matrix for linear part
                            (** y' = Ly **)        
        Optional:
                exp_comp -- method for computing matrix exponential
                            (** default is expm -> scaling & squaring **)         
                D -- array of coefficient matrices for nonlinear part
                            (** default is array(null[]))                           
                history = True to record the evolution of the solution
                            (** default is False **)                            
        
Output:
        y -- numerical solution as an n-size array
        t -- vector of time steps, linspace(t0,tf,nstep)
        
        If time history is output, then the array is shaped
        [rows = time vector length, cols = space data length]
        
Notes:
        nstep = int((tf-t0)/h)+1 since h = (tf - t0)/(nstep-1)
        
Edits:
        15 JUL 2015:
        First working version hle (KJW)
        Unit test added, dy/dt = -2*t*y, h = 0.1, 0.01, 0.001 (KJW)
        Vectorized, i.e. y can be a vector so solvers can be used to solve
        time evolution PDEs (KJW)
        
Solvers:
        y, t = hle(F,y0,tv,L,exp_comp,D,history)
        etd.test?? to see the unit test and an example
"""

from pylab import *
import tools

def linexint(F,y0,t0,tf,h,L,exp_comp = "expm",history = False):
    from scipy.linalg import expm
    # Initialize the number of iterations and the array containing the solution
    # Note: h = (tf - t0)/(nstep-1)
    nstep = int(float(tf-t0)/h)+1
    tv = linspace(t0,tf,nstep)      # Temporal grid
    ny = size(y0)                   # Size of y-data
    # Record the evolution of the solution? If yes, initialize a storage array
    if history:
        yy = zeros([nstep,ny])
    # Set the solution as an array
    # e.g. Solving a PDE on a spatial grid requires y(x,t) to be a vector
    #       [y(x[0],t), y(x[1],t), ..., y(x[N],t)] entry for each time-step
    if ny > 1:
        y = y0*ones(ny)           # Initialize y and force convert to float
        vector_out = True         # solution is vectorized in space
    else:
        y = array([y0*1.])        # Initialize y and force convert to float
        vector_out = False        # Solution will not be vectorized in space
        
    # Step through the integrator
    for j in xrange(nstep):
        y = dot(expm(L*h),y)
    
    return y
        
def testf(u):
    return dot(L,u)
    
def test():
    import diffmat
    
    N = 16
    x = pi*(cos(pi*linspace(0,N,N)/N)+1)/2
    L = diffmat.diff2lag(x)
    u0 = sin(x)
    u = linexint(testf,u0,0,1,1e-3,0.1*L,exp_comp = "expm",history = False)
    
    show(plot(x,u0,'r',x,u,'b'))
    