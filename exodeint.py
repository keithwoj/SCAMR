# -*- coding: utf-8 -*-
"""
Module for implementing ODE integrators
Created on Thu Jun 18 15:56:18 2015

@author: Keith J. Wojciechowski (KJW)

The functions for this module adopt the following convention

y, t = SOLVER(F,y0,t0,tf,h)
y, t = SOLVER(F,y0,t0,tf,h,history = False)

Input:
        F -- Function to be integrated dy/dt = F(y,t)
        y0 -- initial value
        t0 -- initial time
        tf -- final time
        h -- time-step
        
        Optional:
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
        19 JUN 2015:
        First working version RK4e (KJW)
        Unit test added, dy/dt = -2*t*y, h = 0.1, 0.01, 0.001 (KJW)
        Vectorized, i.e. y can be a vector so RK4e can be used to solve
        time evolution PDEs (KJW)
        
Solvers:
        y, t = RK4e(F,y0,t0,tf,h,history)
        exodeint.test?? to see the unit test and an example
"""

from pylab import *
import tools

# Conventional RK4 Solver using evenly-space nodes
def RK4e(F,y0,t0,tf,h,history = False):
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
        if history:
            yy[j,:] = y
        k1 = F(y,tv[j])                   # Euler's method
        k2 = F(y+0.5*h*k1,tv[j]+0.5*h)    # Slope at midpoint
        k3 = F(y+0.5*h*k2,tv[j]+0.5*h)    # Slope at midpoint
        k4 = F(y+h*k3,tv[j]+h)            # Slope at next time-step
        # Integrate (Simpson's method) to obtain solution at next time step
        y += (k1 + 2*(k2 + k3) + k4)*h/6.
    # If saving the history, return the recorded values
    if history:
        if vector_out:
            return yy, tv
        else:
            return yy[:,0], tv      # 1D solution returned as 1D array
    else:
        return y

# ODE Test Function, dy/dt = F(y,t)
def testf(y,t):
    return -2.*t*y

def test():
    t0 = -3
    tf = 3
    y0 = sqrt(pi)/2*exp(-t0**2)
    hvec = array([0.1, 0.01, 0.001])
    print "Scalar Test Results:\n"
    for h in hvec:
        y, tv = RK4e(testf,y0,t0,tf,h,True)
        ytrue = sqrt(pi)/2*exp(-tv**2)
        delta = tools.normd(y-ytrue,tv,'even')
        print("Discrete L2 error is %20.15e for h = %2.1e") % (delta, h)
    y0 = array([1, 2, 3])*sqrt(pi)/2*exp(-t0**2)
    delta = zeros(3)
    print "\nVector Test Results:\n"
    for h in hvec:
        y, tv = RK4e(testf,y0,t0,tf,h,True)
        ytrue = sqrt(pi)/2*exp(-tv**2)
        delta[0] = tools.normd(y[:,0]-ytrue,tv,'even')
        delta[1] = tools.normd(y[:,1]-2*ytrue,tv,'even')
        delta[2] = tools.normd(y[:,2]-3*ytrue,tv,'even')
        print("Discrete L2 error is\n %20.15e, %20.15e, %20.15e\nfor h = %2.1e") % (delta[0],delta[1],delta[2], h)

if __name__ == "__main__":
    
    print "Running Tests"
    test()