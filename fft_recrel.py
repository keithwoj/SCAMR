from pylab import *
from scipy.fftpack import *
'''
********************************************************************************

********************** RECIPROCITY RELATIONS EXAMPLE ***************************

 This script provides an example for deriving the reciprocity relations
 for the discrete Fourier transform (DFT). The reciprocity relations
 describe the relationship among the grid parameters that define the
 spatial (temporal) and frequency grids of a function and its DFT
 respectively.

 The relationship is as follows, define:
       Number of grid points to be N
       Spatial (temporal) grid: [-L/2, L/2], grid spacing dx = L/N
       Frequency grid: [-W/2, W/2], grid spacing dw = W/N
       ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        Reciprocity relations: L*W = N,  dx*dw = 1/N
       ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 So given N grid points, the reciprocity relations describe the
 relationship that exists among L, W, dx, and dw. Knowing these relations
 permits one to construct the frequency grid for the DFT.

 In this example the periodic function considered is u(x) = sin(2*pi*w0*x)
 whose Fourier transform is ut(w0) = [delta(w-w0) - delta(w+w0)]/2i. So we
 expect to see two spikes at -w0 and w0 in the frequency domain. If we
 derived the reciprocity relations correctly, then we will see these two
 spikes are their correct locations.

 First Working Version 7 JAN 2014, Keith J. Wojciechowski (KJW)
********************************************************************************
'''
# Consider a spatial (temporal) grid and a sine function that has an integer 
# number of periods on [-L/2, L/2] with dx = L/N, i.e. x_n = n*dx.
L = 2                          # Spatial (temporal) grid length
w0 = 5                         # INTEGER number of periods
# Number of grid points (try N > 2*w0*L) where N = 2^p for some integer p.
N = 2**(min(where(2**arange(11)>2*w0*L)[0])+1)
xx = linspace(-L/2.,L/2.,N+1)    # Spatial (temporal) grid
x = xx[0:N]; dx = float(L)/N   # Remove last point due to periodicity
u = sin(2*pi*w0*x)             # Sine function considered

# *************************************************************************
# Derive the reciprocity relations for the DFT:
# *************************************************************************
# One of the keys to deriving (and understanding) the reciprocity relations
# is in determining the number of waves of u(x) contained in [-L/2, L/2].
# u(x) = sin(2*pi*w0*x) has wavelength (period) 1/w0, so waves occur with
# spacing 1/w0, i.e. -L/2 + 1/w0, -L/2 + 2*1/w0, ...
# *************************************************************************
wv = arange(-L/2.,L/2.,1./w0)           # waves contained within [-L/2, L/2]
# Display the function with its waves marked by red circles.
figure(1), plot(x,u,'bo-',wv,0*wv,'ro')
axis([-L/2., L/2., -1.1, 1.1])
xlabel('$x$'); ylabel('$u(x) = sin(2\pi\omega_0 x)$')
title('$\omega_0 = $'+str(w0)+', $L = $'+ str(L))
# *************************************************************************
# So the number of waves contained in [-L/2, L/2] is len(wv)
# Count the gaps between the red circles in figure(1) and notice that one wave
# fits between each gap.
# Trigonometry tells us that there are L bins of w0 waves contained 
# within [-L/2, L/2], 
# e.g. if w0 = 6 and L = 3, then there are 3 bins of 6 waves.
# e.g. if w0 = 7 and L = 3, then there are 3 bins of 7 waves.
# e.g. if w0 = 5 and L = 4, then there are 4 bins of 5 waves.
# That is, there are L*w0 waves contained within [-L/2, L/2]
nwaves = L*w0;
# The number of bins provides the natural step size for the frequency
# domain. One way to understand this idea is to think of the number of
# waves per bin as "part" of the "whole" number of waves contained within
# the interval [-L/2, L/2] => frequency domain grid spacing is denoted dw
# => "part"/"whole" = dw = w0/(L*w0) = 1/L
dw = 1./L
# Recall that dw = W/N, so the grid for the frequency domain has length
# W = N*dw = N/L and spans from -W/2 to W/2 with N grid points.
w = N/float(L)
kk = linspace(-w/2.,w/2.,N+1)
k = kk[0:N]
# *************************************************************************
# Alternative versions of the frequency grid (N even):
# *************************************************************************
# k = (1/L)*(-N/2:N/2-1);
# kk = (1/L)*[0:(N/2-1) -N/2:-1]; k = fftshift(kk);
# kk = (1/L)*ifftshift(-N/2:N/2-1); k = fftshift(kk);
# *************************************************************************
# Compute the DFT (via FFT)
ut = fftshift(fft(u));
freq_apprx = max(k[max(abs(ut)) == abs(ut)]);
# Display the result and notice the spike at w (approximately)
figure(2), plot(k,abs(ut)/max(abs(ut)),'r',k,0*k,'ko')
axis([min(k),max(k),-0.1,1.1])
xlabel('$\omega$'); ylabel('Power Spectrum, $|FFT(u)|$')
title('Spikes occur at $|\omega_0| =$ ' + str(freq_apprx))
show()