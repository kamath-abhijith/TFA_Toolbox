import numpy as np

from matplotlib import pyplot as plt
from matplotlib import style
from matplotlib import rcParams

from TFApkg import *

## Support
dt = 0.01
t = np.arange(-5,5,dt)

## Quadratic phase with Gaussian envelope
# Signal Parameters
a = 1
b = 4
w0 = 2

# Time-domain representation
s = (a/np.pi)**(1/4)*np.exp(-a*t**2/2 + 1j*b*t**2/2 + 1j*w0*t)

# Frequency=domain representation
s_hat = np.fft.fft(s)
w = np.fft.fftfreq(s.size)*2*np.pi/dt
s_hat *= dt*np.exp(-1j*w*t[0])/(np.sqrt(2*np.pi))

## Temporal averages
t_mean = np.mean(s)
duration = np.var(s)
print('Temporal mean and duration: ',t_mean,duration)

## Spectral averages
w_mean = signal_moment(w,s_hat,1)
bandwidth = signal_spread(w,s_hat)
print('Spectral mean and bandwidth: ',w_mean,bandwidth)

## Plots
style.use('ggplot')
style.use('dark_background')

rcParams['text.usetex'] = True

fig, plts = plt.subplots(2,2)
plts[0][0].plot(t,s.real)
plts[0][0].set_ylabel(r'$\Re(s)$')
plts[0][0].set_xlabel(r'time (sec)')
plts[1][0].plot(t,s.imag)
plts[1][0].set_ylabel(r'$\Im(s)$')
plts[1][0].set_xlabel(r'time (sec)')

plts[0][1].plot(w,s_hat.real)
plts[0][1].scatter(w_mean,0,color='red')
plts[0][1].set_ylabel(r'$\Re(\hat{s})$')
plts[0][1].set_xlabel(r'Frequency (Hz)')
plts[0][1].set_xlim((-15,15))
plts[1][1].plot(w,s_hat.imag)
plts[1][1].scatter(w_mean,0,color='red')
plts[1][1].set_ylabel(r'$\Im(\hat{s})$')
plts[1][1].set_xlabel(r'Frequency (Hz)')
plts[1][1].set_xlim((-15,15))
plt.show()