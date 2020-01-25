import numpy as np

from matplotlib import pyplot as plt
from matplotlib import style
from matplotlib import rcParams

from scipy.signal import hilbert

from TFApkg import *

## Time support
dt = 1e-4
t = np.arange(-5,5,dt)

## Time domain definition
a = 10
b = 8
c = 0
w0 = 15
s = (1/np.pi)**(1/4)*np.exp(-a*t**2/2)*np.sin(c*t**3/3 + b*t**2/2 + w0*t)

## Analytic signal
sa = hilbert(s)
Hs = np.imag(sa)

ws, s_hat = ctft(t,s)
wsa, sa_hat = ctft(t,sa)

## Global averages
t_mean = signal_moment(t,s)
duration = signal_spread(t,s)

## Plots
style.use('ggplot')
style.use('dark_background')

rcParams['text.usetex'] = True

fig, plts = plt.subplots(2,sharex=True)
plts[0].plot(t,s)
plts[0].set_xlabel(r'time')
plts[0].set_ylabel(r'$s(t)$')
plts[0].set_title('Real-valued Signal')
plts[1].plot(t,Hs)
plts[1].set_xlabel(r'time')
plts[1].set_ylabel(r'$\mathcal{H}s(t)$')
plts[1].set_title('Hilbert Transform')

fig, plts = plt.subplots(2,sharex=True)
plts[0].plot(ws,np.abs(s_hat))
plts[0].scatter(ws_mean,0,color='red')
plts[0].set_ylabel(r'$\Re(\hat{s})$')
plts[0].set_xlabel(r'Frequency (Hz)')
plts[0].set_title(r'Fourier Transform of Real Signal')
plts[1].plot(wsa,np.abs(sa_hat))
plts[1].scatter(wsa_mean,0,color='red')
plts[1].set_ylabel(r'$\Re(\hat{s}_a)$')
plts[1].set_xlabel(r'Frequency (Hz)')
plts[1].set_title(r'Fourier Transform of Analytic Signal')
plts[1].set_xlim((-50,50))
plt.show()