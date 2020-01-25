import numpy as np

from matplotlib import pyplot as plt
from matplotlib import style
from matplotlib import rcParams

from TFApkg import *

## Support
dt = 0.01
t = np.arange(-10,10,dt)

## Cubic phase with Gaussian envelope
# Signal Parameters
a = 1
b = 4
c = 0
w0 = 2

# Time-domain representation
s = (a/np.pi)**(1/4)*np.exp(-a*t**2/2 + 1j*c*t**3/3 + 1j*b*t**2/2 + 1j*w0*t)

print("Signal norm: ",np.linalg.norm(s))

# Frequency domain representation
w,s_hat = ctft(t,s)

## Global averages
t_mean = signal_moment(t,s)
duration = signal_spread(t,s)	

w_mean = signal_moment(w,s_hat)
bandwidth = signal_spread(w,s_hat)

print('Temporal mean and duration: ',t_mean,duration)
print('Spectral mean and bandwidth: ',w_mean,bandwidth)

## Instantaneous frequency and group delay
td, wi = inst_frequency(t,s)
wd, tg = group_delay(t,s)

## Covariance
cov_s = signal_covariance(t,s)
print('Signal covariance: ',cov_s)

## Plots
style.use('ggplot')
style.use('dark_background')

rcParams['text.usetex'] = True

fig, plts = plt.subplots(2,2)
plts[0][0].plot(t,s.real)
plts[0][0].scatter(t_mean,0,color='red')
plts[0][0].set_ylabel(r'$\Re(s)$')
plts[0][0].set_xlabel(r'time (sec)')
plts[1][0].plot(t,s.imag)
plts[1][0].scatter(t_mean,0,color='red')
plts[1][0].set_ylabel(r'$\Im(s)$')
plts[1][0].set_xlabel(r'time (sec)')

plts[0][1].plot(w,s_hat.real)
plts[0][1].scatter(w_mean,0,color='red')
plts[0][1].set_ylabel(r'$\Re(\hat{s})$')
plts[0][1].set_xlabel(r'Frequency (Hz)')
plts[0][1].set_xlim((-30,30))
plts[1][1].plot(w,s_hat.imag)
plts[1][1].scatter(w_mean,0,color='red')
plts[1][1].set_ylabel(r'$\Im(\hat{s})$')
plts[1][1].set_xlabel(r'Frequency (Hz)')
plts[1][1].set_xlim((-30,30))
fig.suptitle('Time and Frequency Descriptions')

plt.figure()
plt.plot(td,wi,'-',color='red',label='Estimate')
plt.plot(t,w0 + b*t + c*t**2,'--',color='green',label='True')
plt.ylabel(r'$\omega_i(t)$')
plt.xlabel(r'time (sec)')
plt.title('Instantaneous Frequency')
plt.legend()

# plt.figure()
# plt.plot(wd,tg,'-',color='red',label='Estimate')
# plt.plot(t,(b/(a**2+b**2))*(w-w0),'--',color='green',label='True')
# plt.ylabel(r'$t_g(\omega)$')
# plt.xlabel(r'Frequency (Hz)')
# plt.title('Group Delay')
# plt.legend()
plt.show()