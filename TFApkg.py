import numpy as np

def ctft(t,x):
    '''
    Computes the 'continuous' time Fourier
    transform using the inbuilt fft function
    to return the complex spectrum with fftshift
    and the range of frequencies

    INPUT:  Time support, t
            Signal, x
    OUTPUT: Frequency support, w
            Complex Fourier transform, x_hat

    Author: Abijith J Kamath
    kamath-abhijith.github.io
    '''
    dt = t[1]-t[0]
    
    x_hat = np.fft.fft(x)
    w = np.fft.fftfreq(x.size)*2*np.pi/dt
    x_hat *= dt*np.exp(-1j*w*t[0])/(np.sqrt(2*np.pi))

    return w, x_hat

def signal_moment(t,x,n=1,*argv):
    '''
    Computes the nth moment over the
    density formed by the signal x
    assuming it has unit norm

    INPUT:  Support vector, t
            Signal, x
            Order, n
    OUTPUT: nth moment

    Author: Abijith J Kamath
    kamath-abhijith.github.io
    '''
    if len(argv)>0:
    	g = argv[0]
    else:
    	g = t

    dt = t[1]-t[0]
    e_density = np.abs(x)**2
    weights = g**n

    return np.dot(weights,e_density)*dt

def signal_spread(t,x):
    '''
    Computes the spread of the signal
    x assuming it has unit norm

    INPUT:  Support vector, t
            Signal, x
    OUTPUT: Spread

    Author: Abijith J Kamath
    kamath-abhijith.github.io
    '''
    _mean = signal_moment(t,x)
    _var = signal_moment(t,x,2)

    return _var - _mean**2

def inst_frequency(t,x):
    '''
    Computes the instantaneous
    frequencies of the signal x

    INPUT:  Support vector, t
            Signal, x
    OUTPUT: Shrunk time support, t
            Instantaneous frequency, inst_freq

    Author: Abijith J Kamath
    kamath-abhijith.github.io
    '''
    dt = t[1]-t[0]
    N = len(t)
    t = t[0:N-1]
    
    inst_phase = np.unwrap(np.angle(x))
    inst_freq = np.diff(inst_phase)/(dt)
    
    return t, inst_freq

def group_delay(t,x):
    w,x_hat = ctft(t,x)
    dw = w[1]-w[0]
    N = len(w)
    w = w[0:N-1]

    phase_spectra = np.unwrap(np.angle(x_hat))
    group_delay = -1*np.diff(phase_spectra)/(dw)

    return w, group_delay

def signal_covariance(t,x):
    N = len(x)
    w,x_hat = ctft(t,x)
    td,inst_freq = inst_frequency(t,x)
    
    t_mean = signal_moment(t,x,1)
    w_mean = signal_moment(w,x_hat,1)
    cov = signal_moment(t,x[0:N-1],1,td*inst_freq)
    return cov - t_mean*w_mean