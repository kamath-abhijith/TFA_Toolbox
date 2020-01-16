import numpy as np

def signal_moment(t,x,n):
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
	dt = t[1]-t[0]
	e_density = np.abs(x)**2
	weights = t**n

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
	_mean = signal_moment(t,x,1)
	_var = signal_moment(t,x,2)

	return _var - _mean**2