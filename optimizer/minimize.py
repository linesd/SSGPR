import numpy as np
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) # suppress np.sqrt Runtime warnings

def minimize(f, X, length, args=(), reduction=None, verbose=True):
	"""
	Minimize a differentiable multivariate function.

	Parameters
	----------
	f : function to minimize. The function must return the value
		of the function (float) and a numpy array of partial
		derivatives of shape (D,) with respect to X, where D is
		the dimensionality of the function.

	X : numpy array - Shape : (D, 1)
		initial guess.

	length : int
		The length of the run. If positive, length gives the maximum
		number of line searches, if negative its absolute value gives
		the maximum number of function evaluations.

	args : tuple
		Tuple of parameters to be passed to the function f.

	reduction : float
		The expected reduction in the function value in the first
		linesearch (if None, defaults to 1.0)

	verbose : bool
		If True - prints the progress of minimize. (default is True)

	Return
	------
	Xs : numpy array - Shape : (D, 1)
		The found solution.

	convergence : numpy array - Shape : (i, D+1)
		Convergence information. The first column is the function values
		returned by the function being minimized. The next D columns are
		the guesses of X during the minimization process.

	i : int
		Number of line searches or function evaluations depending on which
		was selected.

	The function returns when either its length is up, or if no further progress
 	can be made (ie, we are at a (local) minimum, or so close that due to
 	numerical problems, we cannot get any closer)

 	Copyright (C) 2001 - 2006 by Carl Edward Rasmussen (2006-09-08).
 	Coverted to python by David Lines (2019-23-08)
	"""
	INT = 0.1 		# don't reevaluate within 0.1 of the limit of the current bracket
	EXT = 3.0		# extrapolate maximum 3 times the current step size
	MAX = 20		# max 20 function evaluations per line search
	RATIO = 10		# maximum allowed slope ratio
	SIG = 0.1
	RHO = SIG / 2
	# SIG and RHO control the Wolfe-Powell conditions
	# SIG is the maximum allowed absolute ratio between
	# previous and new slopes (derivatives in the search direction), thus setting
	# SIG to low (positive) values forces higher precision in the line-searches.
	# RHO is the minimum allowed fraction of the expected (from the slope at the
	# initial point in the linesearch). Constants must satisfy 0 < RHO < SIG < 1.
	# Tuning of SIG (depending on the nature of the function to be optimized) may
	# speed up the minimization; it is probably not worth playing much with RHO.

	print("Minimizing %s ..." % f)

	if reduction is None:
		red = 1.0
	else:
		red = reduction

	S = 'Linesearch' if length > 0 else 'Function evaluation'

	i = 0 								# run length counter
	is_failed = 0						# no previous line search has failed
	f0, df0 = eval('f')(X, *list(args))			# get initial function value and gradient
	df0 = df0.reshape(-1, 1)
	fX=[]; fX.append(f0)
	Xd=[]; Xd.append(X)
	i += (length < 0)					# count epochs
	s = -df0							# get column vec
	d0 = -s.T @ s 						# initial search direction (steepest) and slope
	x3 = red / (1 - d0)					# initial step is red/(|s|+1)

	while i < abs(length):				# while not finished
		i += (length > 0)				# count iterations

		X0 = X; F0 = f0; dF0 = df0 		# copy current vals
		M = MAX if length > 0 else min(MAX, -length-i)

		while 1:						# extrapolate as long as necessary
			x2 = 0; f2 = f0; d2 = d0; f3 = f0; df3 = df0
			success = False

			while not success and M > 0:
				try:
					M -= 1; i += (length < 0) # count epochs
					f3, df3 = eval('f')(X + x3 * s, *list(args))
					df3 = df3.reshape(-1, 1)
					if np.isnan(f3) or np.isinf(f3) or any(np.isnan(df3)+np.isinf(df3)):
						raise Exception('Either nan or inf in function eval or gradients')
					success = True
				except:	# catch any error occuring in f
					x3 = (x2 + x3) / 2 	# bisect and try again

			if f3 < F0:
				X0 = X + x3 * s; F0 = f3; dF0 = df3 # keep best values

			d3 = df3.T @ s 				# new slope
			if d3 > SIG*d0 or f3 > f0+x3*RHO*d0 or M ==0:
				break					# finished extrapolating

			x1 = x2; f1 = f2; d1 = d2 	# move point 2 to point 1
			x2 = x3; f2 = f3; d2 = d3   # move point 3 to point 2
			A = 6*(f1-f2)+3*(d2+d1)*(x2-x1) # make cubic extrapolation
			B = 3*(f2-f1)-(2*d1+d2)*(x2-x1)
			x3 = x1-d1*(x2-x1)**2/(B+np.sqrt(B*B-A*d1*(x2-x1))) # num. error possible, ok!

			if np.iscomplex(x3) or np.isnan(x3) or np.isinf(x3) or x3 < 0: # num prob | wrong sign
				x3 = x2*EXT
			elif x3 > x2*EXT:
				x3 = x2*EXT
			elif x3 < x2+INT*(x2-x1):
				x3 = x2+INT*(x2-x1)

		while (abs(d3) > -SIG*d0 or f3 > f0+x3*RHO*d0) and M > 0: # keep interpolating

			if d3 > 0 or f3 > f0+x3*RHO*d0:                       # choose subinterval
				x4 = x3; f4 = f3; d4 = d3                         # move point 3 to point 4
			else:
				x2 = x3; f2 = f3; d2 = d3                         # move point 3 to point 2

			if f4 > f0:
				x3 = x2-(0.5*d2*(x4-x2)**2)/(f4-f2-d2*(x4-x2))     # quadratic interpolation
			else:
				A = 6*(f2-f4)/(x4-x2)+3*(d4+d2)            		  # cubic interpolation
				B = 3*(f4-f2)-(2*d2+d4)*(x4-x2)
				x3 = x2+(np.sqrt(B*B-A*d2*(x4-x2)**2)-B)/A         # num. error possible, ok!

			if np.isnan(x3) or np.isinf(x3):
				x3 = (x2+x4)/2               					  # if we had a numerical problem then bisect

			x3 = max(min(x3, x4-INT*(x4-x2)), x2+INT*(x4-x2))     # don't accept too close
			f3, df3 = eval('f')(X + x3 * s, *list(args))
			df3 = df3.reshape(-1,1)
			if f3 < F0:
				X0 = X+x3*s; F0 = f3; dF0 = df3				      # keep best values

			M -= 1; i += (length<0)                               # count epochs?!
			d3 = df3.T @ s                                        # new slope

		if abs(d3) < -SIG*d0 and f3 < f0+x3*RHO*d0:        		  # if line search succeeded
			X = X+x3*s; f0 = f3; fX.append(f0); Xd.append(X)				      # update variables
			if verbose:
				print('%s %6i;  Value %4.6e\r' % (S, i, f0))
			s = (df3.T@df3-df0.T@df3)/(df0.T@df0)*s - df3       # Polack-Ribiere CG direction
			df0 = df3                                             # swap derivatives
			d3 = d0; d0 = df0.T@s
			if d0 > 0:                                    		  # new slope must be negative
				s = -df0.reshape(-1,1); d0 = -s.T@s          					  # otherwise use steepest direction
			x3 = x3 * min(RATIO, d3/(d0-np.finfo(np.double).tiny))# slope ratio but max RATIO
			ls_failed = False                                     # this line search did not fail
		else:
			X = X0; f0 = F0; df0 = dF0                           # restore best point so far
			if ls_failed or i > abs(length):                       # line search failed twice in a row
				break                                             # or we ran out of time, so we give up
			s = -df0.reshape(-1,1); d0 = -s.T@s                                 # try steepest
			x3 = 1/(1-d0)
			ls_failed = True                                     # this line search failed

	convergence =  np.hstack((np.array(fX).reshape(-1,1), np.array(Xd)[:,:,0])) # bundle test_data info
	Xs = X # solution

	return Xs, convergence, i