#############################
#                           #
# Luther Michaels           # 
# ECE 351-52                #
# Lab 3                     #
# September 23, 2021        #
# Discrete Convolution      #
#                           #
#############################

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig

plt.rcParams.update({'font.size': 14}) # Set plot font size.

""" Lab 2: STEP & RAMP """
def u(t): # Step function with input variable t
    y = np.zeros(t.shape) # Initialze an array of zeros.
    for i in range(len(t)): # Run loop for each index of t.
        if t[i] < 0:
            y[i] = 0  # y = 0 for all time t < 0
        else:
            y[i] = 1  # y = 1 if t is GTE to 0
    return y

def r(t):  # Ramp function with input t
    y = np.zeros(t.shape) # Array of zeros

    for i in range(len(t)): # Loop through t.
        if t[i] < 0:
            y[i] = 0  # y = 0 for all time t < 0
        else:
            y[i] = t[i]  # y = t if t is GTE to 0
    return y

""" Part 1, Task 1: USER-DEFINED FUNCTIONS"""
def f_1(t):   # time input
	f = u(t - 2) - u(t - 9) # Function built from step function u(t).
	return f   # function return

def f_2(t):   # Exponential implemented with numpy package.
	f = np.exp(-t) * u(t) 
	return f  

def f_3(t):    # Function built using u(t) and ramp function r(t).
	f = (r(t - 2) * (u(t - 2) - u(t - 3))) + (r(4 - t) * (u(t - 3) - u(t - 4)))
	return f 

steps = 1e-2    # step size
t = np.arange(0, 20 + steps, steps)   # time interval: [0,20]

""" Part 1, Task 2: PLOT USER-DEFINED FUNCTIONS  """
y = f_1(t)
plt.figure(figsize = (10, 7))
plt.subplot(3, 1, 1)   
plt.plot(t, y)
plt.grid()
plt.ylabel('$f_1(t)$')
plt.title('Part 1, Task 2: User-Defined Functions')

y = f_2(t)
plt.subplot(3, 1, 2)   
plt.plot(t, y)
plt.grid()
plt.ylabel('$f_2(t)$')

y = f_3(t)
plt.subplot(3, 1, 3)   
plt.plot(t, y)
plt.grid()
plt.ylabel('$f_3(t)$')
plt.xlabel('t')
plt.show()

""" Part 2, Task 1: USER-DEFINED CONVOLUTION  """
def conv(f1, f2):
	Nf1 = len(f1) # Length of input functions stored in variables.
	Nf2 = len(f2)
	
	# Two arrays of equal length created from input functions.
	f1Extended = np.append(f1, np.zeros((1, Nf2 - 1)))  
	# -1 since index starts at 0
	f2Extended = np.append(f2, np.zeros((1, Nf1 - 1)))  
	# Buffer array with zeros for a size combining both functions.
	result = np.zeros(f1Extended.shape) 
	# Create a result array of same size using reference f1.
	   
	# Iterate through combined size.
	for i in range(Nf2 + Nf1 - 2): # -2 since each index starts at 0
	  # Iterate through original function size of f1.
		for j in range(Nf1): # same function as when creating result
			# condition: length of both functions is greater than first
			if((i - j + 1) > 0):   # +1 as index starts at 0
				try:
			  	# Result sum increased by product of evaluated functions.
					result[i] += f1Extended[j] * f2Extended[i - j + 1]
			  			# accumulation of area of overlap
				except:       
					print(i,j)  # error checking
	return result

t_2 = np.arange(0, (40 + (2 * steps)), steps) # expanded interval to match result

""" Part 2, Task 2: CONVOLUTION OF f_1 & f_2  """
y = conv(f_1(t), f_2(t))
plt.figure(figsize = (10, 7))
plt.subplot(2, 1, 1)
plt.plot(t_2, y)
plt.grid()
plt.title('Part 2, Task 2: $f_1(t)$ * $f_2(t)$')
plt.ylabel('User-Defined')

y = sig.convolve(f_1(t), f_2(t))
plt.subplot(2, 1, 2)
plt.plot(t_2, y)
plt.grid()
plt.ylabel('SCIPY')
plt.xlabel('t')
plt.show()

""" Part 2, Task 3: CONVOLUTION OF f_2 & f_3  """
y = conv(f_2(t), f_3(t))
plt.figure(figsize = (10, 7))
plt.subplot(2, 1, 1)
plt.plot(t_2, y)
plt.grid()
plt.title('Part 2, Task 3: $f_2(t)$ * $f_3(t)$')
plt.ylabel('User-Defined')

y = sig.convolve(f_2(t), f_3(t))
plt.subplot(2, 1, 2)
plt.plot(t_2, y)
plt.grid()
plt.ylabel('SCIPY')
plt.xlabel('t')
plt.show()

""" Part 2, Task 4: CONVOLUTION OF f_1 & f_3  """
y = conv(f_1(t), f_3(t))
plt.figure(figsize = (10, 7))
plt.subplot(2, 1, 1)
plt.plot(t_2, y)
plt.grid()
plt.title('Part 2, Task 4: $f_1(t)$ * $f_3(t)$')
plt.ylabel('User-Defined')

y = sig.convolve(f_1(t), f_3(t))
plt.subplot(2, 1, 2)
plt.plot(t_2, y)
plt.grid()
plt.ylabel('SCIPY')
plt.xlabel('t')
plt.show()