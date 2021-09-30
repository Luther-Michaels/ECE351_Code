#############################
#                           #
# Luther Michaels           # 
# ECE 351-52                #
# Lab 4                     #
# September 30, 2021        #
# System Step Response      #
#    Using Convolution      #
#                           #
#############################

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 14}) # Set plot font size.
steps = 1e-2    # step size
    
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

""" Lab 3: USER-DEFINED CONVOLUTION """
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

t_2 = np.arange(-20, (20 + (0.5 * steps)), steps) # expanded interval to match result
               # Interval must be doubled across both directions.

""" Part 1, Task 1: USER-DEFINED FUNCTIONS """
def h_1(t):
    h = np.exp(-2 * t) * (u(t) - u(t - 3))  # exponential
    return h

def h_2(t):
    h = u(t - 2) - u(t - 6)
    return h

def h_3(t):
    w = 2 * np.pi * 0.25
    h = np.cos(w * t) * u(t)  # cosine
    return h

t = np.arange(-10, 10 + steps, steps)   # time interval: [-10,10]

""" Part 1, Task 2: PLOT USER-DEFINED FUNCTIONS """
y = h_1(t)
plt.figure(figsize = (10, 7))
plt.subplot(3, 1, 1)    # single figure
plt.plot(t, y)
plt.grid()
plt.ylabel('$h_1(t)$')
plt.title('Part 1, Task 2: User-Defined Functions')

y = h_2(t)
plt.subplot(3, 1, 2)   
plt.plot(t, y)
plt.grid()
plt.ylabel('$h_2(t)$')

y = h_3(t)
plt.subplot(3, 1, 3)   
plt.plot(t, y)
plt.grid()
plt.ylabel('$h_3(t)$')
plt.xlabel('t')
plt.show()

""" Part 2, Task 1: PLOT STEP RESPONSE """
y = conv(h_1(t), u(t))
plt.figure(figsize = (10, 7))
plt.subplot(3, 1, 1)   # single figure
plt.plot(t_2, y)
plt.grid()
plt.ylabel('$h_1(t)$ * u(t)')
plt.title('Part 2, Task 1: Step Response ')

y = conv(h_2(t), u(t))
plt.subplot(3, 1, 2)   
plt.plot(t_2, y)
plt.grid()
plt.ylabel('$h_2(t)$ * u(t)')

y = conv(h_3(t), u(t))
plt.subplot(3, 1, 3)   
plt.plot(t_2, y)
plt.grid()
plt.ylabel('$h_3(t)$ * u(t)')
plt.xlabel('t')
plt.show()

""" Part 2, Task 2: PLOT HAND-CALCULATED STEP RESPONSE """
def conv_h_1(t):
    conv_h = (-0.5 * u(t) * ((np.exp(-2 * t)) - 1)) + (0.5 * u(t - 3) * ((np.exp(-2 * (t - 3))) - 1))
    return conv_h   # exponentials

def conv_h_2(t):
    conv_h = ((t - 2) * u(t - 2)) - ((t - 6) * u(t - 6)) 
    return conv_h

def conv_h_3(t):
    w = 2 * np.pi * 0.25
    conv_h = (1 / w) * np.sin(w * t)  * u(t)  # sine
    return conv_h

y = conv_h_1(t_2)
plt.figure(figsize = (10, 7))
plt.subplot(3, 1, 1)  # single figure
plt.plot(t_2, y)
plt.grid()
plt.ylabel('$h_1(t)$ * u(t)')
plt.title('Part 2, Task 2: Hand-Calculated Step Response ')

y = conv_h_2(t_2)
plt.subplot(3, 1, 2)        
plt.plot(t_2, y)
plt.grid()
plt.ylabel('$h_2(t)$ * u(t)')

y = conv_h_3(t_2)
plt.subplot(3, 1, 3)   
plt.plot(t_2, y)
plt.grid()
plt.ylabel('$h_3(t)$ * u(t)')
plt.xlabel('t')
plt.show()