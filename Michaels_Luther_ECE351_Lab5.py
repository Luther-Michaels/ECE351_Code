#################################
#                               #
# Luther Michaels               # 
# ECE 351-52                    #
# Lab 5                         #
# October 7, 2021               #
# Step and Impulse Response     #
# of a RLC Band Pass Filter     #
#                               #
#################################

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig

plt.rcParams.update({'font.size': 14}) # Set plot font size.
steps = 1e-5    # diminished step size
t = np.arange(0, 1.2e-3 + steps, steps)   # time interval between 0 and 1.2 ms
    
""" Lab 2: STEP """
def u(t): # Step function with input variable t
    y = np.zeros(t.shape) # Initialze an array of zeros.
    for i in range(len(t)): # Run loop for each index of t.
        if t[i] < 0:
            y[i] = 0  # y = 0 for all time t < 0
        else:
            y[i] = 1  # y = 1 if t is GTE to 0
    return y

R = 1000   # Initialize RLC circuit parameters as variables outside functions.
L = 27e-3
C = 100e-9
    
""" Part 1, Task 1: PLOT HAND-CALCULATED IMPULSE RESPONSE """
def h(t,R,L,C):   # additional R,L,C inputs
    alpha = (-1 / (2 * R * C))
    omega = (1 / 2) * np.sqrt((1 / (R * C)) ** 2 - (4 * ((1/np.sqrt(L * C)) ** 2)) + 0 * 1j)
    p = alpha + omega
    g = (1 / (R * C)) * p
    g_mag = np.abs(g)        # magnitude
    g_phase = np.angle(g)    # phase
    h = (((g_mag) / np.abs(omega)) * np.exp(alpha * t) * np.sin((np.abs(omega) * t) + g_phase)) * u(t)
    return h

y = h(t,R,L,C)
plt.figure(figsize = (10, 7))
plt.subplot(2, 1, 1)    # subplot 1
plt.plot(t, y)
plt.grid()
plt.title('Part 1: Impulse Response')
plt.ylabel('Hand-Calculated $ h(t) $')

""" Part 1, Task 2: PLOT IMPULSE RESPONSE USING SCIPY ON TRANSFER FUNCTION """
num = [0, (1 / (R * C)), 0]                 # transfer function numerator
den = [1, (1 / (R * C)), (1 / (L * C))]     # transfer function denominator

tout, yout = sig.impulse((num, den), T = t)    # Use scipy to find the impulse response.
plt.subplot(2, 1, 2)  # subplot 2
plt.plot(tout, yout)  # output of scipy impulse
plt.grid()
plt.ylabel('$ h(t) $ using SCIPY')
plt.xlabel('t')
plt.show()

""" Part 2, Task 1: PLOT STEP RESPONSE OF H(s) USING SCIPY"""
tout, yout = sig.step((num, den), T = t)   # Use scipy to find the step response.
plt.figure(figsize = (10, 7))    # tout on same interval as t
plt.subplot(1, 1, 1)
plt.plot(tout, yout)   # output of scipy step
plt.grid()
plt.ylabel('$ h(t)  * u(t) $')
plt.xlabel('t')
plt.title('Part 2, Task 1: Step Response using SCIPY ')
plt.show()