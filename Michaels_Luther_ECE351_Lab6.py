#################################
#                               #
# Luther Michaels               # 
# ECE 351-52                    #
# Lab 6                         #
# October 14, 2021              #
# Partial Fraction Expansion    #
#                               #
#################################

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig

plt.rcParams.update({'font.size': 14}) # Set plot font size.
steps = 1e-5    # step size for good resolution
t = np.arange(0, 2 + steps, steps)   # time interval between 0 and 2s
    
""" Lab 2: STEP """
def u(t): # Step function with input variable t
    y = np.zeros(t.shape) # Initialze an array of zeros.
    for i in range(len(t)): # Run loop for each index of t.
        if t[i] < 0:
            y[i] = 0  # y = 0 for all time t < 0
        else:
            y[i] = 1  # y = 1 if t is GTE to 0
    return y
    
""" Part 1, Task 1: PLOT STEP RESPONSE y(t) """
def y(t):
    y = (0.5 - (0.5 * np.exp(-4 * t)) + np.exp(-6 * t)) * u(t)
    return y

y = y(t)
plt.figure(figsize = (10, 7))
plt.subplot(2, 1, 1)    # subplot 1
plt.plot(t, y)
plt.grid()
plt.title('Part 1, Tasks 1 & 2: Step Response $ y_1(t) $')
plt.ylabel('Hand-Calculated $ y(t) $')

""" Part 1, Task 2: PLOT IMPULSE RESPONSE USING SCIPY ON TRANSFER FUNCTION """
num = [1, 6, 12]    # numerator of H_1(s)
den = [1, 10, 24]   # denominator
tout, yout = sig.step((num, den), T = t)  # step response

plt.subplot(2, 1, 2)    # subplot 2
plt.plot(tout, yout)
plt.grid()
plt.xlabel('t')
plt.ylabel('$ y(t) $ via SCIPY')
plt.show()

""" Part 1, Task 3: PARTIAL FRACTION EXPANSION WITH SCIPY """
num = [0, 1, 6, 12]   # numerator of Y_1(s) with X(s) = u(s)
den = [1, 10, 24, 0]  # denominator
R, P, K = sig.residue(num, den)   # residue
print("R: ", R, "\nP: ", P, "\nK: ", K, "\n")  # Print residue results.

""" PART 2, TASK 1: PARTIAL FRACTION EXPANSION ON HIGH ORDER DIFFERENTIAL """
num = [0, 0, 0, 0, 0, 0, 25250]    # numerator of Y_2(s) with X(s) = u(s)
den = [1, 18, 218, 2036, 9085, 25250, 0]  # denominator
R, P, K = sig.residue(num, den)  # residue
print("R: ", R, "\nP: ", P, "\nK: ", K)  # Print residue results.

""" PART 2, TASK 2: PLOT TIME-DOMAIN STEP RESPONSE """
def s_resp(t):
    y = 0  # summation initialization
    for i in range(len(R)):   # length of residue arrays
        y += (np.abs(R[i]) * np.exp(np.real(P[i]) * t) * np.cos((np.imag(P[i]) * t) + np.angle(R[i])) * u(t))
    return y

t = np.arange(0, 4.5 + steps, steps)  # new time interval [0, 4.5]

y = s_resp(t)
plt.figure(figsize = (10, 7))
plt.subplot(1, 1, 1)
plt.plot(t, y)
plt.grid()
plt.title('Part 2, Task 2: Step Response $ y_2(t) $ derived from Residue Results')
plt.xlabel('t')
plt.ylabel('$ y_2(t) $ with Cosine Method')
plt.ylim(-0.05, 1.22)  # Adjust y-axis limits to match subsequent plot.
plt.show()

""" PART 2, TASK 3: CHECK TIME-DOMAIN RESPONSE with H(s) and SCIPY """
num = [0, 0, 0, 0, 0, 25250]  # numerator of H_2(s)
den = [1, 18, 218, 2036, 9085, 25250]  # denominator
tout, yout = sig.step((num, den), T = t)  # step response

plt.figure(figsize = (10, 7))
plt.subplot(1, 1, 1)
plt.plot(tout, yout)
plt.grid()
plt.title('Part 2, Task 3: Step Response $ y_2(t) $ derived from $ H_2(s) $')
plt.xlabel('t')
plt.ylabel('$ y_2(t) $ via SCIPY')
plt.show()