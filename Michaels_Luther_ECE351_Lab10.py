#########################
#                       #
# Luther Michaels       # 
# ECE 351-52            #
# Lab 10                #
# November 11, 2021     #
# Frequency Response    #
#                       #
#########################

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
import control as con

plt.rcParams.update({'font.size': 14}) # Set plot font size.
steps = 1e-1   # step size for good resolution and reasonable timing
w = np.arange((10**3), (10**6) + steps, steps) # omega interval

def H(R, L, C):  # transfer function
    num = [0, (1 / (R * C)), 0]
    den = [1, (1 / (R * C)), (1 / (L * C))]
    return num, den

def H_magn(w, R, L, C): # magnitude
    m = (w / (R * C)) / np.sqrt((w**4) + ((w**2) * ((1 / ((R**2) * (C**2))) - (2 / (L * C)))) + (1 / ((L**2) * (C**2))))
    return m

def H_phase(w, R, L, C): # phase
    p = 90 - np.arctan((w / (R * C)) / (-(w**2) + (1/ (L * C)))) * (180 / np.pi)  # Convert to degrees.
    for i in range(len(p)): # Lower right-hand side of plot by 180.
        if(p[i] > 90):
            p[i] -= 180
    return p

""" PART 1, TASK 1 """
m = H_magn(w, 1e+3, 27e-3, 100e-9)
dB = 20 * np.log10(m) # decibels
p = H_phase(w, 1e+3, 27e-3, 100e-9)

plt.figure(figsize = (10, 8))
plt.subplot(2, 1, 1) # subplot 1
plt.semilogx(w, dB)  
plt.grid()
plt.title('Part 1, Task 1: Transfer Function H(s) - Bode Plot')
plt.ylabel('Magnitude (dB)')

plt.subplot(2, 1, 2) # subplot 2
plt.semilogx(w, p)
plt.grid()
plt.xlabel('Frequency (rad/s)')
plt.ylabel('Phase (degrees)')
plt.show()

""" PART 1, TASK 2 """
H_num, H_den = H(1e+3, 27e-3, 100e-9) # numerator & denominator
w, dB, p = sig.bode((H_num, H_den), w) # scipy bode components

plt.figure(figsize = (10, 8))
plt.subplot(2, 1, 1) # subplot 1
plt.semilogx(w, dB)  
plt.grid()
plt.title('Part 1, Task 2: Transfer Function H(s) - Bode Plot via SCIPY')
plt.ylabel('Magnitude (dB)')

plt.subplot(2, 1, 2) # subplot 2
plt.semilogx(w, p)
plt.grid()
plt.xlabel('Frequency (rad/s)')
plt.ylabel('Phase (degrees)')
plt.show()

""" PART 1, TASK 3 """
sys = con.TransferFunction(H_num, H_den) # Arrange system for transfer function.

plt.figure(figsize = (10, 8))
plt.title("Part 1, Task 3: Transfer Function Bode Plot in Hz") # title (not shown)
_ = con.bode(sys, w, dB = True, Hz = True, deg = True, plot = True) # bode plot in Hz
plt.show()

""" PART 2, TASK 1 """
fs = 10**5    # sufficient to view all 3 frequencies distinctly
steps = 1 / fs   # as defined in lab manual
t = np.arange(0, 0.01 + steps, steps)  # time interval [0, 0.01]
x = np.cos(2 * np.pi * 100 * t) + np.cos(2 * np.pi * 3024 * t) + np.sin(2 * np.pi * 50000 * t)
                                # middle cosine with middle frequency of low attenuation
plt.figure(figsize = (10, 8))
plt.subplot(1, 1, 1)
plt.plot(t, x)
plt.grid()
plt.title('Part 2, Task 1: Signal x(t)')
plt.xlabel('t (s)')
plt.ylabel('x(t)')
plt.show()

""" PART 2, TASK 2 """
z_num, z_den = sig.bilinear(H_num, H_den, fs=fs)  # digital filter in z-domain

""" PART 2, TASK 3 """
y = sig.lfilter(z_num, z_den, x)  # filtered signal output

""" PART 2, TASK 4 """
plt.figure(figsize = (10, 8))
plt.subplot(1, 1, 1)
plt.plot(t, y)
plt.grid()
plt.title('Part 2, Task 4: Filtered Signal y(t)')
plt.xlabel('t (s)')
plt.ylabel('y(t)')
plt.show()