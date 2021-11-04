##########################
#                        #
# Luther Michaels        # 
# ECE 351-52             #
# Lab 9                  #
# November 4, 2021       #
# Fast Fourier Transform #
#                        #
##########################

import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack as sft

plt.rcParams.update({'font.size': 14}) # Set plot font size.
steps = 1e-2    # step size for good resolution

""" Lab 8 """
def b(k):  # function for b_k with argument k
    b = (2 / (k * np.pi)) * (1 - np.cos(k * np.pi)) 
    return b

def x_series(t, T, N):  # function for x(t) with arguments t, T, and N
    x = 0
    for i in range(1, N + 1):  # k in range [1,N]
        x += b(i) * np.sin((i * 2 * np.pi * t) / T) # summation
    return x

""" Compute the Fast Fourier Transform """
def fft(x, fs, clean):     # function and sampling frequency
    N = len(x)           # Find the length of the signal.
    X_fft = sft.fft(x)   # Perform the fast Fourier transform (fft).
    X_fft_shifted = sft.fftshift(X_fft)  # Shift zero frequency components
                                         # to the center of the spectrum.
    freq = np.arange(-N/2, N/2) * fs/N  # Compute the frequencies 
                                        # for the output signal.
    X_mag = np.abs(X_fft_shifted) / N  # Compute the magnitudes of the signal.
    X_phi = np.angle(X_fft_shifted)   # Compute the phases of the signal.
    
    if(clean): # option to clean phase
        for i in range(len(X_phi)):  # Iterate through phase array.
            if(X_mag[i] < 1e-10):   # Identify minute magnitudes.
                X_phi[i] = 0       # Eliminate phase at this index.
    return freq, X_mag, X_phi

""" Create 5-Subplot Graph of the Fast Fourier Transform """
def plot_fft(x, clean, x_size, y_size, title, x_lim, t):
    freq, X_mag, X_phi = fft(x, 100, clean) # Compute FFT and parameters.

    plt.figure(figsize = (x_size, y_size)) # Set overall plot size.
    plt.subplot(3, 1, 1)  # top subplot
    plt.plot(t, x)
    plt.grid()
    plt.title(title)  # title
    plt.xlabel('t[s]')
    plt.ylabel('x(t)')

    plt.subplot(3, 2, 3)   # middle left subplot
    plt.stem(freq, X_mag)
    plt.grid()
    plt.ylabel('|X(f)|')

    plt.subplot(3, 2, 4)  # middle right subplot
    plt.stem(freq, X_mag)
    plt.xlim(-x_lim, x_lim)  # Limit x-axis.
    plt.grid()

    plt.subplot(3, 2, 5)  # bottom left subplot 
    plt.stem(freq , X_phi)
    plt.grid()
    plt.xlabel('f[Hz]')
    plt.ylabel('/_ X(f)')

    plt.subplot(3, 2, 6)  # bottom right subplot
    plt.stem(freq, X_phi)
    plt.xlim(-x_lim, x_lim)  # Limit x-axis.
    plt.grid()
    plt.xlabel('f[Hz]')
    plt.show()    

t = np.arange(0, 2, steps)  # time interval [0,2]

""" Task 1 """
x = np.cos(2 * np.pi * t)
title = "Task 1: User-Defined FFT of x(t)"
plot_fft(x, 0, 10, 9, title, 2, t) # Plot with defined function and title.

""" Task 2 """
x = 5 * np.sin(2 * np.pi * t)
title = "Task 2: User-Defined FFT of x(t)"
plot_fft(x, 0, 10, 9, title, 2, t)

""" Task 3 """
x = (2 * np.cos((2 * np.pi * 2 * t) - 2)) + (np.sin((2 * np.pi * 6 * t) + 3) ** 2)
title = "Task 3: User-Defined FFT of x(t)"
plot_fft(x, 0, 12, 13, title, 15, t)

""" Task 4 """
x = np.cos(2 * np.pi * t)
title = "Task 4: Clean User-Defined FFT of x(t) from Task 1"
plot_fft(x, 1, 10, 12, title, 2, t)  # clean

x = 5 * np.sin(2 * np.pi * t)
title = "Task 4: Clean User-Defined FFT of x(t) from Task 2"
plot_fft(x, 1, 10, 9, title, 2, t)  # clean

x = (2 * np.cos((2 * np.pi * 2 * t) - 2)) + (np.sin((2 * np.pi * 6 * t) + 3) ** 2)
title = "Task 4: Clean User-Defined FFT of x(t) from Task 3"
plot_fft(x, 1, 12, 13, title, 15, t) # clean

""" Task 5 """
t = np.arange(0, 16, steps)  # time interval [0,16]

x = x_series(t, 8, 15) # Call series function with T = 8 and N = 15.
title = "Task 5: Clean User-Defined FFT of x(t) from Square Wave in Lab 8"
plot_fft(x, 1, 12, 11, title, 2, t) # clean