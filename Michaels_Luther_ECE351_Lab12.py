#############################
#                           #
# Luther Michaels           # 
# ECE 351-52                #
# Lab 12 - Final Project    #
# December 9, 2021          #
# Filter Design             #
#                           #
#############################

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
import scipy.fftpack as sft
import pandas as pd
import control as con

""" FUNCTION DEFINITIONS """
# Enable faster function plotting.
def make_stem(ax, x, y, color='k', style='solid', label='', linewidths=2.5, **kwargs):
    ax.axhline(x[0], x[-1], 0, color='r')
    ax.vlines(x, 0, y, color=color, linestyles=style, label=label, linewidths=linewidths)
    ax.set_ylim ([1.05*y.min(), 1.05*y.max()])
    
# Plot signal.
def plot_sig(time, signal, title):
    fig, ax = plt.subplots(figsize = (10, 7))
    make_stem(ax, time, signal)
    plt.grid()
    plt.title(title)
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude [V]')
    plt.show()

# Fast Fourier Transform
def fft(x, fs):     # function and sampling frequency
    N = len(x)           # Find the length of the signal.
    X_fft = sft.fft(x)   # Perform the fast Fourier transform (fft).
    X_fft_shifted = sft.fftshift(X_fft)  # Shift zero frequency components
                                         # to the center of the spectrum.
    freq = np.arange(-N/2, N/2) * fs/N  # Compute the frequencies 
                                        # for the output signal.
    X_mag = np.abs(X_fft_shifted) / N  # Compute the magnitudes of the signal.
    return freq, X_mag

# Plot FFT.
def plot_fft(frequency, magnitude, title):
    fig, ax = plt.subplots(figsize = (10, 7))   # Plot entire range.
    make_stem(ax, freq, magnitude)
    plt.grid()
    plt.title(title + 'Magnitudes and Corresponding Frequencies')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Magnitude')

    # Plot Low Frequency & Switching Amplifier range.
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize = (10, 10))
    plt.subplot(ax1)
    make_stem(ax1, freq, magnitude)
    plt.grid()
    plt.title(title + 'Magnitudes and Corresponding Frequencies - Low Frequency & Switching Amplifier')
    plt.xlim(0, 1.795e+3)
    plt.ylabel('Magnitude - Low Frequency')

    plt.subplot(ax2)
    make_stem(ax2, freq, magnitude)
    plt.grid()
    plt.xlim(2.005e+3, 100e+3)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel(title + 'Magnitude - Switching Amplifier') 

    # Plot Position Measurement Information.
    fig, ax = plt.subplots(figsize = (10, 7))
    make_stem(ax, freq, magnitude)
    plt.grid()
    plt.title(title + 'Magnitudes and Corresponding Frequencies - Position Measurement Information ')
    plt.xlim(1.795e+3, 2.005e+3)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Magnitude')

    # Plot High Frequency Range.
    fig, ax = plt.subplots(figsize = (10, 7))
    plt.subplot(ax)
    make_stem(ax, freq, magnitude)
    plt.grid()
    plt.title(title + 'Magnitudes and Corresponding Frequencies - High Frequency')
    plt.xlim(100e+3, 500e+3)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Magnitude')
    plt.show()

# Plot Bode.
def plot_bode(freq_Hz, magn_dB, phase_deg, xflag, xlim1, xlim2, title):   
    plt.figure(figsize = (10, 7))  
    plt.subplot(2, 1, 1)
    plt.semilogx(freq_Hz, magn_dB, color='black')  # Plot in black on log scale.
    plt.grid()
    if(xflag):
        plt.xlim(xlim1, xlim2)   # Select Range for Bode Plot.
    plt.title(title)
    plt.ylabel('Magnitude (dB)')

    plt.subplot(2, 1, 2)
    plt.semilogx(freq_Hz, phase_deg, color='black')
    plt.grid()
    if(xflag):
        plt.xlim(xlim1, xlim2)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Phase (deg))')
    plt.show()

""" SETUP """
df = pd.read_csv('NoisySignal.csv')  # Load input signal.
t = df['0']. values    # Assign values from input file.
sensor_sig = df['1'].values

plot_sig(t, sensor_sig, 'Noisy Input Signal')  # Plot input signal.

""" TASK 1: IDENTIFY MAGNITUDES AND CORRESPONDING FREQUENCIES """
freq, magn = fft(sensor_sig, 1e+6)   # Compute FFT and parameters.
plot_fft(freq, magn, 'Unfiltered ')  # Plot FFT over select ranges.

""" TASK 2: DESIGN ANALOG FILTER CIRCUIT """
# Filter component values
R = 2.13e+3
L = 70.3e-3
C = 100e-9

# RLC Bandpass Filter Design
num = [0, (1 / (R * C)), 0]
den = [1, (1 / (R * C)), (1 / (L * C))]

""" TASK 3: GENERATE BODE PLOT """
sys = con.TransferFunction(num, den)  # Generate Bode parameters.
mag, phase, omega = con.bode(sys, freq, dB = True, Hz = True, deg = True, plot = False) # bode plot in Hz

# Convert parameters.
dB = 20 * np.log10(mag)
deg = phase * (180 / np.pi)
Hz = omega / (2 * np.pi)

# Test filter requirements.
x_prev = 0
y_prev = 0

for x,y in zip(Hz, dB):   # Find frequency ranges for required dB values.
    if x >= 0 and x < len(dB):   # Does not function for any filter general case.
        if(y >= -0.3 and y_prev < -0.3):
            three_dB_start = x
        elif(y < -0.3 and y_prev >= -0.3):
            three_dB_end = x_prev
        elif(y > -21 and y_prev <= -21):
            tw_one_dB_start = x_prev    
        elif(y <= -21 and y_prev > -21):
            tw_one_dB_end = x
        elif(y > -30 and y_prev <= -30):
            thirty_dB_start = x_prev
        elif(y <= -30 and y_prev > -30):
            thirty_dB_end = x
    x_prev = x
    y_prev = y
    
# Print intervals.
print('-0.3 dB: [', three_dB_start, ',', three_dB_end, ']')
print('-21 dB: [ 0 ,', tw_one_dB_start, ']')
print('\t\t[', tw_one_dB_end, ', 100000 ]')
print('-30 dB: [ 0 ,', thirty_dB_start, ']')
print('\t\t[', thirty_dB_end, ', 100000 ]')

# Produce Bode plots over relevant frequency ranges.
plot_bode(Hz, dB, deg, 0, 0, 0, 'Filter Bode Plot')  
plot_bode(Hz, dB, deg, 1, 0, 1.795e+3, 'Bode Plot - Low Frequency')
plot_bode(Hz, dB, deg, 1, 1.795e+3, 2.005e+3, 'Bode Plot - Position Measurement Information')
plot_bode(Hz, dB, deg, 1, 2.005e+3, 100e+3, 'Bode Plot - Switching Amplifier')
plot_bode(Hz, dB, deg, 1, 100e+3, 500e+3, 'Bode Plot - High Frequency')

""" TASK 4: FILTER NOISY SENSOR SIGNAL """
z_num, z_den = sig.bilinear(num, den, fs=1e+6)  # Convert filter to digital in Z-domain.
filtered_sig = sig.lfilter(z_num, z_den, sensor_sig)  # Obtain filtered signal.

plot_sig(t, filtered_sig, 'Filtered Signal')  # Plot filtered signal.
freq, magn = fft(filtered_sig, 1e+6)    # Compute FFT and parameters for filtered signal.
plot_fft(freq, magn, 'Filtered ')   # Plot filtered FFT over select ranges.