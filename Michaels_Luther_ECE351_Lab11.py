##############################
#                            #
# Luther Michaels            # 
# ECE 351-52                 #
# Lab 11                     #
# November 18, 2021          #
# Z - Transform Operations   #
#                            #
##############################

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
import scipy.signal as sig

# Copyright (c) 2011 Christopher Felton (Lines 16-81)

# Modified by Drew Owens in Fall 2018 for use in the University of Idaho’s
# Department of Electrical and Computer Engineering Signals and Systems I Lab
# (ECE 351)
#
# Modified by Morteza Soltani in Spring 2019 for use in the ECE 351 of the U of I.
#
# Modified by Phillip Hagen in Fall 2019 for use in the University of Idaho’s
# Department of Electrical and Computer Engineering Signals and Systems I Lab
# (ECE 351)

""" Plot the complex z-plane given a transfer function """
def zplane(b, a, filename = None):

    # get a figure/plot
    ax = plt.subplot(1, 1, 1)

    # create the unit circle
    uc=patches.Circle ((0 ,0),radius=1,fill=False ,color='black',ls='dashed')
    ax.add_patch(uc)

    # the coefficients are less than 1, normalize the coefficients
    if np.max(b) > 1:
        kn = np.max(b)
        b = np.array(b)/float(kn)
    else:
        kn = 1

    if np.max(a) > 1:
        kd = np.max(a)
        a = np.array(a)/float(kd)
    else:
        kd = 1

    # get the poles and zeros
    p = np.roots(a)
    z = np.roots(b)
    k = kn/float(kd)

    # plot the zeros and set marker properties
    t1 = plt.plot(z.real , z.imag , 'o', ms=10,label='Zeros')
    plt.setp(t1 , markersize =10.0, markeredgewidth =1.0)

    # plot the poles and set marker properties
    t2 = plt.plot(p.real , p.imag , 'x', ms=10,label='Poles')
    plt.setp( t2 , markersize =12.0, markeredgewidth =3.0)

    ax.spines['left']. set_position('center')
    ax.spines['bottom']. set_position('center')
    ax.spines['right']. set_visible(False)
    ax.spines['top']. set_visible(False)

    plt.legend ()

    # set the ticks

    # r = 1.5; plt.axis(’scaled ’); plt.axis([-r, r, -r, r])
    # ticks = [-1, -.5, .5, 1]; plt.xticks(ticks); plt.yticks(ticks)

    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)

    return z, p, k

""" TASK 3 """
num = [2, -40, 0]   # transfer function numerator
den = [1, -10, 16]                  # denominator
r, p, k = sig.residuez(num, den)
print('RESIDUES:', r, '\nPOLES: ', p, '\nCOEFFICIENTS: ', k, '\n')

""" TASK 4 """
num = [2, -40]  # Update transfer function for division by z.
zplane(num, den)  # Plot the pole-zero graph.

""" TASK 5 """
w, h = sig.freqz(num, den, whole=True)  # Obtain the magnitude and phase components.
magn = 20 * np.log10(np.abs(h)) # Convert magnitude to decibels.
phase = np.angle(h)  # Isolate phase.

plt.figure(figsize = (10, 8))
plt.subplot(2, 1, 1) # subplot 1
plt.plot(w, magn) # magnitude
plt.grid()
plt.title('Task 5: Frequency Response Plot')
plt.ylabel('Magnitude (dB)')

plt.subplot(2, 1, 2) # subplot 2
plt.plot(w, phase)  # phase
plt.grid()
plt.xlabel('Frequency (rad/s)')
plt.ylabel('Phase (rad)')
plt.show()