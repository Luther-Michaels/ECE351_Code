################################
#                              #
# Luther Michaels              # 
# ECE 351-52                   #
# Lab 8                        #
# October 28, 2021             #
# Fourier Series Approximation #
#             of a Square Wave #
#                              #
################################

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 14}) # Set plot font size.
steps = 1e-5    # step size for good resolution
t = np.arange(0, 20 + steps, steps)  # time interval [0,20]
T = 8  # global assignment

""" Part 1, Task 1: Print Series Terms """
a = 0  # Set all a_k to zero since odd function.

def b(k):  # function for b_k with argument k
    b = (2 / (k * np.pi)) * (1 - np.cos(k * np.pi)) 
    return b

def x(t, N):  # function for x(t) with arguments t and N
    x = 0  
    for i in range(1, N + 1):  # k in range [1,N]
        x += b(i) * np.sin((i * 2 * np.pi * t) / T) # summation
    return x

print("a_0: ", a, "\na_1: ", a, "\nb_1: ", b(1), "\nb_2: ", b(2), "\nb_3: ", b(3), "\n")

""" Part 1, Task 2: Plot Series Approximations """  
y = x(t, 1)  # N = 1
plt.figure(figsize = (10, 7))
plt.subplot(3, 1, 1) # subplot 1
plt.plot(t, y)  
plt.grid()
plt.title('Part 1, Task 2: Fourier Series Approximation x(t)')
plt.ylabel('N = 1')

y = x(t, 3)  # N = 3
plt.subplot(3, 1, 2) # subplot 2
plt.plot(t, y)  
plt.grid()
plt.ylabel('N = 3')

y = x(t, 15)  # N = 15
plt.subplot(3, 1, 3) # subplot 3
plt.plot(t, y)  
plt.grid()
plt.xlabel('t')
plt.ylabel('N = 15')
plt.show()

y = x(t, 50)  # N = 50
plt.figure(figsize = (10, 7))
plt.subplot(3, 1, 1) # subplot 1
plt.plot(t, y)  
plt.grid()
plt.title('Part 1, Task 2: Fourier Series Approximation x(t)')
plt.ylabel('N = 50')

y = x(t, 150)  # N = 150
plt.subplot(3, 1, 2) # subplot 2
plt.plot(t, y)  
plt.grid()
plt.ylabel('N = 150')

y = x(t, 1500)  # N = 1500
plt.subplot(3, 1, 3) # subplot 3
plt.plot(t, y)  
plt.grid()
plt.xlabel('t')
plt.ylabel('N = 1500')
plt.show()