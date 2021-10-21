##########################################
#                                        #
# Luther Michaels                        # 
# ECE 351-52                             #
# Lab 7                                  #
# October 21, 2021                       #
# Block Diagrams and System Stability    #
#                                        #
##########################################

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig

plt.rcParams.update({'font.size': 14})   # Set plot font size.
steps = 1e-5    # step size for good resolution
t = np.arange(0, 2 + steps, steps)   # time interval between 0 and 2s

""" Part 1, Task 1: G(s), A(s), and B(s) in Factored Form """
num_G = [1, 9]  # numerator
den_G = sig.convolve([1, -6, -16], [0, 1, 4])  # expanded denominator
R_G, P_G, K_G = sig.residue(num_G, den_G)    # partial fraction expansion
print("G(s)\nR: ", R_G, "\nP: ", P_G, "\n")

num_A = [0, 1, 4]
den_A = [1, 4, 3]
R_A, P_A, K_A = sig.residue(num_A, den_A)   # partial fraction expansion
print("A(s)\nR: ", R_A, "\nP: ", P_A, "\n")

print("B(s)\nR: [-14. -12.] \n")  # result of arithmetic factorization

""" Part 1, Task 2: Check Results of Part1, Task 1 """
Z_g, P_g, K_g = sig.tf2zpk(num_G, den_G)   # poles & zeros
print("G(s)\nZ: ", Z_g, "\nP: ", P_g, "\nK: ", K_g, "\n")

Z_a, P_a, K_a = sig.tf2zpk(num_A, den_A)   # poles & zeros
print("A(s)\nZ: ", Z_a, "\nP: ", P_a, "\nK: ", K_a, "\n")

B = [1, 26, 168]   # roots/zeros
R_B = np.roots(B)
print("B(s)\nZ: ", R_B, "\n")

""" Part 1, Task 3: PLOT OPEN-LOOP TRANSFER FUNCTION """
num = sig.convolve([1, 9], [1, 4])  # expanded numerator
den = sig.convolve([1, -2, -40, -64], [0, 1, 4, 3])  # expanded denominator

R, P, K = sig.residue(num, den)    # partial fraction expansion
print("H_O(s)\nR: ", R, "\nP: ", P, "\n")

""" Part 1, Task 5: PLOT OPEN-LOOP TRANSFER FUNCTION """
tout, yout = sig.step((num, den), T = t)   # Plot the open-loop response.
plt.figure(figsize = (10, 7))
plt.subplot(1, 1, 1)
plt.plot(tout, yout)
plt.grid()
plt.title('Part 1, Task 5: Open-Loop Transfer Function')
plt.xlabel('t')
plt.ylabel('$ H_O(t) $')
plt.show()

""" PART 2, TASK 2: CLOSED-LOOP TRANSFER FUNCTION """
num = sig.convolve([1, 9], [1, 4])    # expanded numerator
term = sig.convolve([0, 1, 9],[1, 26, 168])  # expanded term in right factor
factor = [1 + term[1], -2 + term[2], -40 + term[3], -64 + term[4]] # total of right factor
den = sig.convolve([0, 1, 4, 3], factor)  # resultant expanded denominator
print("H_C(s)\nNUM: ", num, "\nDEN: ", den, "\n")

""" PART 2, TASK 4: PLOT CLOSED-LOOP TRANSFER FUNCTION """
tout, yout = sig.step((num, den), T = t)  # Plot the closed-loop response.
plt.figure(figsize = (10, 7))
plt.subplot(1, 1, 1)
plt.plot(tout, yout)
plt.grid()
plt.title('Part 2, Task 4: Closed-Loop Transfer Function')
plt.xlabel('t')
plt.ylabel('$ H_C(t) $')
plt.show()