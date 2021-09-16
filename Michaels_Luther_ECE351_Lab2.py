###############################
#                             #
# Luther Michaels             # 
# ECE 351-52                  #
# Lab 2                       #
# September 16, 2021          #
# User-Defined Functions      #
#                             #
###############################

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 14}) # Set plot font size.

""" Part 1, Task 2: COSINE """
steps = 1e-2    # Define step size for all subsequent parts.

def func1(t): # Cosine function with variable t sent
    y = np.cos(t)
    return y # Return y assigned the numpy cosine function.

t = np.arange(0, 10 + steps, steps) # Define time interval from 0 to 10s.
y = func1(t)  # Function call

plt.figure(figsize = (10, 7))   # Plot sizing
plt.subplot(2, 1, 1)
plt.plot(t, y)   # Plot declaration
plt.grid()       # Display a grid on the plot.
plt.title('Part 1: Task 2')  # Plot title
plt.ylabel('y(t) = cos(t)')  # y-axis label
plt.xlabel('t')   # x-axis label
plt.show()    # Required at least once to actually show the plot.

""" Part 2, Task 2: STEP & RAMP """
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

t = np.arange(-1, 1 + steps , steps) # Change the time interval (-1,1).
y = u(t) # Step function call

plt.figure(figsize = (10, 7))
plt.subplot(2, 1, 1) # Use small plots so that both are in a single graph.
plt.plot(t, y)
plt.grid()
plt.ylabel('u(t) - Step Function')
plt.title('Part 2: Task 2')

t = np.arange(-1, 1 + steps , steps) # Ensure proper time interval (-1,1).
y = r(t)

plt.subplot(2, 1, 2)  # Indicate a subplot.
plt.plot(t, y)
plt.grid()
plt.ylabel('r(t) - Ramp Function')
plt.xlabel('t')  # Include a single x-axis label
plt.show()

""" Part 2, Task 3: HANDWRITTEN EQUATION"""
def hand_eqt(t):  # Handwritten equation with time input
    y = r(t) - r(t - 3) + (5 * u(t - 3)) - (2 * u(t - 6)) - (2 * r(t - 6)) 
    return y  

t = np.arange(-5, 10 + steps , steps)  # Change time interval (-5,10).
y = hand_eqt(t) # Call the handwritten equation function.

plt.figure(figsize = (10, 7))
plt.subplot(1, 1, 1)  # Utilize a tall plot.
plt.plot(t, y)
plt.grid()
plt.ylabel('y(t) - Handwritten Equation')
plt.title('Part 2: Task 3')
plt.xlabel('t')
plt.show()

""" Part 3, Task 1: TIME REVERSAL """
t = np.arange(-10, 5 + steps, steps) # Change time interval (-10,5) 
                                     # to view reverse image.
y = hand_eqt(-t) # Call the handwritten function with -t input.

plt.figure(figsize = (10, 7))
plt.subplot(1, 1, 1)   # Tall figure
plt.plot(t, y)
plt.grid()
plt.ylabel('y(-t) - Time Reversal')
plt.title('Part 3: Task 1')
plt.xlabel('t')
plt.show()

""" Part 3, Task 2: TIME SHIFT """
t = np.arange(0, 14 + steps, steps) # Change time interval (0,14) 
                                    # to view right shift image.
y = hand_eqt(t - 4) # Call function with 4s right shift input.

plt.figure(figsize = (10, 7))
plt.subplot(2, 1, 1)  # Small to include 2 plots
plt.plot(t, y)
plt.grid()
plt.ylabel('y(t - 4) - Time-Shift 1')
plt.title('Part 3: Task 2')

t = np.arange(-14, 0 + steps, steps) # Change time interval (-14,0) 
                                     # to view left shift with reversal.
y = hand_eqt((-t) - 4) # Call function with 4s left shift 
                       # and time reversal input.
plt.subplot(2, 1, 2)  # Subplot
plt.plot(t, y)
plt.grid()
plt.ylabel('y(-t - 4) - Time-Shift 2')
plt.xlabel('t')  # Single x-axis label
plt.show()

""" Part 3, Task 3: TIME SCALE """
t = np.arange(-8, 20 + steps, steps) # Change time interval (-8,20) 
                                     # to view time scale expansion.
y = hand_eqt(t/2) # Call function with decreased scale input.

plt.figure(figsize = (10, 7))
plt.subplot(2, 1, 1)
plt.plot(t, y)
plt.grid()
plt.ylabel('y(t/2) - Time Scale 1')
plt.title('Part 3: Task 3')

t = np.arange(-2, 5 + steps, steps) # Change time interval (-2,5) 
                                     # to view time scale contraction.
y = hand_eqt(2 * t) # Call function with increased scale input.

plt.subplot(2, 1, 2) # Subplot
plt.plot(t, y)
plt.grid()
plt.ylabel('y(2t) - Time Scale 2')
plt.xlabel('t') # x-axis label
plt.show()

""" Part 3, Task 3: DIFFERENTIATION """ 
def deriv(t): # General derivative function with time input
    y = np.zeros(t.shape)  # Assign zeroes to a numpy array.
    dy = np.diff(hand_eqt(t))  # Calculate the differences between
                               # consecutive handwritten function outputs.
    dt = np.diff(t) # Calculate changes in time.
    for i in range(len(t) - 1):  # Loop through all t except last value.
        y[i] = dy[i] / dt[i]  # The derivative is given by the change in
                              # Function output divided by change in time.
    return y

t = np.arange(-5, 10 + steps, steps) # Change time interval (-5,10) 
                                     # to view derivative.
y = deriv(t)  # Call derivative function.

plt.figure(figsize = (10, 7))
plt.subplot(1, 1, 1)
plt.plot(t, y)
plt.grid()
plt.ylabel('y\'(t) - Derivative')
plt.title('Part 3: Task 5')
plt.xlabel('t')
plt.axis([-0.5,10,-15,15])  # Change y-axis minimum and maximum
                            # In order to better view derivative.
plt.show()