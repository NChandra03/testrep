import numpy as np
import matplotlib.pyplot as plt

# Define the values for delta
delta_values = np.logspace(-16, 0, num=100)

# Define the functions
def lhs(x, delta):
    return np.cos(x + delta) - np.cos(x)

def rhs(x, delta):
    return -2 * np.sin(x + delta / 2) * np.sin(delta / 2)

# Values of x for the two plots
x_values = [np.pi, 10**6]

# Create plots for each x value
for x in x_values:
    lhs_values = lhs(x, delta_values)
    rhs_values = rhs(x, delta_values)
    
    difference = rhs_values - lhs_values
    
    plt.figure()
    plt.plot(delta_values, difference, label="RHS - LHS")
    plt.xscale('log')
    plt.xlabel('Delta')
    plt.ylabel('RHS - LHS')
    plt.title(f'RHS - LHS for x = {x}')
    plt.legend()
    plt.grid(True)
    plt.show()
