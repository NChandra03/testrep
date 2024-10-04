import numpy as np
import matplotlib.pyplot as plt

# Define the function f(x) = cos(x)
def f(x):
    return np.cos(x)

# Define the forward difference and centered difference formulas
def forward_difference(f, s, h):
    return (f(s + h) - f(s)) / h

def centered_difference(f, s, h):
    return (f(s + h) - f(s - h)) / (2 * h)

# Set the point of interest x = pi/2 and the step sizes
s = np.pi / 2
h = 0.01 * 2.0 ** (-np.arange(0, 10))

# Calculate the derivative approximations using forward and centered difference
forward_diffs = [forward_difference(f, s, h_i) for h_i in h]
centered_diffs = [centered_difference(f, s, h_i) for h_i in h]

# Calculate the errors
forward_error = 1 + np.array(forward_diffs)
centered_error = 1 + np.array(centered_diffs)

# Calculate the shifted errors
shifted_forward_error = np.roll(forward_error, 1)
shifted_centered_error = np.roll(centered_error, 1)

# Plot forward difference errors
plt.figure()
plt.loglog(forward_error[1:], shifted_forward_error[1:], label="Forward Error", marker='o')
plt.loglog(centered_error[1:], shifted_centered_error[1:], label="Centered Error", marker='x')
plt.grid(True)

# Add labels and legend
plt.xlabel('Error')
plt.ylabel('Shifted Error (by 1)')
plt.title('Log-Log Plot of Error vs. Shifted Error')
plt.legend()

# Show plot
plt.show()

