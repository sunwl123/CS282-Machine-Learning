import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Define the function and its gradients
def f(x, y):
    return x ** 2 - y ** 2

def grad_f(x, y):
    return 2 * x, -2 * y

# Initialize variables
x = tf.Variable(10.0)
y = tf.Variable(0.5)

# Set learning rate and epsilon for Adagrad
learning_rate = 0.1
epsilon = 25

# Initialize sum of squares of gradients
grad_squared = 0

# Track values for visualization
x_values = []
y_values = []
f_values = []

# Track if f(x, y) reaches zero
reaches_zero = False

# Start optimization
for i in range(9000):
    with tf.GradientTape() as tape:
        z = f(x, y)
        
    # Compute gradients
    dx, dy = tape.gradient(z, [x, y])
    grad_norm = tf.sqrt(dx**2 + dy**2)
    
    grad_squared += grad_norm**2
    
    # Update parameters
    x.assign_sub(learning_rate / tf.sqrt(grad_squared + epsilon) * dx)
    y.assign_sub(learning_rate / tf.sqrt(grad_squared + epsilon) * dy)
    
    # Track values
    if i % 300 == 0:
        x_values.append(x.numpy())
        y_values.append(y.numpy())
    f_value = f(x, y).numpy()
    f_values.append(f_value)
    
    if i % 100 == 0:
        print(f"Iteration {i}: x = {x.numpy()}, y = {y.numpy()}, f(x, y) = {f_value}")
    
    # Check if f(x, y) reaches zero
    if f_value == 0:
        reaches_zero = True

# Generate meshgrid for plotting
x = np.linspace(-12, 12, 100)
y = np.linspace(-12, 12, 100)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

# Plotting the optimization process
plt.figure(figsize=(16, 4))

# Plot the contour of the objective function
plt.subplot(1, 4, 1)
plt.contour(X, Y, Z, levels=20, cmap='viridis', alpha=0.8)
plt.plot(x_values, y_values, color='r', marker='o')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Optimization Process for (x, y)')

# Plot the optimization process for x
plt.subplot(1, 4, 2)
plt.plot(range(len(x_values)), x_values)
plt.xlabel('Iteration')
plt.ylabel('x')
plt.title('Optimization Process for x')

# Plot the optimization process for y
plt.subplot(1, 4, 3)
plt.plot(range(len(y_values)), y_values)
plt.xlabel('Iteration')
plt.ylabel('y')
plt.title('Optimization Process for y')

# Plot the optimization process for f(x, y)
plt.subplot(1, 4, 4)
plt.plot(range(len(f_values)), f_values)
plt.xlabel('Iteration')
plt.ylabel('f(x, y)')
plt.title('Optimization Process for f(x, y)')

plt.tight_layout()
plt.show()

# Check if f(x, y) reached zero
if reaches_zero:
    print("f(x, y) reached zero during the iteration.")
else:
    print("f(x, y) did not reach zero during the iteration.")
