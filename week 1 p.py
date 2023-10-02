import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
 return 1/(1 + np.exp(-x))


x = np.linspace(-10, 10, 50)
p = sigmoid(x)
plt.xlabel("x")
plt.ylabel("Sigmoid(x)")
plt.plot(x, p)
plt.show()

def der_sigmoid(x):
  return sigmoid(x) * (1- sigmoid(x))

x = np.linspace(-10, 10, 50)
p = der_sigmoid(x)
plt.xlabel("x")
plt.ylabel("Sigmoid(x)")
plt.plot(x, p)
plt.show()

def relu(x):
  return np.maximum(0, x)

x = np.linspace(-10, 10, 50)
p = relu(x)
plt.xlabel("x")
plt.ylabel("RelU(x)")
plt.plot(x, p)
plt.show()

def tanh(x):
    return np.tanh(x)

x = np.linspace(-10, 10, 50)
p = tanh(x)
plt.xlabel("x")
plt.ylabel("tanh(x)")
plt.plot(x, p)
plt.show()

def softmax(x):
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)

x = np.linspace(-10, 10, 50)
p = softmax(x)
plt.xlabel("x")
plt.ylabel("softmax(x)")
plt.plot(x, p)
plt.show()

