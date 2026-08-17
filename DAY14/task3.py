import numpy as np

inputs = np.array([2, 4, 6])

weights = np.array([0.5, 0.2, 0.8])

bias = 1

weighted_sum = np.sum(inputs * weights) + bias

print("Weighted Sum:", weighted_sum)

output = max(0, weighted_sum)

print("Neuron Output:", output)
