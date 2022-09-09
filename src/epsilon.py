"""

    Small utility file to compute epsilon values based on n epochs and
    exploration % needed

    For this project, I found an exploration rate of approx. 66% to work
    the best.

"""
import numpy as np

eps = 1.0
eps_min = 0.05
eps_decay = 0.9991
epochs = 5000

for i in range(epochs):
    eps *= eps_decay
    if eps <= eps_min:
        stop = i
        break

print("With this parameter you will stop epsilon decay after {}% of training".format(stop/epochs*100))
