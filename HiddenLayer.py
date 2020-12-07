import numpy as np


class HiddenLayer:

    def __init__(self, neurons_current, neurons_prev):
        self.weights = np.random.rand(neurons_current, neurons_prev) * 2 - 1
        self.bias = np.random.rand(neurons_current, 1) - 1
        self.activations = None
