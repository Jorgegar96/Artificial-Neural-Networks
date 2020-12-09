import numpy as np


class HiddenLayer:

    def __init__(self, neurons_current=None, neurons_prev=None, layer=None):
        if layer is None:
            self.weights = np.random.rand(neurons_current, neurons_prev) * 2 - 1
            self.bias = np.random.rand(neurons_current, 1) - 1
        else:
            self.weights = []
            self.bias = []
            for neuron in layer:
                for weights in layer[neuron]:
                    self.bias.append([weights['pesos'][0]])
                    self.weights.append(weights['pesos'][1:])
        self.activations = None
