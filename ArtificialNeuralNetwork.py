import numpy as np

from HiddenLayer import HiddenLayer


class ArtificialNeuralNetwork:

    # Dimensions: Corresponds to a list containing the total number of neurons per layer, including
    # input, hidden and output
    def __init__(self, dimensions=2):
        self.layers = {}
        self.initializeNetwork(dimensions)

    def initializeNetwork(self, dimensions):
        for i, dimension in enumerate(dimensions):
            if i != len(dimensions)-1:
                self.layers[f"L{i}"] = HiddenLayer(dimensions[i+1], dimensions[i])
                continue
            break

    def trainNetwork(self, training_data, labels, alpha=0.1):
        error = []
        for i in range(1000):
            y_pred = self.feedForward(training_data)
            error.append(self.calcError(y_pred, labels))
            print(f"Error Iteration:{i} -> {error[i]}")
            self.backPropagation(training_data, labels)

    def feedForward(self, entry_data):
        res = entry_data
        for key in self.layers.keys():
            res = np.dot(np.transpose(self.layers[key].weights).T, res)
            bias = self.layers[key].bias
            res += np.reshape(bias, (len(bias), 1))
            res = self.sigmoid(res)
            self.layers[key].activations = res.copy()

        return res

    def backPropagation(self, entry_data, real_y, alpha=1):
        m = entry_data.shape[1]
        n_lay = len(self.layers) - 1
        delta = {}
        dZ = self.layers[f"L{n_lay}"].activations - real_y
        delta[f"dW{n_lay}"] = 1/m * np.dot(dZ, self.layers[f"L{n_lay-1}"].activations.T)
        delta[f"db{n_lay}"] = 1/m * np.sum(dZ, axis=1, keepdims=True)
        delta[f"dZ{n_lay}"] = dZ
        n_lay -= 1

        #Delta propagation
        while n_lay >= 0:
            term1 = 1/m * np.dot(self.layers[f"L{n_lay+1}"].weights.T, delta[f"dZ{n_lay+1}"])
            term2 = np.multiply(self.layers[f"L{n_lay}"].activations, (1-self.layers[f"L{n_lay}"].activations))
            dZ = np.multiply(term1, term2)
            if n_lay != 0:
                activations = self.layers[f"L{n_lay-1}"].activations
            else:
                activations = entry_data
            delta[f"dW{n_lay}"] = 1/m * np.dot(dZ, activations.T)
            delta[f"db{n_lay}"] = 1/m * np.sum(dZ, axis=1, keepdims=True)
            delta[f"dZ{n_lay}"] = dZ
            n_lay -= 1

        #Weight update
        for i, key in enumerate(self.layers.keys()):
            self.layers[key].weights = self.layers[key].weights - alpha * delta[f"dW{i}"]
            self.layers[key].bias = self.layers[key].bias - alpha * delta[f"db{i}"]

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def linear(self, x):
        return x

    def calcError(self, y_pred, y_real):
        error = -( np.multiply(y_real, np.log2(y_pred)) + np.multiply(1-y_real, np.log2(1 - y_pred)) )
        return np.sum(np.sum(error))

    def printNetwork(self):
        for i, key in enumerate(self.layers):
            print(f"Layer {i}", end='')
            print("-"*100)
            n_elem = len(self.layers[key].weights) + 1
            r_format = "{:<25}" * n_elem
            print(r_format.format("", *[f"Neuron #{x}" for x in range(n_elem)]))
            values = np.transpose(self.layers[key].bias)
            for j in range(len(values)):
                print(r_format.format("Bias", *values[j]))
            values = np.transpose(self.layers[key].weights)
            for j in range(len(values)):
                print(r_format.format(f"Weight {j}", *values[j]))

    def predict(self, features, labels):
        return self.feedForward(features)

    def test(self, X, Y):
        predictions = self.predict(X, Y)
        actual = np.argmax(Y, axis=0)
        pred = np.argmax(predictions, axis=0)
        match = actual == pred
        print(np.sum(match)/len(match))